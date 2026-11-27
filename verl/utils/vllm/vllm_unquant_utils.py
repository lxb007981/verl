# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Refit support for unquantized fused-MoE layers whose kernel prep changes the weight layout.

vLLM hands the unquantized expert weights to a backend-specific prep in
``process_weights_after_loading``. FlashInfer TRT-LLM -- vLLM's default bf16
MoE backend on SM100 -- rewrites ``w13_weight`` / ``w2_weight`` into a 4-D block
layout (``[E, 2H/128, 2I, 64]`` for w13, where the checkpoint layout is
``[E, 2I, H]``), so the per-expert loads of a refit no longer fit the live
parameters (verl-project/verl#7978).

On NPU, vllm-ascend's ``AscendUnquantizedFusedMoEMethod.process_weights_after_loading``
transposes ``w13_weight`` to ``[E, H, 2I]`` and ``w2_weight`` to ``[E, I, H]`` and
installs them through its own imported ``replace_parameter``, which the fold's patch
never sees; those are folded when ``fold_unquantized_moe_params`` exits.

The block layout keeps the byte count (or adds padding: vLLM 0.29 pads the
intermediate dim to a multiple of 128 first), so the checkpoint layout is staged
as a view over the live storage: the refit's loads stream straight into it, and
nothing is buffered or copied while the buckets arrive. After the last bucket,
vLLM's own ``process_weights_after_loading`` re-derives the kernel layout into a
temporary, and ``replace_parameter`` -- patched for the duration of that call --
copies it back into the live storage before the next layer is processed. The
extra memory is one layer's temporary, and the pointers a CUDA graph captured
stay valid. Between staging and folding the live storage holds
checkpoint-layout bytes, so no forward pass may run in between; the refit
already guarantees that.

The checkpoint layout is read from the metadata vLLM records for every layer at
model init for its own weight reloading, so nothing has to be patched before the
engine starts.
"""

import inspect
import logging
import math
import os
from contextlib import contextmanager
from unittest.mock import patch

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MOE_WEIGHT_NAMES = ("w13_weight", "w2_weight")
# Live parameters a staged layer set aside, by name, until the fold puts them back.
_LIVE_ATTR = "_verl_moe_live_params"


def _iter_transferable_param_attrs(src_param):
    """Yield the attributes worth moving onto a rebuilt parameter.

    Methods bound to a tensor are excluded, and that exclusion is the whole
    point of this helper. ``dir()`` reports vLLM's loader entry points
    (``load_merged_column_weight`` and friends) alongside the data attributes,
    and copying one lands a method whose ``self`` is some *other* tensor in the
    destination's instance dict, where it shadows the class. Every later load
    then writes into that stale tensor instead of the parameter it was handed --
    silently, because the shapes still agree.

    Note the test is "bound to a tensor", not "bound to the source": these
    methods are copied forward from rebuild to rebuild, so by the time one does
    damage its ``self`` is usually a temporary from several rebuilds back.
    ``weight_loader`` is bound to the layer, so it survives.
    """
    base_param_attrs = dir(torch.nn.Parameter)
    for attr in dir(src_param):
        if attr in base_param_attrs or attr.startswith("__"):
            continue
        try:
            value = getattr(src_param, attr)
        except AttributeError:
            continue
        if inspect.ismethod(value) and isinstance(value.__self__, torch.Tensor):
            continue
        yield attr, value


def _copy_param_subclass_attrs(dst_param, src_param):
    """Carry vLLM's parameter metadata across a parameter rebuild.

    vLLM stores loader dispatch information (``output_dim``, ``weight_loader``,
    tp state, ...) as plain attributes on ``Parameter`` subclasses. Anything
    that re-wraps a tensor has to bring them along or the next refit will not
    know how to shard the incoming weight.
    """
    if src_param is None:
        return

    for attr, value in _iter_transferable_param_attrs(src_param):
        try:
            setattr(dst_param, attr, value)
        except AttributeError:
            pass

    subclass_type = getattr(src_param, "subclass_type", type(src_param))
    if subclass_type is not torch.nn.Parameter:
        dst_param.subclass_type = subclass_type


def _checkpoint_layouts(layer, layerwise_info) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The expert weights' checkpoint (shape, dtype), as vLLM recorded them at model init."""
    info = layerwise_info.get(layer)
    if info is None:
        return {}
    params, _ = info.restore_metadata
    return {name: (tuple(params[name].shape), params[name].dtype) for name in _MOE_WEIGHT_NAMES if name in params}


def _staging_data(param, shape, dtype):
    """Checkpoint-layout tensor for the refit's loads to write into.

    A view over the live storage whenever that has room -- the kernel layout
    keeps the checkpoint's bytes or pads them -- so staging allocates nothing.
    Otherwise a fresh buffer, all-ones bytes (NaN in bf16/fp16/fp32), so a
    weight the stream fails to write cannot pass for a real one.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    live = param.data
    if live.is_contiguous() and live.nbytes >= nbytes:
        return live.reshape(-1).view(torch.uint8)[:nbytes].view(dtype).view(shape)
    staging = torch.empty(shape, dtype=dtype, device=live.device)
    staging.view(torch.uint8).fill_(0xFF)
    return staging


def unquantized_moe_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The model's unquantized fused-MoE layers."""
    try:
        from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method
    except ImportError:
        return []
    method_cls = unquantized_fused_moe_method.UnquantizedFusedMoEMethod
    return [layer for layer in model.modules() if isinstance(getattr(layer, "quant_method", None), method_cls)]


def stage_unquantized_moe_params(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Give ``load_weights`` checkpoint-layout expert weights where the kernel prep reshaped them.

    Returns the staged layers, to be passed to ``fold_unquantized_moe_params``
    around the post-load processing. Layers whose live weights keep the
    checkpoint layout (Triton, FlashInfer CUTLASS's same-shape gate/up swap)
    already load as-is and are left alone.
    """
    try:
        from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method
        from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO
    except ImportError:
        return []
    # The fold redirects the module-level ``replace_parameter`` that ``_setup_kernel`` writes through.
    if not hasattr(unquantized_fused_moe_method, "replace_parameter"):
        return []

    staged, fresh_bytes = [], 0
    for _, layer in model.named_modules():
        if not isinstance(getattr(layer, "quant_method", None), unquantized_fused_moe_method.UnquantizedFusedMoEMethod):
            continue
        stale = getattr(layer, _LIVE_ATTR, None)
        if stale is not None:
            # An earlier refit failed between staging and folding: start over from the live parameters.
            for name, param in stale.items():
                setattr(layer, name, param)
            delattr(layer, _LIVE_ATTR)

        live = {}
        for name, (shape, dtype) in _checkpoint_layouts(layer, LAYERWISE_INFO).items():
            param = getattr(layer, name, None)
            if not isinstance(param, torch.nn.Parameter) or (tuple(param.shape) == shape and param.dtype == dtype):
                continue
            data = _staging_data(param, shape, dtype)
            if data.data_ptr() != param.data_ptr():
                fresh_bytes += data.nbytes
            staged_param = torch.nn.Parameter(data, requires_grad=False)
            _copy_param_subclass_attrs(staged_param, param)
            # The record goes on the layer before the parameter is swapped out, so a failure part way
            # through still leaves the next refit what it restores from.
            setattr(layer, _LIVE_ATTR, live)
            live[name] = param
            setattr(layer, name, staged_param)

        if live:
            staged.append(layer)

    if staged:
        logger.info("Staged %d unquantized MoE layers in checkpoint layout for the refit", len(staged))
    if fresh_bytes:
        logger.warning(
            "An unquantized MoE kernel layout has no room for the checkpoint layout, so %.1f MB of "
            "checkpoint-layout staging buffers were allocated for this refit",
            fresh_bytes / 2**20,
        )
    return staged


def _fold_into_live_param(layer, param_name, param, new_data):
    """Move post-processing output into the storage the CUDA graph captured."""
    if isinstance(new_data, torch.nn.Parameter):
        new_data = new_data.data

    if new_data.shape != param.shape or new_data.dtype != param.dtype:
        raise RuntimeError(
            f"Unquantized MoE refit re-derived {param_name} as {tuple(new_data.shape)}/{new_data.dtype}, "
            f"but the live parameter is {tuple(param.shape)}/{param.dtype}; "
            "its storage cannot be updated in place."
        )
    if new_data.data_ptr() != param.data_ptr():
        param.data.copy_(new_data)
    setattr(layer, param_name, param)


def _folding_replace_parameter(original):
    def replace_parameter(layer, param_name, new_data, *args, **kwargs):
        live = getattr(layer, _LIVE_ATTR, None)
        param = live.get(param_name) if live and new_data is not None else None
        if param is None:
            return original(layer, param_name, new_data, *args, **kwargs)
        # Into the storage the CUDA graph captured, right away: the temporary is freed before the next layer.
        _fold_into_live_param(layer, param_name, param, new_data)
        # Dropped only once folded: a fold that raises leaves the record the next refit restores from.
        del live[param_name]

    return replace_parameter


@contextmanager
def fold_unquantized_moe_params(layers: list[torch.nn.Module]):
    """Route the staged layers' re-derived weights into their live storage during post-load processing."""
    if not layers:
        yield
        return

    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method

    folding = _folding_replace_parameter(unquantized_fused_moe_method.replace_parameter)
    with patch.object(unquantized_fused_moe_method, "replace_parameter", folding):
        yield

    for layer in layers:
        # Whatever the patch did not see: a backend that bypassed the module-level name still left its
        # kernel layout on the layer, but a staged view left there means nothing re-derived the weight.
        live = getattr(layer, _LIVE_ATTR, None) or {}
        for name, param in list(live.items()):
            current = getattr(layer, name)
            if tuple(current.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"{type(layer).__name__}.{name} was not re-derived into its kernel layout after the refit: "
                    f"the checkpoint-layout view {tuple(current.shape)} is still on the layer"
                )
            _fold_into_live_param(layer, name, param, current)
            del live[name]
        if hasattr(layer, _LIVE_ATTR):
            delattr(layer, _LIVE_ATTR)
