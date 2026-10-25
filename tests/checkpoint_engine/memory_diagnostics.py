# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Opt-in FSDP host-memory diagnostics for the Ascend checkpoint CI job.

Run directly to sample the host while executing the unchanged test command.
Snapshots add latency and can perturb allocation timing. Never alter allocation
arguments, retry a failed allocation, or suppress the original exception.
"""

import argparse
import contextlib
import datetime
import functools
import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"UNAVAILABLE: {exc}\n"


def snapshot(destination, detailed=True):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "meminfo",
        "buddyinfo",
        "vmstat",
        "uptime",
        "sys/kernel/random/boot_id",
        "pressure/memory",
        "self/status",
        "thread-self/status",
        "self/limits",
        "self/cgroup",
        "self/mountinfo",
    ):
        (destination / name.replace("/", "_")).write_text(read("/proc/" + name))
    if detailed:
        (destination / "pagetypeinfo").write_text(read("/proc/pagetypeinfo"))
        (destination / "self_numa_maps").write_text(read("/proc/self/numa_maps"))
        (destination / "self_smaps_rollup").write_text(read("/proc/self/smaps_rollup"))
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        (destination / (node.name + "_meminfo")).write_text(read(node / "meminfo"))
    memberships = [line.split(":", 2) for line in read("/proc/self/cgroup").splitlines() if line.count(":") == 2]
    groups = []
    for line in read("/proc/self/mountinfo").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        root, mount = fields[3:5]
        for _, controllers, group in memberships:
            if fs[0] == "cgroup2" and controllers:
                continue
            if fs[0] == "cgroup" and not (set(controllers.split(",")) & {"memory", "cpuset"} & set(fs[2].split(","))):
                continue
            relative = os.path.relpath(group, root)
            target = Path(mount) / relative if relative != ".." and not relative.startswith("../") else Path(mount)
            if not target.is_dir():
                target = Path(mount)
            while True:
                entry = {"path": str(target), "mount_root": root, "membership": group}
                for item in target.iterdir():
                    if item.name.startswith(("memory.", "cpuset.")) and item.is_file():
                        entry[item.name] = read(item)
                groups.append(entry)
                if target == Path(mount):
                    break
                target = target.parent
    (destination / "cgroups.json").write_text(json.dumps(groups, indent=2))
    (destination / "timestamp.txt").write_text(now() + "\nAncestors above cgroup mount root may be hidden.\n")


def capture(folder, name, command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = f"exit={result.returncode}\n{result.stdout}{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"UNAVAILABLE: {exc}"
    (folder / name).write_text(output)


def record(folder, **details):
    """Diagnostic failures must not replace a workload exception."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        details.update(time=now(), pid=os.getpid(), ppid=os.getppid(), tid=threading.get_native_id())
        (folder / "event.json").write_text(json.dumps(details, indent=2, default=str))
        snapshot(folder, detailed=folder.name in {"installed", "failed", "initial", "final", "collection"})
        (folder / "self_maps").write_text(read("/proc/self/maps"))
    except Exception as exc:
        print(f"Ascend memory diagnostic failed: {exc!r}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def trace_fsdp_initialization():
    """Observe lazy initialization without replacing Tensor.pin_memory or constructors.

    This synchronous scope surrounds get_per_tensor_param only. A handle may
    allocate both a pinned shard and a pinned gradient; its traceback identifies
    which operation failed. Report shard bytes, not an inferred driver request.
    """
    root = os.environ.get("VERL_FSDP_MEMORY_DIAGNOSTICS")
    if not root:
        yield
        return
    from torch.distributed.fsdp._flat_param import FlatParamHandle

    original = FlatParamHandle.init_flat_param_attributes
    worker = Path(root) / "workers" / str(os.getpid()) / str(time.time_ns())
    sequence = itertools.count()
    record(worker / "installed", hook="FlatParamHandle.init_flat_param_attributes")
    try:
        source = inspect.getsourcefile(inspect.unwrap(original))
        (worker / "fsdp-source.py").write_text(read(source))
        (worker / "method-source.txt").write_text(inspect.getsource(original))
    except Exception as exc:
        print(f"Cannot capture FSDP source: {exc!r}", flush=True)

    @functools.wraps(original)
    def initialize(handle, *args, **kwargs):
        event = worker / f"handle-{next(sequence):06d}"
        try:
            param = handle.flat_param
            record(
                event / "before",
                shard_bytes=param.numel() * param.element_size(),
                shape=list(param.shape),
                dtype=str(param.dtype),
                device=str(param.device),
                offload_params=handle._offload_params,
                rank=handle.rank,
                world_size=handle.world_size,
                fqns=getattr(param, "_fqns", []),
                stack=traceback.format_stack(),
            )
        except Exception as exc:
            print(f"Cannot capture shard metadata: {exc!r}", flush=True)
        started = time.monotonic()
        try:
            result = original(handle, *args, **kwargs)
        except Exception:
            record(event / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
            try:
                capture(event / "failed", "kernel.txt", ["dmesg", "-T"])
            except Exception as exc:
                print(f"Cannot capture kernel log: {exc!r}", flush=True)
            raise
        record(event / "after", elapsed_s=time.monotonic() - started)
        return result

    FlatParamHandle.init_flat_param_attributes = initialize
    try:
        yield
    finally:
        FlatParamHandle.init_flat_param_attributes = original


def processes(folder):
    """Capture visible processes, including leftovers from earlier job steps."""
    entries = {}
    for proc in Path("/proc").glob("[0-9]*"):
        entries[proc.name] = {name: read(proc / name) for name in ("status", "cgroup")}
    (folder / "processes.json").write_text(json.dumps(entries))


def collect(root):
    root.mkdir(parents=True, exist_ok=True)
    record(root / "collection")
    processes(root / "collection")
    capture(root, "kernel-after.txt", ["dmesg", "-T"])
    capture(root, "npu-smi-after.txt", ["npu-smi", "info"])
    # Preserve bounded tails of recently modified driver/runtime and Ray logs.
    cutoff = float(read(root / "started.txt").strip()) if (root / "started.txt").exists() else time.time()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    manifest = []
    for base in (Path("/root/ascend/log"), Path("/tmp/ray/session_latest/logs")):
        candidates = []
        for path in base.rglob("*"):
            try:
                if path.is_file() and path.stat().st_mtime >= cutoff:
                    candidates.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for index, (_, path) in enumerate(sorted(candidates, reverse=True)):
            entry = {"source": str(path), "copied": index < 64}
            manifest.append(entry)
            if index >= 64:
                continue
            try:
                target = hashlib.sha256(str(path).encode()).hexdigest()[:16] + ".log"
                with path.open("rb") as stream:
                    size = stream.seek(0, os.SEEK_END)
                    stream.seek(max(0, size - 2 * 1024 * 1024))
                    (logs / target).write_bytes(stream.read())
                entry.update(target=target, original_bytes=size)
            except OSError as exc:
                entry["error"] = str(exc)
    (root / "log-manifest.json").write_text(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if args.collect_only:
        collect(root)
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required after --")
    root.mkdir(parents=True, exist_ok=False)
    (root / "started.txt").write_text(str(time.time()))
    os.environ["VERL_FSDP_MEMORY_DIAGNOSTICS"] = str(root)
    for name, cmd in {
        "kernel-before.txt": ["dmesg", "-T"],
        "npu-smi-before.txt": ["npu-smi", "info"],
        "packages.txt": [sys.executable, "-m", "pip", "list"],
        "revision.txt": ["git", "rev-parse", "HEAD"],
        "uname.txt": ["uname", "-a"],
        "storage.txt": ["df", "-h", "/tmp", "/dev/shm", "/root/.cache/temp"],
    }.items():
        capture(root, name, cmd)
    (root / "driver-version.txt").write_text(read("/usr/local/Ascend/driver/version.info"))
    record(root / "initial", command=command)
    child = subprocess.Popen(command)
    sample = 0
    while child.poll() is None:
        folder = root / "host" / f"sample-{sample:06d}"
        record(folder)
        try:
            processes(folder)
        except Exception as exc:
            print(f"Cannot sample processes: {exc!r}", flush=True)
        sample += 1
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    record(root / "final", exit_code=child.returncode)
    summary = {
        "exit_code": child.returncode,
        "hook_scopes": len(list(root.glob("workers/*/*/installed/event.json"))),
        "handle_calls": len(list(root.glob("workers/*/*/handle-*/before/event.json"))),
        "handle_failures": len(list(root.glob("workers/*/*/handle-*/failed/event.json"))),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if not summary["handle_calls"]:
        print("Diagnostic incomplete: no FSDP handle initialization observed", file=sys.stderr, flush=True)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    sys.exit(main())
