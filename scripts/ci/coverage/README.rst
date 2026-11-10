Ascend PR workload coverage
===========================

``.github/workflows/coverage-collection-npu.yml`` is a separate collection
workflow. It runs the workloads from Ascend PR workflows without instrumenting
ordinary PR checks. The schedule is 05:00 UTC+8 on alternate days of the month
(``0 21 */2 * *``), rather than a strict 48-hour interval. Manual dispatch is
also available. A temporary PR trigger covers collection helpers and source
workflow changes; remove that trigger from the generator after CI validation.

Scope and regeneration
----------------------

The generator discovers PR-triggered ``*_ascend.yml`` workflows and
``npu_unit_tests.yml``. There are currently 13 workflows and 18 declared jobs.
The two jobs disabled by ``if: false`` in the original Megatron/SGLang workflow
stay disabled, leaving 16 runnable jobs. Nightly, multi-node, image-build and
documentation workloads are excluded. CPU test steps inside an Ascend job are
included.

Regenerate after changing an original workflow::

    uv run --no-project --python 3.12 --no-build --with PyYAML==6.0.2 python scripts/ci/coverage/generate_workflow.py

Verify without writing::

    uv run --no-project --python 3.12 --no-build --with PyYAML==6.0.2 python scripts/ci/coverage/generate_workflow.py --check

A validation job checks for drift before allocating NPU runners. Original
commands, environments, images, runner labels, job conditions and dependencies
are copied. Coverage gets a separate concurrency group and larger timeouts.
The NPU workflow's legacy coverage-mode steps are excluded: its normal PR
commands (including the CPU checkpoint test) are instrumented instead.

Collection behavior
-------------------

After dependency installation and before the first workload, the collection
job installs coverage.py 7.8.2 into the ambient Python interpreter with uv.
A dedicated, opt-in ``.pth`` hook preserves the image's existing
``sitecustomize`` and ``PYTHONPATH``. The current Ascend workflows use ambient
Python; a future change introducing another interpreter must install coverage
and the hook there too.

Each original Actions ``run`` block is one collection entry. Its exact script
is retained as ``command.sh``; the manifest also records the source step name
and explicit entrypoint paths. Multiple commands in one block share coverage.
This preserves heredocs, shell state, checkpoint/resume ordering and validation
commands. Different steps invoking the same script remain separate entries.

The wrapper sets coverage identity before executing the entire block, including
any ``ray start`` command. It stops the container-local Ray cluster gracefully
before and after a block so subsequent steps do not reuse workers with stale
coverage identity. Original commands and their exit status are preserved;
cleanup failures also fail the collection step. Coverage uses branch data,
parallel process files, multiprocessing support and SIGTERM handling. It does
not instrument installation or packaging commands.

All source jobs retain their ordinary step-failure and dependency behavior.
Following a failure, later dependent work may be skipped; its manifest entries
remain explicitly incomplete. The finalizer and artifact upload use
``always()``. Hard cancellation or runner loss can still prevent archiving.

Artifacts
---------

Each executed job retains a GitHub Actions artifact for 14 days, named
``ascend-coverage-<source-workflow>__<job-index>-<run-attempt>``. The tar contains::

    manifest.json
    packages.txt
    source-workflow.yml
    npu_pytest_suites.json
    coveragerc
    covstub/verl/...
    coverage/step-NN/command.sh
    coverage/step-NN/coverage.<host>.<pid>.<suffix>
    coverage/step-NN/process-<host>-<pid>.json

The source SHA, run/attempt, original workflow/job, commands, process startup
evidence and per-step status allow later adapters to associate data correctly.
Raw SQLite files are kept unchanged; no coverage combine operation consumes
them. Different jobs and run attempts have separate workspace directories.

``collection_complete`` requires all planned steps to pass, valid branch data
for test/training steps, and data files for the observed interpreter startups.
Preprocessing may legitimately execute no verl code. Missing data, hook errors,
unrun steps and corrupt files are recorded and cause finalization to fail, but
the partial tar is still uploaded. ``worker_coverage_verified`` and
``selection_ready`` remain false: process counts cannot prove that every Ray
actor or engine subprocess was instrumented. Forked processes also do not
necessarily produce a separate interpreter-start record.

There is no precision-test invocation, OBS upload, map generation or PR test
selection. Once the verl adapter is available, these branch-coverage SQLite
files and matching source snapshots can be used to build workflow-level maps.

First CI inspection
-------------------

Inspect a pytest/torchrun job, an FSDP/vLLM job and a runnable SGLang job before
treating collection as reliable. Confirm rank/worker-side verl modules appear
in the raw data, output paths refer to the intended checkout, and later entries
do not contain a previous entry's Ray workers. SIGKILL, abrupt engine shutdown
and overwritten child environments can lose coverage even when training passes.
No local PyTorch execution is needed for generator, lint or packaging checks.
