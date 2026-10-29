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
"""Opt-in, stdlib-only diagnostics for Ascend GRPO actor initialization.

Run this file directly to monitor the unchanged GRPO command on a Linux CI node.
Snapshots add latency and can perturb allocation timing. Never alter allocation
arguments, retry a failed allocation, or suppress the original exception.
"""

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

CGROUP_FILES = (
    "memory.current",
    "memory.max",
    "memory.high",
    "memory.peak",
    "memory.events",
    "memory.events.local",
    "memory.stat",
    "memory.numa_stat",
    "memory.pressure",
    "memory.swap.current",
    "memory.swap.max",
    "memory.swap.events",
    "memory.limit_in_bytes",
    "memory.usage_in_bytes",
    "memory.max_usage_in_bytes",
    "memory.failcnt",
    "memory.oom_control",
    "memory.memsw.limit_in_bytes",
    "memory.memsw.usage_in_bytes",
    "memory.memsw.failcnt",
    "memory.use_hierarchy",
    "cpuset.mems",
    "cpuset.mems.effective",
    "cpuset.effective_mems",
    "cpuset.cpus",
    "cpuset.cpus.effective",
    "cpuset.effective_cpus",
)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"UNAVAILABLE: {exc}\n"


def thread_memory_policy():
    """Read this thread's default NUMA policy without changing it."""
    import ctypes

    try:
        library = ctypes.CDLL("libnuma.so.1", use_errno=True)
        query = library.get_mempolicy
        query.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
        possible = Path("/sys/devices/system/node/possible").read_text().strip()
        max_node = max(int(part.split("-")[-1]) for part in possible.split(",")) + 1
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        words = (max_node + bits - 1) // bits
        mask = (ctypes.c_ulong * words)()
        mode = ctypes.c_int()
        if query(ctypes.byref(mode), mask, words * bits, None, 0):
            raise OSError(ctypes.get_errno(), "get_mempolicy failed")
        return {"mode": mode.value, "nodes": [i for i in range(max_node) if mask[i // bits] & (1 << (i % bits))]}
    except Exception as exc:
        return {"unavailable": repr(exc)}


def snapshot(destination, detailed=True):
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "meminfo",
        "buddyinfo",
        "vmstat",
        "self/status",
        "thread-self/status",
        "thread-self/sched",
        "self/limits",
        "self/cgroup",
        "self/mountinfo",
    ):
        (destination / name.replace("/", "_")).write_text(read("/proc/" + name))
    if detailed:
        (destination / "pagetypeinfo").write_text(read("/proc/pagetypeinfo"))
        (destination / "self_numa_maps").write_text(read("/proc/self/numa_maps"))
        (destination / "thread-memory-policy.json").write_text(json.dumps(thread_memory_policy(), indent=2))
    for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
        (destination / (node.name + "_meminfo")).write_text(read(node / "meminfo"))
    (destination / "cgroups.json").write_text(json.dumps(cgroup_snapshot(), indent=2))
    (destination / "timestamp.txt").write_text(now() + "\nAncestors above cgroup mount root may be hidden.\n")


def cgroup_snapshot(pid="self"):
    """Resolve memberships against this collector's mounts only in shared namespaces."""
    if pid != "self":
        try:
            for namespace in ("mnt", "cgroup"):
                if os.readlink(f"/proc/{pid}/ns/{namespace}") != os.readlink(f"/proc/self/ns/{namespace}"):
                    return [
                        {"unavailable": f"worker {pid} has a different {namespace} namespace; see worker snapshots"}
                    ]
        except OSError as exc:
            return [{"unavailable": repr(exc)}]
    membership = read(f"/proc/{pid}/cgroup")
    if membership.startswith("UNAVAILABLE:"):
        return [{"unavailable": membership}]
    memberships = [line.split(":", 2) for line in membership.splitlines() if line.count(":") == 2]
    groups = []
    for line in read("/proc/self/mountinfo").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue

        def unescape(value):
            for code, char in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
                value = value.replace(code, char)
            return value

        root, mount = map(unescape, fields[3:5])
        for _, controllers, group in memberships:
            if fs[0] == "cgroup2" and controllers:
                continue
            if fs[0] == "cgroup" and not (set(controllers.split(",")) & {"memory", "cpuset"} & set(fs[2].split(","))):
                continue
            group = unescape(group)
            relative = os.path.relpath(group, root)
            target = Path(mount) / relative if relative != ".." and not relative.startswith("../") else Path(mount)
            mapping = "resolved" if target != Path(mount) or group == root else "mount-root fallback"
            if not target.is_dir():
                target = Path(mount)
                mapping = "mount-root fallback; exact membership unavailable"
            while True:
                entry = {"path": str(target), "mount_root": root, "membership": group, "mapping": mapping}
                for name in CGROUP_FILES:
                    item = target / name
                    if item.exists():
                        entry[name] = read(item)
                groups.append(entry)
                if target == Path(mount):
                    break
                target = target.parent
    return groups


def capture(folder, name, command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        output = f"exit={result.returncode}\n{result.stdout}{result.stderr}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"UNAVAILABLE: {exc}"
    (folder / name).write_text(output)


def record(folder, detailed=None, **details):
    """Diagnostic failures must not replace a workload exception."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        details.update(
            time=now(),
            monotonic_ns=time.monotonic_ns(),
            pid=os.getpid(),
            ppid=os.getppid(),
            tid=threading.get_native_id(),
            hostname=socket.gethostname(),
        )
        # Preserve the event/exception even if a later procfs read fails.
        (folder / "event.json").write_text(json.dumps(details, indent=2, default=str))
        snapshot(
            folder,
            detailed=folder.name in {"installed", "failed", "initial", "final"} if detailed is None else detailed,
        )
        (folder / "self_maps").write_text(read("/proc/self/maps"))
    except Exception as exc:
        print(f"Ascend memory diagnostic failed: {exc!r}", file=sys.stderr, flush=True)


@contextmanager
def actor_initialization():
    """Observe actor.reset without wrapping tensors, retrying or synchronizing."""
    root = os.environ.get("VERL_ASCEND_MEMORY_DIAGNOSTICS")
    if not root:
        yield
        return
    folder = Path(root) / "actor_init" / str(os.getpid()) / str(time.monotonic_ns())
    record(folder / "before", detailed=True, stage="actor.reset", stack=traceback.format_stack())
    print(f"ASCEND_MEMORY actor.reset before pid={os.getpid()} tid={threading.get_native_id()}", flush=True)
    started = time.monotonic()
    try:
        yield
    except BaseException:
        # Save the original traceback before slower reads; never replace the exception.
        record(folder / "failed", exception=traceback.format_exc(), elapsed_s=time.monotonic() - started)
        try:
            capture(folder / "failed", "kernel.txt", ["dmesg", "-T"])
        except Exception as exc:
            print(f"Kernel capture failed: {exc!r}", file=sys.stderr, flush=True)
        print(f"ASCEND_MEMORY actor.reset failed pid={os.getpid()}", flush=True)
        raise
    else:
        record(folder / "after", elapsed_s=time.monotonic() - started)
        print(f"ASCEND_MEMORY actor.reset completed pid={os.getpid()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-actors", type=int, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a workload command is required after --")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["VERL_ASCEND_MEMORY_DIAGNOSTICS"] = str(output)
    record(
        output / "initial",
        command=command,
        identity={
            key: os.environ.get(key, "UNAVAILABLE")
            for key in (
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_SHA",
                "RUNNER_NAME",
                "POD_NAME",
                "NODE_NAME",
                "K8S_NODE_NAME",
                "IMAGE_DIGEST",
            )
        },
    )
    capture(output, "revision.txt", ["git", "rev-parse", "HEAD"])
    capture(output, "packages.txt", [sys.executable, "-m", "pip", "list"])
    capture(output, "kernel-before.txt", ["dmesg", "-T"])
    capture(output, "shm.txt", ["df", "-h", "/dev/shm"])
    capture(output, "npu-smi.txt", ["npu-smi", "info"])
    stop = threading.Event()

    def sample():
        sequence = 0
        while not stop.is_set():
            folder = output / "host" / f"sample-{sequence:06d}"
            try:
                snapshot(folder, detailed=False)
                (folder / "clock.json").write_text(json.dumps({"time": now(), "monotonic_ns": time.monotonic_ns()}))
                # Include real actor memberships/status, not only the supervisor's.
                for actor in (output / "actor_init").glob("[0-9]*"):
                    for name in ("status", "cgroup"):
                        (folder / f"actor-{actor.name}-{name}").write_text(read(f"/proc/{actor.name}/{name}"))
                    (folder / f"actor-{actor.name}-cgroups.json").write_text(
                        json.dumps(cgroup_snapshot(actor.name), indent=2)
                    )
                if sequence % 10 == 0:
                    mem = read("/proc/meminfo").splitlines()
                    selected = [line for line in mem if line.startswith(("MemAvailable:", "MemFree:", "Cached:"))]
                    groups = json.loads((folder / "cgroups.json").read_text())
                    compact = [
                        {
                            k: v
                            for k, v in g.items()
                            if k
                            in (
                                "path",
                                "memory.current",
                                "memory.max",
                                "memory.usage_in_bytes",
                                "memory.limit_in_bytes",
                                "memory.events",
                                "memory.failcnt",
                                "memory.oom_control",
                            )
                        }
                        for g in groups
                    ]
                    print(f"ASCEND_MEMORY {now()} {'; '.join(selected)} cgroups={json.dumps(compact)}", flush=True)
            except Exception as exc:
                print(f"Memory sampling unavailable: {exc!r}", flush=True)
            sequence += 1
            # Retain low-frequency monitoring after the target initialization phase.
            completed = len(list((output / "actor_init").glob("*/*/after/event.json")))
            stop.wait(5 if completed >= args.expected_actors else 1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    result = 1
    try:
        result = subprocess.call(command)
        result = result if result >= 0 else 128 - result
    finally:
        stop.set()
        sampler.join(timeout=10)
        record(output / "final", exit_code=result)
    calls = len(list((output / "actor_init").glob("*/*/before/event.json")))
    completed = len(list((output / "actor_init").glob("*/*/after/event.json")))
    failures = [str(p.relative_to(output)) for p in (output / "actor_init").glob("*/*/failed/event.json")]
    summary = {
        "workload_exit_code": result,
        "actor_init_calls": calls,
        "actor_init_completed": completed,
        "actor_init_failures": failures,
        "expected_actors": args.expected_actors,
        "coverage_complete": calls == completed == args.expected_actors and not failures,
    }
    # Never label an uninstrumented passing workload a successful diagnostic.
    diagnostic_result = result or (0 if summary["coverage_complete"] else 2)
    summary["diagnostic_exit_code"] = diagnostic_result
    try:
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    except OSError as exc:
        print(f"Summary write failed: {exc!r}", flush=True)
    print("ASCEND_MEMORY summary=" + json.dumps(summary), flush=True)
    return diagnostic_result


if __name__ == "__main__":
    raise SystemExit(main())
