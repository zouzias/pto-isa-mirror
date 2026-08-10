#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

# A3 simple multi-NPU parallel smoke orchestrator.
# Standard library only. No NPU binaries are executed by this module directly.

import argparse
import dataclasses
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fcntl  # POSIX only
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

# Module-level tracking of live child processes so signal handlers can
# terminate them. Guarded by threads calling Popen; reads iterate defensively.
# Each entry is a ProcGroup holding the Popen and the PGID captured at launch.
_ACTIVE_PROCS = []
_ACTIVE_PROCS_LOCK = threading.Lock()


def reset_active_procs():
    """Clear the tracked process list (used by host-only tests)."""
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.clear()


@dataclass
class ProcGroup:
    """A launched task process plus its process-group id (captured at start)."""
    proc: object
    pgid: object  # int or None


def _new_proc_group(proc):
    """Wrap a Popen into a ProcGroup, capturing its PGID at launch time.

    The PGID is captured while the leader is still alive; after the leader
    exits, os.getpgid(pid) may fail, so we must not re-query it later.
    """
    pgid = None
    if os.name != "nt":
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            pgid = proc.pid  # fall back to the leader pid as the group id
    return ProcGroup(proc=proc, pgid=pgid)


def group_exists(group):
    """Return True if the process group still has any member."""
    if group is None or group.pgid is None:
        return False
    try:
        os.killpg(group.pgid, 0)
        return True
    except OSError:
        return False


def terminate_process_group(group, sig):
    """Send a signal to the saved process group id."""
    if group is None or group.pgid is None:
        return False
    try:
        os.killpg(group.pgid, sig)
        return True
    except OSError:
        return False


def wait_group_gone(group, timeout):
    """Wait until the process group disappears, or timeout.

    Returns True if the group is gone (or was never present), False otherwise.
    """
    if group is None or group.pgid is None:
        # Nothing we can probe; treat as gone if the leader has exited.
        try:
            return group.proc.poll() is not None
        except OSError:
            # Leader may have already been reaped; treat as gone.
            return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not group_exists(group):
            return True
        time.sleep(0.05)
    return not group_exists(group)


def terminate_proc_gracefully(group, timeout=10):
    """TERM the whole process group, wait for it to disappear, then SIGKILL.

    Returns True only if the entire process group is confirmed gone.
    """
    if group is None:
        return True
    proc = group.proc
    # TERM the whole group so run_st.py AND its gtest/NPU children get it.
    if terminate_process_group(group, signal.SIGTERM):
        if wait_group_gone(group, timeout):
            return True
    else:
        # Group not available (e.g. already gone); just wait for the leader.
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Leader still alive; escalate below.
            pass
        if proc.poll() is not None:
            return not group_exists(group)
        # Escalate
        terminate_process_group(group, signal.SIGKILL)
        return wait_group_gone(group, 5)
    # Escalate to SIGKILL on the saved PGID.
    terminate_process_group(group, signal.SIGKILL)
    try:
        proc.terminate()
    except OSError:
        # Process may already be gone; ignore.
        pass
    return wait_group_gone(group, 5)


def finish_task_group(group, timeout=10):
    """Finish a task and confirm its whole process group is gone.

    Called in both normal and exception paths. Waits for the leader to return,
    then checks whether the saved PGID still has members (grandchildren such as
    gtest/NPU processes). If it does, TERM -> wait -> KILL the group. Only after
    the entire group is confirmed gone is the ProcGroup removed from
    _ACTIVE_PROCS. Returns True if the group is confirmed gone, False otherwise.
    """
    if group is None:
        return True
    proc = group.proc
    # Wait for the leader to return (bounded; leader may have already exited).
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Leader still running; it will be handled by the group cleanup below.
        pass
    # If the process group still exists, grandchildren are alive: clean them up.
    if group_exists(group):
        terminate_proc_gracefully(group, timeout=timeout)
    gone = not group_exists(group)
    # Only remove tracking once the whole group is confirmed gone.
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS[:] = [g for g in _ACTIVE_PROCS if g.proc is not proc]
    return gone


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SmokeError(Exception):
    """Base error for the orchestrator."""


class DeviceDiscoveryError(SmokeError):
    """Raised when device allocation cannot be determined safely."""


class ManifestError(SmokeError):
    """Raised when the task manifest is malformed."""


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SmokeTask:
    index: int
    testcase: str
    gtest_filter: object  # Optional[str]
    debug_enable: bool


@dataclass(frozen=True)
class WorkerConfig:
    index: int
    physical_device: int
    build_dir: object  # Path
    log_path: object  # Path


@dataclass(frozen=True)
class WorkerContext:
    """Shared execution context for worker threads."""
    run_st_script: object
    auto_mode: bool
    print_lock: object
    now: object


@dataclass(frozen=True)
class DeviceSources:
    """Candidate device sources for selection, in priority order."""
    explicit: object  # Optional[str] from PTO_ST_PARALLEL_DEVICES
    inherited: object  # list[int] from ASCEND_RT_VISIBLE_DEVICES
    smi_statuses: object  # Optional[list[NpuStatus]]


@dataclass(frozen=True)
class TaskResult:
    task: SmokeTask
    worker_index: int
    physical_device: int
    returncode: int
    elapsed_ms: int


@dataclass(frozen=True)
class NpuStatus:
    device_id: int
    healthy: bool
    busy: bool


@dataclass
class DeviceLock:
    device_id: int
    fd: object  # file object
    path: str


# ---------------------------------------------------------------------------
# Task manifest parsing
# ---------------------------------------------------------------------------

def parse_task_manifest(manifest_path):
    """Parse the TSV manifest into an ordered list of SmokeTask.

    Contract per row: <testcase> TAB <gtest-filter-or-empty> TAB <debug-0-or-1>
    """
    tasks = []
    seen = set()
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ManifestError(f"line {lineno}: expected 3 tab-separated fields, got {len(parts)}")
            testcase, gtest_filter, debug_str = parts
            if not testcase:
                raise ManifestError(f"line {lineno}: empty testcase")
            if debug_str not in ("0", "1"):
                raise ManifestError(f"line {lineno}: invalid debug value {debug_str!r}")
            debug_enable = debug_str == "1"
            triplet = (testcase, gtest_filter, debug_enable)
            if triplet in seen:
                raise ManifestError(f"line {lineno}: duplicate task triple {triplet!r}")
            seen.add(triplet)
            tasks.append(SmokeTask(
                index=len(tasks),
                testcase=testcase,
                gtest_filter=gtest_filter if gtest_filter else None,
                debug_enable=debug_enable,
            ))
    if not tasks:
        raise ManifestError("manifest is empty")
    return tasks


# ---------------------------------------------------------------------------
# Device list parsing
# ---------------------------------------------------------------------------

def parse_device_list(value):
    """Return unique non-negative integer IDs.

    Accepted: "0", "0,3", "3,5,7". Rejected: "0,", "0,0", "-1,3", "0,a".
    """
    if value is None:
        raise ValueError("value is None")
    tokens = value.split(",")
    if any(not t.strip() for t in tokens):
        raise ValueError("empty device entry")
    ids = []
    for tok in tokens:
        tok = tok.strip()
        if not re.fullmatch(r"\d+", tok):
            raise ValueError(f"non-numeric device id {tok!r}")
        ids.append(int(tok))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate device ids")
    return ids


def parse_inherited_visible(value):
    """Parse ASCEND_RT_VISIBLE_DEVICES (comma or space separated). Empty -> []."""
    if value is None:
        return []
    ids = []
    for tok in re.split(r"[, ]+", value.strip()):
        if not tok:
            continue
        if not re.fullmatch(r"\d+", tok):
            raise ValueError(f"non-numeric visible device id {tok!r}")
        ids.append(int(tok))
    return ids


# ---------------------------------------------------------------------------
# npu-smi query
# ---------------------------------------------------------------------------

def query_npu_smi(timeout_seconds=10):
    """Run `npu-smi info`, parse device health and process ownership.

    Returns a list of NpuStatus. Raises DeviceDiscoveryError when the command
    is absent, times out, exits nonzero, or cannot be parsed confidently.
    """
    exe = shutil.which("npu-smi")
    if exe is None:
        raise DeviceDiscoveryError("npu-smi is unavailable")
    try:
        proc = subprocess.run(
            [exe, "info"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        raise DeviceDiscoveryError("npu-smi timed out") from e
    except OSError as e:
        raise DeviceDiscoveryError(f"npu-smi failed to start: {e}") from e
    if proc.returncode != 0:
        raise DeviceDiscoveryError(f"npu-smi exited nonzero: {proc.returncode}")
    return parse_npu_smi(proc.stdout, proc.stderr)


def parse_npu_smi(stdout, stderr=""):
    """Parse the observed 910B4-style `npu-smi info` output.

    Rules:
      - a device row looks like:  | 2     910B4               | OK ...
      - the same block then lists a chip line: | 0 | 0000:C2:00.0 | ...
      - a device is busy if a numeric process row associates a pid with it:
        | 0 | 12345 | ...  (in the Process table).
      - "No running processes found in NPU N" means that device is idle.
      - non-contiguous ids such as 0 and 3 are supported.
    """
    statuses = []
    device_rows = {}
    busy_ids = set()
    idle_confirmed = set()
    proc_section = False
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "Process id" in stripped:
            proc_section = True
            continue
        # Device block header => end of process section once process section starts.
        m = re.match(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(OK|[A-Za-z]+)\s*\|", stripped)
        if m:
            dev_id = int(m.group(1))
            health = m.group(3).strip().lower()
            device_rows[dev_id] = (health == "ok")
            proc_section = False
            continue
        if "No running processes found in NPU" in stripped:
            nm = re.search(r"NPU\s+(\d+)", stripped)
            if nm:
                # Explicitly idle; remove any prior busy marking.
                busy_ids.discard(int(nm.group(1)))
                idle_confirmed.add(int(nm.group(1)))
            continue
        if proc_section:
            # Process row: | <npu_id> | <pid> | <name> | ...
            pm = re.match(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", stripped)
            if pm:
                busy_ids.add(int(pm.group(1)))
                idle_confirmed.discard(int(pm.group(1)))
    if not device_rows:
        raise DeviceDiscoveryError("npu-smi output could not be parsed confidently")
    # Every healthy device must have a confirmed process status: either a busy
    # row or an explicit "No running processes found in NPU N" marker. If the
    # process table is absent or a healthy device got neither, we cannot assume
    # it is idle and must report a probe failure rather than guess.
    for dev_id in device_rows:
        if not device_rows[dev_id]:
            continue  # unhealthy devices are excluded from selection anyway
        if dev_id not in busy_ids and dev_id not in idle_confirmed:
            raise DeviceDiscoveryError(
                f"npu-smi process status for device {dev_id} could not be determined"
            )
    for dev_id in sorted(device_rows):
        statuses.append(NpuStatus(
            device_id=dev_id,
            healthy=device_rows[dev_id],
            busy=dev_id in busy_ids,
        ))
    return statuses


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_devices(requested, sources, assume_available=False, best_effort=False):
    """Select up to `requested` device ids using the documented priority.

    `sources` is a DeviceSources with explicit/inherited/smi_statuses.
    Returns (devices, source) where source is one of:
      "PTO_ST_PARALLEL_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "npu-smi",
      "/dev", or None (insufficient).
    """
    explicit = sources.explicit
    inherited = sources.inherited
    smi_statuses = sources.smi_statuses
    # 1. Explicit operator assignment.
    if explicit is not None:
        ids = parse_device_list(explicit)
        if len(ids) < requested:
            if best_effort:
                return [], "PTO_ST_PARALLEL_DEVICES"
            raise DeviceDiscoveryError(
                f"PTO_ST_PARALLEL_DEVICES has {len(ids)} ids, need {requested}"
            )
        return ids, "PTO_ST_PARALLEL_DEVICES"

    # 2. Inherited visible list.
    if inherited:
        ids = list(dict.fromkeys(inherited))
        if len(ids) >= requested:
            return ids, "ASCEND_RT_VISIBLE_DEVICES"
        # The scheduler assigned a device list but it is insufficient. Do not
        # fall through to npu-smi /dev: that would cross the scheduler's device
        # boundary. Return serial fallback (best-effort) or error.
        if best_effort:
            return [], "ASCEND_RT_VISIBLE_DEVICES"
        raise DeviceDiscoveryError(
            f"ASCEND_RT_VISIBLE_DEVICES has {len(ids)} ids, need {requested}"
        )
    # 3. npu-smi healthy + idle.
    if smi_statuses is not None:
        candidates = [s.device_id for s in smi_statuses if s.healthy and not s.busy]
        if len(candidates) >= requested:
            return candidates, "npu-smi"

    # 4. /dev enumeration only with explicit opt-in.
    if assume_available:
        dev_ids = list_dev_davinci_ids()
        if len(dev_ids) >= requested:
            return dev_ids, "/dev"

    # 5. Best-effort serial fallback.
    if best_effort:
        return [], None
    raise DeviceDiscoveryError(
        "cannot determine NPU allocation safely; set PTO_ST_PARALLEL_DEVICES, "
        "provide ASCEND_RT_VISIBLE_DEVICES, or explicitly use --parallel-assume-available"
    )


def list_dev_davinci_ids():
    """Enumerate exact /dev/davinciN nodes, returning their numeric ids."""
    ids = []
    for ent in sorted(Path("/dev").iterdir()) if Path("/dev").is_dir() else []:
        m = re.fullmatch(r"davinci(\d+)", ent.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


# ---------------------------------------------------------------------------
# Cooperative flock
# ---------------------------------------------------------------------------

def lock_candidate(device_id, lock_dir=None):
    """Acquire a nonblocking exclusive flock for a device, or return None.

    Returns a DeviceLock, or None if the device cannot be locked.
    """
    if fcntl is None:
        raise SmokeError("fcntl unavailable; cooperative locking not supported on this platform")
    lock_dir = Path(lock_dir or os.environ.get("PTO_ST_LOCK_DIR", "/tmp"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"pto-isa-smoke-npu-{device_id}.lock"
    try:
        fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    # Record context for diagnostics (ownership is flock, not file contents).
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
    except OSError:
        # The diagnostic annotation is best-effort; the lock is still valid.
        pass
    return DeviceLock(device_id=device_id, fd=fd, path=str(path))


def release_lock(device_lock):
    """Release a device lock and close its fd, if any."""
    if device_lock is None or device_lock.fd is None:
        return
    if fcntl is not None:
        try:
            fcntl.flock(device_lock.fd, fcntl.LOCK_UN)
        except OSError:
            # Lock fd may already be invalid during teardown; ignore.
            pass
    try:
        os.close(device_lock.fd)
    except OSError:
        # fd may already be closed; ignore during teardown.
        pass


# ---------------------------------------------------------------------------
# Worker build-root isolation
# ---------------------------------------------------------------------------

def link_or_copy(src, dst):
    """Hard-link src to dst, falling back to copy2; return 'link' or 'copy'."""
    try:
        os.link(str(src), str(dst))
        return "link"
    except OSError:
        shutil.copy2(str(src), str(dst))
        return "copy"


def prepare_worker_build(base_build, worker_build):
    """Create a private worker_build/bin tree populated from base_build/bin."""
    base_bin = Path(base_build) / "bin"
    if not base_bin.is_dir():
        raise FileNotFoundError(f"base build/bin not found: {base_bin}")
    worker_bin = Path(worker_build) / "bin"
    worker_bin.mkdir(parents=True, exist_ok=True)
    for entry in base_bin.iterdir():
        if entry.is_file() and not entry.name.endswith((".o", ".d")):
            link_or_copy(entry, worker_bin / entry.name)
    return worker_bin


def remove_parallel_root(root):
    """Safely remove a .smoke-parallel run root if it is below a marker dir."""
    root = Path(root).resolve()
    # The run root must have a `.smoke-parallel` ancestor; never remove from an
    # unchecked path. This guards against dereferencing an arbitrary env var.
    marker = None
    for parent in root.parents:
        if parent.name == ".smoke-parallel":
            marker = parent.resolve()
            break
    if marker is None or marker not in root.parents:
        raise SmokeError(f"refusing to remove path outside .smoke-parallel: {root}")
    if root.exists():
        shutil.rmtree(str(root))


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------

def build_task_command(task, worker, run_st_script, auto_mode):
    """Build an argument list (never a shell string) for one task."""
    cmd = [
        sys.executable,
        os.path.abspath(str(run_st_script)),
        "-r", "npu",
        "-w",
        "-v", "a3",
        "-t", task.testcase,
        "--build-dir", str(worker.build_dir),
    ]
    if task.gtest_filter:
        cmd.extend(["-g", task.gtest_filter])
    if task.debug_enable:
        cmd.append("-d")
    if auto_mode:
        cmd.append("-a")
    return cmd


def run_one_task(task, worker, ctx):
    """Run one task on one worker; return TaskResult."""
    cmd = build_task_command(task, worker, ctx.run_st_script, ctx.auto_mode)
    worker_env = dict(os.environ)
    worker_env["ASCEND_RT_VISIBLE_DEVICES"] = str(worker.physical_device)

    start = ctx.now()
    with ctx.print_lock:
        print(f"[PARALLEL] task={task.index} testcase={task.testcase} "
              f"worker={worker.index} start")
    proc = subprocess.Popen(
        cmd,
        cwd=str(worker.build_dir),
        env=worker_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    group = _new_proc_group(proc)
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.append(group)
    log_handle = open(str(worker.log_path), "a", encoding="utf-8")
    try:
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                log_handle.write(line + "\n")
                log_handle.flush()
                with ctx.print_lock:
                    print(f"[worker-{worker.index}][device-{worker.physical_device}]"
                          f"[{task.testcase}] {line}")
            proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise
    finally:
        log_handle.close()
        proc.stdout.close()
        # Confirm the whole process group is gone before we consider the task
        # finished and drop its tracking. This handles both normal completion
        # (grandchildren may outlive the leader) and the exception path.
        finish_task_group(group, timeout=10)
    elapsed_ms = int((ctx.now() - start) * 1000)
    with ctx.print_lock:
        print(f"[PARALLEL] task={task.index} testcase={task.testcase} "
              f"worker={worker.index} exit={proc.returncode} elapsed_ms={elapsed_ms}")
    return TaskResult(
        task=task,
        worker_index=worker.index,
        physical_device=worker.physical_device,
        returncode=proc.returncode,
        elapsed_ms=elapsed_ms,
    )


def worker_loop(worker, task_queue, ctx, results):
    """Consume tasks from the queue until the sentinel, appending results."""
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break
        task = item
        try:
            result = run_one_task(task, worker, ctx)
        except Exception as e:
            result = TaskResult(
                task=task,
                worker_index=worker.index,
                physical_device=worker.physical_device,
                returncode=-1,
                elapsed_ms=0,
            )
            with print_lock:
                print(f"[PARALLEL][ERROR] task={task.index} testcase={task.testcase} "
                      f"worker={worker.index}: {e}")
        results.append(result)
        task_queue.task_done()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_serial_fallback(tasks, args, print_lock, now):
    """Best-effort serial fallback: run every manifest task in order.

    Uses the base build dir, does not remap device visibility, and returns
    nonzero if any task fails. Returns (exit_code, executed_count).
    """
    print("[PARALLEL][FALLBACK] requested=%d; executing %d tasks serially"
          % (args.workers, len(tasks)))
    executed = 0
    failed = False
    for task in tasks:
        cmd = [
            sys.executable,
            os.path.abspath(str(args.run_st_script)),
            "-r", "npu",
            "-w",
            "-v", "a3",
            "-t", task.testcase,
            "--build-dir", str(args.base_build_dir),
        ]
        if task.gtest_filter:
            cmd.extend(["-g", task.gtest_filter])
        if task.debug_enable:
            cmd.append("-d")
        if args.auto_mode:
            cmd.append("-a")
        # Serial fallback runs from the ST root with the shared build dir and
        # leaves device visibility untouched (inherited environment). It uses a
        # tracked Popen in its own process group so signal cleanup can terminate
        # the whole tree (run_st.py + gtest/NPU children).
        start = now()
        with print_lock:
            print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} start")
        proc = None
        group = None
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(args.st_root), start_new_session=True,
            )
            group = _new_proc_group(proc)
            with _ACTIVE_PROCS_LOCK:
                _ACTIVE_PROCS.append(group)
            proc.wait()
        except Exception as e:
            with print_lock:
                print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} "
                      f"error: {e}")
            failed = True
            executed += 1
            continue
        finally:
            if group is not None:
                # Confirm the whole process group is gone before dropping its
                # tracking (handles grandchildren outliving the leader).
                if not finish_task_group(group, timeout=10):
                    failed = True
                    with print_lock:
                        print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} "
                              f"process group could not be confirmed stopped")
        elapsed_ms = int((now() - start) * 1000)
        executed += 1
        ok = proc.returncode == 0
        with print_lock:
            print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} "
                  f"exit={proc.returncode} elapsed_ms={elapsed_ms} "
                  f"({'ok' if ok else 'FAIL'})")
        if not ok:
            failed = True
    print(f"[PARALLEL][SERIAL] done: executed={executed}/{len(tasks)} "
          f"failed={1 if failed else 0}")
    return (1 if failed else 0), executed


def main():
    parser = argparse.ArgumentParser(description="A3 parallel smoke orchestrator")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--st-root", required=True)
    parser.add_argument("--base-build-dir", required=True)
    parser.add_argument("--run-st-script", required=True)
    parser.add_argument("--run-mode", default="npu")
    parser.add_argument("--soc-version", default="a3")
    parser.add_argument("--auto-mode", action="store_true")
    parser.add_argument("--best-effort", action="store_true")
    parser.add_argument("--assume-available", action="store_true")
    args = parser.parse_args()

    if args.workers < 2:
        raise SmokeError(f"--workers must be >= 2, got {args.workers}")

    tasks = parse_task_manifest(args.manifest)
    now = time.perf_counter
    print_lock = threading.Lock()
    locks = []
    parallel_root = None
    workers = []
    threads = []

    _cleanup_state = {"done": False}

    def cleanup(success):
        if _cleanup_state["done"]:
            return True
        clean = True
        try:
            # 1. Terminate the whole process group of every tracked child,
            #    escalating TERM -> wait -> KILL. Only a group-that-is-confirmed-
            #    gone counts. Collect per-group results.
            with _ACTIVE_PROCS_LOCK:
                groups = list(_ACTIVE_PROCS)
            all_groups_gone = True
            for g in groups:
                if not terminate_proc_gracefully(g, timeout=10):
                    all_groups_gone = False

            # 2. Join worker threads (bounded wait); track whether any is alive.
            all_threads_stopped = True
            for thr in threads:
                thr.join(timeout=5)
                if thr.is_alive():
                    all_threads_stopped = False

            # 3. Only release device locks if every process group is confirmed
            #    gone AND every worker thread has stopped. Otherwise record the
            #    failure (returned to the caller) and keep locks.
            if all_groups_gone and all_threads_stopped:
                for lk in locks:
                    release_lock(lk)
            else:
                clean = False
                print(
                    "[PARALLEL][P0-ERROR] cleanup incomplete: "
                    "all_groups_gone=%s all_threads_stopped=%s; "
                    "device locks NOT released" % (all_groups_gone, all_threads_stopped)
                )

            # 4. Handle the run directory: remove on success, preserve on failure.
            if parallel_root is not None and parallel_root.exists():
                if success:
                    try:
                        remove_parallel_root(parallel_root)
                    except SmokeError as e:
                        print(f"[PARALLEL][WARN] {e}")
                else:
                    print(f"[PARALLEL][INFO] preserving run root for diagnosis: {parallel_root}")
            return clean
        finally:
            _cleanup_state["done"] = True

    def handle_signal(signum, frame):
        print(f"[PARALLEL] received signal {signum}; terminating workers")
        clean = cleanup(success=False)
        sys.exit(128 + signum if clean else 1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- device selection ---
    explicit = os.environ.get("PTO_ST_PARALLEL_DEVICES")
    inherited = parse_inherited_visible(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
    smi_statuses = None
    try:
        smi_statuses = query_npu_smi()
    except DeviceDiscoveryError as e:
        if not (explicit or inherited or args.assume_available or args.best_effort):
            raise
        smi_statuses = None
    devices, source = select_devices(
        args.workers,
        DeviceSources(explicit=explicit, inherited=inherited, smi_statuses=smi_statuses),
        assume_available=args.assume_available,
        best_effort=args.best_effort,
    )
    if not devices:
        if args.best_effort:
            return run_serial_fallback(tasks, args, print_lock, now)[0]
        raise SmokeError(
            "cannot determine NPU allocation safely; set PTO_ST_PARALLEL_DEVICES, "
            "provide ASCEND_RT_VISIBLE_DEVICES, or explicitly use --parallel-assume-available"
        )
    print("[PARALLEL] candidate_pool=%s source=%s"
          % (",".join(str(d) for d in devices), source))

    # --- cooperative locking: iterate the full candidate pool until N locks ---
    for dev in devices:
        if len(locks) >= args.workers:
            break
        lock = lock_candidate(dev)
        if lock is not None:
            locks.append(lock)
    if len(locks) < args.workers:
        for lk in locks:
            release_lock(lk)
        if args.best_effort:
            return run_serial_fallback(tasks, args, print_lock, now)[0]
        raise SmokeError(
            f"could not lock {args.workers} devices (locked {len(locks)})"
        )
    locked_devices = [lk.device_id for lk in locks]
    print("[PARALLEL] requested_workers=%d locked_devices=%s"
          % (args.workers, ",".join(str(d) for d in locked_devices)))

    result_holder = {"rc": 0}
    try:
        # --- private worker build roots ---
        parallel_root = Path(args.st_root) / ".smoke-parallel" / str(os.getpid())
        parallel_root.mkdir(parents=True, exist_ok=True)
        for i, dev in enumerate(locked_devices):
            worker_build = parallel_root / f"worker-{i}" / "build"
            prepare_worker_build(args.base_build_dir, worker_build)
            log_path = parallel_root / f"worker-{i}" / "worker.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            workers.append(WorkerConfig(
                index=i,
                physical_device=dev,
                build_dir=worker_build,
                log_path=log_path,
            ))
            print("[PARALLEL] worker=%d device=%d build_dir=%s"
                  % (i, dev, worker_build))

        # --- dynamic scheduling ---
        task_queue = queue.Queue()
        for t in tasks:
            task_queue.put(t)
        for w in workers:
            task_queue.put(None)  # sentinel per worker

        results = []
        ctx = WorkerContext(
            run_st_script=args.run_st_script,
            auto_mode=args.auto_mode,
            print_lock=print_lock,
            now=now,
        )
        for w in workers:
            thr = threading.Thread(
                target=worker_loop,
                args=(w, task_queue, ctx, results),
            )
            thr.start()
            threads.append(thr)
        for thr in threads:
            thr.join()

        # --- summary (deterministic task-index order) ---
        results.sort(key=lambda r: r.task.index)
        failed = any(r.returncode != 0 for r in results)
        with print_lock:
            for r in results:
                status = "ok" if r.returncode == 0 else "FAIL"
                print(f"[PARALLEL][RESULT] task={r.task.index} testcase={r.task.testcase} "
                      f"worker={r.worker_index} exit={r.returncode} elapsed_ms={r.elapsed_ms} {status}")

        result_holder["rc"] = 1 if failed else 0
    finally:
        # If an exception is propagating out of the try, treat the run as
        # failed so the diagnostic directory is preserved.
        if sys.exc_info()[0] is not None:
            result_holder["rc"] = 1
        # Always release locks, terminate/wait child processes, join threads.
        # Remove the run dir only on success; preserve it on failure/exception.
        # A failed cleanup must override the return code to nonzero: the caller
        # (and the pipeline success marker) must never see a clean run when a
        # process group or worker thread could not be confirmed stopped.
        clean = cleanup(result_holder["rc"] == 0)
        if not clean:
            result_holder["rc"] = 1
            print("[PARALLEL][P0-ERROR] orchestrator returning nonzero because cleanup was incomplete")
    return result_holder["rc"]


if __name__ == "__main__":
    sys.exit(main())