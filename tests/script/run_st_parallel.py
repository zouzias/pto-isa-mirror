#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# coding=utf-8
# A3 simple multi-NPU parallel smoke orchestrator.
# Standard library only. No NPU binaries are executed by this module directly.
# --------------------------------------------------------------------------------

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
_ACTIVE_PROCS = []
_ACTIVE_PROCS_LOCK = threading.Lock()


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
    saw_process_table = False
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "Process id" in stripped:
            proc_section = True
            saw_process_table = True
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
    # Every device must have a confirmed process status: either a busy row or
    # an explicit "No running processes found in NPU N" marker. If the process
    # table was present but a device got neither, we cannot assume it is idle.
    if saw_process_table:
        for dev_id in device_rows:
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

def select_devices(requested, explicit=None, inherited=None, smi_statuses=None,
                   assume_available=False, best_effort=False):
    """Select up to `requested` device ids using the documented priority.

    Returns (devices, source) where source is one of:
      "PTO_ST_PARALLEL_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "npu-smi",
      "/dev", or None (insufficient).
    """
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
        if not best_effort:
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
        pass
    return DeviceLock(device_id=device_id, fd=fd, path=str(path))


def release_lock(device_lock):
    """Release a device lock and close its fd, if any."""
    if device_lock is None:
        return
    try:
        fcntl.flock(device_lock.fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(device_lock.fd)
    except Exception:
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
        str(run_st_script),
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


def run_one_task(task, worker, run_st_script, auto_mode, print_lock, now):
    """Run one task on one worker; return TaskResult."""
    cmd = build_task_command(task, worker, run_st_script, auto_mode)
    worker_env = dict(os.environ)
    worker_env["ASCEND_RT_VISIBLE_DEVICES"] = str(worker.physical_device)

    start = now()
    with print_lock:
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
    )
    with _ACTIVE_PROCS_LOCK:
        _ACTIVE_PROCS.append(proc)
    log_handle = open(str(worker.log_path), "a", encoding="utf-8")
    try:
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                log_handle.write(line + "\n")
                log_handle.flush()
                with print_lock:
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
        proc.wait()
        with _ACTIVE_PROCS_LOCK:
            if proc in _ACTIVE_PROCS:
                _ACTIVE_PROCS.remove(proc)
    elapsed_ms = int((now() - start) * 1000)
    with print_lock:
        print(f"[PARALLEL] task={task.index} testcase={task.testcase} "
              f"worker={worker.index} exit={proc.returncode} elapsed_ms={elapsed_ms}")
    return TaskResult(
        task=task,
        worker_index=worker.index,
        physical_device=worker.physical_device,
        returncode=proc.returncode,
        elapsed_ms=elapsed_ms,
    )


def worker_loop(worker, task_queue, run_st_script, auto_mode, print_lock, results, now):
    """Consume tasks from the queue until the sentinel, appending results."""
    while True:
        item = task_queue.get()
        if item is None:
            task_queue.task_done()
            break
        task = item
        try:
            result = run_one_task(task, worker, run_st_script, auto_mode, print_lock, now)
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
            str(args.run_st_script),
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
        # leaves device visibility untouched (inherited environment).
        start = now()
        with print_lock:
            print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} start")
        try:
            proc = subprocess.run(cmd, cwd=str(args.st_root))
        except Exception as e:
            with print_lock:
                print(f"[PARALLEL][SERIAL] task={task.index} testcase={task.testcase} "
                      f"error: {e}")
            failed = True
            executed += 1
            continue
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

    def cleanup():
        for lk in locks:
            release_lock(lk)
        if parallel_root is not None and parallel_root.exists():
            try:
                remove_parallel_root(parallel_root)
            except SmokeError as e:
                print(f"[PARALLEL][WARN] {e}")

    def handle_signal(signum, frame):
        print(f"[PARALLEL] received signal {signum}; terminating workers")
        with _ACTIVE_PROCS_LOCK:
            procs = list(_ACTIVE_PROCS)
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        cleanup()
        sys.exit(128 + signum)

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
        explicit=explicit,
        inherited=inherited,
        smi_statuses=smi_statuses,
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
        for w in workers:
            thr = threading.Thread(
                target=worker_loop,
                args=(w, task_queue, args.run_st_script, args.auto_mode,
                      print_lock, results, now),
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

        # --- cleanup ---
        if not failed:
            cleanup()
        else:
            print(f"[PARALLEL][INFO] preserving run root for diagnosis: {parallel_root}")
            for lk in locks:
                release_lock(lk)

        return 1 if failed else 0
    finally:
        threads.clear()


if __name__ == "__main__":
    sys.exit(main())