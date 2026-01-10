# --------------------------------------------------------------------------------
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Optional


def _run(cmd, cwd=None, env=None, timeout_sec=None, capture=False, check=True):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        raise TimeoutError(f"timeout after {timeout_sec}s: {' '.join(cmd)}")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=out)
    return out or ""


def _source_ascend_env():
    # Try to source the standard user install location.
    candidates = [
        os.path.expanduser("~/Ascend/ascend-toolkit/set_env.sh"),
        os.path.expanduser("~/Ascend/ascend-toolkit/latest/bin/setenv.bash"),
    ]
    for p in candidates:
        if Path(p).exists():
            script = p
            break
    else:
        raise FileNotFoundError(f"Cannot find Ascend env script under: {candidates}")

    # Use `env -0` to avoid newline issues.
    out = _run(
        ["bash", "-lc", f"source {script} >/dev/null 2>&1 && env -0"],
        capture=True,
        timeout_sec=60,
    )
    new_env = dict(os.environ)
    for item in out.split("\0"):
        if not item:
            continue
        k, v = item.split("=", 1)
        new_env[k] = v
    return new_env


def _parse_npu_smi_info(text):
    devices = {}
    processes = {}  # pid -> (npu_id, proc_name, mem_mb)

    for line in text.splitlines():
        # Device table: `| 0     910B1 | OK | ...`
        m = re.match(r"^\|\s*(\d+)\s+\S+\s*\|\s*(OK|WARN|WARNING|FAULT|ERROR)\s*\|", line, re.IGNORECASE)
        if m:
            npu_id = int(m.group(1))
            health = m.group(2).upper()
            devices[npu_id] = {"health": health}
            continue

        # Process table: `| 0  0 | 12345 | python3 | 102 |`
        m = re.match(r"^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|", line)
        if m:
            npu_id = int(m.group(1))
            pid = int(m.group(2))
            name = m.group(3).strip()
            mem = int(m.group(4))
            processes[pid] = (npu_id, name, mem)

    return devices, processes


class _NpuSmiPoller:
    def __init__(self, env, poll_sec: float):
        self._env = env
        self._poll_sec = max(0.5, float(poll_sec))
        self._lock = threading.Lock()
        self._devices = {}
        self._processes = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def snapshot(self):
        with self._lock:
            return dict(self._devices), dict(self._processes)

    def _run(self):
        while not self._stop.is_set():
            try:
                devices, processes = _npu_smi_info(self._env)
                with self._lock:
                    self._devices = devices
                    self._processes = processes
            except Exception:
                pass
            self._stop.wait(self._poll_sec)


def _sanitize_env_for_npu_tools(env):
    """
    `npu-smi` (and other Ascend tools) may honor visibility env vars and only
    report a subset of devices. For runner scheduling we want physical device
    ids, so strip known visibility filters.
    """
    clean = dict(env or os.environ)
    clean.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    clean.pop("ASCEND_VISIBLE_DEVICES", None)
    return clean


def _npu_smi_info(env):
    out = _run(
        ["npu-smi", "info"],
        env=_sanitize_env_for_npu_tools(env),
        capture=True,
        timeout_sec=30,
        check=False,
    )
    return _parse_npu_smi_info(out)


def _device_procs(processes, npu_id):
    return {pid: (name, mem) for pid, (dev, name, mem) in processes.items() if dev == npu_id}


def _filter_ignored_procs(procs, ignore_proc_re):
    if ignore_proc_re is None:
        return procs
    return {pid: v for pid, v in procs.items() if ignore_proc_re.search(v[0]) is None}


def _wait_device_idle(
    env,
    npu_id,
    timeout_sec,
    poll_sec=1.0,
    ignore_proc_re=None,
    smi: Optional["_NpuSmiPoller"] = None,
    stop_event: Optional[threading.Event] = None,
):
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        if stop_event is not None and stop_event.is_set():
            return False, last
        if ignore_proc_re is not None and ignore_proc_re.pattern == "":
            ignore_proc_re = None
        if smi is not None:
            devices, processes = smi.snapshot()
            if not devices:
                devices, processes = _npu_smi_info(env)
        else:
            devices, processes = _npu_smi_info(env)
        health = devices.get(npu_id, {}).get("health", "UNKNOWN")
        procs = _filter_ignored_procs(_device_procs(processes, npu_id), ignore_proc_re)
        last = (health, procs)
        if health != "OK":
            return False, last
        if not procs:
            return True, last
        if stop_event is not None:
            stop_event.wait(poll_sec)
        else:
            time.sleep(poll_sec)
    return False, last


def _get_free_devices(env, requested=None, ignore_proc_re=None, smi: Optional["_NpuSmiPoller"] = None):
    if smi is not None:
        devices, processes = smi.snapshot()
        if not devices:
            devices, processes = _npu_smi_info(env)
    else:
        devices, processes = _npu_smi_info(env)

    ok = []
    for npu_id, info in sorted(devices.items()):
        if requested is not None and npu_id not in requested:
            continue
        if info.get("health") != "OK":
            continue
        ok.append(npu_id)

    busy = set()
    for _, (npu_id, name, _) in processes.items():
        if ignore_proc_re is not None and ignore_proc_re.search(name) is not None:
            continue
        busy.add(npu_id)
    free = [d for d in ok if d not in busy]
    return free


def _get_ok_devices(env, requested=None, smi: Optional["_NpuSmiPoller"] = None):
    if smi is not None:
        devices, _ = smi.snapshot()
        if not devices:
            devices, _ = _npu_smi_info(env)
    else:
        devices, _ = _npu_smi_info(env)
    ok = []
    for npu_id, info in sorted(devices.items()):
        if requested is not None and npu_id not in requested:
            continue
        if info.get("health") != "OK":
            continue
        ok.append(npu_id)
    return ok


def _extract_a3_testcases():
    cmake_path = Path("tests/npu/a2a3/src/st/testcase/CMakeLists.txt")
    text = cmake_path.read_text(encoding="utf-8", errors="ignore")
    if "set(ALL_TESTCASES" not in text:
        raise RuntimeError(f"Cannot find ALL_TESTCASES in {cmake_path}")
    block = text.split("set(ALL_TESTCASES", 1)[1].split(")", 1)[0]
    testcases = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        testcases.append(line)
    return testcases


def _build_all_a3(env, jobs):
    build_env = dict(env)
    build_env["PTO_ST_JOBS"] = str(jobs)
    st_dir = Path("tests/npu/a2a3/src/st").resolve()
    build_dir = (st_dir / "build").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    # Incremental build (much faster than `build_st.py`, which wipes `build/`).
    _run(["cmake", "-DRUN_MODE=npu", "-DSOC_VERSION=Ascend910B1", ".."], cwd=str(build_dir), env=build_env, timeout_sec=1800)
    _run(["make", "-j", str(jobs)], cwd=str(build_dir), env=build_env, timeout_sec=3600)


def _gen_all_goldens(env, testcases):
    st_dir = Path("tests/npu/a2a3/src/st")
    build_dir = st_dir / "build"
    if not build_dir.exists():
        raise FileNotFoundError(f"build dir not found: {build_dir}")

    for tc in testcases:
        src = st_dir / "testcase" / tc / "gen_data.py"
        if not src.exists():
            raise FileNotFoundError(f"missing gen_data.py: {src}")
        dst = build_dir / "gen_data.py"
        shutil.copyfile(src, dst)
        _run([sys.executable, "gen_data.py"], cwd=str(build_dir), env=env, timeout_sec=600)


def _sanitize_filename(s, limit=180):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    if len(s) > limit:
        return s[:limit]
    return s


def _parse_gtest_list(text):
    tests = []
    current_suite = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):  # suite line like "TMULSTest."
            current_suite = line.strip()
            continue
        if current_suite is None:
            continue
        test_name = line.strip().split("#", 1)[0].strip()
        if not test_name:
            continue
        tests.append(f"{current_suite}{test_name}")
    return tests


def _list_gtests(env, testcase):
    st_dir = Path("tests/npu/a2a3/src/st")
    bin_dir = st_dir / "build" / "bin"
    exe = (bin_dir / testcase).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"missing binary: {exe}")
    out = _run([str(exe), "--gtest_list_tests"], cwd=str(bin_dir), env=env, capture=True, timeout_sec=120)
    tests = _parse_gtest_list(out)
    if not tests:
        raise RuntimeError(f"no gtests found in {exe} (unexpected empty --gtest_list_tests)")
    return tests


def _prepare_sandbox(env, testcase, gtest_filter, attempt, work_dir):
    """
    Create an isolated per-task sandbox so tests can freely write temporary output
    files without polluting the shared build/ golden directories.

    Layout:
      <work_dir>/<task_tag>_try<attempt>/
        bin/<testcase>  (symlink to build/bin/<testcase>)
        <Suite.case>/   (copied golden folder for that gtest case)
    """
    st_dir = Path("tests/npu/a2a3/src/st")
    build_dir = st_dir / "build"
    bin_dir = build_dir / "bin"
    exe_src = (bin_dir / testcase).resolve()
    if not exe_src.exists():
        raise FileNotFoundError(f"missing binary: {exe_src}")

    task_name = testcase if not gtest_filter else f"{testcase}::{gtest_filter}"
    task_tag = _sanitize_filename(task_name)
    sandbox_root = (Path(work_dir) / f"{task_tag}_try{attempt}").resolve()
    sandbox_bin = sandbox_root / "bin"
    sandbox_bin.mkdir(parents=True, exist_ok=True)

    exe_dst = sandbox_bin / testcase
    if not exe_dst.exists():
        os.symlink(str(exe_src), str(exe_dst))

    case_dirs = [gtest_filter] if gtest_filter else _list_gtests(env, testcase)
    for case_dir in case_dirs:
        src_dir = (build_dir / case_dir).resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"missing golden dir: {src_dir}")
        dst_dir = sandbox_root / case_dir
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

    return sandbox_root, sandbox_bin


def _apply_device_env(run_env, physical_device_id: int, device_env_mode: str):
    """
    Map a single worker process to a single physical NPU.

    `physical`:
      - Keep physical numbering; the test must call `aclrtSetDevice(<physical>)`.
      - Clear visibility envs to avoid remapping.

    `visible`:
      - Restrict to a single physical NPU via visibility envs.
      - Set `*_DEVICE_ID=0` so even tests that default to device 0 will run on the
        mapped physical device (CUDA_VISIBLE_DEVICES-style behavior).
    """
    if device_env_mode == "visible":
        run_env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device_id)
        run_env["ASCEND_VISIBLE_DEVICES"] = str(physical_device_id)
        run_env["PTO_ST_PHYSICAL_DEVICE_ID"] = str(physical_device_id)
        for k in ("PTO_ST_DEVICE_ID", "DEVICE_ID", "ACL_DEVICE_ID"):
            run_env[k] = "0"
        return

    if device_env_mode == "physical":
        run_env["PTO_ST_DEVICE_ID"] = str(physical_device_id)
        run_env["DEVICE_ID"] = str(physical_device_id)
        run_env["ACL_DEVICE_ID"] = str(physical_device_id)
        run_env.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        run_env.pop("ASCEND_VISIBLE_DEVICES", None)
        return

    raise ValueError(f"unknown device_env_mode: {device_env_mode}")


def _run_one_binary(
    env,
    device_id,
    testcase,
    timeout_sec,
    gtest_filter=None,
    attempt=1,
    work_dir=None,
    monitor_npu_smi=False,
    smi: Optional["_NpuSmiPoller"] = None,
    device_env_mode: str = "physical",
    run_id: str = "",
    ascend_work_root: str = "/tmp/pto-isa-ascend-work",
    preserve_ascend_work_path: bool = False,
):
    st_dir = Path("tests/npu/a2a3/src/st")
    build_bin_dir = st_dir / "build" / "bin"

    sandbox_root = None
    sandbox_bin_dir = None
    if work_dir:
        sandbox_root, sandbox_bin_dir = _prepare_sandbox(env, testcase, gtest_filter, attempt, work_dir)
        bin_dir = sandbox_bin_dir
    else:
        bin_dir = build_bin_dir

    exe = (bin_dir / testcase).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"missing binary: {exe}")

    run_env = dict(env)
    _apply_device_env(run_env, physical_device_id=int(device_id), device_env_mode=device_env_mode)
    if not preserve_ascend_work_path:
        root = Path(ascend_work_root).resolve()
        dev_dir = (root / (run_id or "no_run_id") / f"dev{int(device_id)}").resolve()
        dev_dir.mkdir(parents=True, exist_ok=True)
        run_env["ASCEND_WORK_PATH"] = str(dev_dir)
        run_env["TMPDIR"] = str(dev_dir / "tmp")
        Path(run_env["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    task_name = testcase if not gtest_filter else f"{testcase}::{gtest_filter}"

    start = time.time()
    npu_seen = False
    max_mem_mb = 0
    npu_health = "UNKNOWN"

    cmd = [str(exe)]
    if gtest_filter:
        cmd.append(f"--gtest_filter={gtest_filter}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(bin_dir),
        env=run_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    pid = proc.pid

    output_tail = deque(maxlen=int(os.environ.get("PTO_ST_OUTPUT_TAIL_LINES", "200")))

    def _drain_stdout():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                output_tail.append(line.rstrip("\n"))
        except Exception:
            # Best-effort; do not let output collection affect test execution.
            return

    drain_thread = threading.Thread(target=_drain_stdout, daemon=True)
    drain_thread.start()

    if monitor_npu_smi:
        poll_interval = float(os.environ.get("PTO_ST_SMI_POLL_SEC", "2.0"))
        last_poll = 0.0
        rc = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break

            now = time.time()
            if now - start > timeout_sec:
                os.killpg(proc.pid, signal.SIGKILL)
                output_tail.append(f"[TIMEOUT] killed after {timeout_sec}s")
                rc = 124
                break

            if now - last_poll >= poll_interval:
                if smi is not None:
                    devices, processes = smi.snapshot()
                    if not devices:
                        devices, processes = _npu_smi_info(env)
                else:
                    devices, processes = _npu_smi_info(env)
                npu_health = devices.get(device_id, {}).get("health", "UNKNOWN")
                if npu_health != "OK":
                    os.killpg(proc.pid, signal.SIGKILL)
                    output_tail.append(f"[NPU] health={npu_health}; killed")
                    rc = 125
                    break
                if pid in processes:
                    npu_seen = True
                    dev_seen, _, mem = processes[pid]
                    if dev_seen != device_id:
                        os.killpg(proc.pid, signal.SIGKILL)
                        output_tail.append(f"[NPU] pid={pid} mapped to dev={dev_seen} (expected {device_id}); killed")
                        rc = 129
                        break
                    max_mem_mb = max(max_mem_mb, mem)
                last_poll = now

            time.sleep(0.2)
    else:
        # Fast path: rely on process exit code + timeout, without calling `npu-smi`
        # during execution (much lower overhead when running on many NPUs).
        rc = None
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output_tail.append(f"[TIMEOUT] killed after {timeout_sec}s")
            rc = 124

    if proc.poll() is None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
    proc.returncode = rc if rc is not None else proc.returncode
    try:
        drain_thread.join(timeout=2)
    except Exception:
        pass

    elapsed = time.time() - start

    return {
        "testcase": testcase,
        "task": task_name,
        "gtest_filter": gtest_filter or "",
        "device": device_id,
        "rc": proc.returncode,
        "elapsed_sec": elapsed,
        "work_dir": str(sandbox_root) if sandbox_root else "",
        "pid": proc.pid,
        "npu_seen": npu_seen,
        "max_mem_mb": max_mem_mb,
        "npu_health": npu_health,
        "output_tail": "\n".join(output_tail),
    }


def _format_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    out = [fmt.format(*headers), sep]
    out.extend(fmt.format(*[str(c) for c in r]) for r in rows)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Build + run A3 NPU ST across free NPUs")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="parallel test jobs (0 means use all selected NPUs)",
    )
    ap.add_argument(
        "--build-jobs",
        type=int,
        default=int(os.environ.get("PTO_ST_BUILD_JOBS", os.environ.get("PTO_ST_JOBS", "32"))),
        help="`make -j` for building ST binaries",
    )
    ap.add_argument("--timeout-sec", type=int, default=int(os.environ.get("PTO_ST_TIMEOUT_SEC", "600")),
                    help="per-testcase binary timeout (detect deadlock)")
    ap.add_argument("--split-gtest", action="store_true",
                    help="split each testcase binary into per-gtest tasks (improves load balance across NPUs)")
    ap.add_argument("--golden-workers", type=int, default=int(os.environ.get("PTO_ST_GOLDEN_WORKERS", "0")),
                    help="0 means auto; parallelism for golden generation (gen_data.py)")
    ap.add_argument("--max-workers", type=int, default=None, help="deprecated: use -j/--jobs (0 means use all selected NPUs)")
    ap.add_argument("--devices", default="", help="comma-separated physical NPU ids to use (default: auto-detect)")
    ap.add_argument("--include-busy-devices", action="store_true",
                    help="use all OK NPUs even if currently busy (wait/retry until idle)")
    ap.add_argument("--free-only", action="store_true",
                    help="only use NPUs that are idle at startup (no waiting for busy devices)")
    ap.add_argument(
        "--device-env-mode",
        choices=["physical", "visible"],
        default=os.environ.get("PTO_ST_DEVICE_ENV_MODE", "visible"),
        help="device selection mode per process (`visible` is most robust for full parallelism)",
    )
    ap.add_argument(
        "--ascend-work-root",
        default=os.environ.get("PTO_ST_ASCEND_WORK_ROOT", "/tmp/pto-isa-ascend-work"),
        help="root for per-device ASCEND_WORK_PATH + TMPDIR (reduces cross-process contention)",
    )
    ap.add_argument(
        "--preserve-ascend-work-path",
        action="store_true",
        help="do not override ASCEND_WORK_PATH/TMPDIR in worker processes",
    )
    ap.add_argument("--ignore-proc-regex", default=os.environ.get("PTO_ST_IGNORE_PROC_REGEX", r"npu-smi"),
                    help="treat NPU processes matching this regex as ignorable when detecting busy/idle")
    ap.add_argument("--testcases", default="", help="comma-separated testcase names to run (default: all A3 ST)")
    ap.add_argument("--skip-build", action="store_true", help="skip build step (assumes `tests/npu/a2a3/src/st/build` exists)")
    ap.add_argument("--skip-golden", action="store_true", help="skip golden generation step (assumes goldens exist)")
    ap.add_argument("--isolate", action="store_true", help="run each testcase in a private sandbox (slower)")
    ap.add_argument("--monitor-npu-smi", action="store_true",
                    help="poll `npu-smi` during test execution (slower, but can catch unhealthy devices)")
    ap.add_argument(
        "--smi-poll-sec",
        type=float,
        default=float(os.environ.get("PTO_ST_SMI_POLL_SEC", "1.0")),
        help="shared `npu-smi` polling interval for scheduler/monitoring (seconds)",
    )
    ap.add_argument("--work-dir", default="/tmp/pto-isa-st-work",
                    help="per-task sandbox root (each test runs in its own folder)")
    ap.add_argument("--device-idle-wait-sec", type=int, default=int(os.environ.get("PTO_ST_DEVICE_IDLE_WAIT_SEC", "30")),
                    help="wait for a device to become idle between tests (helps recover after kill)")
    ap.add_argument("--device-busy-backoff-sec", type=float, default=float(os.environ.get("PTO_ST_DEVICE_BUSY_BACKOFF_SEC", "1.0")),
                    help="sleep before retrying a busy device (only affects scheduling, not correctness)")
    args = ap.parse_args()

    env = _source_ascend_env()
    run_id = os.environ.get("PTO_ST_RUN_ID") or f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    work_dir = None
    if args.isolate:
        work_dir = (Path(args.work_dir) / run_id).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] run_id={run_id}")
    if work_dir:
        print(f"[INFO] work_dir={work_dir}")
    sys.stdout.flush()

    ignore_proc_re = None
    if args.ignore_proc_regex.strip():
        ignore_proc_re = re.compile(args.ignore_proc_regex)

    requested = None
    if args.devices.strip():
        requested = {int(x) for x in args.devices.split(",") if x.strip() != ""}

    smi = _NpuSmiPoller(env, poll_sec=args.smi_poll_sec)
    smi.start()
    try:
        run_jobs_env_default = int(os.environ.get("PTO_ST_JOBS_RUN", "0"))
        run_jobs = args.jobs if args.jobs is not None else run_jobs_env_default
        # Backward-compat: `--max-workers` historically controlled run concurrency.
        if args.max_workers is not None and args.jobs is None:
            run_jobs = args.max_workers

        ok_devices = _get_ok_devices(env, requested=requested, smi=smi)
        if not ok_devices:
            raise RuntimeError("No NPUs with Health=OK found (or all requested NPUs are unhealthy).")
        if args.free_only:
            free = _get_free_devices(env, requested=set(ok_devices), ignore_proc_re=ignore_proc_re, smi=smi)
            if not free:
                raise RuntimeError("No free NPUs with Health=OK found (or all requested NPUs are busy).")
            desired_workers = len(free) if run_jobs == 0 else min(run_jobs, len(free))
            device_pool = free[:desired_workers]
        else:
            desired_workers = len(ok_devices) if run_jobs == 0 else min(run_jobs, len(ok_devices))
            device_pool = ok_devices[:desired_workers]

        # Note: We always start one worker per device in the pool. Workers wait
        # for their device to become idle before pulling tasks, so busy devices
        # do not create scheduling bubbles.
        print(f"[INFO] NPU pool: {device_pool} (workers={desired_workers}, free_only={args.free_only})")
        devices = list(device_pool)
        sys.stdout.flush()

        testcases = _extract_a3_testcases()
        if args.testcases.strip():
            wanted = {x.strip() for x in args.testcases.split(",") if x.strip()}
            testcases = [t for t in testcases if t in wanted]
            missing = sorted(wanted - set(testcases))
            if missing:
                raise RuntimeError(f"unknown testcase(s): {missing}")
        print(f"[INFO] A3 testcases: {len(testcases)}")
        sys.stdout.flush()

        if not args.skip_build:
            print("[INFO] building all A3 NPU ST (incremental)...")
            sys.stdout.flush()
            _build_all_a3(env, jobs=args.build_jobs)

        golden_workers = args.golden_workers
        if golden_workers <= 0:
            golden_workers = min(max(1, os.cpu_count() or 1), 16)

        if not args.skip_golden:
            print(f"[INFO] generating golden data (parallel workers={golden_workers})...")
            sys.stdout.flush()
            st_dir = Path("tests/npu/a2a3/src/st")
            build_dir = st_dir / "build"
            if not build_dir.exists():
                raise FileNotFoundError(f"build dir not found: {build_dir}")

            def _gen_one(tc):
                src = (st_dir / "testcase" / tc / "gen_data.py").resolve()
                if not src.exists():
                    raise FileNotFoundError(f"missing gen_data.py: {src}")
                stamp = (build_dir / f".gen_data_{tc}.stamp").resolve()
                if stamp.exists() and stamp.stat().st_mtime >= src.stat().st_mtime:
                    return
                # Copy to an isolated filename in build_dir to avoid races on a shared
                # `build/gen_data.py` when running goldens in parallel.
                dst = (build_dir / f"gen_data_{tc}.py").resolve()
                shutil.copyfile(src, dst)
                _run([sys.executable, str(dst)], cwd=str(build_dir), env=env, timeout_sec=600)
                stamp.touch()

            with ThreadPoolExecutor(max_workers=golden_workers) as pool:
                futs = [pool.submit(_gen_one, tc) for tc in testcases]
                for f in futs:
                    f.result()

        tasks = []
        if args.split_gtest:
            print("[INFO] splitting binaries into per-gtest tasks...")
            sys.stdout.flush()
            for tc in testcases:
                for g in _list_gtests(env, tc):
                    tasks.append({"testcase": tc, "gtest_filter": g})
            print(f"[INFO] total tasks after split: {len(tasks)}")
            sys.stdout.flush()
        else:
            tasks = [{"testcase": tc, "gtest_filter": None} for tc in testcases]

        for i, t in enumerate(tasks, start=1):
            t["idx"] = i
        total_tasks = len(tasks)

        print("[INFO] running testcase binaries in parallel (1 process per NPU)...")
        sys.stdout.flush()
        results = []
        failures = []

        work = Queue()
        for t in tasks:
            work.put(t)

        results_lock = threading.Lock()
        attempts = {}
        attempts_lock = threading.Lock()
        stop_event = threading.Event()

        device_lock = threading.Lock()
        active_devices = set()
        worker_threads = []

        def _disable_device(dev, health):
            with device_lock:
                active_devices.discard(dev)
            print(f"[WARN] dev={dev} disabled (health={health})")
            sys.stdout.flush()

        def _device_worker(dev: int):
            while not stop_event.is_set():
                # Do not reserve a task before the device is idle (prevents busy
                # devices from "holding" tasks and creating bubbles).
                idle_ok, last = _wait_device_idle(
                    env,
                    dev,
                    timeout_sec=max(1, int(args.device_idle_wait_sec)),
                    poll_sec=0.5,
                    ignore_proc_re=ignore_proc_re,
                    smi=smi,
                    stop_event=stop_event,
                )
                if stop_event.is_set():
                    return
                if not idle_ok:
                    health, _procs = last or ("UNKNOWN", {})
                    if health != "OK":
                        _disable_device(dev, health)
                        return
                    stop_event.wait(max(0.0, float(args.device_busy_backoff_sec)))
                    continue

                try:
                    task = work.get(timeout=0.2)
                except Empty:
                    with device_lock:
                        if not active_devices:
                            return
                    continue

                res = None
                try:
                    with attempts_lock:
                        key = (task["testcase"], task.get("gtest_filter") or "")
                        attempt = attempts.get(key, 0) + 1
                        attempts[key] = attempt
                    print(
                        f"[{task['idx']}/{total_tasks}] RUN  dev={dev} "
                        f"{task['testcase']} filter={task.get('gtest_filter') or '-'} try={attempt}"
                    )
                    sys.stdout.flush()
                    res = _run_one_binary(
                        env,
                        dev,
                        task["testcase"],
                        args.timeout_sec,
                        gtest_filter=task.get("gtest_filter"),
                        attempt=attempt,
                        work_dir=str(work_dir) if work_dir else None,
                        monitor_npu_smi=args.monitor_npu_smi,
                        smi=smi,
                        device_env_mode=args.device_env_mode,
                        run_id=run_id,
                        ascend_work_root=args.ascend_work_root,
                        preserve_ascend_work_path=args.preserve_ascend_work_path,
                    )
                    res["idx"] = task["idx"]
                except Exception as e:
                    res = {
                        "testcase": task["testcase"],
                        "task": task["testcase"] if not task.get("gtest_filter") else f"{task['testcase']}::{task['gtest_filter']}",
                        "gtest_filter": task.get("gtest_filter") or "",
                        "device": dev if dev is not None else -1,
                        "rc": 128,
                        "elapsed_sec": 0.0,
                        "log": "",
                        "pid": -1,
                        "npu_seen": False,
                        "max_mem_mb": 0,
                        "npu_health": "UNKNOWN",
                        "error": f"runner exception: {type(e).__name__}: {e}",
                        "idx": task.get("idx", 0),
                    }
                finally:
                    if res is not None:
                        with results_lock:
                            results.append(res)
                            status = "PASS" if res["rc"] == 0 else "FAIL"
                            print(
                                f"[{res.get('idx', 0)}/{total_tasks}] {status:4s} dev={res['device']} "
                                f"{res.get('task', res['testcase'])} sec={res['elapsed_sec']:.1f} rc={res['rc']}"
                            )
                            sys.stdout.flush()
                            if res["rc"] != 0:
                                failures.append(res)
                    work.task_done()

        def _start_device(dev: int):
            with device_lock:
                if dev in active_devices:
                    return
                active_devices.add(dev)
            t = threading.Thread(target=_device_worker, args=(dev,), daemon=True)
            worker_threads.append(t)
            t.start()

        for d in device_pool:
            _start_device(d)

        # Wait for all tasks to complete.
        while True:
            with results_lock:
                done = len(results)
            if done >= total_tasks:
                break
            with device_lock:
                if not active_devices:
                    stop_event.set()
                    raise RuntimeError("No active NPUs left (all devices unhealthy/busy).")
            time.sleep(0.2)

        stop_event.set()
        for t in worker_threads:
            t.join(timeout=1.0)

        results_sorted = sorted(results, key=lambda r: (r.get("idx", 0), r.get("testcase", ""), r.get("gtest_filter", "")))
        rows = []
        for r in results_sorted:
            if args.monitor_npu_smi:
                npu_col = "Y" if r.get("npu_seen") else "N"
                mem_col = str(r.get("max_mem_mb", 0))
            else:
                npu_col = "-"
                mem_col = "-"
            rows.append([
                r.get("task", r["testcase"]),
                str(r["device"]),
                "PASS" if r["rc"] == 0 else "FAIL",
                f"{r['elapsed_sec']:.1f}",
                str(r["rc"]),
                npu_col,
                mem_col,
            ])

        print("\n[SUMMARY] results:")
        print(_format_table(
            rows,
            headers=["TASK", "DEV", "STATUS", "SEC", "RC", "NPU", "MAXMEM"],
        ))

        if failures:
            print("\n[SUMMARY] failures:")
            for f in sorted(failures, key=lambda r: (r.get("idx", 0), r.get("testcase", ""), r.get("gtest_filter", ""))):
                extra = f.get("error", "")
                extra = f" ({extra})" if extra else ""
                print(f" - {f.get('task', f['testcase'])} dev={f['device']} rc={f['rc']}{extra}")
                tail = (f.get("output_tail") or "").strip()
                if tail:
                    print("   ---- output tail ----")
                    for line in tail.splitlines()[-40:]:
                        print(f"   {line}")
            raise SystemExit(1)

        final_devices, final_processes = _npu_smi_info(env)
        leftovers = []
        for dev in devices:
            procs = _filter_ignored_procs(_device_procs(final_processes, dev), ignore_proc_re)
            if procs:
                leftovers.append((dev, list(procs.keys())))
        if leftovers:
            print("\n[WARN] leftover NPU processes detected:")
            for dev, pids in leftovers:
                print(f" - dev={dev} pids={pids} health={final_devices.get(dev, {}).get('health', 'UNKNOWN')}")

        print("\n[SUMMARY] all A3 NPU ST testcases passed")
    finally:
        smi.stop()


if __name__ == "__main__":
    main()
