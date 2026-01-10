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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue


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


def _npu_smi_info(env):
    out = _run(["npu-smi", "info"], env=env, capture=True, timeout_sec=30, check=False)
    return _parse_npu_smi_info(out)


def _device_procs(processes, npu_id):
    return {pid: (name, mem) for pid, (dev, name, mem) in processes.items() if dev == npu_id}


def _wait_device_idle(env, npu_id, timeout_sec, poll_sec=1.0):
    deadline = time.time() + timeout_sec
    last = None
    while time.time() < deadline:
        devices, processes = _npu_smi_info(env)
        health = devices.get(npu_id, {}).get("health", "UNKNOWN")
        procs = _device_procs(processes, npu_id)
        last = (health, procs)
        if health != "OK":
            return False, last
        if not procs:
            return True, last
        time.sleep(poll_sec)
    return False, last


def _get_free_devices(env, requested=None):
    devices, processes = _npu_smi_info(env)

    ok = []
    for npu_id, info in sorted(devices.items()):
        if requested is not None and npu_id not in requested:
            continue
        if info.get("health") != "OK":
            continue
        ok.append(npu_id)

    busy = {npu_id for (_, (npu_id, _, _)) in processes.items()}
    free = [d for d in ok if d not in busy]
    return free


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
    cmd = [sys.executable, "tests/script/build_st.py", "-r", "npu", "-v", "a3", "-t", "all"]
    build_env = dict(env)
    build_env["PTO_ST_JOBS"] = str(jobs)
    _run(["bash", "-lc", " ".join(map(str, cmd))], env=build_env, timeout_sec=3600)


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


def _run_one_binary(env, device_id, testcase, timeout_sec, log_dir):
    st_dir = Path("tests/npu/a2a3/src/st")
    bin_dir = st_dir / "build" / "bin"
    exe = (bin_dir / testcase).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"missing binary: {exe}")

    run_env = dict(env)
    # Map the chosen physical device to logical device 0 for code paths that call `aclrtSetDevice(0)`.
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
    run_env["ASCEND_VISIBLE_DEVICES"] = str(device_id)
    run_env["DEVICE_ID"] = "0"
    run_env["ACL_DEVICE_ID"] = "0"

    log_path = Path(log_dir) / f"a3_npu_{testcase}_dev{device_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    npu_seen = False
    max_mem_mb = 0
    npu_health = "UNKNOWN"

    with open(log_path, "w", encoding="utf-8", errors="ignore") as logf:
        proc = subprocess.Popen(
            [str(exe)],
            cwd=str(bin_dir),
            env=run_env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )
        pid = proc.pid

        poll_interval = float(os.environ.get("PTO_ST_SMI_POLL_SEC", "2.0"))
        last_poll = 0.0
        while True:
            rc = proc.poll()
            if rc is not None:
                break

            now = time.time()
            if now - start > timeout_sec:
                os.killpg(proc.pid, signal.SIGKILL)
                logf.write(f"\n[TIMEOUT] killed after {timeout_sec}s\n")
                rc = 124
                break

            if now - last_poll >= poll_interval:
                devices, processes = _npu_smi_info(env)
                npu_health = devices.get(device_id, {}).get("health", "UNKNOWN")
                if npu_health != "OK":
                    os.killpg(proc.pid, signal.SIGKILL)
                    logf.write(f"\n[NPU] health={npu_health}; killed\n")
                    rc = 125
                    break
                if pid in processes:
                    npu_seen = True
                    dev_seen, _, mem = processes[pid]
                    if dev_seen != device_id:
                        os.killpg(proc.pid, signal.SIGKILL)
                        logf.write(f"\n[NPU] pid={pid} mapped to dev={dev_seen} (expected {device_id}); killed\n")
                        rc = 129
                        break
                    max_mem_mb = max(max_mem_mb, mem)
                last_poll = now

            time.sleep(0.2)

        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        proc.returncode = rc if rc is not None else proc.returncode

    elapsed = time.time() - start

    return {
        "testcase": testcase,
        "device": device_id,
        "rc": proc.returncode,
        "elapsed_sec": elapsed,
        "log": str(log_path),
        "pid": proc.pid,
        "npu_seen": npu_seen,
        "max_mem_mb": max_mem_mb,
        "npu_health": npu_health,
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
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("PTO_ST_JOBS", "32")), help="make -j")
    ap.add_argument("--timeout-sec", type=int, default=int(os.environ.get("PTO_ST_TIMEOUT_SEC", "600")),
                    help="per-testcase binary timeout (detect deadlock)")
    ap.add_argument("--max-workers", type=int, default=0, help="0 means use all free NPUs")
    ap.add_argument("--devices", default="", help="comma-separated physical NPU ids to use (default: auto-detect free)")
    ap.add_argument("--log-dir", default="/tmp/pto-isa-st-logs", help="where to write per-testcase logs")
    ap.add_argument("--device-idle-wait-sec", type=int, default=int(os.environ.get("PTO_ST_DEVICE_IDLE_WAIT_SEC", "30")),
                    help="wait for a device to become idle between tests (helps recover after kill)")
    args = ap.parse_args()

    env = _source_ascend_env()

    requested = None
    if args.devices.strip():
        requested = {int(x) for x in args.devices.split(",") if x.strip() != ""}

    free = _get_free_devices(env, requested=requested)
    if not free:
        raise RuntimeError("No free NPUs with Health=OK found (or all requested NPUs are busy).")

    workers = len(free) if args.max_workers == 0 else min(args.max_workers, len(free))
    devices = free[:workers]
    print(f"[INFO] using NPUs: {devices} (workers={workers})")
    sys.stdout.flush()

    testcases = _extract_a3_testcases()
    print(f"[INFO] A3 testcases: {len(testcases)}")
    sys.stdout.flush()

    print("[INFO] building all A3 NPU ST once...")
    sys.stdout.flush()
    _build_all_a3(env, jobs=args.jobs)

    print("[INFO] generating golden data for all testcases (sequential)...")
    sys.stdout.flush()
    _gen_all_goldens(env, testcases)

    print("[INFO] running testcase binaries in parallel (1 process per NPU)...")
    sys.stdout.flush()

    results = []
    failures = []

    work = Queue()
    for tc in testcases:
        work.put(tc)

    results_lock = threading.Lock()

    def _worker(device_id):
        while True:
            try:
                tc = work.get_nowait()
            except Empty:
                return

            try:
                idle_ok, last = _wait_device_idle(env, device_id, timeout_sec=args.device_idle_wait_sec)
                if not idle_ok:
                    health, procs = last or ("UNKNOWN", {})
                    res = {
                        "testcase": tc,
                        "device": device_id,
                        "rc": 126,
                        "elapsed_sec": 0.0,
                        "log": "",
                        "pid": -1,
                        "npu_seen": False,
                        "max_mem_mb": 0,
                        "npu_health": health,
                        "error": f"device not idle before start; health={health} procs={list(procs.keys())}",
                    }
                else:
                    print(f"[RUN ] {tc} dev={device_id}")
                    sys.stdout.flush()
                    res = _run_one_binary(env, device_id, tc, args.timeout_sec, args.log_dir)

                    idle_ok_after, last_after = _wait_device_idle(env, device_id, timeout_sec=args.device_idle_wait_sec)
                    if not idle_ok_after:
                        health, procs = last_after or ("UNKNOWN", {})
                        if res.get("rc", 0) == 0:
                            res["rc"] = 127
                        res["npu_health"] = health
                        res["error"] = f"device not idle after run; health={health} procs={list(procs.keys())}"
            except Exception as e:
                res = {
                    "testcase": tc,
                    "device": device_id,
                    "rc": 128,
                    "elapsed_sec": 0.0,
                    "log": "",
                    "pid": -1,
                    "npu_seen": False,
                    "max_mem_mb": 0,
                    "npu_health": "UNKNOWN",
                    "error": f"runner exception: {type(e).__name__}: {e}",
                }
            finally:
                with results_lock:
                    results.append(res)
                    status = "PASS" if res["rc"] == 0 else "FAIL"
                    print(
                        f"[{status}] {res['testcase']} dev={res['device']} "
                        f"time={res['elapsed_sec']:.1f}s rc={res['rc']} log={res.get('log','')}"
                    )
                    sys.stdout.flush()
                    if res["rc"] != 0:
                        failures.append(res)
                work.task_done()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for dev in devices:
            pool.submit(_worker, dev)
        work.join()

    results_sorted = sorted(results, key=lambda r: r["testcase"])
    rows = []
    for r in results_sorted:
        rows.append([
            r["testcase"],
            str(r["device"]),
            "PASS" if r["rc"] == 0 else "FAIL",
            f"{r['elapsed_sec']:.1f}",
            str(r["rc"]),
            "Y" if r.get("npu_seen") else "N",
            str(r.get("max_mem_mb", 0)),
            os.path.basename(r.get("log", "")) if r.get("log") else "",
        ])

    print("\n[SUMMARY] results:")
    print(_format_table(
        rows,
        headers=["TESTCASE", "DEV", "STATUS", "SEC", "RC", "NPU", "MAXMEM", "LOG"],
    ))

    if failures:
        print("\n[SUMMARY] failures:")
        for f in sorted(failures, key=lambda r: r["testcase"]):
            extra = f.get("error", "")
            extra = f" ({extra})" if extra else ""
            print(f" - {f['testcase']} dev={f['device']} rc={f['rc']} log={f.get('log','')}{extra}")
        raise SystemExit(1)

    final_devices, final_processes = _npu_smi_info(env)
    leftovers = []
    for dev in devices:
        procs = _device_procs(final_processes, dev)
        if procs:
            leftovers.append((dev, list(procs.keys())))
    if leftovers:
        print("\n[WARN] leftover NPU processes detected:")
        for dev, pids in leftovers:
            print(f" - dev={dev} pids={pids} health={final_devices.get(dev, {}).get('health', 'UNKNOWN')}")

    print("\n[SUMMARY] all A3 NPU ST testcases passed")


if __name__ == "__main__":
    main()
