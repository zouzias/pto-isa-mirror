#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""
run-pto.py

Master runner for this repository:
  - Auto-detect host arch + Ascend NPU (via npu-smi when present)
  - Run ST (npu/sim/cpu) with parallel execution where supported
  - Run demos (CPU demos via tests/run_cpu.py)
  - Run manual kernels (kernels/manual/* run.sh) and capture basic perf signals
  - Print a tidy results table

This script is intentionally an orchestrator: it reuses existing build/run entrypoints
where they already exist (e.g. tests/script/run_st_parallel.py).
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
_PRINT_LOCK = threading.Lock()


def _now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    out = [fmt.format(*headers), sep]
    out.extend(fmt.format(*[str(c) for c in r]) for r in rows)
    return "\n".join(out)


def _run(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
    capture: bool = True,
    check: bool = False,
) -> Tuple[int, str, float]:
    start = time.perf_counter()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    out = ""
    try:
        out, _ = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass
        return 124, (out or "") + f"\n[run-pto] timeout after {timeout_sec}s: {' '.join(cmd)}\n", time.perf_counter() - start

    rc = int(proc.returncode or 0)
    elapsed = time.perf_counter() - start
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd, output=out)
    return rc, out or "", elapsed


def _run_tee(
    cmd: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_sec: Optional[int] = None,
    echo: bool = True,
) -> Tuple[int, str, float]:
    """
    Run a subprocess while streaming stdout to console and capturing it.

    Useful for long-running, multi-test runners where we want intermediate progress logs
    (e.g. which testcase is currently running) and still want the final output for parsing.
    """
    start = time.perf_counter()
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    buf: List[str] = []
    q: "Queue[Optional[str]]"
    from queue import Empty, Queue

    q = Queue()

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                buf.append(line)
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = None if timeout_sec is None else (time.perf_counter() + float(timeout_sec))
    timed_out = False

    while True:
        if deadline is not None and time.perf_counter() >= deadline:
            timed_out = True
            break
        try:
            item = q.get(timeout=0.2)
        except Empty:
            item = None
            # If the process already exited and reader drained, we're done.
            if proc.poll() is not None and not t.is_alive():
                break
            continue
        if item is None:
            break
        if echo:
            with _PRINT_LOCK:
                sys.stdout.write(item)
                sys.stdout.flush()

    if timed_out:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass

    try:
        rc = int(proc.wait(timeout=5) or 0)
    except subprocess.TimeoutExpired:
        rc = 124

    t.join(timeout=1)
    out = "".join(buf)
    if timed_out:
        rc = 124
        out += f"\n[run-pto] timeout after {timeout_sec}s: {' '.join(cmd)}\n"
    return rc, out, time.perf_counter() - start


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def _prepend_env_path(env: Dict[str, str], var: str, path: str) -> None:
    if not path:
        return
    cur = env.get(var, "")
    env[var] = path if not cur else f"{path}:{cur}"


def _source_ascend_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Best-effort loader for Ascend env variables, consistent with tests/script/run_st.py.
    """
    base = dict(base_env or os.environ)
    candidates: List[str] = []
    ascend_home = (base.get("ASCEND_HOME_PATH") or "").strip()
    if ascend_home:
        candidates.extend(
            [
                str(Path(ascend_home) / "bin" / "setenv.bash"),
                str(Path(ascend_home) / "set_env.sh"),
            ]
        )
    candidates.extend(
        [
            str(Path.home() / "Ascend" / "ascend-toolkit" / "set_env.sh"),
            str(Path.home() / "Ascend" / "ascend-toolkit" / "bin" / "setenv.bash"),
            str(Path.home() / "Ascend" / "ascend-toolkit" / "latest" / "bin" / "setenv.bash"),
        ]
    )

    script = next((p for p in candidates if Path(p).exists()), None)
    if script is None:
        return base

    bash = shutil.which("bash") or "bash"
    rc, out, _ = _run([bash, "-lc", f"source {script} >/dev/null 2>&1 && env -0"], capture=True, timeout_sec=60)
    if rc != 0:
        return base

    new_env = dict(base)
    for item in out.split("\0"):
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        new_env[k] = v
    return new_env


def _detect_host() -> Dict[str, str]:
    arch = (platform.machine() or "unknown").lower()
    sysname = (platform.system() or "unknown").lower()
    return {"arch": arch, "os": sysname}


def _npu_smi_exists() -> bool:
    return shutil.which("npu-smi") is not None


def _parse_npu_smi_devices(text: str) -> List[Dict[str, str]]:
    devices: List[Dict[str, str]] = []
    for line in text.splitlines():
        # Row example:
        # | 0     910B1               | OK            | ...
        m = re.match(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(OK|WARN|WARNING|FAULT|ERROR)\s*\|", line, re.IGNORECASE)
        if not m:
            continue
        devices.append({"id": m.group(1), "name": m.group(2), "health": m.group(3).upper()})
    return devices


def _detect_soc_from_npu_smi(env: Dict[str, str]) -> Optional[Dict[str, str]]:
    # Prefer `smi-npu` when available (some environments ship a wrapper with that name),
    # otherwise fall back to `npu-smi`.
    smi_cmd = None
    if shutil.which("smi-npu"):
        smi_cmd = ["smi-npu", "info"]
    elif _npu_smi_exists():
        smi_cmd = ["npu-smi", "info"]
    else:
        return None

    rc, out, _ = _run(smi_cmd, env=env, timeout_sec=10, capture=True)
    if rc != 0 or not out.strip():
        return None

    devices = _parse_npu_smi_devices(out)
    if not devices:
        return None

    names = {d["name"] for d in devices}

    # Heuristics: map npu-smi Name column to soc family.
    # - a3: 910B1 / 910B2 / 910B? (Ascend910B*)
    # - a5: 910_9599 or similar (rarely shown in Name column; also match "9599")
    # - a2: 910 (fallback)
    joined = " ".join(sorted(names))
    if re.search(r"\b910_?9599\b", joined) or "9599" in joined:
        return {"soc": "a5", "soc_name": "Ascend910_9599"}
    if re.search(r"\b910B\d\b", joined) or "910B" in joined:
        return {"soc": "a3", "soc_name": "Ascend910B1"}
    if re.search(r"\b910\b", joined):
        return {"soc": "a2", "soc_name": "Ascend910"}
    return None


def _soc_config(soc: str) -> Dict[str, object]:
    """
    Map short soc code to CMake SOC_VERSION name and ST directory layout.
    """
    if soc == "a3":
        return {
            "soc": "a3",
            "soc_name": "Ascend910B1",
            "st_dir": (REPO_ROOT / "tests" / "npu" / "a2a3" / "src" / "st").resolve(),
        }
    if soc == "a2":
        return {
            "soc": "a2",
            "soc_name": "Ascend910",
            "st_dir": (REPO_ROOT / "tests" / "npu" / "a2a3" / "src" / "st").resolve(),
        }
    if soc == "a5":
        return {
            "soc": "a5",
            "soc_name": "Ascend910_9599",
            "st_dir": (REPO_ROOT / "tests" / "npu" / "a5" / "src" / "st").resolve(),
        }
    raise ValueError(f"unsupported soc: {soc}")


def _extract_st_testcases(st_dir: Path) -> List[str]:
    testcase_cmake = (st_dir / "testcase" / "CMakeLists.txt").resolve()
    txt = testcase_cmake.read_text(encoding="utf-8", errors="replace")
    if "set(ALL_TESTCASES" not in txt:
        raise RuntimeError(f"cannot find ALL_TESTCASES in {testcase_cmake}")
    block = txt.split("set(ALL_TESTCASES", 1)[1].split(")", 1)[0]
    out: List[str] = []
    for line in block.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.append(line)
    if not out:
        raise RuntimeError(f"empty ALL_TESTCASES in {testcase_cmake}")
    return out


def _list_cpu_st_gen_scripts() -> List[Path]:
    root = (REPO_ROOT / "tests" / "cpu" / "st" / "testcase").resolve()
    return sorted(p for p in root.glob("*/gen_data.py") if p.is_file())


def _detect_cpu_st_binaries(build_dir: Path) -> Dict[str, Path]:
    bin_dir = (build_dir / "bin").resolve()
    if not bin_dir.exists():
        return {}
    out: Dict[str, Path] = {}
    for p in bin_dir.iterdir():
        if not p.is_file():
            continue
        # match all_cpu_tests.py behavior: run everything in build/bin
        if os.name == "nt":
            if p.suffix.lower() != ".exe":
                continue
            out[p.stem] = p
        else:
            if not os.access(p, os.X_OK):
                continue
            out[p.name] = p
    return out


def _extract_perf_signals(text: str, limit: int = 6) -> List[str]:
    """
    Extract a few human-readable performance lines from program output.
    """
    if not text:
        return []
    patterns = [
        re.compile(r"\b(TFLOPS|GFLOPS|TOPS|GOPS|GB/s|GiB/s|BW|Bandwidth|throughput|latency|p\d+|us|ms)\b", re.I),
        re.compile(r"^\s*time\s+", re.I),
        re.compile(r"^\s*\[\s*PERF\s*\]", re.I),
        re.compile(r"^\s*\[\s*PROFILE\s*\]", re.I),
    ]
    lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        # Strip common logger prefixes (e.g. tests/run_cpu.py uses logging module).
        if " - INFO: " in s:
            s = s.split(" - INFO: ", 1)[1].strip()
        elif s.startswith("INFO:"):
            s = s[len("INFO:") :].strip()
        if any(p.search(s) for p in patterns):
            if len(s) > 160:
                s = s[:157] + "..."
            lines.append(s)
            if len(lines) >= limit:
                break
    return lines


@dataclass
class TaskResult:
    category: str
    backend: str
    name: str
    status: str
    sec: float
    rc: int
    detail: str = ""
    perf: List[str] = field(default_factory=list)


def _run_st_npu_parallel(
    *,
    soc: str,
    env: Dict[str, str],
    timeout_sec: int,
    jobs: int,
    devices: str,
    testcases: str,
    split_gtest: bool,
    verbose: bool,
    progress: bool,
) -> List[TaskResult]:
    runner = (REPO_ROOT / "tests" / "script" / "run_st_parallel.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(f"missing {runner}")

    cmd = [sys.executable, str(runner), "-v", soc, "--timeout-sec", str(int(timeout_sec))]
    if jobs > 0:
        cmd.extend(["-j", str(int(jobs))])
    if devices.strip():
        cmd.extend(["--devices", devices.strip()])
    if testcases.strip():
        cmd.extend(["--testcases", testcases.strip()])
    if split_gtest:
        cmd.append("--split-gtest")

    _log(f"[run-pto] ST(npu) cmd: {' '.join(cmd)}")
    rc, out, sec = _run_tee(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        timeout_sec=max(timeout_sec * 4, 1800),
        echo=bool(progress or verbose),
    )

    # Parse the per-testcase summary table emitted by run_st_parallel.py.
    results: List[TaskResult] = []
    summary_lines = out.splitlines()
    in_table = False
    headers: List[str] = []
    for ln in summary_lines:
        if ln.strip() == "[SUMMARY] results:":
            in_table = True
            headers = []
            continue
        if not in_table:
            continue
        if not ln.strip():
            if headers:
                break
            continue
        if ln.strip().startswith("TASK"):
            headers = re.split(r"\s{2,}", ln.strip())
            continue
        if headers and set(ln.strip()) == {"-"}:
            continue
        if not headers:
            continue
        cols = re.split(r"\s{2,}", ln.strip())
        if cols and cols[0] and cols[0].strip("-") == "":
            # Separator row like "-----  ---  ---- ..."
            continue
        if len(cols) < 4:
            continue
        task = cols[0]
        dev = cols[1]
        status = cols[2]
        sec_s = cols[3]
        rc_s = cols[4] if len(cols) > 4 else ("0" if status == "PASS" else str(rc))
        try:
            tsec = float(sec_s)
        except Exception:
            tsec = 0.0
        try:
            trc = int(rc_s)
        except Exception:
            trc = 0 if status == "PASS" else rc
        results.append(
            TaskResult(
                category="st",
                backend="npu",
                name=f"{task} (dev={dev})",
                status=status,
                sec=tsec,
                rc=trc,
            )
        )

    # If parsing failed (unexpected format), fall back to a single aggregate row.
    if not results:
        results.append(
            TaskResult(
                category="st",
                backend="npu",
                name=f"st_npu_all ({soc})",
                status="PASS" if rc == 0 else "FAIL",
                sec=sec,
                rc=rc,
                detail="(unparsed run_st_parallel output)",
            )
        )
    return results


def _build_st(
    *,
    st_dir: Path,
    build_dir: Path,
    run_mode: str,
    soc_name: str,
    env: Dict[str, str],
    jobs: int,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmake_cmd = ["cmake", "-U", "TEST_CASE", f"-DRUN_MODE={run_mode}", f"-DSOC_VERSION={soc_name}", ".."]
    rc, out, _ = _run(cmake_cmd, cwd=build_dir, env=env, timeout_sec=1800, capture=True)
    if rc != 0:
        raise RuntimeError(f"cmake failed (rc={rc})\n{out}")
    make_jobs = jobs if jobs > 0 else int(os.environ.get("PTO_ST_BUILD_JOBS", "32"))
    rc, out, _ = _run(["make", "-j", str(int(make_jobs))], cwd=build_dir, env=env, timeout_sec=3600, capture=True)
    if rc != 0:
        raise RuntimeError(f"make failed (rc={rc})\n{out}")


def _gen_st_goldens(
    *,
    st_dir: Path,
    build_dir: Path,
    testcases: List[str],
    env: Dict[str, str],
    jobs: int,
    verbose: bool,
    progress: bool,
) -> None:
    # Use per-testcase filename to avoid races when running in parallel.
    workers = jobs if jobs > 0 else min(max(1, os.cpu_count() or 1), 16)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(tc: str) -> None:
        if progress:
            _log(f"[run-pto] [GOLDEN] {tc}")
        src = (st_dir / "testcase" / tc / "gen_data.py").resolve()
        if not src.exists():
            raise FileNotFoundError(f"missing gen_data.py: {src}")
        dst = (build_dir / f"gen_data_{tc}.py").resolve()
        shutil.copyfile(src, dst)
        rc, out, _ = _run([sys.executable, str(dst)], cwd=build_dir, env=env, timeout_sec=600, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        if rc != 0:
            raise RuntimeError(f"golden gen failed for {tc} (rc={rc})\n{out}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, tc) for tc in testcases]
        for f in as_completed(futs):
            f.result()


def _run_st_sim_parallel(
    *,
    soc: str,
    soc_name: str,
    st_dir: Path,
    env: Dict[str, str],
    timeout_sec: int,
    jobs: int,
    testcases_csv: str,
    verbose: bool,
    progress: bool,
) -> List[TaskResult]:
    # Separate build dir so users can keep NPU build intact.
    build_dir = (st_dir / "build_sim").resolve()
    if progress:
        _log(f"[run-pto] building ST(sim) in {build_dir}")
    _build_st(st_dir=st_dir, build_dir=build_dir, run_mode="sim", soc_name=soc_name, env=env, jobs=jobs)

    testcases = _extract_st_testcases(st_dir)
    if testcases_csv.strip():
        wanted = {x.strip() for x in testcases_csv.split(",") if x.strip()}
        testcases = [t for t in testcases if t in wanted]
        missing = sorted(wanted - set(testcases))
        if missing:
            raise RuntimeError(f"unknown testcase(s): {missing}")

    # Sim runtime libs (mirrors demos/manual kernels scripts).
    if env.get("ASCEND_HOME_PATH"):
        _prepend_env_path(env, "LD_LIBRARY_PATH", f"{env['ASCEND_HOME_PATH']}/tools/simulator/{soc_name}/lib")

    _gen_st_goldens(
        st_dir=st_dir,
        build_dir=build_dir,
        testcases=testcases,
        env=env,
        jobs=jobs,
        verbose=verbose,
        progress=progress,
    )

    bin_dir = (build_dir / "bin").resolve()
    if not bin_dir.exists():
        raise FileNotFoundError(f"missing bin dir: {bin_dir}")

    # Run one testcase binary per worker.
    max_workers = jobs if jobs > 0 else min(max(1, os.cpu_count() or 1), 16)

    results: List[TaskResult] = []
    pending = list(testcases)

    # Simple worker loop (avoid bringing in heavy scheduling; output is captured anyway).
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(tc: str) -> TaskResult:
        exe = (bin_dir / tc).resolve()
        if not exe.exists():
            return TaskResult(category="st", backend="sim", name=tc, status="MISSING", sec=0.0, rc=2)
        if progress:
            _log(f"[run-pto] [RUN] ST(sim) {tc}")
        rc, out, sec = _run([str(exe)], cwd=bin_dir, env=env, timeout_sec=timeout_sec, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        if progress:
            _log(f"[run-pto] [DONE] ST(sim) {tc} rc={rc} sec={sec:.1f}")
        return TaskResult(
            category="st",
            backend="sim",
            name=tc,
            status="PASS" if rc == 0 else "FAIL",
            sec=sec,
            rc=rc,
            perf=_extract_perf_signals(out),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, tc): tc for tc in pending}
        for f in as_completed(futs):
            results.append(f.result())

    results.sort(key=lambda r: r.name)
    return results


def _run_cpu_st_parallel(
    *,
    env: Dict[str, str],
    timeout_sec: int,
    jobs: int,
    compiler: str,
    verbose: bool,
    progress: bool,
) -> List[TaskResult]:
    src_dir = (REPO_ROOT / "tests" / "cpu" / "st").resolve()
    build_dir = (REPO_ROOT / "build_cpu_st").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    if progress:
        _log(f"[run-pto] building CPU ST in {build_dir}")

    build_env = dict(env)
    if compiler:
        # all_cpu_tests.py uses /usr/bin/<compiler>; keep it flexible here.
        build_env["CXX"] = compiler if os.path.isabs(compiler) else (shutil.which(compiler) or compiler)

    rc, out, _ = _run(["cmake", "-S", str(src_dir), "-B", str(build_dir)], env=build_env, timeout_sec=1800, capture=True)
    if rc != 0:
        raise RuntimeError(f"cmake cpu-st failed (rc={rc})\n{out}")
    build_jobs = jobs if jobs > 0 else 8
    rc, out, _ = _run(["cmake", "--build", str(build_dir), "--parallel", str(int(build_jobs))], env=build_env, timeout_sec=3600, capture=True)
    if rc != 0:
        raise RuntimeError(f"build cpu-st failed (rc={rc})\n{out}")

    # Generate goldens in the build dir (matches tests/script/all_cpu_tests.py behavior).
    for gen in _list_cpu_st_gen_scripts():
        if progress:
            _log(f"[run-pto] [GOLDEN] cpu-st {gen.parent.name}")
        rc, out, _ = _run([sys.executable, str(gen)], cwd=build_dir, env=build_env, timeout_sec=600, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        if rc != 0:
            raise RuntimeError(f"cpu-st golden gen failed: {gen} (rc={rc})\n{out}")

    bins = _detect_cpu_st_binaries(build_dir)
    if not bins:
        raise RuntimeError(f"no cpu-st executables found under {build_dir / 'bin'}")

    max_workers = jobs if jobs > 0 else min(max(1, os.cpu_count() or 1), 16)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(name: str, exe: Path) -> TaskResult:
        if progress:
            _log(f"[run-pto] [RUN] ST(cpu) {name}")
        rc, out, sec = _run([str(exe)], cwd=exe.parent, env=build_env, timeout_sec=timeout_sec, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        if progress:
            _log(f"[run-pto] [DONE] ST(cpu) {name} rc={rc} sec={sec:.1f}")
        return TaskResult(
            category="st",
            backend="cpu",
            name=name,
            status="PASS" if rc == 0 else "FAIL",
            sec=sec,
            rc=rc,
            perf=_extract_perf_signals(out),
        )

    results: List[TaskResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_one, name, exe): name for name, exe in bins.items()}
        for f in as_completed(futs):
            results.append(f.result())
    results.sort(key=lambda r: r.name)
    return results


def _run_cpu_demos(*, verbose: bool, progress: bool) -> List[TaskResult]:
    demos = ["gemm", "flash_attn", "mla"]
    runner = (REPO_ROOT / "tests" / "run_cpu.py").resolve()
    if not runner.exists():
        return []
    results: List[TaskResult] = []
    for d in demos:
        if progress:
            _log(f"[run-pto] [RUN] demo(cpu) {d}")
        cmd = [sys.executable, str(runner), "--demo", d]
        if verbose:
            cmd.append("--verbose")
        rc, out, sec = _run(cmd, cwd=REPO_ROOT, env=dict(os.environ), timeout_sec=1800, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        results.append(
            TaskResult(
                category="demo",
                backend="cpu",
                name=f"cpu_demo:{d}",
                status="PASS" if rc == 0 else "FAIL",
                sec=sec,
                rc=rc,
                perf=_extract_perf_signals(out),
            )
        )
        if progress:
            _log(f"[run-pto] [DONE] demo(cpu) {d} rc={rc} sec={sec:.1f}")
    return results


def _run_baseline_demos(
    *,
    backend: str,
    soc_name: str,
    env: Dict[str, str],
    verbose: bool,
    progress: bool,
) -> List[TaskResult]:
    """
    Baseline demos are NPU/sim oriented and ship their own run.sh entrypoints.
    """
    if backend not in ("npu", "sim"):
        return []
    script = (REPO_ROOT / "demos" / "baseline" / "gemm_basic" / "run.sh").resolve()
    if not script.exists():
        return []
    cmd = ["bash", str(script), "-r", backend, "-v", soc_name]
    if progress:
        _log(f"[run-pto] [RUN] demo({backend}) baseline:gemm_basic")
    rc, out, sec = _run(cmd, cwd=script.parent, env=env, timeout_sec=3600, capture=True)
    if verbose and out.strip():
        sys.stdout.write(out)
        sys.stdout.flush()
    if progress:
        _log(f"[run-pto] [DONE] demo({backend}) baseline:gemm_basic rc={rc} sec={sec:.1f}")
    return [
        TaskResult(
            category="demo",
            backend=backend,
            name="baseline:gemm_basic",
            status="PASS" if rc == 0 else "FAIL",
            sec=sec,
            rc=rc,
            perf=_extract_perf_signals(out),
        )
    ]


def _run_manual_kernels(
    *,
    backend: str,
    soc_name: str,
    env: Dict[str, str],
    verbose: bool,
    npu_id: int,
    progress: bool,
) -> List[TaskResult]:
    if backend not in ("npu", "sim"):
        return []

    kernels = [
        ("flash_atten", (REPO_ROOT / "kernels" / "manual" / "a2a3" / "flash_atten" / "run.sh").resolve(), True),
        ("gemm_performance", (REPO_ROOT / "kernels" / "manual" / "a2a3" / "gemm_performance" / "run.sh").resolve(), False),
    ]
    results: List[TaskResult] = []
    for name, script, supports_npu_id in kernels:
        if not script.exists():
            continue
        cmd = ["bash", str(script), "-r", backend, "-v", soc_name]
        if backend == "npu" and supports_npu_id:
            cmd.extend(["-n", str(int(npu_id))])
        if progress:
            _log(f"[run-pto] [RUN] kernel({backend}) manual:{name}")
        rc, out, sec = _run(cmd, cwd=script.parent, env=env, timeout_sec=7200, capture=True)
        if verbose and out.strip():
            sys.stdout.write(out)
            sys.stdout.flush()
        if progress:
            _log(f"[run-pto] [DONE] kernel({backend}) manual:{name} rc={rc} sec={sec:.1f}")
        results.append(
            TaskResult(
                category="kernel",
                backend=backend,
                name=f"manual:{name}",
                status="PASS" if rc == 0 else "FAIL",
                sec=sec,
                rc=rc,
                perf=_extract_perf_signals(out),
            )
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="PTO master runner (ST + demos + manual kernels)")
    ap.add_argument(
        "--backend",
        choices=["auto", "npu", "sim", "cpu"],
        default="auto",
        help="Execution backend for ST and manual kernels (auto: prefer NPU if present, else CPU)",
    )
    ap.add_argument("--soc", choices=["auto", "a2", "a3", "a5"], default="auto", help="Target Ascend SoC family")
    ap.add_argument(
        "--run",
        default="st,demo,kernel",
        help="Comma list: st,demo,kernel (default: st,demo,kernel)",
    )
    ap.add_argument("--timeout-sec", type=int, default=int(os.environ.get("PTO_ST_TIMEOUT_SEC", "120")))
    ap.add_argument("-j", "--jobs", type=int, default=0, help="Parallel workers (0: auto)")
    ap.add_argument("--devices", default="", help="Comma-separated physical NPU ids (NPU ST only)")
    ap.add_argument("--st-testcases", default="", help="Comma-separated ST testcase binaries to run (NPU/SIM only)")
    ap.add_argument("--split-gtest", action="store_true", help="Split NPU ST binaries into per-gtest tasks (NPU only)")
    ap.add_argument("--cpu-compiler", default=os.environ.get("CXX", ""), help="C++ compiler for CPU ST (default: env CXX or system)")
    ap.add_argument("--npu-id", type=int, default=0, help="Manual kernels NPU id (when backend=npu)")
    ap.add_argument("--include-baseline", action="store_true", help="Also run baseline NPU/sim demos (e.g. demos/baseline/gemm_basic)")
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print intermediate progress logs (which testcase/demo/kernel is running)",
    )
    ap.add_argument("--verbose", action="store_true", help="Print subcommand output")
    args = ap.parse_args()

    host = _detect_host()
    base_env = dict(os.environ)
    ascend_env = _source_ascend_env(base_env)
    soc_detected = _detect_soc_from_npu_smi(ascend_env)

    backend = args.backend
    if backend == "auto":
        backend = "npu" if soc_detected else "cpu"

    soc = args.soc
    if soc == "auto":
        soc = soc_detected["soc"] if soc_detected else "a3"
        # If no NPU detected and backend is cpu, soc is irrelevant; keep default for printing.

    run_set = {x.strip() for x in args.run.split(",") if x.strip()}

    cfg = None
    if backend in ("npu", "sim"):
        cfg = _soc_config(soc)

    _log(f"[run-pto] host={host['os']}:{host['arch']} backend={backend} soc={soc}")
    if soc_detected:
        _log(f"[run-pto] detected_npu_soc={soc_detected['soc']} soc_name={soc_detected['soc_name']}")

    results: List[TaskResult] = []

    if "st" in run_set:
        if backend == "npu":
            if cfg is None:
                raise RuntimeError("internal error: missing soc config for npu")
            results.extend(
                _run_st_npu_parallel(
                    soc=str(cfg["soc"]),
                    env=ascend_env,
                    timeout_sec=int(args.timeout_sec),
                    jobs=int(args.jobs),
                    devices=args.devices,
                    testcases=args.st_testcases,
                    split_gtest=bool(args.split_gtest),
                    verbose=bool(args.verbose),
                    progress=bool(args.progress),
                )
            )
        elif backend == "sim":
            if cfg is None:
                raise RuntimeError("internal error: missing soc config for sim")
            results.extend(
                _run_st_sim_parallel(
                    soc=str(cfg["soc"]),
                    soc_name=str(cfg["soc_name"]),
                    st_dir=Path(cfg["st_dir"]),
                    env=ascend_env,
                    timeout_sec=max(int(args.timeout_sec), 600),
                    jobs=int(args.jobs),
                    testcases_csv=args.st_testcases,
                    verbose=bool(args.verbose),
                    progress=bool(args.progress),
                )
            )
        else:  # cpu
            results.extend(
                _run_cpu_st_parallel(
                    env=base_env,
                    timeout_sec=max(int(args.timeout_sec), 600),
                    jobs=int(args.jobs),
                    compiler=args.cpu_compiler,
                    verbose=bool(args.verbose),
                    progress=bool(args.progress),
                )
            )

    if "demo" in run_set:
        results.extend(_run_cpu_demos(verbose=bool(args.verbose), progress=bool(args.progress)))
        if args.include_baseline and backend in ("npu", "sim") and cfg is not None:
            results.extend(
                _run_baseline_demos(
                    backend=backend,
                    soc_name=str(cfg["soc_name"]),
                    env=ascend_env,
                    verbose=bool(args.verbose),
                    progress=bool(args.progress),
                )
            )

    if "kernel" in run_set:
        if backend in ("npu", "sim"):
            if cfg is None:
                raise RuntimeError("internal error: missing soc config for kernels")
            results.extend(
                _run_manual_kernels(
                    backend=backend,
                    soc_name=str(cfg["soc_name"]),
                    env=ascend_env,
                    verbose=bool(args.verbose),
                    npu_id=int(args.npu_id),
                    progress=bool(args.progress),
                )
            )

    # Render summary.
    rows: List[List[str]] = []
    failures = 0
    for r in results:
        ok = r.status.upper() in ("PASS", "OK") or (r.rc == 0 and r.status.upper() != "FAIL")
        if not ok:
            failures += 1
        perf = r.perf[0] if r.perf else ""
        rows.append([r.category, r.backend, r.name, r.status, f"{r.sec:.1f}", str(r.rc), perf])

    if rows:
        print()
        print(_format_table(rows, headers=["CAT", "BACKEND", "NAME", "STATUS", "SEC", "RC", "PERF (first hit)"]))
        print()
        print(f"[run-pto] total={len(results)} failed={failures}")
    else:
        print("[run-pto] no tasks executed (check --run / backend selection)")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
