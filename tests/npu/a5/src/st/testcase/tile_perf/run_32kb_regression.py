#!/usr/bin/env python3
"""
Robust tile_perf regression runner.

Features:
  - Up to 5 tests in parallel, each in its own working directory
  - Monitors instr_log.dump: if no new instruction retires in STALL_TIMEOUT (60s), kills the process
  - Max MAX_RETRIES (3) per test case
  - Each test gets its own dump directory to avoid file conflicts
  - Results written to stdout and CSV

Usage:
    python3 run_32kb_regression.py [--parallel 5] [--stall-timeout 60] [--max-retries 3]
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Thread

# ── Paths ────────────────────────────────────────────────────────────────────

BUILD_DIR = Path("/home/happybot/pto-isa/tests/npu/a5/src/st/build")
BIN = BUILD_DIR / "bin" / "tile_perf"
RESULTS_CSV = Path(__file__).parent / "regression_results.csv"

# ── Test matrix: 32KB (fp32 = 8192 elem, fp16 = 16384 elem) ─────────────────

TESTS_32KB = [
    # (filter, op, dtype, shape, elements)
    ("*TADD_float_1x8192*",  "TADD",  "fp32", "1x8192",  8192),
    ("*TADD_float_32x256*",  "TADD",  "fp32", "32x256",  8192),
    ("*TADD_float_16x512*",  "TADD",  "fp32", "16x512",  8192),
    ("*TEXP_float_1x8192*",  "TEXP",  "fp32", "1x8192",  8192),
    ("*TEXP_float_32x256*",  "TEXP",  "fp32", "32x256",  8192),
    ("*TEXP_float_16x512*",  "TEXP",  "fp32", "16x512",  8192),
    ("*TADDS_float_1x8192*", "TADDS", "fp32", "1x8192",  8192),
    ("*TADDS_float_32x256*", "TADDS", "fp32", "32x256",  8192),
    ("*TADDS_float_16x512*", "TADDS", "fp32", "16x512",  8192),
    ("*TADD_half_1x16384*",  "TADD",  "fp16", "1x16384", 16384),
    ("*TADD_half_64x256*",   "TADD",  "fp16", "64x256",  16384),
    ("*TADD_half_32x512*",   "TADD",  "fp16", "32x512",  16384),
    ("*TEXP_half_1x16384*",  "TEXP",  "fp16", "1x16384", 16384),
    ("*TEXP_half_64x256*",   "TEXP",  "fp16", "64x256",  16384),
    ("*TEXP_half_32x512*",   "TEXP",  "fp16", "32x512",  16384),
    ("*TADDS_half_1x16384*", "TADDS", "fp16", "1x16384", 16384),
    ("*TADDS_half_64x256*",  "TADDS", "fp16", "64x256",  16384),
    ("*TADDS_half_32x512*",  "TADDS", "fp16", "32x512",  16384),
]


# ── EPC extraction ──────────────────────────────────────────────────────────

def extract_vf_epc(dump_dir: Path, elements: int):
    """Extract VF EPC from A5 instr_log.dump using vf_real_execute_time."""
    instr_log = dump_dir / "core0.veccore0.instr_log.dump"
    if not instr_log.exists() or instr_log.stat().st_size == 0:
        return None, None, None
    content = instr_log.read_text()
    pattern = r"\[(\d+)\].*VF.*vf_real_execute_time:\s*(\d+)"
    matches = re.findall(pattern, content)
    if not matches:
        return None, None, None
    all_times = [int(t) for _, t in matches]
    warm_times = all_times[1:] if len(all_times) > 1 else all_times
    avg_warm = sum(warm_times) / len(warm_times)
    epc = elements / avg_warm if avg_warm > 0 else 0
    return epc, avg_warm, len(matches)


def extract_total_tick(stdout: str):
    m = re.search(r"Total tick:\s*(\d+)", stdout)
    return int(m.group(1)) if m else None


# ── Stall-monitored test runner ─────────────────────────────────────────────

def get_dump_file_size(dump_dir: Path) -> int:
    """Get total size of all dump files to detect progress."""
    total = 0
    for f in dump_dir.glob("*.dump"):
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


def run_single_test(filt, op, dtype, shape, elements, env, stall_timeout, max_retries):
    """Run a single test with stall detection and retry logic."""
    test_id = f"{op}_{dtype}_{shape}"

    for attempt in range(1, max_retries + 1):
        # Each test gets its own working directory for dump isolation
        work_dir = BUILD_DIR / f"_perf_run_{test_id}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Clean old dumps
        for f in work_dir.glob("*.dump"):
            f.unlink()

        local_env = env.copy()
        # Some simulators use NPU_DUMP_PATH or just write to cwd
        local_env["NPU_DUMP_PATH"] = str(work_dir)

        proc = subprocess.Popen(
            [str(BIN), f"--gtest_filter={filt}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=local_env,
            cwd=str(work_dir),
        )

        # Monitor for stall: check dump file growth every 10s
        last_size = 0
        last_change_time = time.time()
        timed_out = False

        while proc.poll() is None:
            time.sleep(10)
            cur_size = get_dump_file_size(work_dir)
            now = time.time()

            if cur_size != last_size:
                last_size = cur_size
                last_change_time = now
            elif now - last_change_time > stall_timeout:
                # Stalled — no dump file growth in stall_timeout seconds
                timed_out = True
                try:
                    # Kill entire process group
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                break

        stdout_data, stderr_data = proc.communicate(timeout=10)
        stdout_str = stdout_data.decode("utf-8", errors="replace")

        if timed_out:
            status = "STALL"
            print(f"  [{test_id}] attempt {attempt}/{max_retries}: stalled after {stall_timeout}s, killed", flush=True)
            if attempt < max_retries:
                continue
            # Final attempt also stalled
            return {
                "test_id": test_id, "op": op, "dtype": dtype, "shape": shape,
                "elements": elements, "epc": None, "vf_cycles": None,
                "n_vf": None, "tick": None, "status": f"STALL(x{max_retries})",
                "attempts": attempt,
            }

        if proc.returncode != 0:
            status = "FAIL"
            print(f"  [{test_id}] attempt {attempt}/{max_retries}: exit code {proc.returncode}", flush=True)
            if attempt < max_retries:
                continue
            return {
                "test_id": test_id, "op": op, "dtype": dtype, "shape": shape,
                "elements": elements, "epc": None, "vf_cycles": None,
                "n_vf": None, "tick": None, "status": f"FAIL(rc={proc.returncode})",
                "attempts": attempt,
            }

        # Success — extract metrics
        tick = extract_total_tick(stdout_str)
        epc, avg_cy, n_vf = extract_vf_epc(work_dir, elements)

        return {
            "test_id": test_id, "op": op, "dtype": dtype, "shape": shape,
            "elements": elements, "epc": epc, "vf_cycles": avg_cy,
            "n_vf": n_vf, "tick": tick, "status": "PASS",
            "attempts": attempt,
        }

    # Should not reach here
    return {
        "test_id": test_id, "op": op, "dtype": dtype, "shape": shape,
        "elements": elements, "epc": None, "vf_cycles": None,
        "n_vf": None, "tick": None, "status": "ERROR",
        "attempts": max_retries,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robust tile_perf regression")
    parser.add_argument("--parallel", type=int, default=5, help="Max parallel tests")
    parser.add_argument("--stall-timeout", type=int, default=60, help="Kill if no dump progress in N seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per test")
    args = parser.parse_args()

    env = os.environ.copy()

    print(f"tile_perf 32KB regression — {len(TESTS_32KB)} tests")
    print(f"  parallel={args.parallel}, stall_timeout={args.stall_timeout}s, max_retries={args.max_retries}")
    print(f"  binary: {BIN}")
    print(flush=True)

    hdr = f"{'Op':<6} {'Dtype':<5} {'Shape':<10} {'Elems':>6} {'VF_EPC':>8} {'VF_cy':>7} {'#VF':>4} {'Tick':>8} {'Status':<14} {'Att':>3}"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep, flush=True)

    results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for filt, op, dtype, shape, elements in TESTS_32KB:
            fut = pool.submit(
                run_single_test,
                filt, op, dtype, shape, elements,
                env, args.stall_timeout, args.max_retries,
            )
            futures[fut] = (op, dtype, shape)

        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)

            epc_s = f"{r['epc']:.2f}" if r['epc'] is not None else "N/A"
            cy_s = f"{r['vf_cycles']:.0f}" if r['vf_cycles'] is not None else "N/A"
            vf_s = str(r['n_vf']) if r['n_vf'] is not None else "N/A"
            tk_s = str(r['tick']) if r['tick'] is not None else "N/A"

            print(f"{r['op']:<6} {r['dtype']:<5} {r['shape']:<10} {r['elements']:>6} {epc_s:>8} {cy_s:>7} {vf_s:>4} {tk_s:>8} {r['status']:<14} {r['attempts']:>3}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * len(hdr)}")
    passed = [r for r in results if r['status'] == 'PASS' and r['epc'] is not None]
    failed = [r for r in results if r['status'] != 'PASS']
    print(f"Total: {len(results)} | Passed: {len(passed)} | Failed: {len(failed)}")

    if passed:
        # Group by op
        ops = sorted(set(r['op'] for r in passed))
        for op in ops:
            op_results = [r for r in passed if r['op'] == op]
            avg_epc = sum(r['epc'] for r in op_results) / len(op_results)
            print(f"  {op}: avg EPC = {avg_epc:.2f} ({len(op_results)} tests)")

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_id", "op", "dtype", "shape", "elements", "epc", "vf_cycles", "n_vf", "tick", "status", "attempts"])
        w.writeheader()
        for r in sorted(results, key=lambda x: x["test_id"]):
            w.writerow(r)
    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
