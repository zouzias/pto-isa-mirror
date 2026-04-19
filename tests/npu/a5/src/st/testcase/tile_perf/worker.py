#!/usr/bin/env python3
"""
PTO Regression Worker - Runs tests and extracts performance metrics.

A5: Uses vf_real_execute_time from instr_log.dump
A2A3: Uses first VEC pop → last VEC retire per iteration (Lok's method)
"""

import os
import re
import sys
import json
import time
import sqlite3
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
PTO_ISA_ROOT = Path.home() / "pto-isa"
CANN_ROOT = Path("/usr/local/Ascend/cann-9.0.0-alpha.1")
DB_PATH = Path(__file__).parent / "results.db"

# Architecture configs
ARCH_CONFIG = {
    "a5": {
        "soc": "Ascend910_9599",
        "sim_lib": CANN_ROOT / "aarch64-linux/simulator/Ascend910_9599/lib",
        "build_dir": PTO_ISA_ROOT / "tests/npu/a5/src/st/build",
        "test_dir": PTO_ISA_ROOT / "tests/npu/a5/src/st/testcase",
    },
    "a2a3": {
        "soc": "Ascend910B1",
        "sim_lib": CANN_ROOT / "aarch64-linux/simulator/Ascend910B1/lib",
        "build_dir": PTO_ISA_ROOT / "tests/npu/a2a3/src/st/build",
        "test_dir": PTO_ISA_ROOT / "tests/npu/a2a3/src/st/testcase",
    },
}

# Map PTO op to vector instruction pattern
OP_TO_INSTR = {
    "TADD": r"VADD",
    "TADDS": r"VADDS|VADD",  # VADDS or VADD with scalar
    "TEXP": r"VEXP",
    "TMUL": r"VMUL",
    "TMULS": r"VMULS|VMUL",
    "TROWSUM": r"VADD|VREDUCE",    # Row reduce uses VADD tree or VREDUCE
    "TCOLSUM": r"VADD|VREDUCE",    # Col reduce uses VADD tree or VREDUCE
    "TEXPANDS": r"VBCAST|VMOV",    # Scalar broadcast
    "TROWEXPAND": r"VBCAST|VMOV",  # Row broadcast (col vector expand)
    "TCOLEXPAND": r"VBCAST|VMOV",  # Col broadcast (row vector expand)
}


def init_db():
    """Initialize database with schema."""
    conn = sqlite3.connect(DB_PATH)
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        conn.executescript(schema_path.read_text())
    conn.commit()
    return conn


def get_env(arch: str) -> dict:
    """Get environment variables for running simulator."""
    cfg = ARCH_CONFIG[arch]
    env = os.environ.copy()
    
    cann_env = CANN_ROOT / "set_env.sh"
    if cann_env.exists():
        result = subprocess.run(
            f"source {cann_env} && env",
            shell=True, capture_output=True, text=True, executable="/bin/bash"
        )
        for line in result.stdout.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cfg['sim_lib']}:{ld_path}"
    
    return env


def parse_a5_vf_epc(dump_dir: Path, elements: int) -> dict:
    """
    Parse A5 VF EPC from vf_real_execute_time in instr_log.dump.
    This excludes icache overhead - pure VF compute time.
    """
    instr_log = dump_dir / "core0.veccore0.instr_log.dump"
    
    if not instr_log.exists():
        return {"error": f"instr_log not found: {instr_log}"}
    
    content = instr_log.read_text()
    pattern = r"\[(\d+)\].*VF.*vf_real_execute_time:\s*(\d+)"
    matches = re.findall(pattern, content)
    
    if matches:
        # Skip first cold VF (icache miss), use warm average
        warm_times = [int(t) for _, t in matches[1:]] if len(matches) > 1 else [int(matches[0][1])]
        avg_warm = sum(warm_times) / len(warm_times)
        vf_epc = elements / avg_warm if avg_warm > 0 else 0
        
        return {
            "method": "vf_real_execute_time",
            "vf_count": len(matches),
            "cold_cycles": int(matches[0][1]) if matches else 0,
            "warm_times": warm_times,
            "avg_warm_cycles": avg_warm,
            "elements": elements,
            "vf_epc": vf_epc,
        }
    
    return {"error": "No VF timing found in logs"}


def parse_a5_instr_epc(dump_dir: Path, op_type: str = "TADD") -> dict:
    """Parse per-instruction EPC from EXU dump (for reference)."""
    exu_file = dump_dir / "core0.veccore0.rvec.EXU.dump"
    if not exu_file.exists():
        return {"error": f"EXU dump not found"}
    
    content = exu_file.read_text()
    op_map = {"TADD": "RV_VADD", "TADDS": "RV_VADDS", "TEXP": "RV_VEXP"}
    instr = op_map.get(op_type, "RV_VADD")
    
    pattern = r'\[(\d+)\].*?' + instr + r'.*?retire\s+(\d+).*?ele_cnt\s+(\d+)'
    
    latencies = []
    ele_counts = []
    for match in re.finditer(pattern, content, re.IGNORECASE):
        issue = int(match.group(1))
        retire = int(match.group(2))
        ele_cnt = int(match.group(3))
        latency = retire - issue
        if latency > 0 and ele_cnt > 0:
            latencies.append(latency)
            ele_counts.append(ele_cnt)
    
    if not latencies:
        return {"error": f"No {instr} found"}
    
    warm_lat = latencies[2:] if len(latencies) > 2 else latencies
    warm_ele = ele_counts[2:] if len(ele_counts) > 2 else ele_counts
    avg_lat = sum(warm_lat) / len(warm_lat)
    avg_ele = sum(warm_ele) / len(warm_ele)
    
    return {
        "instruction_count": len(latencies),
        "avg_latency": avg_lat,
        "avg_ele_cnt": avg_ele,
        "instr_epc": avg_ele / avg_lat if avg_lat > 0 else 0,
    }


def parse_a2a3_vf_epc(dump_dir: Path, elements: int, op_type: str = "TADD") -> dict:
    """
    Parse A2A3 EPC using first VEC pop → last VEC retire per iteration.
    
    Lok's method for 2D shapes (multiple VEC ops per iteration):
    1. Find pipe_barrier cycles to delimit iterations
    2. Between barriers: first VEC pop → last VEC retire = vector cycles
    3. Skip first 2 cold iterations, use warm average
    """
    instr_log = dump_dir / "core0.veccore0.instr_log.dump"
    popped_log = dump_dir / "core0.veccore0.instr_popped_log.dump"
    
    if not instr_log.exists():
        return {"error": f"instr_log not found"}
    if not popped_log.exists():
        return {"error": f"popped_log not found"}
    
    # Get instruction pattern for this op
    instr_pattern = OP_TO_INSTR.get(op_type, r"VADD|VEXP|VMUL")
    
    # Parse barriers from instr_log.dump
    barrier_pattern = r"\[(\d+)\].*(?:pipe_barrier|BAR\s+PIPE[:\s]*ALL|SEND_BARRIER)"
    barrier_cycles = []
    with open(instr_log) as f:
        for line in f:
            m = re.search(barrier_pattern, line, re.IGNORECASE)
            if m:
                barrier_cycles.append(int(m.group(1)))
    
    # Parse VEC pop times from popped_log.dump
    vec_pop = []
    with open(popped_log) as f:
        for line in f:
            if re.search(instr_pattern, line, re.IGNORECASE):
                m = re.search(r"\[(\d+)\]", line)
                if m:
                    vec_pop.append(int(m.group(1)))
    
    # Parse VEC retire times from instr_log.dump
    vec_retire = []
    with open(instr_log) as f:
        for line in f:
            if re.search(instr_pattern, line, re.IGNORECASE):
                m = re.search(r"\[(\d+)\]", line)
                if m:
                    vec_retire.append(int(m.group(1)))
    
    if not vec_pop and not vec_retire:
        return {"error": f"No {instr_pattern} found in logs"}
    
    # If barriers found, group by iteration
    iteration_cycles = []
    
    if len(barrier_cycles) >= 2:
        for i in range(len(barrier_cycles) - 1):
            bar_start = barrier_cycles[i]
            bar_end = barrier_cycles[i + 1]
            
            # Find first pop after bar_start
            first_pop = None
            for p in vec_pop:
                if p > bar_start and p < bar_end:
                    first_pop = p
                    break
            
            # Find last retire before bar_end
            last_retire = None
            for r in reversed(vec_retire):
                if r > bar_start and r < bar_end:
                    last_retire = r
                    break
            
            if first_pop is not None and last_retire is not None:
                vec_cycles = last_retire - first_pop
                if vec_cycles > 0:
                    iteration_cycles.append(vec_cycles)
    
    # Fallback: if no good iteration data, use simple pop→retire pairs
    if not iteration_cycles and vec_pop and vec_retire:
        # Match by count - assume same order
        for i in range(min(len(vec_pop), len(vec_retire))):
            lat = vec_retire[i] - vec_pop[i]
            if lat > 0:
                iteration_cycles.append(lat)
    
    if not iteration_cycles:
        return {"error": "Could not extract iteration cycles"}
    
    # Skip first 2 cold iterations
    warm_cycles = iteration_cycles[2:] if len(iteration_cycles) > 2 else iteration_cycles
    
    if not warm_cycles:
        warm_cycles = iteration_cycles  # use all if not enough
    
    avg_warm = sum(warm_cycles) / len(warm_cycles)
    vf_epc = elements / avg_warm if avg_warm > 0 else 0
    
    return {
        "method": "first_pop_to_last_retire",
        "barrier_count": len(barrier_cycles),
        "vec_pop_count": len(vec_pop),
        "vec_retire_count": len(vec_retire),
        "iteration_count": len(iteration_cycles),
        "warm_cycles": warm_cycles,
        "avg_warm_cycles": avg_warm,
        "elements": elements,
        "vf_epc": vf_epc,
    }


def get_git_info() -> dict:
    """Get current git commit and branch."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(PTO_ISA_ROOT), "rev-parse", "--short", "HEAD"],
            text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(PTO_ISA_ROOT), "branch", "--show-current"],
            text=True
        ).strip()
        return {"commit": commit, "branch": branch}
    except:
        return {"commit": "unknown", "branch": "unknown"}


def list_tests(arch: str) -> list:
    """List available tests."""
    cfg = ARCH_CONFIG[arch]
    bin_path = cfg["build_dir"] / "bin" / "tile_perf"
    
    if not bin_path.exists():
        return []
    
    env = get_env(arch)
    result = subprocess.run(
        [str(bin_path), "--gtest_list_tests"],
        capture_output=True, text=True, env=env
    )
    
    tests = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith("Running") and not line.endswith("."):
            tests.append(f"TilePerfTest.{line}")
    
    return tests


def run_test(arch: str, test_name: str, output_dir: Path) -> dict:
    """Run a single test and return results."""
    cfg = ARCH_CONFIG[arch]
    bin_path = cfg["build_dir"] / "bin" / "tile_perf"
    
    test_short = test_name.replace("TilePerfTest.", "")
    dump_dir = output_dir / arch / test_short
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    env = get_env(arch)
    env["NPU_DUMP_PATH"] = str(dump_dir)
    
    start = time.time()
    result = subprocess.run(
        [str(bin_path), f"--gtest_filter={test_name}"],
        capture_output=True, text=True, env=env, cwd=str(dump_dir)
    )
    elapsed = time.time() - start
    
    # Parse test name: TADD_float_16x256
    parts = test_short.split("_")
    op_type = parts[0] if parts else "TADD"
    dtype = parts[1] if len(parts) > 1 else "float"
    shape = parts[2] if len(parts) > 2 else "64x64"
    
    if "x" in shape:
        h, w = map(int, shape.split("x"))
        elements = h * w
    else:
        elements = int(shape)
    
    # Parse tick
    tick = None
    for line in result.stdout.splitlines():
        if "Tick:" in line or "tick:" in line:
            m = re.search(r'[Tt]ick[:\s]+(\d+)', line)
            if m:
                tick = int(m.group(1))
    
    # Extract VF EPC (main metric)
    vf_result = {}
    instr_result = {}
    
    if arch == "a5":
        vf_result = parse_a5_vf_epc(dump_dir, elements)
        instr_result = parse_a5_instr_epc(dump_dir, op_type)
    else:
        vf_result = parse_a2a3_vf_epc(dump_dir, elements, op_type)
    
    return {
        "test_name": test_short,
        "arch": arch,
        "op_type": op_type,
        "dtype": dtype,
        "shape": shape,
        "elements": elements,
        "tick": tick,
        "elapsed": elapsed,
        "returncode": result.returncode,
        "vf_epc": vf_result.get("vf_epc"),
        "vf_cycles": vf_result.get("avg_warm_cycles"),
        "instr_epc": instr_result.get("instr_epc"),
        "instr_latency": instr_result.get("avg_latency"),
        "log_path": str(dump_dir),
        "error": vf_result.get("error") or instr_result.get("error"),
    }


def save_result(conn: sqlite3.Connection, run_id: int, result: dict):
    """Save test result to database."""
    conn.execute("""
        INSERT INTO results (
            run_id, test_name, arch, dtype, shape, elements,
            total_tick, vf_cycles, pop_retire_cycles, epc,
            instr_epc, vf_epc, status, error_msg, log_path, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        result["test_name"],
        result["arch"],
        result["dtype"],
        result["shape"],
        result["elements"],
        result.get("tick"),
        result.get("vf_cycles"),
        result.get("instr_latency"),
        result.get("vf_epc"),  # main EPC
        result.get("instr_epc"),
        result.get("vf_epc"),
        "pass" if result["returncode"] == 0 else "fail",
        result.get("error"),
        result["log_path"],
        datetime.now().isoformat(),
    ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="PTO Regression Worker")
    parser.add_argument("--arch", choices=["a5", "a2a3"], default="a5")
    parser.add_argument("--tests", default="*", help="Test filter pattern")
    parser.add_argument("--output", type=Path, default=Path("/tmp/pto_regress"))
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    args = parser.parse_args()
    
    conn = init_db()
    git = get_git_info()
    
    cur = conn.execute("""
        INSERT INTO runs (commit_hash, branch, arch, started_at)
        VALUES (?, ?, ?, ?)
    """, (git["commit"], git["branch"], args.arch, datetime.now().isoformat()))
    run_id = cur.lastrowid
    conn.commit()
    
    print(f"=== PTO Regression Run #{run_id} ===")
    print(f"Commit: {git['commit']} | Branch: {git['branch']} | Arch: {args.arch}")
    if args.arch == "a5":
        print("EPC Method: A5 vf_real_execute_time")
    else:
        print("EPC Method: A2A3 first_pop → last_retire")
    
    all_tests = list_tests(args.arch)
    
    import fnmatch
    if args.tests == "*":
        tests = all_tests
    else:
        pattern = f"TilePerfTest.{args.tests}"
        tests = [t for t in all_tests if fnmatch.fnmatch(t, pattern)]
    
    print(f"[{args.arch}] {len(tests)} tests matching '{args.tests}'")
    
    for i, test in enumerate(tests, 1):
        short = test.replace("TilePerfTest.", "")
        print(f"[{args.arch}] ({i}/{len(tests)}) {short}")
        
        result = run_test(args.arch, test, args.output)
        save_result(conn, run_id, result)
        
        if result.get("vf_epc"):
            print(f"    ✅ VF EPC: {result['vf_epc']:.2f} | VF cy: {result.get('vf_cycles', 0):.0f} | elem: {result['elements']}")
        else:
            print(f"    ❌ {result.get('error', 'unknown')}")
    
    conn.execute("""
        UPDATE runs SET finished_at = ?, status = 'complete'
        WHERE id = ?
    """, (datetime.now().isoformat(), run_id))
    conn.commit()
    
    print(f"\n=== Run #{run_id} Complete ===")
    conn.close()


if __name__ == "__main__":
    main()
