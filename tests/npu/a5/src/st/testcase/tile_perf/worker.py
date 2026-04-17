#!/usr/bin/env python3
"""
PTO Regression Worker - Runs tests and extracts performance metrics.

EPC Metrics:
- vf_epc: VF throughput from pop→retire latency (vf_real_execute_time)
- instr_epc: Per-instruction EPC from EXU dump (for reference)

VF pop→retire is the correct metric - pure VF compute time without icache overhead.
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
    Parse VF throughput EPC from pop→retire latency.
    Uses vf_real_execute_time which excludes icache overhead.
    
    Source files:
    - instr_popped_log.dump: VF dispatch (pop) cycles
    - instr_log.dump: VF retire cycles + vf_real_execute_time
    """
    instr_log = dump_dir / "core0.veccore0.instr_log.dump"
    popped_log = dump_dir / "core0.veccore0.instr_popped_log.dump"
    
    if not instr_log.exists():
        return {"error": f"instr_log not found: {instr_log}"}
    
    # Method 1: Use vf_real_execute_time directly from retire log
    content = instr_log.read_text()
    pattern = r"\[(\d+)\].*VF.*vf_real_execute_time:\s*(\d+)"
    matches = re.findall(pattern, content)
    
    if matches:
        # Skip first cold VF, use warm average
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
    
    # Method 2: Fallback to pop→retire calculation
    if not popped_log.exists():
        return {"error": "No VF timing found in logs"}
    
    # Parse pop times
    pop_times = {}
    for line in popped_log.read_text().splitlines():
        if "VF" in line:
            m = re.search(r"\[(\d+)\].*ID:\s*(\d+)\)\s*VF", line)
            if m:
                pop_times[int(m.group(2))] = int(m.group(1))
    
    # Parse retire times
    retire_times = {}
    for line in instr_log.read_text().splitlines():
        if "VF" in line:
            m = re.search(r"\[(\d+)\].*ID:\s*(\d+)\)\s*VF", line)
            if m:
                retire_times[int(m.group(2))] = int(m.group(1))
    
    # Calculate pop→retire latencies
    latencies = []
    for instr_id in sorted(pop_times.keys()):
        if instr_id in retire_times:
            latencies.append(retire_times[instr_id] - pop_times[instr_id])
    
    if not latencies:
        return {"error": "No matching VF pop/retire pairs"}
    
    # Skip first cold, use warm
    warm = latencies[1:] if len(latencies) > 1 else latencies
    avg_warm = sum(warm) / len(warm)
    vf_epc = elements / avg_warm if avg_warm > 0 else 0
    
    return {
        "method": "pop_retire_delta",
        "vf_count": len(latencies),
        "cold_cycles": latencies[0] if latencies else 0,
        "warm_times": warm,
        "avg_warm_cycles": avg_warm,
        "elements": elements,
        "vf_epc": vf_epc,
    }


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


def parse_a2a3_vf_epc(dump_dir: Path, elements: int) -> dict:
    """Parse A2A3 VF EPC from pop→retire."""
    instr_log = dump_dir / "core0.veccore0.instr_log.dump"
    popped_log = dump_dir / "core0.veccore0.instr_popped_log.dump"
    
    if not instr_log.exists() or not popped_log.exists():
        return {"error": "A2A3 logs not found"}
    
    # Parse VF pop/retire times
    pop_times = {}
    for line in popped_log.read_text().splitlines():
        if "VF" in line:
            m = re.search(r"\[(\d+)\].*ID:\s*(\d+)\)\s*VF", line)
            if m:
                pop_times[int(m.group(2))] = int(m.group(1))
    
    retire_times = {}
    for line in instr_log.read_text().splitlines():
        if "VF" in line:
            m = re.search(r"\[(\d+)\].*ID:\s*(\d+)\)\s*VF", line)
            if m:
                retire_times[int(m.group(2))] = int(m.group(1))
    
    latencies = []
    for instr_id in sorted(pop_times.keys()):
        if instr_id in retire_times:
            latencies.append(retire_times[instr_id] - pop_times[instr_id])
    
    if not latencies:
        return {"error": "No VF pairs found"}
    
    warm = latencies[1:] if len(latencies) > 1 else latencies
    avg_warm = sum(warm) / len(warm)
    vf_epc = elements / avg_warm if avg_warm > 0 else 0
    
    return {
        "vf_count": len(latencies),
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
        vf_result = parse_a2a3_vf_epc(dump_dir, elements)
    
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
    print(f"Commit: {git['commit']} | Branch: {git['branch']}")
    print(f"EPC Method: VF pop→retire (vf_real_execute_time)")
    
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
            print(f"    ✅ VF EPC: {result['vf_epc']:.2f} | VF cy: {result.get('vf_cycles', 0):.0f}")
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
