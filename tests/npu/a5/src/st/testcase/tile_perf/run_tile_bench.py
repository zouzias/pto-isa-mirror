#!/usr/bin/env python3
"""
Tile Performance Benchmark Runner
Runs A5 and/or A2A3 tile benchmarks and outputs comparison CSV.

Usage:
    python3 run_tile_bench.py --arch a5          # Run A5 only
    python3 run_tile_bench.py --arch a2a3        # Run A2A3 only
    python3 run_tile_bench.py --arch both        # Run both (default)
    python3 run_tile_bench.py --arch both --output results.csv
    python3 run_tile_bench.py --cases "TADD,TADDS" --shapes "32x64"
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Configuration
CANN_PATH = "/usr/local/Ascend/cann-9.0.0-alpha.1"
A5_SOC = "Ascend910_9599"
A2A3_SOC = "Ascend910B1"
PTO_ISA_ROOT = Path.home() / "pto-isa"
TILE_PERF_PATH = PTO_ISA_ROOT / "tests/npu/a5/src/st/testcase/tile_perf"

@dataclass
class TestCase:
    op: str
    dtype: str
    tile_h: int
    tile_w: int
    valid_h: int
    valid_w: int
    scalar: Optional[float] = None
    
    @property
    def elements(self) -> int:
        return self.valid_h * self.valid_w
    
    @property
    def shape_str(self) -> str:
        return f"{self.valid_h}x{self.valid_w}"
    
    @property
    def test_name(self) -> str:
        if self.op == "TADD":
            return f"TADDBenchTest.case_{self.dtype}_{self.valid_h}x{self.valid_w}"
        elif self.op == "TADDS":
            return f"TADDSBenchTest.case_{self.dtype}_{self.valid_h}x{self.valid_w}"
        else:
            return f"{self.op}BenchTest.case_{self.dtype}_{self.valid_h}x{self.valid_w}"

@dataclass
class BenchResult:
    arch: str
    test_case: TestCase
    total_ticks: int
    warm_cycles: float
    epc: float
    raw_latencies: List[int]

def parse_input_csv(csv_path: Path) -> List[TestCase]:
    """Parse input.csv to get test cases."""
    cases = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            op = row[0].strip()
            dtype = row[1].strip()
            tile_h = int(row[2])
            tile_w = int(row[3])
            valid_h = int(row[4])
            valid_w = int(row[5])
            scalar = float(row[6]) if len(row) > 6 and row[6].strip() else None
            cases.append(TestCase(op, dtype, tile_h, tile_w, valid_h, valid_w, scalar))
    return cases

def setup_env(arch: str) -> dict:
    """Setup environment variables for simulator."""
    env = os.environ.copy()
    soc = A5_SOC if arch == "a5" else A2A3_SOC
    sim_lib = f"{CANN_PATH}/aarch64-linux/simulator/{soc}/lib"
    
    # Source set_env.sh equivalent
    env["ASCEND_HOME_PATH"] = CANN_PATH
    ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{sim_lib}:{CANN_PATH}/aarch64-linux/lib64:{ld_path}"
    
    return env

def get_build_dir(arch: str) -> Path:
    """Get build directory for architecture."""
    return PTO_ISA_ROOT / f"tests/npu/{arch}/src/st/build"

def build_tile_perf(arch: str) -> bool:
    """Build tile_perf for given architecture."""
    build_dir = get_build_dir(arch)
    soc = A5_SOC if arch == "a5" else A2A3_SOC
    
    # Clean and create build dir
    if build_dir.exists():
        subprocess.run(["rm", "-rf", str(build_dir)], check=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    
    env = setup_env(arch)
    
    # CMake
    cmake_cmd = [
        "cmake", "..",
        f"-DTEST_CASE=tile_perf",
        f"-DRUN_MODE=sim",
        f"-DSOC_VERSION={soc}"
    ]
    result = subprocess.run(cmake_cmd, cwd=build_dir, env=env, 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake failed for {arch}: {result.stderr}", file=sys.stderr)
        return False
    
    # Make
    result = subprocess.run(["make", "-j4"], cwd=build_dir, env=env,
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Make failed for {arch}: {result.stderr}", file=sys.stderr)
        return False
    
    print(f"Built tile_perf for {arch.upper()}")
    return True

def run_single_test(arch: str, test_case: TestCase, output_dir: Path) -> Optional[BenchResult]:
    """Run a single test case and parse results."""
    build_dir = get_build_dir(arch)
    binary = build_dir / "bin/tile_perf"
    
    if not binary.exists():
        print(f"Binary not found: {binary}", file=sys.stderr)
        return None
    
    # Create output directory for this test
    test_output_dir = output_dir / test_case.test_name
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy golden files if they exist
    golden_dir = TILE_PERF_PATH / test_case.test_name
    if golden_dir.exists():
        for f in golden_dir.glob("*.bin"):
            subprocess.run(["cp", str(f), str(test_output_dir)], check=True)
    
    env = setup_env(arch)
    
    # Run test
    cmd = [str(binary), f"--gtest_filter={test_case.test_name}"]
    result = subprocess.run(cmd, cwd=test_output_dir, env=env,
                          capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"Test failed: {test_case.test_name} on {arch}", file=sys.stderr)
        return None
    
    # Parse total ticks
    total_ticks = 0
    for line in result.stdout.split('\n'):
        m = re.search(r'Total tick:\s*(\d+)', line)
        if m:
            total_ticks = int(m.group(1))
            break
    
    # Parse instruction logs for EPC
    instr_log = test_output_dir / "core0.veccore0.instr_log.dump"
    popped_log = test_output_dir / "core0.veccore0.instr_popped_log.dump"
    
    latencies = []
    warm_cycles = 0.0
    epc = 0.0
    
    if arch == "a5":
        # A5: Parse VF vf_real_execute_time
        latencies = parse_a5_vf_times(instr_log)
    else:
        # A2A3: Parse VADD pop->retire
        latencies = parse_a2a3_vadd_latencies(instr_log, popped_log)
    
    if len(latencies) > 1:
        warm_latencies = latencies[1:]  # Skip first (cold)
        warm_cycles = sum(warm_latencies) / len(warm_latencies)
        epc = test_case.elements / warm_cycles if warm_cycles > 0 else 0
    
    return BenchResult(
        arch=arch,
        test_case=test_case,
        total_ticks=total_ticks,
        warm_cycles=warm_cycles,
        epc=epc,
        raw_latencies=latencies
    )

def parse_a5_vf_times(log_file: Path) -> List[int]:
    """Extract vf_real_execute_time from A5 instruction log."""
    times = []
    if not log_file.exists():
        return times
    with open(log_file) as f:
        for line in f:
            m = re.search(r'vf_real_execute_time:\s*(\d+)', line)
            if m:
                times.append(int(m.group(1)))
    return times

def parse_a2a3_vadd_latencies(instr_log: Path, popped_log: Path) -> List[int]:
    """Extract VADD pop->retire latencies from A2A3 logs."""
    if not instr_log.exists() or not popped_log.exists():
        return []
    
    # Parse retired VADDs
    retired = {}
    with open(instr_log) as f:
        for line in f:
            if 'VADD' in line and 'VEC' in line:
                m_cycle = re.search(r'\[(\d+)\]', line)
                m_id = re.search(r'Id:(\d+)', line)
                if m_cycle and m_id:
                    retired[int(m_id.group(1))] = int(m_cycle.group(1))
    
    # Parse popped VADDs
    popped = {}
    with open(popped_log) as f:
        for line in f:
            if 'VADD' in line and 'VEC' in line:
                m_cycle = re.search(r'\[(\d+)\]', line)
                m_id = re.search(r'Id:(\d+)', line)
                if m_cycle and m_id:
                    popped[int(m_id.group(1))] = int(m_cycle.group(1))
    
    # Calculate latencies
    latencies = []
    for instr_id in sorted(popped.keys()):
        if instr_id in retired:
            latencies.append(retired[instr_id] - popped[instr_id])
    
    return latencies

def write_results_csv(results: List[BenchResult], output_path: Path):
    """Write benchmark results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'arch', 'op', 'dtype', 'shape', 'elements',
            'total_ticks', 'warm_cycles', 'epc', 
            'theory_epc', 'efficiency_pct', 'raw_latencies'
        ])
        
        for r in results:
            theory_epc = 64.0  # 64 lanes
            efficiency = (r.epc / theory_epc * 100) if theory_epc > 0 else 0
            
            writer.writerow([
                r.arch.upper(),
                r.test_case.op,
                r.test_case.dtype,
                r.test_case.shape_str,
                r.test_case.elements,
                r.total_ticks,
                f"{r.warm_cycles:.1f}",
                f"{r.epc:.2f}",
                f"{theory_epc:.1f}",
                f"{efficiency:.1f}",
                ';'.join(map(str, r.raw_latencies[:5]))  # First 5 latencies
            ])
    
    print(f"Results written to: {output_path}")

def write_comparison_csv(a5_results: List[BenchResult], a2a3_results: List[BenchResult], 
                         output_path: Path):
    """Write side-by-side comparison CSV."""
    # Index results by test name
    a5_by_test = {r.test_case.test_name: r for r in a5_results}
    a2a3_by_test = {r.test_case.test_name: r for r in a2a3_results}
    
    all_tests = set(a5_by_test.keys()) | set(a2a3_by_test.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'op', 'dtype', 'shape', 'elements',
            'a5_warm_cy', 'a5_epc', 'a2a3_warm_cy', 'a2a3_epc',
            'speedup', 'notes'
        ])
        
        for test_name in sorted(all_tests):
            a5_r = a5_by_test.get(test_name)
            a2a3_r = a2a3_by_test.get(test_name)
            
            # Get test case info from whichever result exists
            tc = (a5_r or a2a3_r).test_case
            
            a5_cy = f"{a5_r.warm_cycles:.1f}" if a5_r else "N/A"
            a5_epc = f"{a5_r.epc:.2f}" if a5_r else "N/A"
            a2a3_cy = f"{a2a3_r.warm_cycles:.1f}" if a2a3_r else "N/A"
            a2a3_epc = f"{a2a3_r.epc:.2f}" if a2a3_r else "N/A"
            
            # Calculate speedup (A2A3 / A5)
            speedup = ""
            notes = ""
            if a5_r and a2a3_r and a5_r.epc > 0:
                ratio = a2a3_r.epc / a5_r.epc
                speedup = f"{ratio:.2f}x"
                if ratio > 2:
                    notes = "A2A3 faster"
                elif ratio < 0.5:
                    notes = "A5 faster"
            
            writer.writerow([
                tc.op, tc.dtype, tc.shape_str, tc.elements,
                a5_cy, a5_epc, a2a3_cy, a2a3_epc,
                speedup, notes
            ])
    
    print(f"Comparison written to: {output_path}")

def print_summary(results: List[BenchResult]):
    """Print summary table to console."""
    print("\n" + "="*80)
    print(f"{'Arch':<6} {'Op':<8} {'Shape':<10} {'Elements':<10} {'Warm Cy':<12} {'EPC':<10}")
    print("="*80)
    
    for r in results:
        print(f"{r.arch.upper():<6} {r.test_case.op:<8} {r.test_case.shape_str:<10} "
              f"{r.test_case.elements:<10} {r.warm_cycles:<12.1f} {r.epc:<10.2f}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Tile Performance Benchmark Runner")
    parser.add_argument("--arch", choices=["a5", "a2a3", "both"], default="both",
                       help="Architecture to benchmark (default: both)")
    parser.add_argument("--input", type=Path, default=TILE_PERF_PATH / "input.csv",
                       help="Input CSV with test cases")
    parser.add_argument("--output", type=Path, default=Path("tile_perf_results.csv"),
                       help="Output CSV file")
    parser.add_argument("--comparison", type=Path, default=Path("tile_perf_comparison.csv"),
                       help="Comparison CSV file (when running both)")
    parser.add_argument("--cases", type=str, help="Comma-separated ops to run (e.g., TADD,TADDS)")
    parser.add_argument("--shapes", type=str, help="Comma-separated shapes to run (e.g., 32x64,1x2048)")
    parser.add_argument("--no-build", action="store_true", help="Skip build step")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/tile_perf_bench"),
                       help="Working directory for dumps")
    
    args = parser.parse_args()
    
    # Parse test cases
    if not args.input.exists():
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    test_cases = parse_input_csv(args.input)
    print(f"Loaded {len(test_cases)} test cases from {args.input}")
    
    # Filter cases if specified
    if args.cases:
        ops = [o.strip().upper() for o in args.cases.split(",")]
        test_cases = [tc for tc in test_cases if tc.op.upper() in ops]
    
    if args.shapes:
        shapes = [s.strip() for s in args.shapes.split(",")]
        test_cases = [tc for tc in test_cases if tc.shape_str in shapes]
    
    if not test_cases:
        print("No test cases to run after filtering", file=sys.stderr)
        sys.exit(1)
    
    print(f"Running {len(test_cases)} test cases")
    
    # Determine architectures to run
    archs = ["a5", "a2a3"] if args.arch == "both" else [args.arch]
    
    # Build
    if not args.no_build:
        for arch in archs:
            if not build_tile_perf(arch):
                print(f"Failed to build for {arch}", file=sys.stderr)
                sys.exit(1)
    
    # Create work directory
    args.work_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = []
    a5_results = []
    a2a3_results = []
    
    for arch in archs:
        arch_work_dir = args.work_dir / arch
        arch_work_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRunning {arch.upper()} benchmarks...")
        
        for tc in test_cases:
            print(f"  {tc.test_name}...", end=" ", flush=True)
            result = run_single_test(arch, tc, arch_work_dir)
            if result:
                all_results.append(result)
                if arch == "a5":
                    a5_results.append(result)
                else:
                    a2a3_results.append(result)
                print(f"EPC={result.epc:.2f}")
            else:
                print("FAILED")
    
    # Print summary
    print_summary(all_results)
    
    # Write results
    write_results_csv(all_results, args.output)
    
    # Write comparison if both architectures
    if args.arch == "both" and a5_results and a2a3_results:
        write_comparison_csv(a5_results, a2a3_results, args.comparison)

if __name__ == "__main__":
    main()
