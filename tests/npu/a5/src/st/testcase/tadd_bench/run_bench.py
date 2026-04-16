#!/usr/bin/env python3
"""TADD Benchmark Runner v2 - Reads cases from input.csv, parses EXU+LSU logs"""
import os, csv, re
from pathlib import Path
from collections import defaultdict

NUM_VF = 10
WARMUP = 2

def parse_vf_times(log_file):
    """Extract vf_real_execute_time from instr_log.dump"""
    times = []
    if not os.path.exists(log_file): return times
    with open(log_file) as f:
        for line in f:
            m = re.search(r"vf_real_execute_time:\s*(\d+)", line)
            if m: times.append(int(m.group(1)))
    return times

def parse_epu_instr(epu_dump):
    """Parse EXU dump for compute/predicate instructions"""
    counts = defaultdict(int)
    if not os.path.exists(epu_dump): return counts
    with open(epu_dump) as f:
        for line in f:
            m = re.search(r"instr_name\s+(RV_\w+)", line)
            if m: counts[m.group(1)] += 1
    return counts

def parse_lsu_instr(lsu_dump):
    """Parse LSU dump for load/store instructions (count retires only)"""
    counts = defaultdict(int)
    if not os.path.exists(lsu_dump): return counts
    with open(lsu_dump) as f:
        for line in f:
            # Count VLD retires
            if "LSU_RETIRE" in line and "RV_VLD" in line:
                m = re.search(r"instr_name\s+(RV_VLD\w*)", line)
                if m: counts[m.group(1)] += 1
            # Count VST releases (stores don't have LSU_RETIRE)
            elif "SEND_RELEASE.ST" in line and "RV_VST" in line:
                m = re.search(r"instr_name\s+(RV_VST\w*)", line)
                if m: counts[m.group(1)] += 1
    return counts

def categorize(epu_counts, lsu_counts):
    """Categorize instructions into RVECSU/EX_PRED/EX_COMPUTE/LD/LP/ST/OTHER"""
    cats = {"RVECSU": 0, "EX_PRED": 0, "EX_COMPUTE": 0, "LD": 0, "LP": 0, "ST": 0, "OTHER": 0}
    
    # EXU instructions
    for instr, cnt in epu_counts.items():
        if "PLT" in instr or "PRED" in instr or "MASK" in instr:
            cats["EX_PRED"] += cnt
        elif any(x in instr for x in ["ADD","MUL","SUB","MAX","MIN","ABS","NEG","DIV","RCP","SQRT","EXP","AND","OR","XOR","SHL","SHR","CVT"]):
            cats["EX_COMPUTE"] += cnt
        elif "SEND" in instr or "SYNC" in instr:
            cats["RVECSU"] += cnt
        elif "LOOP" in instr or "BR" in instr:
            cats["LP"] += cnt
        else:
            cats["OTHER"] += cnt
    
    # LSU instructions
    for instr, cnt in lsu_counts.items():
        if "VLD" in instr:
            cats["LD"] += cnt
        elif "VST" in instr:
            cats["ST"] += cnt
    
    return cats

def read_input_csv(csv_path):
    """Read test cases from input.csv"""
    cases = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 6:
                op, dtype, tile_h, tile_w, valid_h, valid_w = parts[:6]
                cases.append({
                    'op': op.strip(),
                    'dtype': dtype.strip(),
                    'tile_h': int(tile_h),
                    'tile_w': int(tile_w),
                    'valid_h': int(valid_h),
                    'valid_w': int(valid_w)
                })
    return cases

def dtype_to_gtest(dtype):
    """Convert dtype to gtest case naming"""
    mapping = {
        'float32': 'float',
        'float16': 'half',
        'bfloat16': 'bfloat16',
        'int32': 'int32',
        'int16': 'int16',
        'int8': 'int8',
        'uint8': 'uint8'
    }
    return mapping.get(dtype, dtype)

def process(build_dir, case_info):
    """Process a single test case and extract metrics"""
    op = case_info['op']
    dtype = case_info['dtype']
    h, w = case_info['valid_h'], case_info['valid_w']
    elems = h * w
    
    # Derive test name from op/dtype/shape
    dtype_gtest = dtype_to_gtest(dtype)
    test_name = f"{op.upper()}BenchTest.case_{dtype_gtest}_{h}x{w}"
    
    # Determine bytes per element
    dtype_bytes = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int32': 4, 'int16': 2, 'int8': 1, 'uint8': 1}
    elem_bytes = dtype_bytes.get(dtype, 4)
    
    # Log paths
    log_dir = f"{build_dir}/{test_name}"
    instr_log = f"{log_dir}/core0.veccore0.instr_log.dump"
    epu_dump = f"{log_dir}/core0.veccore0.rvec.EXU.dump"
    lsu_dump = f"{log_dir}/core0.veccore0.rvec.LSU.dump"
    
    vf = parse_vf_times(instr_log)
    if len(vf) < NUM_VF:
        print(f"  SKIP {test_name}: only {len(vf)} VFs (need {NUM_VF})")
        return None
    
    warm = vf[WARMUP]
    last8 = vf[WARMUP:]
    avg = sum(last8) / len(last8)
    epc = elems / warm
    bpc = epc * elem_bytes
    
    epu_counts = parse_epu_instr(epu_dump)
    lsu_counts = parse_lsu_instr(lsu_dump)
    cats = categorize(epu_counts, lsu_counts)
    
    per_vf = {k: v / NUM_VF for k, v in cats.items()}
    vl = elems / 64  # VL = 64 for fp32
    per_vl = {k: v / vl if vl > 0 else 0 for k, v in per_vf.items()}
    
    return {
        "name": test_name,
        "op": op,
        "dtype": dtype,
        "shape": f"{h}x{w}",
        "elems": elems,
        "bytes": elems * elem_bytes,
        "vf_cold": vf[1],
        "vf_warm": warm,
        "last8_avg": avg,
        "epc": epc,
        "bpc": bpc,
        "per_vf": per_vf,
        "per_vl": per_vl
    }

def main():
    script_dir = Path(__file__).parent
    bd = script_dir.parent.parent / "build"
    csv_path = script_dir / "input.csv"
    
    cases = read_input_csv(csv_path)
    if not cases:
        print(f"No cases found in {csv_path}")
        return
    
    print(f"Found {len(cases)} test cases in input.csv")
    results = []
    
    for c in cases:
        r = process(str(bd), c)
        if r:
            results.append(r)
            print(f"{r['op']:6} {r['dtype']:8} {r['shape']:12} | warm={r['vf_warm']:4} | EPC={r['epc']:5.1f} | B/cycle={r['bpc']:6.1f} | LD/vl={r['per_vl']['LD']:.1f} ST/vl={r['per_vl']['ST']:.1f}")
    
    if not results:
        print("No results to write")
        return
    
    # Write results CSV
    with open(bd / "tadd_bench_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test_name", "op", "dtype", "shape", "elements", "bytes", "vf_cold", "vf_warm", "last8_avg", "epc", "bytes_per_cycle"])
        for r in results:
            wr.writerow([r["name"], r["op"], r["dtype"], r["shape"], r["elems"], r["bytes"],
                         r["vf_cold"], r["vf_warm"], f"{r['last8_avg']:.1f}", f"{r['epc']:.2f}", f"{r['bpc']:.2f}"])
    
    # Write breakdown CSV
    with open(bd / "tadd_bench_breakdown.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test_name", "op", "dtype", "shape", "elements",
                     "RVECSU/vf", "EX_PRED/vf", "EX_COMPUTE/vf", "LD/vf", "LP/vf", "ST/vf", "OTHER/vf",
                     "RVECSU/vl", "EX_PRED/vl", "EX_COMPUTE/vl", "LD/vl", "LP/vl", "ST/vl", "OTHER/vl"])
        for r in results:
            pv, pl = r["per_vf"], r["per_vl"]
            wr.writerow([r["name"], r["op"], r["dtype"], r["shape"], r["elems"],
                         f"{pv['RVECSU']:.1f}", f"{pv['EX_PRED']:.1f}", f"{pv['EX_COMPUTE']:.1f}",
                         f"{pv['LD']:.1f}", f"{pv['LP']:.1f}", f"{pv['ST']:.1f}", f"{pv['OTHER']:.1f}",
                         f"{pl['RVECSU']:.2f}", f"{pl['EX_PRED']:.2f}", f"{pl['EX_COMPUTE']:.2f}",
                         f"{pl['LD']:.2f}", f"{pl['LP']:.2f}", f"{pl['ST']:.2f}", f"{pl['OTHER']:.2f}"])
    
    print(f"\nResults: {bd}/tadd_bench_results.csv")
    print(f"Breakdown: {bd}/tadd_bench_breakdown.csv")

if __name__ == "__main__":
    main()
