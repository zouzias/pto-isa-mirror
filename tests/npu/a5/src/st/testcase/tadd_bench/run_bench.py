#!/usr/bin/env python3
"""TADD Benchmark Runner - Parses CA model logs and outputs CSV results"""
import os, csv, re
from pathlib import Path
from collections import defaultdict

NUM_VF = 10
WARMUP = 2

def parse_vf_times(log_file):
    times = []
    if not os.path.exists(log_file): return times
    with open(log_file) as f:
        for line in f:
            m = re.search(r"vf_real_execute_time:\s*(\d+)", line)
            if m: times.append(int(m.group(1)))
    return times

def parse_instr(epu_dump):
    counts = defaultdict(int)
    if not os.path.exists(epu_dump): return counts
    with open(epu_dump) as f:
        for line in f:
            m = re.search(r"instr_name\s+(\w+)", line)
            if m: counts[m.group(1)] += 1
    return counts

def categorize(counts):
    cats = {"RVECSU": 0, "EX_PRED": 0, "EX_COMPUTE": 0, "LD": 0, "LP": 0, "ST": 0, "OTHER": 0}
    for instr, cnt in counts.items():
        if "PLT" in instr or "PRED" in instr or "MASK" in instr:
            cats["EX_PRED"] += cnt
        elif instr.startswith("RV_V") and any(x in instr for x in ["ADD","MUL","SUB","MAX","MIN","ABS","NEG","DIV","RCP","SQRT","EXP"]):
            cats["EX_COMPUTE"] += cnt
        elif "VLD" in instr or "LOAD" in instr:
            cats["LD"] += cnt
        elif "VST" in instr or "STORE" in instr:
            cats["ST"] += cnt
        elif "SEND" in instr or "SYNC" in instr:
            cats["RVECSU"] += cnt
        elif "LOOP" in instr or "BR" in instr:
            cats["LP"] += cnt
        else:
            cats["OTHER"] += cnt
    return cats

def process(build_dir, name, h, w):
    elems = h * w
    log = f"{build_dir}/{name}/core0.veccore0.instr_log.dump"
    epu = f"{build_dir}/{name}/core0.veccore0.rvec.EXU.dump"
    vf = parse_vf_times(log)
    if len(vf) < NUM_VF:
        print(f"  SKIP {name}: only {len(vf)} VFs")
        return None
    warm = vf[WARMUP]
    last8 = vf[WARMUP:]
    avg = sum(last8)/len(last8)
    epc = elems/warm
    bpc = epc*4
    cats = categorize(parse_instr(epu))
    per_vf = {k: v/NUM_VF for k,v in cats.items()}
    vl = elems/64
    per_vl = {k: v/vl if vl>0 else 0 for k,v in per_vf.items()}
    return {"name":name,"shape":f"{h}x{w}","elems":elems,"bytes":elems*4,
            "vf_cold":vf[1],"vf_warm":warm,"last8_avg":avg,"epc":epc,"bpc":bpc,
            "per_vf":per_vf,"per_vl":per_vl}

def main():
    bd = Path(__file__).parent.parent.parent / "build"
    cases = [("TADDBenchTest.case_float_64x64",64,64),("TADDBenchTest.case_float_8x512",8,512),
             ("TADDBenchTest.case_float_1x4096",1,4096),("TADDBenchTest.case_float_64x128",64,128),
             ("TADDBenchTest.case_float_16x512",16,512),("TADDBenchTest.case_float_1x8192",1,8192),
             ("TADDBenchTest.case_float_128x128",128,128),("TADDBenchTest.case_float_32x512",32,512),
             ("TADDBenchTest.case_float_1x16384",1,16384)]
    results = []
    for n,h,w in cases:
        r = process(str(bd),n,h,w)
        if r:
            results.append(r)
            print(f"{r['shape']:12} | warm={r['vf_warm']:4} | EPC={r['epc']:5.1f} | B/cycle={r['bpc']:6.1f}")

    with open(bd/"tadd_bench_results.csv","w",newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test_name","shape","elements","bytes","vf_cold","vf_warm","last8_avg","epc","bytes_per_cycle"])
        for r in results:
            wr.writerow([r["name"],r["shape"],r["elems"],r["bytes"],r["vf_cold"],r["vf_warm"],f"{r['last8_avg']:.1f}",f"{r['epc']:.2f}",f"{r['bpc']:.2f}"])

    with open(bd/"tadd_bench_breakdown.csv","w",newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test_name","shape","elements","RVECSU/vf","EX_PRED/vf","EX_COMPUTE/vf","LD/vf","LP/vf","ST/vf","OTHER/vf",
                     "RVECSU/vl","EX_PRED/vl","EX_COMPUTE/vl","LD/vl","LP/vl","ST/vl","OTHER/vl"])
        for r in results:
            pv,pl = r["per_vf"],r["per_vl"]
            wr.writerow([r["name"],r["shape"],r["elems"],
                f"{pv['RVECSU']:.1f}",f"{pv['EX_PRED']:.1f}",f"{pv['EX_COMPUTE']:.1f}",f"{pv['LD']:.1f}",f"{pv['LP']:.1f}",f"{pv['ST']:.1f}",f"{pv['OTHER']:.1f}",
                f"{pl['RVECSU']:.2f}",f"{pl['EX_PRED']:.2f}",f"{pl['EX_COMPUTE']:.2f}",f"{pl['LD']:.2f}",f"{pl['LP']:.2f}",f"{pl['ST']:.2f}",f"{pl['OTHER']:.2f}"])
    print(f"\nResults: {bd}/tadd_bench_results.csv")
    print(f"Breakdown: {bd}/tadd_bench_breakdown.csv")

if __name__ == "__main__": main()
