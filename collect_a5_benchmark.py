#!/usr/bin/env python3
"""
collect_a5_benchmark.py

Runs a curated set of A5 ST testcases in sim mode, one gtest case at a
time (using -g/--without-build), collects metrics from each per-case
core0_summary_log, and writes one CSV row per test case.

A5 metrics extracted:
  kernal_ticks   : "kernal total ticks : <value>"   (kernel only, no dispatch)
  system_ticks   : "system total ticks : <value>"   (wall-clock)
  simd_cycles    : "rvec_veccore0_simd_busy_cycle | <value>"  (VEC ops)
  mac_cycles     : "cube.cube_cubecore0_mac_busy_cycle | <value>"  (MATMUL)

EPC = valid_rows × valid_cols / kernal_ticks  (0 if kernal_ticks not found)

CSV columns:
  testcase, gtest, soc,
  src_dtype, dst_dtype,
  tile_rows, tile_cols, valid_rows, valid_cols,
  kernal_ticks, system_ticks, simd_cycles, mac_cycles,
  total_elements, epc

Usage:
  # Source CANN env first:
  #   source /home/ivanmang/Ascend/cann/bin/setenv.bash
  python3 collect_a5_benchmark.py [--out a5_benchmark.csv]
  python3 collect_a5_benchmark.py --testcases tadd tmul    (subset)
  python3 collect_a5_benchmark.py --no-run                  (re-parse existing logs)
"""

import os
import re
import sys
import csv
import shutil
import argparse
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
TESTS_DIR = REPO / "tests"
SCRIPT = TESTS_DIR / "script" / "run_st.py"
ST_DIR = TESTS_DIR / "npu" / "a5" / "src" / "st"
BUILD_DIR = ST_DIR / "build"
TESTCASE_DIR = ST_DIR / "testcase"
SOC = "a5"
RUN_MODE = "sim"

# Curated set — same instructions as A2A3 where available in A5
CURATED = [
    "tabs", "tadd", "tsub", "tmul", "tdiv", "tneg",
    "texp", "tlog", "trsqrt", "tsqrt", "trecip", "trelu",
    "tmax", "tmin",
    "tcvt",
    "tquant",
    "trowsum", "tcolsum",
    "tmatmul",
    "tload", "tstore", "tassign", "tfillpad", "tmov",
]

CSV_HEADER = [
    "testcase", "gtest", "soc",
    "src_dtype", "dst_dtype",
    "tile_rows", "tile_cols",
    "valid_rows", "valid_cols",
    "kernal_ticks", "system_ticks",
    "simd_cycles", "mac_cycles",
    "total_elements", "epc",
]

# VEC instructions → use simd_cycles for EPC; MTX → use mac_cycles
VEC_TESTCASES = {
    "tabs", "tadd", "tsub", "tmul", "tdiv", "tneg",
    "texp", "tlog", "trsqrt", "tsqrt", "trecip", "trelu",
    "tmax", "tmin", "tcvt", "tquant",
    "trowsum", "tcolsum",
    "tload", "tstore", "tassign", "tfillpad", "tmov",
}
MTX_TESTCASES = {"tmatmul"}


# ---------------------------------------------------------------------------
# Parse TEST_F parameters from main.cpp
# Returns: {case_name: (suite, src_dtype, dst_dtype, tR, tC, vR, vC)}
# ---------------------------------------------------------------------------

def parse_main_cpp(tc_name: str) -> dict:
    fpath = TESTCASE_DIR / tc_name / "main.cpp"
    if not fpath.exists():
        return {}
    text = fpath.read_text(errors="replace")

    def normalize_dtype(raw: str) -> str:
        t = raw.strip().replace(" ", "")
        t = t.replace("std::", "")
        mapping = {
            "aclFloat16": "half",
            "aclFloat32": "float",
            "float": "float",
            "double": "double",
            "int8_t": "int8",
            "uint8_t": "uint8",
            "int16_t": "int16",
            "uint16_t": "uint16",
            "int32_t": "int32",
            "uint32_t": "uint32",
            "int64_t": "int64",
            "uint64_t": "uint64",
            "bfloat16": "bfloat16",
            "aclBfloat16": "bfloat16",
            "half": "half",
        }
        return mapping.get(t, t)

    def parse_from_body(body: str):
        # TAddSTestFramework<caseId, T, dstTileRow, dstTileCol, row, vaildRow, col, srcVaildCol>
        m = re.search(
            r'TAddSTestFramework\s*<\s*\d+\s*,\s*([^,>]+)\s*,'
            r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            tR, tC = int(m.group(2)), int(m.group(3))
            vR, vC = int(m.group(5)), int(m.group(7))
            return dtype, dtype, tR, tC, vR, vC

        # test_tand<T, kTRows_, kTCols_, vRows, vCols, ...>
        m = re.search(
            r'test_tand\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            tR, tC, vR, vC = map(int, m.groups()[1:5])
            return dtype, dtype, tR, tC, vR, vC

        # test_taxpy<T, kTRows_, kTCols_, vRows, vCols[, U]>
        m = re.search(
            r'test_taxpy\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)'
            r'(?:\s*,\s*([^,>]+))?',
            body,
        )
        if m:
            dst_dtype = normalize_dtype(m.group(1))
            tR, tC, vR, vC = map(int, m.groups()[1:5])
            src_dtype = normalize_dtype(m.group(6)) if m.group(6) else dst_dtype
            return src_dtype, dst_dtype, tR, tC, vR, vC

        # test_vci_b32/b16<T, ROW, COL, ...>
        m = re.search(
            r'test_vci_b(?:32|16)\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            tR, tC = int(m.group(2)), int(m.group(3))
            return dtype, dtype, tR, tC, tR, tC

        # TCOLCMAX/TCOLCMIN (argmax/argmin)
        m = re.search(
            r'TCOLC(?:MAX|MIN)TestFramework\s*<\s*\d+\s*,\s*([^,>]+)\s*,'
            r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            src_dtype = normalize_dtype(m.group(1))
            dst_row = int(m.group(4))
            col = int(m.group(5))
            valid_col = int(m.group(6))
            return src_dtype, "uint32", dst_row, col, dst_row, valid_col

        # TCOLEXPANDTestFramework<T, src_row, src_col, src_validCol, dst_row, dst_col, dst_validRow, dst_validCol>
        m = re.search(
            r'TCOLEXPANDTestFramework\s*<\s*([^,>]+)\s*,\s*'
            r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            dst_row, dst_col = int(m.group(5)), int(m.group(6))
            vR, vC = int(m.group(7)), int(m.group(8))
            return dtype, dtype, dst_row, dst_col, vR, vC

        # test_tcolexpandadd/div/max/min/mul/sub<T, dstRow, dstCol, src1Row, src1Col>
        m = re.search(
            r'test_tcolexpand\w*\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            dst_row, dst_col = int(m.group(2)), int(m.group(3))
            return dtype, dtype, dst_row, dst_col, dst_row, dst_col

        # TCOLSumTestFramework<T, cols, src_row, src_validRow, IsBinary>
        m = re.search(
            r'TCOLSumTestFramework\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(?:true|false)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            cols = int(m.group(2))
            return dtype, dtype, 1, cols, 1, cols

        # TCOLMax/Min/ProdTestFramework<T, cols, src_row, src_validRow>
        m = re.search(
            r'TCOL(?:Max|Min|Prod)TestFramework\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            cols = int(m.group(2))
            return dtype, dtype, 1, cols, 1, cols

        # test_tconcat<T, dstH, dstW, src0H, src0W, src1H, src1W, vRows, vCols0, vCols1>
        m = re.search(
            r'test_tconcat\s*<\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
            r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            tR, tC = int(m.group(2)), int(m.group(3))
            vR, vC0, vC1 = int(m.group(8)), int(m.group(9)), int(m.group(10))
            return dtype, dtype, tR, tC, vR, vC0 + vC1

        # test_tconcat<dtype, idxType, dstH, dstW, src0H, src0W, src1H, src1W, vRows, vCols0, vCols1>
        m = re.search(
            r'test_tconcat\s*<\s*([^,>]+)\s*,\s*([^,>]+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,'
            r'\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)',
            body,
        )
        if m:
            dtype = normalize_dtype(m.group(1))
            tR, tC = int(m.group(3)), int(m.group(4))
            vR, vC0, vC1 = int(m.group(9)), int(m.group(10)), int(m.group(11))
            return dtype, dtype, tR, tC, vR, vC0 + vC1

        return None

    results = {}

    # Strategy 0: parse TEST_F body to extract dtype/tile/valid from actual ST calls
    testf_blocks_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*\{([^}]*)\}',
        re.MULTILINE | re.DOTALL
    )
    for block_m in testf_blocks_re.finditer(text):
        suite, case_name, body = block_m.group(1), block_m.group(2), block_m.group(3)
        parsed = parse_from_body(body)
        if parsed:
            src_dtype, dst_dtype, tR, tC, vR, vC = parsed
            results[case_name] = (suite, src_dtype, dst_dtype, tR, tC, vR, vC)

    # Helper: split-based parser for case_<dtype>_<NxN> names
    def _parse_case_name(case_name: str):
        """Return (dtype, tR, tC, vR, vC) or None."""
        if not case_name.startswith("case_"):
            return None
        parts = case_name[5:].split("_")
        dtype_parts, dim_pairs = [], []
        for p in parts:
            m = re.fullmatch(r'(\d+)x(\d+)', p)
            if m:
                dim_pairs.append((int(m.group(1)), int(m.group(2))))
            elif not dim_pairs:
                dtype_parts.append(p)
        if not dtype_parts or not dim_pairs:
            return None
        dtype = "_".join(dtype_parts)
        tR, tC = dim_pairs[0]
        vR, vC = dim_pairs[-1]
        return dtype, tR, tC, vR, vC

    # Strategy 1a: tload/tfillpad GT/VT encoding
    fillpad_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*(case_(\w+?)_GT_((?:\d+_)*\d+)_VT_(\d+)_(\d+)\w*)\s*\)',
        re.MULTILINE
    )
    for m in fillpad_re.finditer(text):
        suite, case, dtype = m.group(1), m.group(2), m.group(3)
        if case in results:
            continue
        gt_nums = [int(x) for x in m.group(4).split('_') if x]
        vtR, vtC = int(m.group(5)), int(m.group(6))
        gtR, gtC = gt_nums[-2], gt_nums[-1]
        results[case] = (suite, dtype, dtype, vtR, vtC, gtR, gtC)

    # Strategy 1b: tquant (sym/asym)
    tquant_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*'
        r'(case_([a-zA-Z0-9]+)_(?:sym|asym)_([a-zA-Z0-9]+)_(\d+)x(\d+)_\w+)\s*\)',
        re.MULTILINE
    )
    for m in tquant_re.finditer(text):
        suite, case, dst, src, R, C = (
            m.group(1), m.group(2), m.group(3), m.group(4),
            int(m.group(5)), int(m.group(6)),
        )
        if case in results:
            continue
        results[case] = (suite, src, dst, R, C, R, C)

    # Strategy 1c: A5 tquant mxfp8 variant
    tquant_mx_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*(case_mxfp8_([a-zA-Z0-9]+)_(\d+)x(\d+)_\w+)\s*\)',
        re.MULTILINE
    )
    for m in tquant_mx_re.finditer(text):
        suite, case, src, R, C = (
            m.group(1), m.group(2), m.group(3),
            int(m.group(4)), int(m.group(5)),
        )
        if case in results:
            continue
        results[case] = (suite, src, "mxfp8", R, C, R, C)

    # Strategy 2: split-based general case_<dtype>_<NxN> names
    all_case_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*(case_\w+)\s*\)',
        re.MULTILINE
    )
    for m in all_case_re.finditer(text):
        suite, case = m.group(1), m.group(2)
        if case in results:
            continue
        p = _parse_case_name(case)
        if p:
            dtype, tR, tC, vR, vC = p
            results[case] = (suite, dtype, dtype, tR, tC, vR, vC)

    # Strategy 3: tcvt-style <mode>_<src>_<dst>_<R>x<C>
    tcvt_re = re.compile(
        r'TEST_F\s*\(\s*(\w+)\s*,\s*'
        r'(\w+?_([a-zA-Z0-9]+?)_([a-zA-Z0-9]+?)_(\d+)x(\d+))\s*\)',
        re.MULTILINE
    )
    for m in tcvt_re.finditer(text):
        suite, case, src, dst, R, C = (
            m.group(1), m.group(2), m.group(3), m.group(4),
            int(m.group(5)), int(m.group(6)),
        )
        if case in results:
            continue
        results[case] = (suite, src, dst, R, C, R, C)

    return results


def list_testcases() -> list:
    """Return all testcase directory names with a main.cpp."""
    if not TESTCASE_DIR.exists():
        return []
    return sorted(
        p.name
        for p in TESTCASE_DIR.iterdir()
        if p.is_dir() and (p / "main.cpp").exists()
    )


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------

def run_build(tc_name: str) -> bool:
    """Build testcase binary and generate test data.

    Replicates the cmake/make/gen_data steps of run_st.py WITHOUT running the
    full GTest suite.  Skipping the all-cases sim run saves ~N_cases × 60s
    per testcase (A5 simulator takes ~60s per case to initialize 32 AICs).

    Requires setenv.bash already sourced: cmake, bisheng, and ASCEND_HOME_PATH
    must all be set in the environment before calling this script.
    """
    print(f"  [BUILD] {tc_name}")
    build_dir = ST_DIR / "build"

    # Clean + recreate build dir (mirrors run_st.py build_project behavior)
    if build_dir.exists():
        shutil.rmtree(str(build_dir))
    build_dir.mkdir(parents=True, exist_ok=True)

    # cmake configure
    cmake_cmd = [
        "cmake",
        f"-DRUN_MODE={RUN_MODE}",
        "-DSOC_VERSION=Ascend910_9599",
        f"-DTEST_CASE={tc_name}",
        "..",
    ]
    r = subprocess.run(
        cmake_cmd, cwd=str(build_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if r.returncode != 0:
        print(f"  [ERROR] cmake failed:\n{r.stdout[-800:]}")
        return False

    # make -j<N>
    cpu_count = os.cpu_count() or 4
    r = subprocess.run(
        ["make", "-j", str(cpu_count)],
        cwd=str(build_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if r.returncode != 0:
        print(f"  [ERROR] make failed:\n{r.stdout[-800:]}")
        return False

    # gen_data.py (copy from testcase dir and run from build dir; non-fatal)
    gen_src = ST_DIR / "testcase" / tc_name / "gen_data.py"
    if gen_src.exists():
        shutil.copy(str(gen_src), str(build_dir / "gen_data.py"))
        subprocess.run(
            [sys.executable, "gen_data.py"],
            cwd=str(build_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return True


def run_single_case(tc_name: str, gtest_filter: str) -> bool:
    """
    Run one gtest case directly, bypassing run_st.py overhead.

    run_st.py -w does: rm -rf build/T* + gen_data + binary invocation.
    We skip the first two (build/golden already exist) and call the binary
    directly. This avoids the rm/gen_data overhead per case.

    Binary: tests/npu/a5/src/st/build/bin/<testcase>
    Log:    CAMODEL_LOG_PATH=../<filter>  → build/<filter>/core0_summary_log
    Library: $ASCEND_HOME_PATH/tools/simulator/Ascend910_9599/lib must be in LD_LIBRARY_PATH
    """
    bin_dir = ST_DIR / "build" / "bin"
    binary = bin_dir / tc_name
    if not binary.exists():
        return False
    env = os.environ.copy()
    env["CAMODEL_LOG_PATH"] = f"../{gtest_filter}"
    # Add simulator lib if not already present
    ascend_home = env.get("ASCEND_HOME_PATH", "")
    if ascend_home:
        sim_lib = f"{ascend_home}/tools/simulator/Ascend910_9599/lib"
        existing = env.get("LD_LIBRARY_PATH", "")
        if sim_lib not in existing:
            env["LD_LIBRARY_PATH"] = f"{sim_lib}:{existing}"
    r = subprocess.run(
        [str(binary), f"--gtest_filter={gtest_filter}"],
        cwd=str(bin_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Log parsing — A5 core0_summary_log
# ---------------------------------------------------------------------------

def read_metrics(log_path: Path) -> dict:
    """
    Extract A5 performance metrics from core0_summary_log.
    Returns dict with keys: kernal_ticks, system_ticks, simd_cycles, mac_cycles
    (any may be None if not present in log).
    """
    metrics = {
        "kernal_ticks": None,
        "system_ticks": None,
        "simd_cycles": None,
        "mac_cycles": None,
    }
    try:
        text = log_path.read_text(errors="replace")
        m = re.search(r'kernal\s+total\s+ticks\s*:\s*(\d+)', text)
        if m:
            metrics["kernal_ticks"] = int(m.group(1))
        m = re.search(r'system\s+total\s+ticks\s*:\s*(\d+)', text)
        if m:
            metrics["system_ticks"] = int(m.group(1))
        m = re.search(r'rvec_veccore0_simd_busy_cycle\s*\|\s*(\d+)', text)
        if m:
            metrics["simd_cycles"] = int(m.group(1))
        m = re.search(r'cube\.cube_cubecore0_mac_busy_cycle\s*\|\s*(\d+)', text)
        if m:
            metrics["mac_cycles"] = int(m.group(1))
    except Exception:
        pass
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A5 ST benchmark collector")
    parser.add_argument("--out", default=str(REPO / "a5_benchmark.csv"),
                        help="Output CSV path (default: a5_benchmark.csv)")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running; collect from existing build dirs only")
    parser.add_argument("--testcases", nargs="*", default=None,
                        help="Override curated list (space-separated testcase names)")
    args = parser.parse_args()

    testcases = args.testcases if args.testcases else list_testcases()
    if not testcases:
        print("[ERROR] No testcases found under tests/npu/a5/src/st/testcase")
        sys.exit(1)
    out_path = Path(args.out)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()

        total_rows = 0
        failed_build = []

        for tc in testcases:
            print(f"\n{'='*60}")
            print(f"Testcase: {tc}")

            params = parse_main_cpp(tc)
            if not params:
                print(f"  [WARN] Could not parse params for {tc}")

            if not args.no_run:
                ok = run_build(tc)
                if not ok:
                    print(f"  [ERROR] Build failed for {tc} — skipping")
                    failed_build.append(tc)
                    continue

            # Choose primary cycle counter for EPC
            use_mac = tc in MTX_TESTCASES

            collected = 0
            for case_name, p in params.items():
                suite, src_dtype, dst_dtype, tR, tC, vR, vC = p
                gtest_filter = f"{suite}.{case_name}"
                log_path = BUILD_DIR / gtest_filter / "core0_summary_log"

                # Per-case run: skip only if --no-run.
                if not args.no_run:
                    run_single_case(tc, gtest_filter)

                m = read_metrics(log_path)
                kt = m["kernal_ticks"]
                if kt is None:
                    print(f"  [MISS] {gtest_filter}")
                    continue

                total_elements = vR * vC
                primary_cycles = m["mac_cycles"] if use_mac else m["simd_cycles"]
                denom = primary_cycles if primary_cycles else kt
                epc = round(total_elements / denom, 6) if denom and total_elements > 0 else ""

                writer.writerow({
                    "testcase":       tc,
                    "gtest":          gtest_filter,
                    "soc":            "a5",
                    "src_dtype":      src_dtype,
                    "dst_dtype":      dst_dtype,
                    "tile_rows":      tR,
                    "tile_cols":      tC,
                    "valid_rows":     vR,
                    "valid_cols":     vC,
                    "kernal_ticks":   kt,
                    "system_ticks":   m["system_ticks"] or "",
                    "simd_cycles":    m["simd_cycles"] or "",
                    "mac_cycles":     m["mac_cycles"] or "",
                    "total_elements": total_elements,
                    "epc":            epc,
                })
                csv_file.flush()
                collected += 1
                print(f"  [OK] {gtest_filter}: kernal={kt}, simd={m['simd_cycles']}, mac={m['mac_cycles']}, epc={epc}")

            total_rows += collected
            print(f"  → {collected}/{len(params)} rows collected")

    print(f"\n{'='*60}")
    print(f"Done. {total_rows} rows → {out_path}")
    if failed_build:
        print(f"Build failures: {', '.join(failed_build)}")
    print("="*60)


if __name__ == "__main__":
    main()
