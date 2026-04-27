---
name: Cube Matmul N-Buffer Profiling (msprof / trace.json)
description: 'Profile a passing PTO cube/matmul N-buffer kernel on the A5 simulator (Ascend950PR_9599, dav-c310-cube) and produce trace.json + per-pipe busy-cycle data for pipeline analysis. USE WHEN: a `CubeMatmulNBufTest.*` config already passes correctness and you need msprof outputs (`trace.json`, `instr_exe.csv`, `core0_summary_log`, `core0.cubecore0_su_perf_summary_log`) to identify the long pole (MTE2 / MTE1 / CUBE-MAC / FIXP / SCALAR), to compare two N-buffer configurations, or to feed Chrome trace viewer / MindStudio Insight. Covers the gotchas hit on this dev box: missing golden bins under the cwd, `LD_LIBRARY_PATH` for the A5 sim libs, `chmod 700` on the output dir, and where artifacts land relative to the build tree. Sibling skill: `cube-matmul-nbuffer-debug` (for the *correctness* failure mode).'
license: CANN Open Software License Agreement Version 2.0
---

# Cube Matmul N-Buffer Profiling — `msprof op simulator` Recipe

This skill captures the exact steps used to generate `trace.json` and the
companion CSV/log artifacts for the `CubeMatmulNBufTest.bnbuf8_K16_8KB`
configuration (B-tile N-buffer, 8 B slots × 8 KiB, A=[32,128] big-load
ping-pong) under `tests/npu/a5/src/st/testcase/cube_matmul_nbuf/`.

The same recipe works for any other `CubeMatmulNBufTest.<cfg>` once that
config passes correctness via `run_st.py`.

## Decision: when to use this skill

| Situation | Apply this skill? |
|-----------|-------------------|
| Need `trace.json` for pipeline-overlap inspection | YES |
| Need per-pipe busy cycles (`mte2/mte1/cube.mac/fixp/scalar`) | YES |
| Need per-instruction cycles (`instr_exe.csv`) for hot-instr ranking | YES |
| Comparing two configs side-by-side after they both pass | YES |
| Test is **failing** correctness (~99% wrong) | NO — use sibling `cube-matmul-nbuffer-debug` first |
| Test never built or compile error | NO — fix build first via `run_st.py` |

Profiling **only works on a binary that already runs to completion under the
sim**. If `run_st.py` fails the gtest, msprof will exit early with no usable
data.

## Prerequisites (verify in order)

1. **CANN env**: must be `cann_9b2`, NOT the default `ascend-toolkit`.
   ```bash
   source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
   command -v msprof   # → /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/bin/msprof
   echo "$ASCEND_HOME_PATH"
   ```
2. **Sim runtime libs on `LD_LIBRARY_PATH`** (msprof rejects without them):
   ```bash
   export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
   ```
3. **Test passes via the standard runner** (build + golden generated):
   ```bash
   python3 tests/script/run_st.py -r sim -v a5 -t cube_matmul_nbuf \
       -g 'CubeMatmulNBufTest.bnbuf8_K16_8KB'
   # expect: bad count: 0,  [  PASSED  ] 1 test.
   ```

## One-shot profiling command

Run from repo root.

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH

CFG=bnbuf8_K16_8KB                    # any CubeMatmulNBufTest.* name
OUT=/tmp/msprof_${CFG}
rm -rf "$OUT" && mkdir -p "$OUT" && chmod 700 "$OUT"   # msprof refuses g/o-writable dirs

cd tests/npu/a5/src/st/build/bin
# CAMODEL_LOG_PATH is where the simulator drops core*_summary_log etc.
export CAMODEL_LOG_PATH="$(pwd)/../CubeMatmulNBufTest.${CFG}"
mkdir -p "$CAMODEL_LOG_PATH"

msprof op simulator \
    --soc-version=Ascend950PR_9599 \
    --output="$OUT" \
    ./cube_matmul_nbuf --gtest_filter=CubeMatmulNBufTest.${CFG}
```

Wall-clock: ~60–120 s for one config of this kernel (the A5 sim is ~10–20×
slower than A3; do not assume a hang).

Successful tail:

```
[INFO] Total tick: 23608
[       OK ] CubeMatmulNBufTest.bnbuf8_K16_8KB
[  PASSED  ] 1 test.
[INFO]  Profiling results saved in /tmp/msprof_bnbuf8_K16_8KB/OPPROF_<ts>_<id>
```

## Where the artifacts land

```
$OUT/OPPROF_<timestamp>_<id>/
├── dump/                                 # raw simulator traces (large, optional)
└── simulator/
    ├── trace.json                        # ← top-level Chrome-trace timeline
    ├── visualize_data.bin                # MindStudio Insight payload
    └── core0.cubecore0/
        ├── trace.json                    # cubecore-only timeline
        ├── core0.cubecore0_instr_exe.csv # per-instr cycles + pipe + opcode
        └── core0.cubecore0_code_exe.csv  # empty unless built with -g
```

Plus, in the `CAMODEL_LOG_PATH` directory (sibling of `bin/`):

```
tests/npu/a5/src/st/build/CubeMatmulNBufTest.<CFG>/
├── core0_summary_log                       # ← per-pipe busy cycles (rich on A5)
├── core0.cubecore0_su_perf_summary_log     # scalar PMU stalls (icache, payloadQ, …)
└── core0.cubecore0.*.dump                  # raw sub-unit logs (very large)
```

## Archive into the testcase tree

Convention used by this testcase (matches commit `b4f61057` for buf4/buf8):

```bash
DST=tests/npu/a5/src/st/testcase/cube_matmul_nbuf/profiling/${CFG}
SRC=$(ls -1d "$OUT"/OPPROF_* | tail -1)
mkdir -p "$DST"
cp "$SRC/simulator/trace.json"                                  "$DST/trace.json"
cp "$SRC/simulator/core0.cubecore0/trace.json"                  "$DST/core0.cubecore0.trace.json"
cp "$SRC/simulator/core0.cubecore0/core0.cubecore0_instr_exe.csv" "$DST/instr_exe.csv"
cp tests/npu/a5/src/st/build/CubeMatmulNBufTest.${CFG}/core0_summary_log                  "$DST/core0_summary_log"
cp tests/npu/a5/src/st/build/CubeMatmulNBufTest.${CFG}/core0.cubecore0_su_perf_summary_log "$DST/core0.cubecore0_su_perf_summary_log"
```

Do **not** copy the `dump/` tree (hundreds of MB and not consumed by any
downstream tool here).

## Reading the artifacts (A5 / dav-c310)

### 1. `core0_summary_log` — first stop, identifies the long pole

Sample for `bnbuf8_K16_8KB`:

```
kernal total ticks : 22677
fixp_cubecore0_busy_cycle             | 1420
mte3_cubecore0_su_busy_cycle          | 0
mte2_cubecore0_su_busy_cycle          | 20056    ← 88% of total
mte1_cubecore0_su_busy_cycle          | 4375     ← 19%
cube.cube_cubecore0_busy_cycle        | 3648     ← 16%
cube.cube_cubecore0_mac_busy_cycle    | 3584
CCU.scalar_cubecore0_su_busy_cycle    | 4487     ← 20%
```

Rule of thumb: the kernel is bound by whichever `*_busy_cycle` is closest to
`kernal total ticks`. Above, MTE2 = 88 % → memory-bandwidth bound; further
reductions in scalar/cube overhead won't move the wall.

### 2. `instr_exe.csv` — per-instruction ranking

The first two rows are usually the dominant MTE2 ND2NZ pair (B-tile loads
for an 8-buf B kernel):

```
instr,addr,pipe,call_count,cycles,running_time(us),detail
MOV_OUT_TO_L1_MULTI_ND2NZ,…,MTE2,64,102084,11.13,…
MOV_SPR_XN,…,MTE2,64,84762,11.00,…   (the SPR setup for ND2NZ)
MOV_OUT_TO_L1_MULTI_ND2NZ,…,MTE2,8,11297, 6.28,…  (A big-load, once per outer)
```

Quick aggregation by pipe:

```bash
awk -F, 'NR>1 {sum[$3]+=$5} END {for (p in sum) print p, sum[p]}' "$DST/instr_exe.csv" \
  | sort -k2 -nr
```

### 3. `core0.cubecore0_su_perf_summary_log` — scalar stall breakdown

Use to confirm whether scalar is *blocked on the MTE2 payload queue*
(the classic memory-bound symptom):

```
su_issue_mte2_payloadQ_stall_cycle    ← high for memory-bound, MTE2-heavy kernels
su_issue_icache_miss_cycle            ← high → code footprint too large
su_issue_branch_stall_cycle
```

### 4. `trace.json` — visualize the pipeline

```bash
python3 -c "import json; print('events=', len(json.load(open('$DST/trace.json'))['traceEvents']))"
# bnbuf8_K16_8KB → events= 1977
```

Open in Chrome: `chrome://tracing` → *Load* → select `trace.json`.
Look for:
- contiguous MTE2 bars vs gaps (gaps = under-utilized GM bandwidth)
- whether CUBE bars overlap MTE2 (good) or strictly follow it (bad)
- `set_flag` / `wait_flag` events stacking up (sync-bound)

## Profiling multiple configs in one shot

```bash
for CFG in bnbuf2_K16_8KB bnbuf4_K16_8KB bnbuf8_K16_8KB \
           bnbuf2_K32_16KB bnbuf4_K32_16KB; do
    OUT=/tmp/msprof_${CFG}
    rm -rf "$OUT" && mkdir -p "$OUT" && chmod 700 "$OUT"
    ( cd tests/npu/a5/src/st/build/bin
      export CAMODEL_LOG_PATH="$(pwd)/../CubeMatmulNBufTest.${CFG}"
      mkdir -p "$CAMODEL_LOG_PATH"
      msprof op simulator --soc-version=Ascend950PR_9599 \
        --output="$OUT" ./cube_matmul_nbuf \
        --gtest_filter=CubeMatmulNBufTest.${CFG}
    ) 2>&1 | tail -3
done
```

Plan ~10–15 min wall-clock for the full sweep.

## Gotchas observed on this dev box

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Cannot open ../CubeMatmulNBufTest.golden/A_gm.bin` | Golden bins not generated (binary launched directly without `run_st.py` first) | Run `python3 tests/npu/a5/src/st/testcase/cube_matmul_nbuf/gen_data.py` from `tests/npu/a5/src/st/build/` so the `CubeMatmulNBufTest.golden/` dir is created next to `bin/` |
| `msprof` rejects output dir | dir is group/other-writable | `chmod 700 $OUT` |
| `libesl_top_wrapper.so: cannot open …` | Sim libs missing from `LD_LIBRARY_PATH` | `export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH` |
| `WARN  Child process killed by signal 11` followed by parser still running | The gtest itself failed (often the golden-file issue above) — sim wrote partial dumps | Fix the gtest first; do not trust the OPPROF output of a crashed run |
| `Kernel missed debug_line information` | `code_exe.csv` will be empty (no source-line hotspots) | Acceptable for pipeline analysis; add `-g` to the kernel build only if you specifically need source attribution |
| Empty `trace.json` (`events=0`) | Build was `npu` mode, or run wrapper script, or sim crashed early | Confirm the build is `sim` mode (`run_st.py -r sim -v a5`) and the binary completes the gtest standalone |
| `0` for `mte3_cubecore0_su_busy_cycle` | Expected — cube-only kernel, no UB→GM stores | Not a bug |

## Cross-config comparison checklist

When building a per-config performance table (as in `REPORT.md` Section 13):

1. Profile every config with the loop above.
2. From each `core0_summary_log`, capture:
   `kernal total ticks`, `mte2_*_busy`, `mte1_*_busy`, `cube.cube_*_mac_busy`,
   `fixp_*_busy`, `CCU.scalar_*_busy`.
3. **Do not compare absolute ticks across SoC versions** — only within A5.
4. The "ceiling" for an MTE2-bound kernel ≈ `mte2_*_busy_cycle` of the most
   bandwidth-efficient config (single contiguous big-tile load wins).

## Anti-patterns

- Running `msprof` on a binary the gtest cannot pass — the OPPROF dir is
  half-empty and metrics are meaningless.
- Profiling under the default `ascend-toolkit/set_env.sh` — sim libs path
  resolution differs; use `cann_9b2` consistently with the rest of this repo.
- Comparing tick counts across SoC versions (A3 vs A5) — clocks and pipes
  differ; compare *percentages* within one SoC.
- Re-running `python3 tests/script/run_st.py` (without `--without-build`)
  between profiling runs — it nukes `build/` and your `OPPROF_*` directories
  in `/tmp` survive but the matching `core*_summary_log` under
  `build/CubeMatmulNBufTest.<CFG>/` is gone. Archive both **before** any
  rebuild.

## Verified result for `bnbuf8_K16_8KB`

Artifacts committed under
[tests/npu/a5/src/st/testcase/cube_matmul_nbuf/profiling/bnbuf8_K16_8KB/](../../../tests/npu/a5/src/st/testcase/cube_matmul_nbuf/profiling/bnbuf8_K16_8KB/):

| File | Size | Purpose |
|------|-----:|---------|
| `trace.json` | ~480 KB, 1977 events | Chrome-trace timeline |
| `core0.cubecore0.trace.json` | ~450 KB | Cubecore-only timeline |
| `instr_exe.csv` | ~20 KB | Per-instruction cycles & pipe |
| `core0_summary_log` | ~1 KB | Per-pipe busy-cycle counts |
| `core0.cubecore0_su_perf_summary_log` | ~4 KB | Scalar PMU stalls |

Headline numbers (A5 sim, kernel total = 22 677 ticks):
**MTE2 88 %, scalar 20 %, MTE1 19 %, CUBE 16 %, FIXP 6 %** → memory-bandwidth
bound; B-tile N-buffer with 8 buffers does saturate MTE2.
