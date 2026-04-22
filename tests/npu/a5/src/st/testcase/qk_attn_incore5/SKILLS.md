# SKILLS.md — How to Run PyPTO Cases: End-to-End Workflow

This document captures the full workflow for developing and validating PyPTO-generated kernels
on the A5 simulator, based on the Qwen3-32B decode `incore_5` case.

---

## Overview

```
PyPTO DSL (Python)
    ↓ compile_program()
30 MLIR passes → .pto files (PTO assembly)
    ↓ ptoas --pto-arch=a5
CCE C++ kernel files
    ↓ CCE compiler (pto_mix_st CMake macro)
A5 sim binary
    ↓ run_st.py / msprof
Correctness result + trace.json profiling data
    ↓ gen_pipeline_svg.py
Pipeline SVG diagram
```

---

## Part 1: Write and Compile the PyPTO Model

### 1.1 Environment setup

```bash
# On server: happybot@192.168.0.106
source ~/venv/bin/activate
export PTOAS_ROOT=/home/happybot/PTOAS-official/build/tools/ptoas
```

### 1.2 PyPTO import pattern

```python
import sys
sys.path.insert(0, '/home/happybot/pypto/python')
sys.path.insert(0, '/home/happybot/pypto-lib')

import pypto.language as pl
from pypto.runtime import compile_program, BackendType, OptimizationStrategy
```

### 1.3 Minimal PyPTO program structure

```python
class MyModel:
    @pl.function(type=pl.FunctionType.Opaque)
    def forward(self, ...):
        # Use pl.matmul, pl.add, pl.mul etc.
        with pl.at(level=pl.Level.CORE_GROUP, ...):
            result = pl.matmul(a, b)
            # Optional: add Vec op after matmul to enable split=UP_DOWN
            scaled = pl.mul(result, scale)
        return scaled

program = pl.program(MyModel)(...)
```

**Critical rules:**
- Top-level method MUST be decorated with `@pl.function(type=pl.FunctionType.Opaque)`
- A bare `@pl.program` class with undecorated methods fails: "Class contains no @pl.function decorated methods"

### 1.4 Enable split=UP_DOWN (TPUSH/TPOP pattern)

To generate TPUSH/TPOP cross-core C2V pipeline:

```python
# ✅ Correct: single matmul + Vec op per split=UP_DOWN block
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
    qk = pl.matmul(q, k, transpose_b=True)
    scores = pl.mul(qk, attn_scale)   # Vec op → creates Acc→Vec boundary

# ❌ Wrong: two matmuls in one split=UP_DOWN block
with pl.at(level=pl.Level.CORE_GROUP,
           optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
    gate = pl.matmul(x, w_gate)
    up   = pl.matmul(x, w_up)    # ExpandMixedKernel will fail!
    out  = pl.mul(gate, pl.silu(up))
```

**Rule:** Only ONE matmul accumulation chain per `split=UP_DOWN` `pl.at` block. Multiple matmuls
in the same block cause `ExpandMixedKernel` error:
> `Function must order cross-core tpop chains as 'tpop -> use -> tfree -> next tpop'`

### 1.5 Compile the program

```python
from pypto.runtime import compile_program, BackendType, OptimizationStrategy

work_dir = '/home/happybot/npu_skills/seg_topics/pypto_stack_overview_qwen_study/my_kernel'
results = compile_program(
    program,
    work_dir=work_dir,
    strategy=OptimizationStrategy.Default,
    backend_type=BackendType.Ascend950,   # A5
    dump_passes=True                       # saves 30 pass dump .py files
)
```

Output files:
- `passes_dump/00_frontend.py` … `29_after_Simplify.py` (30 pass dumps)
- `ptoas/` — `.pto` PTO assembly files, one per incore function
- `cce/` — `.cpp` CCE C++ kernels after ptoas compilation

### 1.6 Run ptoas manually (for inspection)

```bash
env PTOAS_ROOT=/home/happybot/PTOAS-official/build/tools/ptoas \
    python3 -c "
from pypto.runtime import compile_pto_to_cce
compile_pto_to_cce(
    pto_file='my_kernel.pto',
    out_file='my_kernel.cpp',
    arch='a5',
    level='level3',
    enable_insert_sync=True
)"
```

Or directly:
```bash
$PTOAS_ROOT/ptoas my_kernel.pto \
    --pto-arch=a5 --pto-level=level3 --enable-insert-sync \
    -o my_kernel.cpp
```

**Notes:**
- `--pto-level=level3` is required for Qwen3 kernels (uses addr operands)
- `--enable-insert-sync` required for sync insertion analysis

---

## Part 2: Inspect Pass Dumps

The 30 compile passes are named:

| Pass # | Name | What to look for |
|--------|------|-----------------|
| 00 | Frontend | Original DSL → tensor ops |
| 10 | OutlineIncoreScopes | `tensor.matmul(q, k, b_trans=True)` separated into incore functions |
| 12 | ConvertTensorToTileOps | `tile.load(..., transpose=True, target_memory=Mat)` |
| 15 | InferTileMemorySpace | `tile.move` inserted (MTE1 path) |
| 19 | ExpandMixedKernel | AIC (`@pl.function(type=AIC)`) vs AIV assigned; TPUSH/TPOP generated |
| 29 | Final Simplify | Final IR, ready for ptoas |

Key pass 19 markers for split=UP_DOWN:
```python
# AIC side — look for:
tile.push(acc, pipe)
# AIV side — look for:
tile.pop(pipe) → vec_tile
```

For no_split (scope1/scope2), pass 19 has:
```python
tensor.store(acc, gm_tensor)  # direct L0C→GM, no TPUSH
```

---

## Part 3: Write a CCE ST Testcase

### 3.1 Testcase structure (mixed AIC+AIV)

```
tests/npu/a5/src/st/testcase/<testname>/
├── CMakeLists.txt           # uses pto_mix_st macro
├── <testname>_kernel.cpp    # AIC (#if __DAV_CUBE__) + AIV (#if __DAV_VEC__)
├── main.cpp                 # GTest harness
└── gen_data.py              # Input/output data generator
```

### 3.2 CMakeLists.txt

```cmake
pto_mix_st(<testname>)
```

Register in `tests/npu/a5/src/st/testcase/CMakeLists.txt`:
```cmake
list(APPEND ALL_TESTCASES ... <testname>)
```

### 3.3 Kernel structure

```cpp
// In <testname>_kernel.cpp

__global__ AICORE void RunMyKernel(__gm__ float *output, __gm__ bfloat16_t *input, ...)
{
    // ── AIC body ──────────────────────────────────────────────
#if defined(__DAV_CUBE__)
    {
        auto pipe = TPipe<0, Direction::DIR_C2V, SLOT_SIZE, DEPTH>(nullptr, 0, 0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        // ... TLOAD, TMOV, TMATMUL, TPUSH ...
        for (...) {
            TLOAD(k_tile, k_gm);
            TLOAD(q_tile, q_gm);
            TMATMUL(acc, q_tile, k_tile);
            TPUSH<PipeType, AccType, TILE_UP_DOWN>(pipe, acc);
        }
    }
#endif

    // ── AIV body ──────────────────────────────────────────────
#if defined(__DAV_VEC__)
    {
        auto pipe = TPipe<0, Direction::DIR_C2V, SLOT_SIZE, DEPTH>(nullptr, 0, 0);
        int64_t subblk = get_subblockid();  // 0 or 1
        for (...) {
            TPOP<PipeType, VecType, TILE_UP_DOWN>(pipe, vec_tile);
            TMULS(out_tile, vec_tile, scale);
            TFREE<PipeType, TILE_UP_DOWN>(pipe);
            TSTORE(gm_view, out_tile);
        }
    }
#endif

    ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
}
```

### 3.4 Reference: existing TPUSH/TPOP testcases

```bash
ls ~/pto-isa-ptoas-st/tests/npu/a5/src/st/testcase/ | grep tpush
# tpushpop_cv, tpushpop_vc, tpushpop_dir_both, tpushpop_cv_nosplit, tpushpop_vc_nosplit
```

---

## Part 4: Build and Run on A5 Sim

### 4.1 Build

```bash
cd ~/pto-isa-ptoas-st
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh

python3 tests/script/build_st.py -r sim -v a5 -t <testname>
# Output: tests/npu/a5/src/st/build/bin/<testname>
```

### 4.2 Run (correctness)

```bash
python3 tests/script/run_st.py -r sim -v a5 -t <testname> \
    -g '<TestSuiteName>.<case_name>'
```

### 4.3 Run all testcases

```bash
python3 tests/script/build_st.py -r sim -v a5 -t all
```

---

## Part 5: Profile with msprof

### 5.1 Profile command

```bash
source /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh

TESTBIN=tests/npu/a5/src/st/build/bin/<testname>
OUTDIR=/tmp/msprof_<testname>
mkdir -p $OUTDIR && chmod 700 $OUTDIR  # must be mode 700!

SIMLIB=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib
export LD_LIBRARY_PATH="$SIMLIB:$LD_LIBRARY_PATH"

msprof op simulator \
    --soc-version=Ascend950PR_9599 \
    --output=$OUTDIR \
    $TESTBIN --gtest_filter='<TestSuiteName>.<case_name>'
```

**Important:**
- Profile the final binary, NOT `bash run.sh`
- Output dir must be `chmod 700` — msprof rejects group/world-writable dirs
- A5 sim profiling is ~200x slower than real hardware (small kernel = ~3.5 min)

### 5.2 Locate trace.json

```bash
# trace.json is in simulator/ subdirectory (NOT OPPROF root!)
ls $OUTDIR/OPPROF_*/simulator/
# trace.json                    ← combined all-core trace
# core0.cubecore0/trace.json    ← AIC only
# core0.veccore0/trace.json     ← AIV-0 only
# core0.veccore1/trace.json     ← AIV-1 only
# core0.cubecore0/core0.cubecore0_instr_exe.csv
# core0.veccore0/core0.veccore0_instr_exe.csv
```

**Note:** Unlike `cube_matmul_nbuf` (hardware profiling), the A5 sim places trace.json inside
`simulator/` not at the OPPROF root. Always check `find $OUTDIR -name trace.json`.

### 5.3 Copy profiling data to repo

```bash
PROFDIR=tests/npu/a5/src/st/testcase/<testname>/profiling
mkdir -p $PROFDIR

cp $OUTDIR/OPPROF_*/simulator/trace.json $PROFDIR/
cp $OUTDIR/OPPROF_*/simulator/core0.cubecore0/core0.cubecore0_instr_exe.csv $PROFDIR/
cp $OUTDIR/OPPROF_*/simulator/core0.veccore0/core0.veccore0_instr_exe.csv  $PROFDIR/
```

---

## Part 6: Generate Pipeline SVG

```bash
TRACE=tests/npu/a5/src/st/testcase/<testname>/profiling/trace.json

# Startup window (0-10us, most informative)
python3 ~/npu_skills/performance/gen_pipeline_svg.py $TRACE \
    --out tests/npu/a5/src/st/testcase/<testname>/profiling/pipeline_startup.svg \
    --title "<testname> Pipeline (0-10us startup)" \
    --tmin 0.0 --tmax 10.0

# Full run
python3 ~/npu_skills/performance/gen_pipeline_svg.py $TRACE \
    --out tests/npu/a5/src/st/testcase/<testname>/profiling/pipeline_full.svg \
    --title "<testname> Full Run Pipeline" \
    --tmin 0.0 --tmax 44.0
```

**Script location:** `~/npu_skills/performance/gen_pipeline_svg.py`  
**Important:** Never generate SVG via SSH heredoc — double quotes get stripped. Always:
- Write script locally → `scp` to server → run remotely, OR
- Use the committed script at `~/npu_skills/performance/gen_pipeline_svg.py`

**SVG colour key:**

| Colour | Stage | Pipe |
|--------|-------|------|
| 🟠 Orange | MTE2 ND2NZ | GM→L1 DMA |
| 🟢 Green | MTE1 LOAD | L1→L0A/B |
| 🔴 Red | MMAD | TMATMUL |
| 🟡 Dark orange | FIXP | FIX_L0C_TO_DST / TPUSH drain |
| 🔵 Steel blue | SCALAR | Control flow / address |
| 🟡 Amber | TPUSH | C2V push (AIC side) |
| 🟡 Yellow | TPOP | C2V pop (AIV side) |
| 🩵 Light blue | VEC | TMULS / vector ALU |
| 🔵 Teal | TSTORE | AIV→GM output |

---

## Part 7: Reference Files

| File | Location | Purpose |
|------|----------|---------|
| Qwen3 model (no_split) | `~/pypto-lib/examples/models/qwen3/qwen3_32b_decode.py` | Base model, chunked_loop_optimizer |
| Qwen3 model (split=UP_DOWN) | `~/pypto-lib/examples/models/qwen3/qwen3_32b_decode_scope3.py` | split=UP_DOWN at lines 67, 141 |
| gemm_eltwise example | `~/pypto-lib/examples/intermediate/gemm_eltwise.py` | Minimal split=UP_DOWN example |
| incore_5 pass dumps | `~/pypto/qwen3_a5/passes_dump/` | 30 pass dumps for Qwen3 A5 |
| pypto_stack study | `~/npu_skills/seg_topics/pypto_stack_overview_qwen_study/` | Full study with 17-function map |
| Pipeline reconstruction skill | `~/npu_skills/performance/pipeline-reconstruction.md` | Interval reconstruction from sim logs |
| msprof skill | `~/npu_skills/performance/msprof-simulator-profiling.md` | msprof op simulator workflow |
| gen_pipeline_svg.py | `~/npu_skills/performance/gen_pipeline_svg.py` | trace.json → SVG |
| ST testcase reference | `tests/npu/a5/src/st/testcase/qk_attn_incore5/` | This case (TPUSH/TPOP pattern) |
| nbuf testcase | `tests/npu/a5/src/st/testcase/cube_matmul_nbuf/` | N-buffer matmul (pure AIC, no TPUSH) |

---

## Quick reference: key identifiers

```
Server:        happybot@192.168.0.106
CANN env:      /usr/local/Ascend/cann_9b2/cann-9.0.0-beta.2/set_env.sh
venv:          ~/venv/bin/activate
pypto source:  ~/pypto/python
pypto-lib:     ~/pypto-lib
PTOAS_ROOT:    /home/happybot/PTOAS-official/build/tools/ptoas
Target:        BackendType.Ascend950 (A5, dav-c310, Ascend950PR_9599)
ptoas flags:   --pto-arch=a5 --pto-level=level3 --enable-insert-sync
```
