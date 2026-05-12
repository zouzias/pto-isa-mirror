# moe_segmented_ffn_top1

## Purpose

First end-to-end per-expert FFN on A3 auto mode:

```text
GEMM1 → ReLU → GEMM2
```

inside the proven top-1 segmented expert loop (§A13 / §A14 / §A15 / §A16 /
§A17). Reuses the host-padded segment layout from §A15; reuses the cube
TMATMUL skeleton from §A16; reuses the FIX-pipe `ReluPreMode::NormalRelu`
TSTORE from §A17. Adds **one** new piece on top: a TSTORE that **combines**
the FIX-pipe ReLU AND the FP32 accumulator → FP16 GM down-cast in a single
call, so the hidden state lands in GM as FP16 and can be re-loaded directly
into the next GEMM's Mat tile.

Still **not** router / top-K / capacity / multi-core / backward. Assumes
the packed-token layout and expert metadata already exist (this is what
the proven §A13 forward permute produces).

## Where this sits in the MoE pipeline

```text
moe_top1_permute             (proven, §A13)
    ↓ packed_tokens, expert_count, expert_start, token_to_packed
moe_segmented_ffn_top1       (THIS) — GEMM1 → ReLU → GEMM2
    ↓ packed_output
moe_top1_unpermute           (proven, §A14)
    ↓ output (original token order)
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-cube`,
auto mode). Single AICORE; no mix arch; no second kernel.

## Implementation choice — single cube kernel

Considered before coding:

- **Option A (chosen)**: one cube kernel, two TMATMULs back-to-back per
  inner microtile, hidden state on GM scratch as FP16, written via one
  TSTORE that fuses ReLU + FP32→FP16 down-cast in the FIX pipe. The cube
  tiles are reused across both GEMMs (with our shape `kH = kF = kO = 64`
  every cube tile in GEMM1 has the same dimensions as in GEMM2). One GM
  roundtrip per inner tile for the hidden state; scratch buffer is fixed
  `TILE_M * F * sizeof(half)` = 16 KB, allocated once on the device.
- Option B: two separate kernels in the same project, hidden state on a
  full-size `T_PADDED * F` FP16 GM intermediate. More mechanical but uses
  more GM (the full hidden state at once rather than one tile's worth).
- Option C: GEMM1 stores FP32 hidden with ReLU (the proven §A17 form),
  then a separate vec kernel casts FP32→FP16. Requires mix arch and a
  second kernel; rejected.

Option A is the simplest single-kernel design that reuses only patterns
from confirmed-built A3 references. Its **only** new piece is the combined
TSTORE template-arg form
`<AccTile, GlobalDataHiddenFp16, AtomicType::AtomicNone, ReluPreMode::NormalRelu>`
applied to an `AccTile<float>` with a `GlobalTensor<half, …>` destination —
see "Assumptions" below.

## What the kernel does (per inner microtile)

```cpp
// GEMM1: A1 (FP16 packed_tokens slice) × W1[e] (FP16) → C1 (FP32 in L0C)
TLOAD(aMatTile, a1Global);          // GM → L1
TLOAD(bMatTile, b1Global);          // GM → L1
TMOV (aTile, aMatTile);             // L1 → L0A
TMOV (bTile, bMatTile);             // L1 → L0B
TMATMUL(cTile, aTile, bTile);       // cube
// FIX-pipe TSTORE: fuses ReLU + FP32 → FP16 down-cast.
TSTORE<AccTile, GlobalDataHiddenFp16,
       AtomicType::AtomicNone, ReluPreMode::NormalRelu>(hGlobal, cTile);

// GEMM2: A2 = hidden_scratch (FP16) × W2[e] (FP16) → C2 (FP32 in L0C)
TLOAD(aMatTile, hGlobal);           // GM → L1 (REUSE same Mat tile)
TLOAD(bMatTile, b2Global);          // GM → L1 (REUSE same Mat tile)
TMOV (aTile, aMatTile);             // L1 → L0A (REUSE)
TMOV (bTile, bMatTile);             // L1 → L0B (REUSE)
TMATMUL(cTile, aTile, bTile);       // cube (REUSE Acc tile, overwrites C1)
TSTORE(c2Global, cTile);            // plain FP32 store
```

All five cube tiles (`Mat ×2`, `Left`, `Right`, `Acc`) are declared once
outside both loops; the auto allocator pins their L1/L0 addresses and the
auto-sync pass inserts MTE2 → MTE1 → M → FIX fences twice per inner iter.

## Inputs / outputs

**Inputs (GM)**
- `packed_tokens [T_PADDED, H]` — `float16`
- `expert_count [E]` — `int32` (PADDED)
- `expert_start [E]` — `int32` (PADDED)
- `w1 [E, H, F]` — `float16`
- `w2 [E, F, O]` — `float16`

**Output (GM)**
- `packed_output [T_PADDED, O]` — `float32` (final FFN output)

**Device-only scratch (allocated by main.cpp, reused by the kernel)**
- `hidden_scratch [TILE_M, F]` — `float16`, sized `TILE_M * F * sizeof(half)`
  = 16 KB; lifetime is the kernel invocation only. Contents on entry are
  don't-care.

## Shape

```text
T = 256 real tokens
TILE_M = 128
H = F = O = 64
num_experts = 4
single AICORE
host-padded expert segments
FP16 inputs/weights
FP32 final output
```

With `H = F = O = 64`, the cube tile dimensions are identical between
GEMM1 and GEMM2, so one set of tile aliases covers both. If you pick
different shapes later you may need separate `TileMatA / TileMatB / Left /
Right / Acc` aliases for the second GEMM.

## Tail policy

Same as §A15 / §A16 / §A17: host pads each expert's count to a multiple
of `TILE_M`. Padded rows in `packed_tokens` are zero, so GEMM1 produces
zero, ReLU keeps it zero, GEMM2 produces zero. Padded rows therefore
appear as zeros in the final golden — comparable directly to the device
output. No `SetValidRow`, no partial-tile stores.

## How to generate data

```bash
python ./scripts/gen_data.py
```

Writes:

```text
./input/input_packed_tokens.bin       (T_PADDED * H float16)
./input/input_expert_count.bin        (E         int32)
./input/input_expert_start.bin        (E         int32)
./input/input_w1.bin                  (E * H * F float16)
./input/input_w2.bin                  (E * F * O float16)
./output/golden_packed_output.bin     (T_PADDED * O float32; final FFN out)
./output/golden_hidden_relu.bin       (T_PADDED * F float32; post-ReLU; debug)
./output/t_padded.txt                 (int)
./output/expert_count_real.bin        (E int32; debug only)
```

Inputs are integers in `[-3, 4]` cast to FP16 (exact). The Python golden
also down-casts the post-ReLU hidden state to FP16 before GEMM2 — this
matches the device kernel's behaviour exactly. With this distribution the
GEMM products and accumulator partial sums fit in FP32 and FP16 ranges
without rounding, so the final output should be bit-exact.

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

On success:

```text
test data success
test success
```

## DEBUG mode (isolates Assumption A.combined)

If the main FFN run fails (e.g., final output is all zero), build emits a
**second** executable, `./moe_segmented_ffn_top1_debug`, that runs only
GEMM1 + ReLU + FP32→FP16 down-cast and writes the post-ReLU hidden state
to a full-size FP16 GM buffer the host reads back. This isolates whether
the **combined-mode TSTORE** form
`TSTORE<AccTile<float>, GlobalTensor<half, ...>, AtomicNone, NormalRelu>`
(Assumption A.combined) writes the correct data — separately from the
GEMM2 / tile-reuse question (Assumption A.reuse) in the main FFN kernel.

Run order after a normal `bash run.sh` build:

```bash
cd build
./moe_segmented_ffn_top1_debug                         # writes ./output/debug_hidden_fp16.bin
python ../scripts/compare_hidden_debug.py              # compares vs ./output/golden_hidden_fp16.bin
```

Or all in one shot (the `run.sh` already runs `gen_data.py` and compiles
both binaries — the debug binary is linked against the same kernel shared
library):

```bash
bash run.sh -r npu -v Ascend910B1
cd build
./moe_segmented_ffn_top1_debug
python ../scripts/compare_hidden_debug.py
```

Interpretation:

- **`compare_hidden_debug.py` PASSes**: the combined-mode TSTORE writes the
  correct FP16 hidden state. A.combined is OK at this shape. The main FFN
  failure is on the **GEMM2 / tile-reuse** side (Assumption A.reuse) — next
  step is to declare separate Mat/Left/Right/Acc tiles for GEMM2 in
  `moe_segmented_ffn_top1_kernel.cpp` (the prepared Fallback F2 in the
  failure protocol).
- **`compare_hidden_debug.py` FAILs** (especially if the device hidden is
  all zeros): A.combined is broken. Switch the main kernel to **Fallback
  F1** (store FP32 hidden with ReLU using the proven §A17 form, then a
  separate cast kernel) or **F2** (FP32→FP16 no-ReLU store + separate vec
  ReLU). Both are documented in the kernel header and the failure-protocol
  section below.

The debug binary uses **exactly the same** GEMM1 body and TSTORE template
arguments as the main FFN kernel — the only difference is that the TSTORE
destination is a per-tile offset into a full-size FP16 buffer (so the host
can read back all hidden values at once) instead of a fixed reused
scratch. This means a PASS on the debug binary directly validates A.combined
at the per-iter shape used by the main kernel.

The standard `compare_outputs.py` (final-output diff) **also** picks up
`debug_hidden_fp16.bin` if present and prints a one-line summary at the
end so you can confirm both verdicts (hidden state OK / final output OK)
from a single command.

## How to compare against the Python reference

```bash
python ./scripts/compare_outputs.py
```

In addition to the standard first-mismatch breakdown
(`{flat, row, col, expert, segment, tile_m0, real-vs-pad}`), this script
loads `./output/golden_hidden_relu.bin` and reports the post-ReLU hidden
state for the first mismatching row — useful for distinguishing:

- **GEMM1 wrong** — the device hidden state differs from the golden;
- **ReLU wrong** — hidden state has wrong sign pattern;
- **FP32→FP16 cast wrong** — small-magnitude residuals at every position;
- **GEMM2 wrong** — final output differs even though the golden hidden
  state is the same.

## Auto-mode constraints honored

All §A17 constraints, plus:

- Two cube TMATMULs in a single auto-mode kernel; no manual sync between
  them; the auto-sync pass handles both fence chains.
- The intermediate state goes through a **single** device-resident
  scratch buffer, not multiple kernel launches.
- No `_IMPL` calls; no `TASSIGN` literal addresses; no `Tile::data()`;
  no raw CCE intrinsics; no `Event<>`; no `TPipe` / `TPUSH` / `TPOP`;
  no double buffering; no `SetValidRow` / partial-tile stores; no
  A5-only instructions.
- Host boundary: `uint8_t*` for typed buffers + `int32_t*` for metadata
  + non-template `…Fp16` wrapper at the kernel TU boundary (the §E13
  pattern — main.cpp never names `half`).

## Patterns reused from earlier confirmed kernels

- scalar GM read of `int32_t` metadata                              — §11.4 / §11.6 / §11.7
- runtime scalar GM row offset for `GlobalTensor`                   — §11.4 / §11.5 / §11.6 / §11.7
- nested (expert, microtile) loop                                   — §11.6 / §11.7
- host-padded expert segment layout                                 — §11.6 / §11.7
- cube `TMATMUL` inside the segmented loop, FP16 → FP32 combo       — §11.7
- FIX-pipe ReLU via `TSTORE<…, ReluPreMode::NormalRelu>`            — §11.8
- non-template `…Fp16` host wrapper hiding `half` from main.cpp     — §11.7

## NEW: what this kernel is the first to validate

If it passes, these become Resolved-by-experiment for the tested shape:

- **A.combined** — `TSTORE<AccTile<float>, GlobalTensor<half, …>,
  AtomicType::AtomicNone, ReluPreMode::NormalRelu>(dst_fp16, src_fp32_acc)`
  in a single call simultaneously applies ReLU **and** FP32 → FP16
  down-cast in the FIX pipe. The two features are individually proven
  in `tstore_acc2gm` (`tilingKey=4` FP32→FP16 no ReLU; `tilingKey=21`
  ReLU no cast); their combination has no in-tree precedent.
- **A.reuse** — Reusing the same Mat / Left / Right / Acc tiles across
  two back-to-back TMATMUL calls in the same inner iter (with a
  TSTORE-then-TLOAD chain in between) is auto-sync-safe — the
  allocator does not need duplicate tiles for the second GEMM.
- **Two cube GEMMs in a single auto-mode kernel** with one FIX-pipe
  TSTORE bridging them.

## What this prototype does NOT prove

Even if it passes:

- Top-K (`topK > 1`); weighted combine; router / argmax.
- Capacity / drop policy; fallback expert.
- Multi-core (`block_idx`) work split.
- Dynamic tail handling with `validM` / `SetValidRow` / `SetValidShape`;
  partial-tile stores.
- Activation other than `NormalRelu` (GELU, SiLU, etc.).
- Bias path (`TMATMUL_BIAS`).
- `bf16` / `int8` / `fp32`-input GEMMs; non-`(128, 64, 64)` shapes.
- Backward pass; performance characterisation.
- The case where `kH != kF != kO` and the two GEMMs need DIFFERENT cube
  tile aliases — this kernel deliberately picks the simplest shape that
  lets one tile set cover both GEMMs.

## Failure protocol

If it fails, report:

1. The first meaningful compiler / runtime error verbatim.
2. Whether the failure is in:
   - **compile**: TSTORE template arg combination (likely Assumption
     A.combined failing — the FIX-pipe lowering does not support
     `Acc<float> -> dst<half>` together with `NormalRelu`);
   - **link**: missing `launchMoeSegmentedFfnTop1Fp16` symbol;
   - **runtime, GEMM1 wrong**: hidden state in `compare_outputs.py`'s
     dump differs from golden — suggests the new TSTORE combo wrote the
     wrong data;
   - **runtime, ReLU wrong**: hidden state has negatives where golden has
     zeros (or vice versa) — A.combined dropped the ReLU half of the
     fusion;
   - **runtime, GEMM2 wrong but GEMM1 correct**: tile-reuse race
     (Assumption A.reuse) — the second TMATMUL read stale data from the
     reused Mat / Left / Right tiles before the first GEMM's pipeline
     drained;
   - **host harness**: scratch buffer not allocated, wrong byte count,
     etc.

**Prepared fallbacks (do NOT apply until first failure is known)**:

- **F1**: split the TSTORE into a two-step path — `TSTORE<AccTile,
  GlobalDataHiddenFp32, AtomicType::AtomicNone, ReluPreMode::NormalRelu>`
  to a `T_PADDED * F` FP32 scratch; then a separate vector kernel (or a
  separate cube kernel that does a `TCVT`-equivalent) casts FP32→FP16
  before GEMM2. Requires mix arch OR a second kernel; documented but
  not applied.
- **F2**: declare separate Mat / Left / Right / Acc tiles for GEMM2 if
  A.reuse turns out to be the problem. Roughly doubles the L0/L1
  footprint; still within budget.

Do **not** silently fall back. Report which assumption failed first.

## After it passes

Future milestones (NOT to implement until the user asks):

- `moe_router_top1` — router GEMM + argmax produces `expert_id`,
  upstream of `moe_top1_permute`.
- `moe_full_topk_1_end_to_end` — chain `router` → `permute` → `ffn` →
  `unpermute` in one orchestrator main.cpp, single AICORE.
- `moe_topk_gt_1` — `topK > 1` with weighted combine.
- `moe_multi_core` — `block_idx` partitioning with per-core histogram.
- `moe_segmented_ffn_dynamic_tail` — retire the host-padding tail
  policy via `SetValidRow` once it is validated.
