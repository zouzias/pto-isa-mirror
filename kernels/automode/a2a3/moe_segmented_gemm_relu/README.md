# moe_segmented_gemm_relu

## Purpose

Add ReLU activation after the per-expert GEMM proven by §A16
[moe_segmented_gemm_one_layer](../moe_segmented_gemm_one_layer/).

The activation is **fused into the L0C → GM store** via the
`ReluPreMode::NormalRelu` overload of the public `TSTORE` wrapper. There
is **no** cube → vec handoff, **no** separate vector kernel, **no** mix-arch
build. The kernel is otherwise byte-for-byte the §A16 cube kernel — only the
final `TSTORE` template arg list changes:

```diff
- TSTORE(cGlobal, cTile);
+ TSTORE<AccTile, GlobalDataC, AtomicType::AtomicNone, ReluPreMode::NormalRelu>(
+       cGlobal, cTile);
```

This is **not** the full FFN yet — only GEMM1 + activation. GEMM2 lands in a
later milestone.

## Activation reference

- Public wrapper declaration:
  [include/pto/common/pto_instr.hpp:251-258](../../../../include/pto/common/pto_instr.hpp)
  ```cpp
  template <typename TileData, typename GlobalData,
            AtomicType atomicType = AtomicType::AtomicNone,
            ReluPreMode reluPreMode, typename... WaitEvents>
  PTO_INST RecordEvent TSTORE(GlobalData &dst, TileData &src, WaitEvents &...events);
  ```
- Enum:
  [include/pto/common/type.hpp:255-259](../../../../include/pto/common/type.hpp)
  ```cpp
  enum class ReluPreMode : uint8_t { NoRelu = 0, NormalRelu = 1 };
  ```
- Confirmed-built A3 cube reference using this exact form:
  [tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp:88-93](../../../../tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp) —
  `TSTORE<AccTile, GlobalDataOut, atomicTypeEnum, ReluPreMode::NormalRelu>(dstGlobal, cTile);`.
  `tstore_acc2gm` is in `ALL_TESTCASES` and builds under `pto_cube_st`
  (same cube-arch + auto-mode recipe as this project).

## Where this sits in the MoE pipeline

```text
moe_top1_permute             (proven, §A13)
    ↓
moe_segmented_identity       (proven, §A15) — fake +1.0 expert compute
moe_segmented_gemm_one_layer (proven, §A16) — one expert GEMM, no activation
moe_segmented_gemm_relu      (THIS) — GEMM + ReLU (fused via TSTORE/FIX pipe)
    ↓ (eventually)
moe_segmented_ffn            (future) — GEMM1 + activation + GEMM2
    ↓
moe_top1_unpermute           (proven, §A14)
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with
`--cce-aicore-arch=dav-c220-cube` — same as §A16; no mix arch).

## Why fused-in-TSTORE over a separate vector kernel

Three options were considered before coding (per the prompt):

- **Strategy A — public `TRELU` wrapper.** Exists in
  [include/pto/common/pto_instr.hpp:1502-1506](../../../../include/pto/common/pto_instr.hpp);
  used by `tests/npu/a2a3/src/st/testcase/trelu/trelu_kernel.cpp`. But it
  operates on `Tile<TileType::Vec, T, ...>` (a UB tile), not directly on
  L0C accumulator output. Routing through it would require:
  1. moving the cube `AccTile` (L0C) to UB (via `TFIX` or a `TSTORE` to a
     scratch GM area then `TLOAD` back into a Vec tile), or
  2. splitting the kernel and switching to mix-arch
     (`--cce-aicore-arch=dav-c220`). Neither has a confirmed auto-mode
     reference in this stack.
- **Strategy B — `TMAXS(dst, src, 0.0f)`.** Same problem as A: operates
  on Vec tiles.
- **Strategy C — separate `moe_segmented_relu` vector kernel** reading
  the §A16 FP32 GEMM output and writing FP32 ReLU output. Would work but
  doubles the GM-roundtrip and the kernel count.
- **Strategy D (chosen) — fused `TSTORE<..., ReluPreMode::NormalRelu>`**.
  Confirmed-built in `tstore_acc2gm`. Same cube-arch build. Single kernel.
  No extra tile, no extra GM roundtrip. The ReLU happens during the FIX
  pipe's L0C → GM drain.

## What the kernel does

Identical to §A16 except for the final `TSTORE`:

```cpp
for (e = 0; e < kNumExperts; ++e) {
    int32_t start = expert_start[e];
    int32_t count = expert_count[e];
    GlobalDataB bGlobal(expert_weight + e * (kH * kO));

    for (m0 = 0; m0 < count; m0 += kTileM) {
        size_t row = size_t(start) + size_t(m0);
        GlobalDataA aGlobal(packed_tokens + row * kH);
        GlobalDataC cGlobal(packed_output + row * kO);

        TLOAD (aMatTile, aGlobal);
        TLOAD (bMatTile, bGlobal);
        TMOV  (aTile, aMatTile);
        TMOV  (bTile, bMatTile);
        TMATMUL(cTile, aTile, bTile);
        TSTORE<AccTile, GlobalDataC, AtomicType::AtomicNone,
               ReluPreMode::NormalRelu>(cGlobal, cTile);
    }
}
```

## Inputs / outputs

Same shapes and dtypes as §A16:

- `packed_tokens [T_PADDED, H]` — `float16`
- `expert_count [E]` — `int32` (PADDED)
- `expert_start [E]` — `int32` (PADDED)
- `expert_weight [E, H, O]` — `float16`
- `packed_output [T_PADDED, O]` — `float32` (POST-ReLU; cube FP32 accumulator
  with FIX-pipe `max(., 0)` applied)

## Shape

`T = 256, H = K = 64, O = N = 64, E = 4, kTileM = 128`, FP16 × FP16 → FP32.

The Python data generator widens the input range to `[-4, 4]` so the
pre-ReLU GEMM output contains both negative values (clipped to zero) and
positive values (passed through). Without negatives ReLU degenerates to
identity and the test cannot distinguish "ReLU fired" from "ReLU was a
no-op." `gen_data.py` prints the clip-vs-pass-through statistics so this
is observable at data-generation time.

## How to generate data

```bash
python ./scripts/gen_data.py
```

Writes:

```text
./input/input_packed_tokens.bin       (T_PADDED * H float16)
./input/input_expert_count.bin        (E         int32)
./input/input_expert_start.bin        (E         int32)
./input/input_expert_weight.bin       (E * H * O float16)
./output/golden_packed_output.bin     (T_PADDED * O float32; POST-ReLU)
./output/golden_gemm_output.bin       (T_PADDED * O float32; PRE-ReLU; debug)
./output/t_padded.txt                 (int)
./output/expert_count_real.bin        (E int32; debug only)
```

## How to build / run

```bash
bash run.sh -r npu -v Ascend910B1
```

On success:

```text
test data success
test success
```

## How to compare against the Python reference

```bash
python ./scripts/compare_outputs.py
```

In addition to the §A16 first-mismatch breakdown
(`{flat, row, col, expert, segment, tile_m0, real-vs-pad}`), this script
also reports:

- the **PRE-ReLU** value at the first mismatch position (and whether the
  golden value was clipped to zero by ReLU);
- a failure-mode breakdown:
  - mismatches on positions where the golden was **CLIPPED** by ReLU →
    "device skipped ReLU";
  - mismatches on positions where the golden was **PASS-THROUGH** (>= 0) →
    "GEMM itself is wrong, not ReLU".

## Auto-mode constraints honored

All §A16 constraints, plus:

- ReLU is invoked through the public `TSTORE<..., ReluPreMode>` template
  argument — **not** through any `*_IMPL` call or kernel-level
  `pipe_barrier` / `set_flag`. The activation lives entirely inside the
  FIX-pipe `TSTORE` lowering.
- No `TRELU` / `TMAXS` invocation in the kernel; no Vec tile introduced;
  no L0C → UB intermediate.

## Patterns reused from earlier confirmed kernels

- scalar GM read of `int32_t` metadata                              — §11.4 / §11.6
- runtime scalar GM row offset for `GlobalTensor`                   — §11.4 / §11.5 / §11.6
- nested (expert, microtile) loop                                   — §11.6
- host-padded expert segment layout                                 — §11.6
- cube `TMATMUL` inside the segmented loop, FP16 → FP32 combo       — §11.7
- non-template `…Fp16` host wrapper hiding `half` from main.cpp     — §11.7

## NEW: what this kernel is the first to validate

If it passes, this becomes Resolved-by-experiment for the tested shape:

- **Fused ReLU in the L0C → GM TSTORE** via
  `TSTORE<TileData, GlobalData, AtomicType::AtomicNone,
  ReluPreMode::NormalRelu>(dst, src)` inside the per-expert segmented
  loop. No cube → vec handoff; no separate vector kernel.
- **FIX-pipe activation fusion** is auto-mode-safe with the canonical
  FP16 × FP16 → FP32 cube combo at `(M, K, N) = (128, 64, 64)`.

## What this prototype does NOT prove

Even if it passes:

- Full MoE FFN (GEMM1 + activation + GEMM2).
- Standalone `TRELU` / `TMAXS` on Vec tiles in auto-mode kernel code
  (the standalone vec-arch path is unexercised by this milestone — the
  ReLU happens entirely in the FIX pipe here).
- Cube → Vec handoff via `TFIX` / explicit tile-move.
- Non-ReLU activations (GELU, SiLU, swish, etc.). Only `NormalRelu` mode
  is tested. Other modes (`NoRelu` is trivial; future enums) untested.
- Bias path; SplitK; TF32; INT8; BF16.
- Dynamic tail handling; multi-core; `topK > 1`; weighted combine; backward.
- Performance.

## Failure protocol

If it fails, report:

1. The first meaningful compiler / runtime error verbatim.
2. Whether the failure is:
   - **compile**: `TSTORE` template arg list mismatch (e.g., wrong arg
     order or wrong default-arg interaction);
   - **link**: missing `launchMoeSegmentedGemmReluFp16` symbol;
   - **runtime**: the device output is wrong → look at
     `compare_outputs.py`'s "failure-mode breakdown":
     - all mismatches on CLIPPED positions → device didn't apply ReLU,
       falling back to the same shape as §A16's GEMM output. Suggests
       the `ReluPreMode::NormalRelu` template arg didn't propagate.
     - all mismatches on PASS-THROUGH positions → the GEMM itself
       regressed (would mean the §A16 baseline also broke);
     - mismatches on both → likely a tile-layout / TSTORE-overload
       issue.

**Prepared fallback (do NOT apply until first failure is known)**: split
into a separate `moe_segmented_relu` vector kernel that consumes the FP32
GEMM output from §A16 (Strategy C). This is more code but uses only
patterns already proven (segmented vec loop, public `TRELU` wrapper).

## After it passes

The next milestone is **moe_segmented_ffn** — GEMM1 + activation + GEMM2
per expert microtile. Do not implement until this kernel passes.
