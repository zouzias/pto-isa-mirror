# moe_segmented_ffn_top1

## Purpose

First in-tree milestone implementing the full top-1 segmented MoE FFN inside
a single auto-mode A3 kernel:

```text
per expert e, per microtile m0:
  scratch       = ReLU(packed_tokens @ w1[e])     # GEMM1 + activation
  packed_output = scratch        @ w2[e]          # GEMM2
```

The fresh implementation is a direct composition of two confirmed-built
milestones with one new fused-TSTORE assumption (documented below). It is
**not** a refactor of any deleted predecessor — the design starts from the
known-good references only.

## Where this sits in the MoE pipeline

```text
moe_top1_permute             (proven, §A13)
    ↓
moe_segmented_identity       (proven, §A15) — fake +1.0 expert compute
moe_segmented_gemm_one_layer (proven, §A16) — one expert GEMM, no activation
moe_segmented_gemm_relu      (proven, §A17) — GEMM + ReLU fused into TSTORE
moe_segmented_ffn_top1       (THIS) — GEMM1 + ReLU + GEMM2
    ↓
moe_top1_unpermute           (proven, §A14)
```

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-cube`
— same as §A16 / §A17; no mix-arch).

## What the kernel does

Single AICORE, no `block_idx` work split. Reuses §A15's nested expert /
microtile loop on host-padded segments. Each inner iteration runs two
back-to-back cube GEMMs through one set of L1 / L0A / L0B / L0C tiles:

```cpp
for (e = 0; e < kNumExperts; ++e) {
    int32_t start = expert_start[e];
    int32_t count = expert_count[e];
    GlobalDataB1 b1Global(w1 + e * (kH * kF));
    GlobalDataB2 b2Global(w2 + e * (kF * kH));

    for (m0 = 0; m0 < count; m0 += kTileM) {
        size_t row  = size_t(start) + size_t(m0);
        GlobalDataA1 a1Global(packed_tokens + row * kH);
        GlobalDataC1 c1Global(scratch       + row * kF);
        GlobalDataA2 a2Global(scratch       + row * kF);
        GlobalDataC2 c2Global(packed_output + row * kH);

        // GEMM1 — FP16 × FP16 → FP32 acc → FP16 GM (fused ReLU + downcast).
        TLOAD(a1MatTile, a1Global);
        TLOAD(b1MatTile, b1Global);
        TMOV (a1Tile,  a1MatTile);
        TMOV (b1Tile,  b1MatTile);
        TMATMUL(c1Tile, a1Tile, b1Tile);
        TSTORE<AccTile1, GlobalDataC1, AtomicType::AtomicNone,
               ReluPreMode::NormalRelu>(c1Global, c1Tile);

        // GEMM2 — FP16 × FP16 → FP32 (no activation).
        TLOAD(a2MatTile, a2Global);
        TLOAD(b2MatTile, b2Global);
        TMOV (a2Tile,  a2MatTile);
        TMOV (b2Tile,  b2MatTile);
        TMATMUL(c2Tile, a2Tile, b2Tile);
        TSTORE(c2Global, c2Tile);
    }
}
```

Auto-sync inserts the MTE2 → MTE1 → M → FIX fences within and between the
two GEMMs on the same scratch row.

## Interface

Non-template host wrapper (see `compile_error_logbook.md §E13` for why
`half` must not appear in `main.cpp`):

```cpp
extern "C" void launchMoeSegmentedFfnTop1Fp16(
    uint8_t *packed_output,
    uint8_t *packed_tokens,
    int32_t *expert_count,
    int32_t *expert_start,
    uint8_t *w1,
    uint8_t *w2,
    uint8_t *scratch,
    void    *stream);
```

The underlying templated launcher is parameterised over `<TOut, TIn,
TWeight, TScratch>` with a single explicit instantiation
`<float, half, half, half>`.

## Dtype contract

| Buffer            | Dtype     | Shape                 | Notes                                |
|-------------------|-----------|-----------------------|--------------------------------------|
| `packed_tokens`   | float16   | `[T_PADDED, kH]`      |                                      |
| `w1`              | float16   | `[kE, kH, kF]`        |                                      |
| `w2`              | float16   | `[kE, kF, kH]`        |                                      |
| `scratch`         | float16   | `[T_PADDED, kF]`      | post-ReLU; kernel writes before reads |
| `packed_output`   | float32   | `[T_PADDED, kH]`      | cube FP32 accumulator                |
| `expert_count`    | int32     | `[kE]`                | PADDED counts (multiples of kTileM)  |
| `expert_start`    | int32     | `[kE]`                | PADDED starts (prefix sum)           |

Accumulation: FP32 inside the cube AccTile for both GEMM1 and GEMM2.

**No `float*`-reinterpreted-as-`half*` anywhere.** The host allocates
`scratch` on device but never copies host bytes into it; the kernel
populates it before reading.

## Tail policy

Host-padded counts only. `expert_count[e]` is rounded up to the next
multiple of `kTileM` (= 128) by `scripts/gen_data.py`; padded rows in
`packed_tokens` are zero. Padded rows produce zero throughout the FFN
(zero × anything = 0, ReLU(0) = 0, etc.) so the kernel processes them
identically to real rows. No `SetValidRow` / `SetValidShape` /
partial-tile stores anywhere — those remain Unknown for auto mode.

If `expert_count[e]` is **not** padded, the inner `for (m0 = 0; m0 <
count; m0 += kTileM)` loop is undefined: a partial trailing tile would
read uninitialised tokens (and write past the real row range into the
adjacent expert). Do not pass unpadded counts.

## Shape for first milestone

```text
T          = 256                # number of real tokens
T_PADDED   = sum(padded counts) # depends on histogram; gen_data prints it
kH         = 64                 # hidden dim
kF         = 64                 # FFN intermediate dim
kE         = 4                  # experts
kTileM     = 128                # M dim per cube tile
GEMM1: (M, K, N) = (kTileM, kH, kF) = (128, 64, 64)  FP16 × FP16 → FP32
GEMM2: (M, K, N) = (kTileM, kF, kH) = (128, 64, 64)  FP16 × FP16 → FP32
```

`kH == kF == 64` is intentional for this first milestone — every GEMM
microtile is identical in shape, so the tile aliases are reused verbatim
from §A16 / §A17. Generalising to `kH != kF` is straightforward (the
template machinery already separates the two dimensions internally) but
would mean exercising new tile shapes; that is deferred.

A single F tile per microtile only. Multi-F-tile (e.g. F > 64 covered by
multiple N tiles) is **not** supported by this milestone; it would
require either splitting GEMM1's N or using a SplitN accumulation pattern
that has no auto-mode reference yet.

## NEW assumption introduced by this milestone

GEMM1 ends with a single TSTORE that does THREE things at once:

1. drain L0C → GM,
2. cast FP32 (accumulator) → FP16 (GM),
3. apply `max(., 0)` (ReluPreMode::NormalRelu) along the way.

The **same dtype combo + ReLU** is in `ALL_TESTCASES` for the **NZ-layout**
TSTORE — `tstore_acc2gm` `LaunchTStoreAcc2gmNz2nz<21>` instantiates
`<0, float, float, half, …, 1>` at
[tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp:627](../../../../tests/npu/a2a3/src/st/testcase/tstore_acc2gm/tstore_acc2gm_kernel.cpp#L627).

But the §A17 entry of `known_good_kernel_examples.md` flagged
the **ND-layout** form of this combination as Unknown:

> The **combination** of `ReluPreMode::NormalRelu` AND down-cast
> `AccTile<float>` → GM `half` in the same TSTORE is NOT validated.
> `tstore_acc2gm` shows the two features individually (`tilingKey=4`
> does FP32→FP16 without ReLU; `tilingKey=21` does ReLU with no dtype
> change) but not combined.

(Note: that summary actually understates the NZ coverage — `tilingKey=21`
for Nz2nz **does** combine them; only the ND/Nz2nd variant is missing.)

So this milestone's single new piece is the **ND-layout** TSTORE of FP32
Acc to FP16 GM with `ReluPreMode::NormalRelu`. If that works, the rest
of the kernel is a direct composition of §A16 (GEMM2) and §A17 (GEMM1
minus the dtype change).

## Inputs / outputs

Already covered by the dtype-contract table above. Files produced by
`scripts/gen_data.py` and consumed by `main.cpp`:

```text
./input/input_packed_tokens.bin       (T_PADDED * kH float16)
./input/input_expert_count.bin        (kE          int32 )  PADDED counts
./input/input_expert_start.bin        (kE          int32 )  PADDED starts
./input/input_w1.bin                  (kE * kH * kF float16)
./input/input_w2.bin                  (kE * kF * kH float16)
./output/golden_packed_output.bin     (T_PADDED * kH float32; full FFN)
./output/golden_scratch.bin           (T_PADDED * kF float16; post-ReLU; debug)
./output/golden_gemm1_output.bin      (T_PADDED * kF float32; PRE-ReLU; debug)
./output/t_padded.txt                 (single int line)
./output/expert_count_real.bin        (kE int32; debug only)
```

## How to generate data

```bash
python ./scripts/gen_data.py
```

`gen_data.py` mirrors the kernel exactly:

- builds `packed_tokens` from a stable `argsort(expert_id)` permutation;
- runs per-expert FP32 GEMM1 over FP16 inputs;
- applies `np.maximum(., 0)` and casts to FP16 (matches the kernel's fused
  TSTORE);
- runs per-expert FP32 GEMM2 over the FP16 scratch and FP16 w2;
- writes the FP32 result.

Token / weight values are random integers in `[-4, 4]`. With `kH = 64` the
pre-ReLU GEMM1 magnitude is bounded by `64 × 4 × 4 = 1024`, inside the
FP16 exact-integer regime (≤ 2048), so scratch is bit-exact even though
it is FP16. Both clipped and pass-through values appear (the printout
shows the histogram). For larger `kH` this exact-integer property may
not hold and the comparison would need a tolerance.

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

Reports first-mismatch location with expert / segment / tile-index
breakdown plus stage-attribution hints (was the pre-ReLU GEMM1 value at
this position negative? was scratch zero? does the magnitude of the
device output match scratch · w2 or GEMM1 · w2?).

## Auto-mode constraints honored

Inherits the full §A16 / §A17 list; the highlights:

- Single AICORE.
- Static tile shapes; no `SetValidRow` / `SetValidShape` / partial-tile
  stores.
- All ten tiles (`Mat ×2`, `Left`, `Right`, `Acc` for each of GEMM1 and
  GEMM2) declared once outside both loops; auto allocator pins every
  address; auto-sync handles MTE2 / MTE1 / M / FIX fences both within
  and across the two GEMMs.
- No `TASSIGN` literal-address aliasing; no `#ifndef __PTO_AUTO__`
  manual-sync; no `Tile::data()` from kernel; no `*_IMPL` calls; no
  raw CCE intrinsics; no `Event<>`; no `TPipe` / `TPUSH` / `TPOP`; no
  double buffering; no A5-only instructions.
- Only `ReluPreMode::NormalRelu`; no GELU / SiLU / LeakyReLU.

## Deliverable summary (what the prompt asked for)

1. **Which known-good milestones were copied / adapted.**
   - §A15 moe_segmented_identity — nested (expert, microtile) loop
     shape, scalar GM metadata reads, host-padded segment layout.
   - §A16 moe_segmented_gemm_one_layer — GEMM2 (TLOAD / TMOV / TMATMUL
     / TSTORE) and the FP16 × FP16 → FP32 cube combo, plus the
     `uint8_t*` host boundary and non-template launcher wrapper.
   - §A17 moe_segmented_gemm_relu — GEMM1 + fused ReLU in TSTORE; only
     the destination GM dtype is changed from float to half.

2. **Assumed dtype contract.**
   `packed_tokens, w1, w2, scratch` = FP16; `packed_output` = FP32;
   accumulation = FP32 (FIX-pipe Acc tiles). See the table above.

3. **`expert_count` padded or real.**
   PADDED. Host-padded to a multiple of `kTileM = 128`. Padded rows are
   zero in `packed_tokens` and remain zero through the full FFN.

4. **Tails supported or intentionally excluded.**
   Intentionally excluded. The kernel relies on host padding; an
   unpadded count would invoke undefined behaviour on the trailing
   partial tile. No `SetValidRow` support is attempted.

5. **GEMM1 → ReLU → scratch → GEMM2: known-good or new assumption?**
   Mostly known-good. **One new piece**: fused FP32 Acc → FP16 GM +
   `ReluPreMode::NormalRelu` in **ND-layout** TSTORE. The same combo is
   validated in NZ layout by `tstore_acc2gm Nz2nz tilingKey=21` (line
   627). ReLU-in-TSTORE alone (without downcast) is validated by §A17.
   FP32 Acc → FP16 GM alone (without ReLU) is validated by
   `tstore_acc2gm Nz2nd tilingKey=4`. The fused-three-way variant in
   ND layout is the one new assumption.

6. **How to build / run / compare.** See the sections above.

## What this milestone does NOT prove

Even if it passes:

- Activations other than `NormalRelu`. GELU / SiLU / LeakyReLU remain
  unvalidated and are explicitly out of scope.
- Non-equal `kH` and `kF`.
- Multi-F or multi-K tile (Split-K / Split-N) accumulation.
- Bias path on either GEMM.
- TF32, INT8, BF16, MX, FP8 paths.
- Multi-core `block_idx` partitioning.
- `topK > 1` (weighted combine).
- Dynamic-tail / `SetValidRow` / partial-tile stores.
- The full router (no GEMM router here; expert IDs are precomputed by
  the data generator and turned into the standard packed layout).
- Backward.
- Performance.

## Prepared fallback (do not apply until first failure is known)

If the fused FP32 Acc → FP16 GM + ReLU TSTORE in ND layout does not
behave correctly, split the kernel into three confirmed-built stages:

1. **§A17 gemm_relu** — write `scratch_fp32` (T_PADDED × kF float32)
   with ReLU fused into the TSTORE. No downcast.
2. A separate **FP32 → FP16 cast pass** (a small vec-arch kernel or a
   host-side cast for the prototype). FP16 input is required by GEMM2.
3. **§A16 gemm_one_layer** — read the FP16 scratch, multiply by `w2`,
   write FP32 `packed_output`.

Stage 2 has no in-tree confirmed-built auto-mode reference; if it is
needed, the cleanest path is a host-side cast for the first prototype
(slower but trivially correct), then a separate vec-arch milestone for
the device-side cast.

## Failure protocol

If it fails, report verbatim:

1. The first meaningful compiler / runtime error.
2. Whether the failure is:
   - **compile**: TSTORE template arg list mismatch on the GEMM1 TSTORE
     (the new piece);
   - **link**: missing `launchMoeSegmentedFfnTop1Fp16` symbol;
   - **runtime**: device output is wrong → look at
     `compare_outputs.py`'s stage-attribution hints.

After it passes, capture this milestone as a new entry in
`docs_for_ai/known_good_kernel_examples.md` (next §A slot) and move the
"FP32 Acc → FP16 GM + ReLU in ND-layout TSTORE" assumption from
`docs_for_ai/assumptions_to_verify.md` into its "Resolved by experiments
/ build runs" section.
