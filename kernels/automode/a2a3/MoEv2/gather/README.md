# gather

Auto-mode A3 prototype. Unpack and accumulate per-expert FFN outputs back into per-token rows, with **softmax routing weights** when `kTopK > 1`. Generic over `kTopK ∈ {1, 2, 4, 8, 16}`.

## What it does

```
weights = softmax(outVal, axis=1)       # only computed when kTopK > 1
for r in [0, kT·kTopK):
    t = A_id[r]
    k = rank_id[r]
    if kTopK == 1:
        C[t] = B[r]                     # row reorder only
    else:
        C[t] += weights[t, k] * B[r]    # C is zero-initialized before launch
```

For `kTopK == 1` the softmax is degenerate (single-element softmax = 1.0), and each token appears exactly once in `A_id`. The kernel takes a **fast path** that skips pass-1 softmax, skips `TMULS`, and directly stores each packed row into its original token row.

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `B` (input)         | `(kT·kTopK + 64, kH)` | fp32 | first `kT·kTopK` rows consulted |
| `A_id` (input)      | `(kT·kTopK + 64)`     | int32 | first `kT·kTopK` consulted |
| `rank_id` (input)   | `(kT·kTopK + 64)`     | int32 | only used when `kTopK > 1` |
| `outVal` (input)    | `(kT, kPadded)`       | fp32 | cols `kTopK..kPadded-1` host-padded with `-1e30` |
| `weights_scratch` (GM scratch) | `(kT, kPadded)` | fp32 | written by pass 1, read by pass 2 |
| `C` (output)        | `(kT, kH)`            | fp32 | zero-initialized by host before launch; only required for `kTopK > 1` |

`kPadded = max(8, kTopK)` — softmax tile column padding for the 32-byte UB alignment requirement on fp32 (`Cols * 4 % 32 == 0` ⇒ `Cols % 8 == 0`).

## Softmax composition (pass 1, only when `kTopK > 1`)

Pure tile-op recipe, lifted from the manual-mode FA softmax macro at [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp:54-60](../../../../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp#L54-L60):

```cpp
TLOAD          valTile     <- outVal              // (kT, kPadded)
TROWMAX        maxTile     <- valTile             // (kT, 8) per-row max
TROWEXPANDSUB  tmpTile     <- valTile - maxTile   // broadcast subtract
TEXP           expTile     <- exp(tmpTile)
TROWSUM        sumTile     <- expTile             // (kT, 8) per-row sum
TROWEXPANDDIV  weightTile  <- expTile / sumTile   // softmax
TSTORE         weights_scratch <- weightTile
```

No libc math (no `expf`), no scalar UB reads, no manual `pipe_barrier(PIPE_V)` — trusts auto-mode RAW dependency analysis through the five-instruction chain.

### Why outVal columns `kTopK..kPadded-1` need `-1e30` padding

`TROWMAX`, `TROWEXPANDSUB`, `TEXP`, `TROWSUM` all read the full tile width. For `kTopK ∈ {1, 2, 4}` (where `kPadded = 8 > kTopK`), the unused columns must not corrupt the softmax. Filling them with `-1e30` makes the pipeline neutralize them automatically:

```
-1e30 - real_max  ≈ -inf
exp(-inf)         =  0
0 contributes nothing to the row sum
0 / real_sum      =  0   →  padding cols of weights_scratch end up at 0.0
```

`rank_id` from scatter only takes values in `[0, kTopK)`, so we never look up a padding column in pass 2.

## Target platform

A3 / Ascend 910B1. Vec target (`--cce-aicore-arch=dav-c220-vec`).

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`).
- Static softmax tiles (pass 1) and static row tiles (pass 2), all declared once outside any loop; auto allocator pins their UB addresses.
- `pipe_barrier(PIPE_ALL)` at the start of each pass-2 row iteration (hardware-confirmed cross-iter auto-sync guard).
- No `TASSIGN` aliasing, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe`/`TPUSH`/`TPOP`, no double buffering.

## Initialization

The kernel does **not** pre-zero `C`. The host driver calls `aclrtMemset(cDev, ..., 0x00)` before launch (0x00 bytes in fp32 = `+0.0f`). That memset is only required for `kTopK > 1`, where the weighted path reads `C[t]`, accumulates into it, and stores it back. For `kTopK == 1`, the fast path is a direct `TLOAD(B[r]) → TSTORE(C[A_id[r]])` row reorder.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

`scripts/gen_data.py` writes a self-contained `(B, A_id, rank_id, outVal, C_golden)` set — no scatter / expert_ffn / moe_topk_padded dependency. The C++ driver memsets `C` to zero, fires the kernel, and compares against `golden_C.bin` at `1e-3` abs tolerance (softmax adds `TEXP` / divide rounding on top of fp32 accumulation).

## Sweeping kTopK / kT / kH

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kTopK`
- `main.cpp`                : `constexpr int kT/kH/kTopK`
- `gather_kernel.cpp`       : `namespace gather_cfg { constexpr unsigned ... }`

`kPadded` is derived from `kTopK` in all three files — it updates automatically when `kTopK` is patched (the sweep.sh sed patterns leave `kPadded = max(8, kTopK)` alone).

## UB budget (pass 1 softmax tiles, when `kTopK > 1`)

| `kT` | `kTopK` | `kPadded` | Sum of six static tiles (worst case) |
|---|---|---|---|
| 256 | 1   | 8  | unused (kTopK=1 fast path) |
| 256 | 4   | 8  | ~40 KB |
| 256 | 16  | 16 | ~96 KB |
| 512 | 4   | 8  | ~80 KB |
| 512 | 16  | 16 | ~192 KB — **right at the UB ceiling** |

For `kT=512, kTopK=16` the auto allocator must reuse UB slots aggressively (e.g., alias `weightTile` onto `expTile` once the divide writes it). If the build fails there, the fix is to chunk pass 1 (e.g., process 256 tokens at a time across two outer iterations).

## Known limitations (v1)

- Single AICORE; no `block_idx` work split.
- fp32 only (matches `expert_ffn`'s fp32 output; an fp16 path would need a different accumulator type).
- For `kTopK > 1` the accumulation is serial and order-dependent in fp32 (catastrophic cancellation possible for adversarial inputs; not a concern for the v1 test distribution at this magnitude).
- No double-buffering.
- The pass-1 chain `TROWMAX → TROWEXPANDSUB → TEXP → TROWSUM → TROWEXPANDDIV` is the same composition as the in-tree FA softmax macro **plus** the final divide. The macro is manual-mode and inserts `pipe_barrier(PIPE_V)` between phases; this auto-mode kernel relies on `__PTO_AUTO__` to insert the RAW edges. If outputs come back wrong, that's the first thing to suspect.

## Pattern sources

- Softmax recipe: [tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp:54-60](../../../../tests/npu/a2a3/src/st/testcase/tfa/pto_macro_fa_softmax.hpp#L54-L60).
- Pass-2 row TLOAD → TADD → TSTORE skeleton: [moe_top1_unpermute_kernel.cpp](../../../../kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp) (§A14 / §11.5).
