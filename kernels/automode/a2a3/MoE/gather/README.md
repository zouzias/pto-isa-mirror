# gather

Auto-mode A3 prototype. Unpack and accumulate per-expert FFN outputs back into per-token rows. Generic over `kTopK ∈ {1, 2, 4, 8, 16}`.

## What it does

```
C := 0
for r in [0, kT·kTopK):
    C[A_id[r]] += B[r]
```

For `kTopK = 1` each `C[t]` is written exactly once → equivalent to a permute. For `kTopK > 1` multiple packed rows map to the same token row → genuine scatter-add. The kernel uses the same `TLOAD → TADD → TSTORE` per-row sequence in both cases, so there's no code-path switch.

| Buffer | Shape | dtype | Notes |
|---|---|---|---|
| `B` (input)  | `(kT·kTopK + 16, kH)` | fp32 | only first `kT·kTopK` rows consulted |
| `A_id` (input) | `(kT·kTopK + 16)` | int32 | only first `kT·kTopK` entries consulted |
| `C` (output) | `(kT, kH)` | fp32 | zero-initialized by host before launch |

## Target platform

A3 / Ascend 910B1. Vec target (`--cce-aicore-arch=dav-c220-vec`).

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`).
- Static row tiles declared **once outside** the row loop; auto allocator pins their UB addresses.
- `pipe_barrier(PIPE_ALL)` at the start of each row iteration (hardware-confirmed cross-iter auto-sync guard from `topk_kernel.cpp` and the placeholder `expert_ffn` row loops).
- GlobalTensor reconstructed per iteration with `(base + runtime offset)`.
- No `TASSIGN` aliasing, no `Tile::data()` in kernel, no `*_IMPL` calls, no raw CCE intrinsics, no `Event<>`, no manual sync, no `TPipe` / `TPUSH` / `TPOP`, no double buffering.

## Initialization

The kernel does **not** pre-zero `C`. The host driver calls `aclrtMemset(cDev, ..., 0x00)` before launch (0x00 bytes in fp32 = `+0.0f`), so the first `TLOAD` of any `C[t]` reads zeros. This makes the same `TLOAD → TADD → TSTORE` code correct for both kTopK=1 (where each row is written once) and kTopK>1 (where rows accumulate).

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

`scripts/gen_data.py` writes a self-contained `(B, A_id, C_golden)` set — no scatter / expert_ffn dependency. The C++ driver memsets `C` to zero, fires the kernel, and compares against `golden_C.bin` at `1e-4` abs tolerance.

## Sweeping kTopK / kT / kH

Three places to update together:

- `scripts/gen_data.py`     : `kT`, `kH`, `kTopK`
- `main.cpp`                : `constexpr int kT/kH/kTopK`
- `gather_kernel.cpp`       : `namespace gather_cfg { constexpr unsigned ... }`

## Known limitations (v1)

- Single AICORE; no block_idx work split.
- fp32 only (matches `expert_ffn`'s fp32 output; an fp16 path would need a different accumulator type).
- For `kTopK > 1` the accumulation is serial and order-dependent in fp32 (catastrophic cancellation possible for adversarial inputs; not a concern for the v1 test distribution).
- No double-buffering.
