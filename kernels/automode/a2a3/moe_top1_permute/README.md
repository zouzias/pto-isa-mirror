# moe_top1_permute (Milestone 1b)

## Purpose

First real top-1 MoE forward permute on A3 auto mode: the kernel itself
performs expert grouping (histogram + prefix sum + counters), then packs
tokens into expert-grouped order and emits the per-expert metadata the rest
of the MoE pipeline will consume.

This is **not** a port of M1a's host-precomputed gather; the device computes
the packing order from `expert_id[T]`. M1a (host-precomputed) is a separate
prerequisite sanity test.

## Target platform

A3 only (`PTO_NPU_ARCH_A2A3`, compiled with `--cce-aicore-arch=dav-c220-vec`).

## What the kernel does

Three sequential passes, single AICORE:

```cpp
// Pass 1: histogram
int32_t count[E] = {0};
for (t = 0; t < T; ++t) count[expert_id[t]]++;
expert_count[*] = count[*];

// Pass 2: prefix sum
int32_t start[E];
start[0] = 0;
for (e = 1; e < E; ++e) start[e] = start[e-1] + count[e-1];
expert_start[*] = start[*];

// Pass 3: pack
int32_t counter[E] = {0};
for (t = 0; t < T; ++t) {
    e          = expert_id[t];
    slot       = counter[e]++;
    packed_pos = start[e] + slot;
    token_to_packed[t]            = packed_pos;
    packed_tokens[packed_pos, :]  = tokens[t, :];
}
```

The kernel preserves original within-expert token order (stable
permutation), which matches `np.argsort(expert_id, kind="stable")` in the
Python golden generator.

## Inputs / outputs

**Inputs (GM)**
- `tokens [T, H]` — `float32`
- `expert_id [T]` — `int32`

**Outputs (GM)**
- `packed_tokens [T, H]` — `float32`
- `expert_count [E]` — `int32`
- `expert_start [E]` — `int32`
- `token_to_packed [T]` — `int32`

## Auto-mode constraints honored

- Single AICORE (`<<<1, nullptr, stream>>>`); no `block_idx` work split.
- Static row tile (`Tile<Vec, T, 1, H, RowMajor, 1, H>`) declared once,
  reused across iterations — auto allocator pins its UB address.
- Per-expert `count[E]`, `start[E]`, `counter[E]` are small local int32
  arrays (`E = 4`), register/stack resident — no UB tile, no `TASSIGN`,
  no `Tile::data()`.
- `GlobalTensor` reconstructed per iteration with `base + runtime_offset`.
- No `TASSIGN` aliasing; no `Tile::data()` in kernel; no `*_IMPL` calls;
  no raw CCE intrinsics; no `Event<>`; no manual sync; no `TPipe` /
  `TPUSH` / `TPOP`; no double buffering.

## Shape

- `T = 256` tokens
- `H = 64` hidden dim
- `E = 4`  experts
- dtype: `float32`

## Build

```bash
bash run.sh -r npu -v Ascend910B1
```

(or `-r sim` for the simulator). Mirrors the
[add_tile_array](../add_tile_array) build harness; only the executable
name differs.

## Run / compare against Python reference

`run.sh` first invokes `scripts/gen_data.py`, which writes:

- `./input/input_tokens.bin`
- `./input/input_expert_id.bin`
- `./output/golden_packed_tokens.bin`
- `./output/golden_expert_count.bin`
- `./output/golden_expert_start.bin`
- `./output/golden_token_to_packed.bin`

The driver writes `./output/output_*.bin` siblings and validates each
against its golden. On success it prints:

```
packed_tokens    : success
expert_count     : success
expert_start     : success
token_to_packed  : success
test data success
test success
```

## Known limitations (v1)

- Single AICORE; no multi-core dispatch.
- Fixed shape (`T=256`, `H=64`, `E=4`); no dynamic shapes, no tail handling.
- `float32` only.
- `topK = 1`; no router GEMM, no top-K selection, no weighted combine, no
  capacity/drop policy, no fallback expert, no backward pass.
- No expert FFN; this kernel only emits the dispatch metadata + packed
  token rows. The FFN lands in M3 / M4 / M5.

## Assumptions this milestone exercises

If something here breaks, the first compiler / runtime failure should
identify which assumption gave way:

- **A1** — scalar GM read of `int32_t` from kernel code
  (`int32_t e = expert_id[t];`) is auto-mode-safe. Already exercised on
  `uint32_t` in M1a; this milestone confirms it on `int32_t` too.
- **A2** — runtime scalar usable as multiplier in
  `GlobalTensor srcGlobal(tokens + t * H);`.
- **A5 (new)** — scalar GM **write** from kernel code
  (`expert_count[e] = count[e];`, `token_to_packed[t] = packed_pos;`) is
  auto-mode-safe. Symmetric to A1; likely yes; flagged because no current
  in-tree auto-mode A3 example writes scalar GM ints from kernel code.
- **A6 (new)** — small device-local `int32_t arr[E]` (E = 4) indexed by a
  runtime scalar, with `arr[e]++` updates, is auto-mode-safe. Stack/register
  resident; should be standard C++ device codegen but flagged because the
  auto-mode pass may transform loops in unexpected ways.
- **A7 (new)** — auto-sync correctly orders the scalar GM writes
  (`token_to_packed[t] = packed_pos;`) against the `TLOAD`/`TSTORE` pair in
  the same pass-3 loop iteration. The writes target a different GM buffer
  than the row copies, so liveness is independent — should be fine.

## Failure protocol

If the direct M1b implementation fails to compile or produces wrong output:

1. Report the **first meaningful** compiler / runtime error verbatim.
2. Identify which of A1 / A2 / A5 / A6 / A7 failed (or a new one).
3. Propose the smallest fallback reproducer (e.g., kernel doing only the
   scalar histogram with no row copy, or only the row copy with literal
   indices).

Do not silently fall back to M1a-style host-precomputed shape and call M1b
done.
