# GQA Flash Attention (A5 DN)

Grouped Query Attention (GQA) DN kernel for Ascend A5, modeled after [`../flash_atten`](../flash_atten) (single-head MHA flash attention).

GQA extends MHA by allowing multiple query heads to share a single K/V head, reducing KV cache size while maintaining model quality. See [`Projects/attention/pytorch/GQA.py`](../../../../../../attention/pytorch/GQA.py) and [`Projects/attention/skills/npu/A5/a5-fa-implementation.md`](../../../../../../attention/skills/npu/A5/a5-fa-implementation.md) for the full algorithm context.

---

## Algorithm (per query head)

GQA computes the same per-head flash attention as MHA, but iterates over multiple Q heads sharing each K/V head:

```
for each kv_group (0 .. NUM_KV_HEADS):
    for each q_head in this group:
        O[q_head] = softmax(Q[q_head] @ K[kv_group]^T / sqrt(head_size)) @ V[kv_group]
```

Per-head stages:

| Stage | Computation | Inner dim |
|-------|-------------|-----------|
| QK | `Q [S0, head_size] @ K^T [head_size, S1]` | `head_size` |
| Softmax | streaming softmax on QK scores, scale `1/sqrt(head_size)` | — |
| PV | `P [S0, S1] @ V [S1, head_size]` | `head_size` |
| GU | running output rescale / divide | — |

Where `NUM_QUERIES_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS` is the group size.

---

## Comparison with `flash_atten` (MHA)

Both kernels share the same **4-stage Cube–Vec pipeline**:

```
QK (Cube) -> P/softmax (Vec) -> PV (Cube) -> GU (Vec)
```

Synchronization (`TSync_Custom`), FIFO layout, UB path modes (`FIFO_MODE`), and macros (`pto_macro_fa_dn_softmax.hpp`, `pto_macro_fa_dn_gu.hpp`, `pto_macro_dn_matmul.hpp`) are reused unchanged.

### High-level data flow

```
MHA (flash_atten):  1 Q head, 1 KV head
  Q [S0, head_size]  x  K^T [head_size, S1]  ->  softmax  ->  P x V [S1, head_size]  ->  O

GQA (this folder):  NUM_Q heads, NUM_KV heads (NUM_Q >= NUM_KV)
  Q[q] [S0, head_size]  x  K[kv]^T [head_size, S1]  ->  softmax  ->  P x V[kv] [S1, head_size]  ->  O[q]
  ^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^^^^^
  per-head MHA QK           shared K across query group           per-head MHA PV
```

### Pipeline stage comparison

| Stage | `flash_atten` (MHA) | `GQA` (this impl) | Code change |
|-------|---------------------|-------------------|-------------|
| **compute_qk** | `Q [Cube_S0, HEAD_SIZE]` x `K [Cube_S1, HEAD_SIZE]` | same | **No** |
| **compute_p** | streaming softmax, scale `1/sqrt(HEAD_SIZE)` | same | No |
| **compute_pv** | load `V [Cube_S1, HEAD_SIZE]` from GM, `P @ V` | same | No |
| **compute_gu** | running O update | same | No |

**Summary:** the kernel code is **structurally identical** to MHA flash attention. All GQA-specific logic (Q-head iteration, K/V sharing) is handled at the **host level** in `main.cpp`, which calls `LaunchTGQA` once per Q head with the appropriate Q and shared K/V pointers. The DN kernel itself is reused from flash_atten with renamed constants (`kGqa*` instead of `kFa*`).

### Host-level GQA orchestration

| Aspect | MHA (`fa_performance_dn/main.cpp`) | GQA (`main.cpp`) |
|--------|-------------------------------------|-------------------|
| Launch calls | 1 call per test case | `NUM_Q_HEADS` calls per test case (one per Q head) |
| Q pointer | `qDevice` (single head) | `qDevice + q_head * S0 * HEAD_SIZE` (offset per head) |
| K pointer | `kDevice` (single head) | `kDevice + kv_group * HEAD_SIZE * S1` (shared per group) |
| V pointer | `vDevice` (single head) | `vDevice + kv_group * S1 * HEAD_SIZE` (shared per group) |
| O pointer | `oDevice` (single head) | `oDevice + q_head * S0 * HEAD_SIZE` (per head output) |
| Golden dir | flat directory | `h_q{q_head}_g_kv{kv_group}` per-head subdirectory |
| K input file | `kt.bin` (transposed) | `kt_all.bin` (all KV heads, transposed) |

### Memory / bandwidth

For `NUM_Q_HEADS=8`, `NUM_KV_HEADS=2`, `head_size=128`, `S0=128`, `S1=512`:

| Buffer | MHA (8 heads) | GQA (8Q, 2KV) |
|--------|---------------|----------------|
| KV cache (GM) | 8 x `(S1 x 128 x 2B)` K + 8 x `(S1 x 128 x 2B)` V | 2 x `(S1 x 128 x 2B)` K + 2 x `(S1 x 128 x 2B)` V |
| Total KV bytes | 8 x 2 x `512 x 128 x 2` = 2 MB | 2 x 2 x `512 x 128 x 2` = 512 KB |
| Per-head kernel GM | same (Q+S0xHEAD, K+S1xHEAD, V+S1xHEAD) | same |

GQA reduces KV cache to `NUM_KV_HEADS / NUM_Q_HEADS` of MHA, with `NUM_QUERIES_PER_KV` serial kernel launches per KV group. When `NUM_KV_HEADS = 1` (MQA), KV cache is minimized to a single head.

---

## Special cases

- **MQA (Multi-Query Attention):** `NUM_KV_HEADS = 1` — all query heads share a single KV head. Use `--num-q-heads 8 --num-kv-heads 1`.
- **MHA (Multi-Head Attention):** `NUM_KV_HEADS = NUM_Q_HEADS` — each query head has its own KV head, equivalent to standard flash attention. Use `--num-q-heads 8 --num-kv-heads 8`.

---

## Current scope and follow-ups

This implementation uses the **host-level loop** approach (approach 1 in `a5-fa-implementation.md`). It does **not** yet fuse the multi-head loop inside the kernel:

| Approach | Status |
|----------|--------|
| Host-level per-head iteration (serial kernel launches) | **Implemented** |
| Fused multi-head loop inside kernel (single launch, all Q heads) | **Not yet** — would reduce kernel launch overhead |

A fused multi-head kernel would iterate over `NUM_QUERIES_PER_KV` Q heads inside the same AI Core, reusing the K/V data already prefetched into L2/L1, eliminating repeated H2D transfers for shared K/V.

---

## Directory layout

```
kernels/manual/a5/GQA/
├── scripts/
│   ├── gen_data.py              # GQA golden (per-head MHA with shared K/V)
│   ├── generate_cases.py        # Case config with NUM_Q_HEADS, NUM_KV_HEADS
│   └── validate_buffer_usage.py # L1/UB checks (same as MHA per-head)
├── gqa_performance_dn_kernel.cpp # DN kernel (reused from flash_atten, renamed constants)
├── gqa_performance_kernel.h     # Kernel header (LaunchTGQA)
├── main.cpp                     # Host driver (per-head loop over Q heads)
├── pto_macro_*.hpp              # Shared with flash_atten (softmax, GU, matmul)
├── run.sh                       # Convenience script
└── CMakeLists.txt               # DN-only build
```

---

## Case format

```
NUM_Q_HEADS,NUM_KV_HEADS,HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]
```

`NUM_Q_HEADS` must be divisible by `NUM_KV_HEADS`.

Example:

```bash
source /usr/local/Ascend/cann_9b2/cann/bin/setenv.bash
cd kernels/manual/a5/GQA

# GQA: 8 Q heads, 2 KV heads (4 queries per KV group)
bash run.sh -r sim -v Ascend950PR_9599 --cases "8,2,128,128,512,128,128" -p 2 -m 1 --mode_dn

# MQA: 8 Q heads, 1 KV head (extreme sharing)
bash run.sh -r sim -v Ascend950PR_9599 --cases "4,1,128,128,512,128,128" -p 2 -m 1 --mode_dn
```

If the run succeeds, the output prints:

```text
test success
```

---

## References

- [`../flash_atten`](../flash_atten) — MHA DN reference kernel
- [`../MLA`](../MLA) — MLA DN reference kernel
- [`Projects/attention/pytorch/GQA.py`](../../../../../../attention/pytorch/GQA.py) — PyTorch GQA reference
- [`Projects/attention/skills/npu/A5/a5-fa-implementation.md`](../../../../../../attention/skills/npu/A5/a5-fa-implementation.md) — MHA/GQA/MLA/DSA kernel reuse analysis
- [`Projects/npu_skills/architecture/a5-core.md`](../../../../../../npu_skills/architecture/a5-core.md) — A5 memory hierarchy
