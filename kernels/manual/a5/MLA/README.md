# MLA Flash Attention (A5 DN)

Multi-head Latent Attention (MLA) DN kernel for Ascend A5, modeled after [`../flash_atten`](../flash_atten) (single-head MHA flash attention).

This folder implements the **weight-absorbed inference** path from DeepSeek-V2 MLA: the KV cache stores only a compressed latent vector `c_kv`, and the QK matmul runs in latent space. See [`Projects/attention/pytorch/MLA.py`](../../../../../../attention/pytorch/MLA.py) and [`Projects/attention/skills/npu/A5/a5-fa-implementation.md`](../../../../../../attention/skills/npu/A5/a5-fa-implementation.md) for the full mechanism context.

---

## Algorithm (single query head)

At inference, with `W_uk` merged into the query projection offline:

| Stage | Computation | Inner dim |
|-------|-------------|-----------|
| QK | `q_absorbed [S0, kv_latent_dim] @ c_kv^T [kv_latent_dim, S1]` | `kv_latent_dim` |
| Softmax | streaming softmax on QK scores, scale `1/sqrt(head_size)` | — |
| PV | `P [S0, S1] @ V [S1, head_size]` | `head_size` |
| GU | running output rescale / divide | — |

Where:

- `q_absorbed = q_raw @ W_uk` (computed offline in golden; stored as `q.bin`)
- `c_kv [S1, kv_latent_dim]` is the compressed KV cache (no full K stored)
- `V [S1, head_size] = c_kv @ W_uv` (computed in golden; stored as `v.bin`)

Golden generation in `scripts/gen_data.py` follows this path. Weight absorption skips K up-projection entirely at runtime.

---

## Comparison with `flash_atten` (MHA)

Both kernels share the same **4-stage Cube–Vec pipeline**:

```
QK (Cube) → P/softmax (Vec) → PV (Cube) → GU (Vec)
```

Synchronization (`TSync_Custom`), FIFO layout, UB path modes (`FIFO_MODE`), and macros (`pto_macro_fa_dn_softmax.hpp`, `pto_macro_fa_dn_gu.hpp`, `pto_macro_dn_matmul.hpp`) are reused unchanged.

### High-level data flow

```
MHA (flash_atten):
  Q [S0, head_size]  ×  K [head_size, S1]  →  softmax  →  P × V [S1, head_size]  →  O

MLA (this folder):
  Q [S0, kv_latent_dim]  ×  c_kv [kv_latent_dim, S1]  →  softmax  →  P × V [S1, head_size]  →  O
  ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^^^^^^^^^
  weight-absorbed query      compressed KV cache                      same as MHA (pre-reconstructed V)
```

### Pipeline stage comparison

| Stage | `flash_atten` (MHA) | `MLA` (this impl) | Code change |
|-------|---------------------|-------------------|-------------|
| **compute_qk** | `Q [Cube_S0, HEAD_SIZE]` × `K [Cube_S1, HEAD_SIZE]` | `Q [Cube_S0, KV_LATENT_DIM]` × `c_kv [Cube_S1, KV_LATENT_DIM]` | **Yes — main difference** |
| **compute_p** | streaming softmax, scale `1/sqrt(HEAD_SIZE)` | same | No (scale still uses `head_size`, not `kv_latent_dim`) |
| **compute_pv** | load `V [Cube_S1, HEAD_SIZE]` from GM, `P @ V` | same | No (loads pre-reconstructed `v.bin`) |
| **compute_gu** | running O update | same | No |

**Summary:** almost all kernel algorithmic differences are in **`compute_qk`** and the supporting L1 tile setup in `runTMLA` (Q/c_kv tile types, strides, prefetch sizes). Softmax, PV, and GU are structurally identical to `flash_atten`.

### `compute_qk` — what changed

| Aspect | MHA (`fa_performance_dn_kernel.cpp`) | MLA (`mla_performance_dn_kernel.cpp`) |
|--------|--------------------------------------|---------------------------------------|
| Q GM layout | `[Cube_S0, HEAD_SIZE]` DN | `[Cube_S0, KV_LATENT_DIM]` DN |
| K-side GM buffer | `k` → `[Cube_S1, HEAD_SIZE]` | `c_kv` → `[Cube_S1, KV_LATENT_DIM]` |
| Q block stride | `block_offset * HEAD_SIZE` | `block_offset * KV_LATENT_DIM` |
| K/c_kv tile offset | `s1_index * HEAD_SIZE` | `s1_index * KV_LATENT_DIM` |
| Matmul | `pto_macro_matmul<Cube_S1, HEAD_SIZE, Cube_S0>(k, q, …)` | `pto_macro_matmul<Cube_S1, KV_LATENT_DIM, Cube_S0>(c_kv, q, …)` |
| L1 tiles | `TileMatQData [HEAD_SIZE, Cube_S0]`, `TileMatKData [Cube_S1, HEAD_SIZE]` | `TileMatQData [KV_LATENT_DIM, Cube_S0]`, `TileMatCKVData [Cube_S1, KV_LATENT_DIM]` |

The QK **output shape** is unchanged: `[Cube_S1, Cube_S0]` per sub-tile, so downstream softmax/PV/GU logic does not need to change.

### Host and golden differences

| | MHA | MLA |
|---|-----|-----|
| Case tuple | `HEAD_SIZE,S0,S1,CUBE_S0[,TILE_S1]` | `HEAD_SIZE,KV_LATENT_DIM,S0,S1,CUBE_S0[,TILE_S1]` |
| Case name | `case_float_H_{H}_S0_{S0}_S1_{S1}` | `case_float_H_{H}_L_{L}_S0_{S0}_S1_{S1}` |
| Inputs | `q.bin`, `kt.bin`, `v.bin` | `q.bin` (absorbed), `c_kv.bin`, `w_uv.bin`, `v.bin` |
| Q size | `S0 × head_size` | `S0 × kv_latent_dim` |
| K-side size | `head_size × S1` | `S1 × kv_latent_dim` |
| Launch API | `LaunchTFA(q, k, v, …)` | `LaunchTMLA(q, c_kv, v, …)` |

### Memory / bandwidth

For a single head with `kv_latent_dim = 256`, `head_size = 128`:

| Buffer | MHA GM read (K side) | MLA GM read (K side) |
|--------|----------------------|----------------------|
| K / c_kv per sequence | `S1 × 128 × 2 B` (K) + `S1 × 128 × 2 B` (V) | `S1 × 256 × 2 B` (c_kv only for QK) + `S1 × 128 × 2 B` (V for PV) |

MLA saves storing full K in the KV cache. QK GM bandwidth scales with `kv_latent_dim` instead of `head_size`. L1 footprint grows when `kv_latent_dim > head_size` (e.g. default case uses 384 KB L1 vs 224 KB for the equivalent MHA case); `scripts/validate_buffer_usage.py` checks this at case generation time.

---

## Current scope and follow-ups

This implementation uses the **weight-absorbed QK** path (approach 2 in `a5-fa-implementation.md`). It does **not** yet fuse V up-projection on-NPU:

| Approach | Status |
|----------|--------|
| Weight absorption for QK (`q_absorbed @ c_kv^T`) | **Implemented** |
| V reconstruction in golden (`v = c_kv @ W_uv`) | **Implemented** (`gen_data.py`) |
| Fused `c_kv @ W_uv` inside `compute_pv` (L1, no GM write) | **Not yet** — kernel loads `v.bin` like MHA |

Adding fused V up-projection would extend `compute_pv` with an extra Cube matmul before `P @ V`, keeping `W_uv [KV_LATENT_DIM, HEAD_SIZE]` resident in L1.

---

## Directory layout

```
kernels/manual/a5/MLA/
├── scripts/
│   ├── gen_data.py              # MLA golden (weight absorption + V reconstruct)
│   ├── generate_cases.py        # Case config with KV_LATENT_DIM
│   └── validate_buffer_usage.py # L1/UB checks for MLA tile shapes
├── mla_performance_dn_kernel.cpp
├── mla_performance_kernel.h
├── main.cpp
├── pto_macro_*.hpp              # Shared with flash_atten (softmax, GU, matmul)
├── run.sh
└── CMakeLists.txt
```

---

## Case format

```
HEAD_SIZE,KV_LATENT_DIM,S0,S1,CUBE_S0[,TILE_S1]
```

Example (equivalent MHA shape `128,128,512,128,128` but with `kv_latent_dim=256`):

```bash
source /usr/local/Ascend/cann_9b2/cann/bin/setenv.bash
cd kernels/manual/a5/MLA
bash run.sh -r sim -v Ascend950PR_9599 --cases "128,256,128,512,128,128" -p 2 -m 1
```

---

## References

- [`../flash_atten`](../flash_atten) — MHA DN reference kernel
- [`Projects/attention/pytorch/MLA.py`](../../../../../../attention/pytorch/MLA.py) — PyTorch MLA reference
- [`Projects/attention/skills/npu/A5/a5-fa-implementation.md`](../../../../../../attention/skills/npu/A5/a5-fa-implementation.md) — MHA/GQA/MLA/DSA kernel reuse analysis
- [`Projects/npu_skills/architecture/a5-core.md`](../../../../../../npu_skills/architecture/a5-core.md) — A5 memory hierarchy
