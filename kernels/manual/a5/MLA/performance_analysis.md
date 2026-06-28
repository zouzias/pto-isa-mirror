# MLA Performance Analysis: Why MLA Underperforms MHA/GQA on A5 NPU

## Root Cause: QK matmul inner dimension is larger than head_size

The MLA kernel uses weight-absorbed inference: `q_absorbed @ c_kv^T` instead of `q @ K^T`.
This changes the QK matmul **inner dimension** from `HEAD_SIZE` to `KV_LATENT_DIM`.

| | MHA | MLA (L=256) | MLA (L=64) |
|---|-----|-------------|-------------|
| QK inner dim | 128 | **256** | **64** |
| PV inner dim | 128 | 128 | 128 |

---

## Three compounding bottlenecks at kv_latent_dim=256

### 1. QK computation takes 2x more Cube core cycles

With `CUBE_S0=128, CUBE_S1=128`, `calculateFittingCubeK` returns 128 (L0 buffer limit).

- MHA: `Tile_K=128 → kSegments=1` (one TMATMUL call per QK tile)
- MLA L=256: `Tile_K=256 → kSegments=2` (two TMATMUL calls per QK tile)

The QK stage is the bottleneck on the Cube core, which also runs PV.
The 4-stage pipeline (`QK → P → PV → GU`) throughput is limited by the slowest Cube stage.
When QK doubles, the entire pipeline slows.

### 2. L1 buffer pressure increases 37%

| Buffer | MHA | MLA (L=256) |
|--------|-----|-------------|
| q_mat | 1x(128 x 128 x 2) = 32 KB | 1x(256 x 128 x 2) = 64 KB |
| k / c_kv_mat | 2x(128 x 128 x 2) = 64 KB | 2x(128 x 256 x 2) = 128 KB |
| p_mat | 3x(128 x 128 x 2) = 96 KB | 96 KB |
| v_mat | 2x(128 x 128 x 2) = 64 KB | 64 KB |
| **Total L1** | **256 KB** | **352 KB** |

More L1 pressure means fewer double-buffer slots and less room for pipeline overlap tuning.

### 3. GM bandwidth increases 50% on the K/V side

MHA reads `K + V` per S1 tile: `S1 x 128 + S1 x 128 = 256 elements`.
MLA reads `c_kv + V`: `S1 x 256 + S1 x 128 = 384 elements` (1.5x).
The weight absorption saves storing full K in the KV cache, but the kernel still reads
c_kv (which is 2x larger than K) from GM for every tile.

---

## Why kv_latent_dim=64 performs better (109% of MHA)

QK inner dim=64 → `kSegments=1` with smaller FLOPs.
L1 drops to 208 KB (less than MHA's 256 KB).
GM bandwidth drops to 0.75x of MHA.
The Cube core bottleneck is relieved because QK is faster, making PV the dominant Cube stage —
same as MHA.
Net result: pipeline runs slightly faster than MHA despite lower per-TMATMUL compute density.

---

## V up-projection fusion — in progress

V reconstruction (`c_kv @ W_uv`) is now being fused inside the kernel.
See `mla_performance_opt.md` for detailed design, bug fixes, and current status.

Expected effects:

- Eliminate the `v.bin` GM read (save `S1 x HEAD_SIZE x 2B` per tile)
- Reuse `c_kv` already resident in L1 for the second matmul
- Turn `compute_pv` into two Cube matmuls (V reconstruct + P@V), keeping `W_uv` in L1
- Reduce GM bandwidth but **increase Cube core load** (two matmuls in PV stage),
  potentially further exacerbating the Cube bottleneck

Current status: QK correct, PV still broken (intermediate check TSTORE race condition).

---

## Summary

| Factor | kv_latent_dim=256 vs MHA | kv_latent_dim=64 vs MHA |
|--------|--------------------------|-------------------------|
| QK FLOPs | 2x (inner dim doubled) | 0.5x (inner dim halved) |
| L1 pressure | +37% (352 vs 256 KB) | -19% (208 vs 256 KB) |
| GM bandwidth (K-side) | +50% | -25% |
| Cube core pipeline | QK bottleneck (slow) | PV bottleneck (same as MHA) |
| **Measured perf** | **83%** | **109%** |

The core insight: **MLA's KV cache compression is a memory optimization, not a compute optimization.**
When `kv_latent_dim > head_size`, the compute cost of QK increases, creating a Cube core pipeline
bottleneck that overwhelms any memory savings.
MLA only gains NPU throughput when `kv_latent_dim < head_size`, where QK becomes cheaper and
the pipeline balance improves.
