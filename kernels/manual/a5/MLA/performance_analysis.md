# MLA Performance Analysis: Why MLA Underperforms MHA/GQA on A5 NPU

## Root Cause: QK matmul inner dimension is larger than head_size

The MLA kernel uses weight-absorbed inference: `q_absorbed @ c_kv^T` instead of `q @ K^T`.
This changes the QK matmul **inner dimension** from `HEAD_SIZE` to `KV_LATENT_DIM`.

| | MHA | MLA (L=256) | MLA (L=64) |
|---|-----|-------------|-------------|
| QK inner dim | 128 | **256** | **64** |
| PV inner dim | 128 | 128 | 128 |

---

## Three compounding factors at kv_latent_dim=256

### 1. Pipeline is Vec (softmax) bound — extra Cube work creates sync stalls

With `CUBE_S0=128, CUBE_S1=128`, `calculateFittingCubeK` returns 128 (L0 buffer limit).

- MHA: `Tile_K=128 → kSegments=1` (one TMATMUL call per QK tile)
- MLA L=256: `Tile_K=256 → kSegments=2` (two TMATMUL calls per QK tile)

Simulator profile data (block_end timestamps) shows **Vec finishes last** in both MHA and MLA:

| Version | Cube (AIC) end | Vec (AIV) end | Vec lags Cube |
|---------|---------------|---------------|---------------|
| Baseline (V from GM) | 21,928 | 22,874 | +946 ticks |
| V recon TMOV | 27,238 | 28,139 | +901 ticks |

The pipeline is **Vec (softmax) bound**: softmax + GU takes more wall-clock time per tile
than QK + PV on Cube. The 4-stage Cube–Vec pipeline (`QK → P → PV → GU`) throughput
is limited by the Vec side.

When QK doubles (kSegments=2), Cube takes longer per tile, which **increases
Cube→Vec sync stalls**: Vec must wait longer for PV data before starting GU.
These stalls extend Vec's total wall-clock time, slowing the pipeline even though
Vec's compute workload (softmax + GU) is unchanged.

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
The reduced Cube workload shortens Cube→Vec sync stalls, so Vec can proceed
through softmax → GU with fewer wait gaps. Net result: pipeline runs slightly
faster than MHA despite lower per-TMATMUL compute density.

---

## V reconstruction — implemented (TMOV L0C→L1 path)

V reconstruction (`c_kv @ W_uv`) is fused inside the kernel using direct TMOV (L0C→L1)
instead of GM roundtrip (TSTORE→GM→TLOAD). See `mla_performance_opt.md` for detailed
design, bug fixes, and session history.

### Implementation

- V recon TMATMUL produces vReconsAccTile in L0C (float)
- TMOV(vMatTile, vReconsAccTile) moves data directly from L0C to L1 (auto float→half)
- PIPE_MTE1→PIPE_FIX reverse sync with priming prevents binary flag collision
- PIPE_FIX→PIPE_MTE1 forward sync ensures V data ready before PV TEXTRACT
- No GM access needed for V data — eliminates `v.bin` GM read entirely

### Performance results (kv_latent_dim=256, HEAD_SIZE=128, S1=512)

| Version | Total Ticks | vs Baseline | Description |
|---------|------------|-------------|-------------|
| Baseline (V from GM) | 22,891 | — | V loaded directly from GM |
| V recon TMOV L0C→L1 | 28,155 | +23% slower | V = c_kv × W_uv, TMOV L0C→L1 |
| V recon GM roundtrip (old) | 40,009 | +75% slower | V = c_kv × W_uv, TSTORE→GM→TLOAD |

### Why V reconstruction is slower than baseline at kv_latent_dim=256

V reconstruction eliminates the V GM read (saving `S1 × HEAD_SIZE × 2B` per tile) and
reuses c_kv already resident in L1. However, it adds an extra TMATMUL (c_kv × W_uv) with
inner dimension = kv_latent_dim = 256, which increases the Cube workload in compute_pv.

Since the pipeline is **Vec (softmax) bound**, adding Cube work doesn't directly slow
the pipeline — Cube has slack time while Vec processes softmax. Instead, the slowdown
comes from **Cube→Vec sync stalls**: the extra V recon TMATMUL delays PV data delivery
to Vec, extending the wait gap between softmax and GU. Vec's compute workload is unchanged,
but its total wall-clock time (compute + sync waits) increases.

Profile data confirms this mechanism:

| | Cube time increase | Vec time increase | Vec > Cube gap |
|--|--------------------|-------------------|----------------|
| Baseline → V recon | +5,310 ticks | +5,265 ticks | ~900 ticks (constant) |

Cube and Vec increase by roughly the same amount — the extra Cube work propagates
as sync stall delay to Vec, which finishes last and determines pipeline throughput.

**Net effect**: GM bandwidth saved (no V GM read), but Cube→Vec sync stalls worsen
because PV data is delayed by the V recon TMATMUL. Since Vec is the bottleneck,
any increase in Vec's wall-clock time directly slows the pipeline.

### When V reconstruction would provide a performance gain

V reconstruction provides a per-kernel throughput gain only when the V recon TMATMUL is
cheaper than the V GM TLOAD it replaces. This happens when:

- **kv_latent_dim < head_size**: V recon TMATMUL inner dim is smaller, adding less Cube
  overhead. The shorter Cube work means shorter sync stalls for Vec, and the saved GM
  bandwidth (V TLOAD latency) outweighs the extra stall time.
- **GM bandwidth is the bottleneck**: If the pipeline is limited by GM access rather than
  Vec compute, eliminating the V GM read frees up bandwidth for other stages.

At kv_latent_dim=256 > head_size=128, neither condition holds. V reconstruction's benefit
is **system-level** (smaller KV cache → longer context, no V stored in GM), not per-kernel
throughput. For long-sequence inference, the reduced KV cache size may enable longer contexts
that outweigh the 23% per-token throughput loss.

---

## Summary

| Factor | kv_latent_dim=256 vs MHA | kv_latent_dim=64 vs MHA |
|--------|--------------------------|-------------------------|
| QK FLOPs | 2x (inner dim doubled) | 0.5x (inner dim halved) |
| L1 pressure | +37% (352 vs 256 KB) | -19% (208 vs 256 KB) |
| GM bandwidth (K-side) | +50% | -25% |
| Pipeline bottleneck | Vec (softmax) bound, Cube→Vec stalls worsened | Vec bound, stalls reduced |
| **Measured perf** | **83%** | **109%** |

| V reconstruction effect | kv_latent_dim=256 | kv_latent_dim=64 |
|------------------------|--------------------|--------------------|
| Extra Cube TMATMUL | inner dim=256 (same as QK) | inner dim=64 (cheap) |
| GM bandwidth saved | V TLOAD eliminated | V TLOAD eliminated |
| **vs baseline** | **+23% slower** (longer Cube→Vec stalls) | **expected faster** (light TMATMUL + no V GM read) |
| System benefit | smaller KV cache, no V in GM | smaller KV cache, no V in GM |

The core insight: **MLA's KV cache compression is a memory optimization, not a compute optimization.**
The pipeline is Vec (softmax) bound; adding Cube work (larger QK, V recon) worsens Cube→Vec
sync stalls and extends Vec's wall-clock time, slowing the pipeline.
MLA only gains NPU throughput when `kv_latent_dim < head_size`, where QK is cheaper and
the Cube→Vec stall gap shrinks, letting Vec proceed faster.
V reconstruction amplifies this pattern: it helps when `kv_latent_dim < head_size` (light extra
Cube work + no V GM read → shorter stalls) but hurts when `kv_latent_dim > head_size` (heavy
extra Cube work → longer stalls on already-delayed Vec).

### Potential optimization directions

Since the pipeline is Vec (softmax) bound, performance improvements should focus on:

1. **Speed up softmax/GU on Vec**: reduce Vec compute time per tile (fewer Vec ops,
   more efficient manual assembly). Any Vec time reduction directly improves throughput.
2. **Reduce Cube→Vec sync stall latency**: minimize the gap between "P ready on Vec"
   and "PV output delivered back to Vec". Overlapping V recon TMATMUL with softmax
   (e.g., moving V recon to compute_qk so PV starts immediately after P is ready)
   does not reduce total iteration time — the V recon delay shifts from one wait to
   another. Real improvement requires making Cube deliver PV data faster to Vec.
3. **System-level evaluation**: V recon eliminates v.bin GM read and reduces KV cache
   128x. For long-sequence inference, this system benefit may outweigh per-kernel loss.
