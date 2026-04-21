# cube_matmul_nbuf — Performance Report

**Date:** 2026-04-21  
**Platform:** Ascend 950B (A5) — `Ascend950PR_9599` simulator  
**Kernel:** `RunCubeMatmulNBuf`  
**Problem size:** M=32, K_total=1024, N=256, fp16 → fp32  
**Total FLOPs:** 2 × 32 × 1024 × 256 = **16,777,216 MACs**

---

## 1. Correctness

All 5 configurations pass numerical verification:

| Config | Status | Max Diff | Bad Count |
|--------|--------|----------|-----------|
| buf2_ktile16_8KB  | ✅ PASS | 2.67e-05 | 0 |
| buf4_ktile16_8KB  | ✅ PASS | 2.67e-05 | 0 |
| buf8_ktile16_8KB  | ✅ PASS | 2.67e-05 | 0 |
| buf2_ktile32_16KB | ✅ PASS | 2.67e-05 | 0 |
| buf4_ktile32_16KB | ✅ PASS | 2.67e-05 | 0 |

---

## 2. Performance — Cycle Counts (Total Ticks)

### 2a. K_tile=16 (8 KB B-tile) — effect of N-buffering

| Config | #Bufs | K_tile | Total Ticks | vs 2-buf | vs 4-buf |
|--------|-------|--------|-------------|----------|----------|
| buf2_ktile16_8KB | 2 | 16 | 28,117 | baseline | +15.4% |
| buf4_ktile16_8KB | 4 | 16 | 24,364 | **−13.4%** | baseline |
| buf8_ktile16_8KB | 8 | 16 | 24,886 | **−11.5%** | +2.1% |

**Key finding:** Going from 2→4 buffers saves **3,753 ticks (−13.4%)**.  
Going from 4→8 buffers gives no additional gain (+2.1% overhead) — the
4-buffer depth is sufficient to hide the MTE2 GM-load latency for this tile size.

### 2b. K_tile=32 (16 KB B-tile) — effect of larger tile

| Config | #Bufs | K_tile | Total Ticks | vs K16 2-buf | vs K16 4-buf |
|--------|-------|--------|-------------|--------------|--------------|
| buf2_ktile32_16KB | 2 | 32 | 22,919 | **−18.5%** | **−6.0%** |
| buf4_ktile32_16KB | 4 | 32 | 22,916 | **−18.5%** | **−6.0%** |

**Key finding:** K_tile=32 nearly eliminates the benefit of extra buffering —
2-buf and 4-buf K32 are essentially identical (3-tick difference, noise level).
The larger tile amortises the pipeline overhead enough that 2 buffers suffice.

---

## 3. Pipeline Utilisation (reference: standalone cube_matmul_4buf, K_tile=16)

From the A5 sim `core0_summary_log` for the passing 4-buf K16 baseline:

| Pipeline Unit | Busy Cycles | % of Kernel Ticks (24,100) |
|--------------|-------------|---------------------------|
| MTE2 (GM→L1) | 21,615 | **89.7%** |
| Cube MAC     | 3,584       | 14.9% |
| Cube total   | 3,648       | 15.1% |
| MTE1 (L1→L0) | 4,248       | 17.6% |
| Scalar       | 2,596       | 10.8% |

**The kernel is heavily MTE2-bound** (DMA from GM dominates).  
MAC utilisation is only 15% — the cube unit is mostly idle waiting for data.

---

## 4. Analysis: Why N-Buffering Helps (K_tile=16)

With **2 buffers (ping-pong)**, MTE2 loads tile `k+1` while CUBE consumes tile `k`.
For K_tile=16, each A-tile is only **32×16×2 = 1 KB** and takes ~350 cycles to load
from GM. The CUBE finishes the MMAD in ~60 cycles (3584/64 ≈ 56 cycles/MMAD).  
The pipeline stalls because MTE2 cannot stay far enough ahead.

With **4 buffers**, MTE2 can queue 3 loads ahead of CUBE, absorbing the GM latency
and reducing head-of-line blocking. This accounts for the 13.4% improvement.

With **8 buffers**, the extra L1/L0 slots add instruction-scheduling overhead
(more `get_buf`/`rls_buf` operations, larger code size → icache pressure) that
offsets any additional prefetch depth benefit. Net result: slightly worse than 4-buf.

For **K_tile=32**, each tile is 2× larger (2 KB A-tile, 16 KB B-tile). The longer
DMA naturally keeps the pipeline deeper and 2 buffers already provide enough
overlap. N-buffering beyond 2 shows no benefit.

---

## 5. Recommendations

| Scenario | Recommendation |
|----------|---------------|
| Small tile (K_tile ≤ 16) | Use **4 buffers** — best cycle count, fits in 32-ID budget (9 IDs) |
| Large tile (K_tile ≥ 32) | Use **2 buffers** — sufficient overlap, simpler code, fewer IDs |
| Even larger tiles | May not need buffering at all (tile load > CUBE latency) |

---

## 6. Summary

```
Config               Ticks    Improvement over 2-buf K16 baseline
─────────────────────────────────────────────────────────────────
buf2_ktile16_8KB     28,117   baseline
buf4_ktile16_8KB     24,364   −13.4%  ← optimal for K16
buf8_ktile16_8KB     24,886   −11.5%  (worse than 4-buf)
buf2_ktile32_16KB    22,919   −18.5%  ← K32 is better baseline
buf4_ktile32_16KB    22,916   −18.5%  (same as 2-buf K32)
```

N-buffering is effective for **small K-tiles** where the compute-to-load
ratio is low. For K_tile=16 on A5, **4-buffer is the sweet spot**.  
Deeper buffering (8+) brings diminishing returns due to scheduling overhead.

---

## 7. Files

| File | Description |
|------|-------------|
| `cube_matmul_nbuf_kernel.cpp` | Kernel — 5 configs (2/4/8-buf × K16/K32) |
| `main_nbuf.cpp` | GTest harness |
| `gen_data.py` | Data generator — `[M,K]×[K,N]` row-major layout |
| `README.md` | Layout documentation and build instructions |
| `REPORT.md` | This performance report |
