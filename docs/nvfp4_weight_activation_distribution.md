# NVFP4 Weight & Activation Distribution Analysis

**Model:** microsoft/phi-2  
**Date:** 2026-03-04  
**Quantization:** NVFP4 (E2M1) with block-scaled quantization (block_size=16)

## Executive Summary

This report analyzes how NVFP4 (4-bit floating point) quantization affects both **weights** and **activations** in the Phi-2 LLM. NVFP4 uses E2M1 format with only 8 representable magnitude levels: `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.

### Summary Comparison

| Bucket | Weights | Activations |
|--------|---------|-------------|
| **0.0 (underflow)** | 40.30% | 34.24% |
| **0.5** | 45.39% | 48.62% |
| **1.0** | 14.32% | 17.14% |
| **1.5+** | 0.00% | 0.00% |

**Key finding:** Weights suffer *higher* underflow than activations (40.3% vs 34.2%), meaning weight quantization is actually the bigger precision bottleneck in NVFP4.

---

## Weights Distribution

### Per-Layer Weight Distribution (fc1 & fc2)

| Layer | fc1 0.0% | fc1 0.5% | fc1 1.0% | fc2 0.0% | fc2 0.5% | fc2 1.0% |
|-------|----------|----------|----------|----------|----------|----------|
| 0 | 41.8 | 44.2 | 14.1 | 50.4 | 36.4 | 13.2 |
| 1 | 38.8 | 46.4 | 14.8 | 41.4 | 44.7 | 14.0 |
| 2 | 39.0 | 46.3 | 14.7 | 42.2 | 43.8 | 13.9 |
| 3 | 38.9 | 46.4 | 14.8 | 40.4 | 45.3 | 14.2 |
| 4 | 39.0 | 46.3 | 14.7 | 40.8 | 45.1 | 14.1 |
| 5 | 39.1 | 46.2 | 14.7 | 40.1 | 45.6 | 14.3 |
| 6 | 39.1 | 46.2 | 14.7 | 40.0 | 45.7 | 14.3 |
| 7 | 39.2 | 46.1 | 14.7 | 40.0 | 45.7 | 14.3 |
| 8 | 39.2 | 46.1 | 14.7 | 39.9 | 45.8 | 14.4 |
| 9 | 39.2 | 46.1 | 14.7 | 39.8 | 45.8 | 14.4 |
| 10 | 39.3 | 46.0 | 14.6 | 39.8 | 45.8 | 14.4 |
| 11 | 39.4 | 46.0 | 14.6 | 39.6 | 45.9 | 14.4 |
| 12 | 39.4 | 46.0 | 14.6 | 39.6 | 45.9 | 14.5 |
| 13 | 39.5 | 45.9 | 14.6 | 39.5 | 46.0 | 14.5 |
| 14 | 39.5 | 45.9 | 14.6 | 39.5 | 46.0 | 14.5 |
| 15 | 39.6 | 45.9 | 14.5 | 39.4 | 46.0 | 14.5 |
| 16 | 39.6 | 45.9 | 14.5 | 39.4 | 46.1 | 14.5 |
| 17 | 39.7 | 45.8 | 14.5 | 39.3 | 46.1 | 14.6 |
| 18 | 39.7 | 45.8 | 14.5 | 39.3 | 46.1 | 14.6 |
| 19 | 39.8 | 45.8 | 14.5 | 39.2 | 46.2 | 14.6 |
| 20 | 39.8 | 45.7 | 14.5 | 39.2 | 46.2 | 14.6 |
| 21 | 39.9 | 45.7 | 14.4 | 39.1 | 46.2 | 14.7 |
| 22 | 39.9 | 45.7 | 14.4 | 39.1 | 46.3 | 14.7 |
| 23 | 40.0 | 45.6 | 14.4 | 39.0 | 46.3 | 14.7 |
| 24 | 40.0 | 45.6 | 14.4 | 39.0 | 46.3 | 14.7 |
| 25 | 40.1 | 45.6 | 14.4 | 38.9 | 46.4 | 14.7 |
| 26 | 40.1 | 45.5 | 14.3 | 38.9 | 46.4 | 14.7 |
| 27 | 40.2 | 45.5 | 14.3 | 38.8 | 46.4 | 14.8 |
| 28 | 40.2 | 45.5 | 14.3 | 38.8 | 46.5 | 14.8 |
| 29 | 40.3 | 45.4 | 14.3 | 38.7 | 46.5 | 14.8 |
| 30 | 40.3 | 45.4 | 14.3 | 38.7 | 46.5 | 14.8 |
| 31 | 40.4 | 45.4 | 14.2 | 38.6 | 46.5 | 14.8 |

**Observations:**
- **Layer 0 fc2 is worst:** 50.4% zeros — half of all weight values collapse to zero!
- **fc2 layers generally worse than fc1** in early layers
- **Gradual improvement** as layer depth increases
- **Upper buckets (1.5+) completely unused** — same as activations

---

## Activations Distribution

### Per-Layer Activation Distribution (fc1 & fc2)

| Layer | fc1 0.0% | fc1 0.5% | fc1 1.0% | fc2 0.0% | fc2 0.5% | fc2 1.0% |
|-------|----------|----------|----------|----------|----------|----------|
| 0 | 38.9 | 29.6 | 31.6 | 45.1 | 41.6 | 13.3 |
| 1 | 23.0 | 48.0 | 29.0 | 42.2 | 43.9 | 13.9 |
| 2 | 23.3 | 48.6 | 28.2 | 41.1 | 44.6 | 14.3 |
| 3 | 20.2 | 54.8 | 25.0 | 40.5 | 45.0 | 14.6 |
| 4 | 22.2 | 54.9 | 22.9 | 40.3 | 45.3 | 14.5 |
| 5 | 27.2 | 52.0 | 20.8 | 40.8 | 44.8 | 14.4 |
| 6 | 27.2 | 52.2 | 20.5 | 40.2 | 45.3 | 14.5 |
| 7 | 25.4 | 54.2 | 20.5 | 40.2 | 45.4 | 14.4 |
| 8 | 26.1 | 53.8 | 20.1 | 40.6 | 44.8 | 14.6 |
| 9 | 26.6 | 53.5 | 19.9 | 40.3 | 45.5 | 14.2 |
| 10 | 28.4 | 52.9 | 18.7 | 40.1 | 45.4 | 14.5 |
| 11 | 29.5 | 52.5 | 18.0 | 40.3 | 45.2 | 14.5 |
| 12 | 28.8 | 52.7 | 18.5 | 39.6 | 45.7 | 14.7 |
| 13 | 27.9 | 53.4 | 18.7 | 39.6 | 45.8 | 14.6 |
| 14 | 28.3 | 53.2 | 18.6 | 40.3 | 45.6 | 14.1 |
| 15 | 29.5 | 52.6 | 17.9 | 39.8 | 45.6 | 14.6 |
| 16 | 28.6 | 52.9 | 18.6 | 40.0 | 45.5 | 14.5 |
| 17 | 31.8 | 50.9 | 17.3 | 39.6 | 46.2 | 14.2 |
| 18 | 30.9 | 51.7 | 17.4 | 39.3 | 46.2 | 14.5 |
| 19 | 27.8 | 53.6 | 18.7 | 39.9 | 45.7 | 14.5 |
| 20 | 29.3 | 52.6 | 18.1 | 39.9 | 45.6 | 14.5 |
| 21 | 28.0 | 53.7 | 18.3 | 40.4 | 45.0 | 14.6 |
| 22 | 27.6 | 53.9 | 18.6 | 40.1 | 45.4 | 14.6 |
| 23 | 26.8 | 54.4 | 18.8 | 40.5 | 45.4 | 14.1 |
| 24 | 25.6 | 55.4 | 19.0 | 40.2 | 45.4 | 14.4 |
| 25 | 26.2 | 55.2 | 18.6 | 40.5 | 45.2 | 14.3 |
| 26 | 26.4 | 55.0 | 18.6 | 40.9 | 44.8 | 14.3 |
| 27 | 26.9 | 54.8 | 18.3 | 41.9 | 44.0 | 14.2 |
| 28 | 27.3 | 54.9 | 17.8 | 44.4 | 42.3 | 13.4 |
| 29 | 26.2 | 56.2 | 17.6 | **46.1** | 40.4 | 13.5 |
| 30 | 28.1 | 54.2 | 17.7 | 44.9 | 41.6 | 13.5 |
| 31 | 27.8 | 53.0 | 19.3 | 43.8 | 42.8 | 13.4 |

**Observations:**
- **Layer 0 fc1 has unusual distribution:** Higher 1.0% bucket (31.6%) than other layers
- **fc2 layers have higher underflow** (~40-46%) vs fc1 (~20-30%)
- **Final layers (28-31) fc2 spike to 44-46%** underflow — output projections degrade

---

## Side-by-Side Comparison

### fc1 Layers: Weights vs Activations

```
             Weights                    Activations
Layer   0.0%   0.5%   1.0%        0.0%   0.5%   1.0%
  0    41.8   44.2   14.1        38.9   29.6   31.6  ← Act has unusual 1.0 spike
  5    39.1   46.2   14.7        27.2   52.0   20.8
 15    39.6   45.9   14.5        29.5   52.6   17.9
 25    40.1   45.6   14.4        26.2   55.2   18.6
 31    40.4   45.4   14.2        27.8   53.0   19.3
```

**Insight:** Weights have ~12-15% more zeros than activations in fc1 layers.

### fc2 Layers: Weights vs Activations

```
             Weights                    Activations
Layer   0.0%   0.5%   1.0%        0.0%   0.5%   1.0%
  0    50.4   36.4   13.2        45.1   41.6   13.3  ← Layer 0 worst for both
  5    40.1   45.6   14.3        40.8   44.8   14.4
 15    39.4   46.0   14.5        39.8   45.6   14.6
 25    38.9   46.4   14.7        40.5   45.2   14.3
 31    38.6   46.5   14.8        43.8   42.8   13.4  ← Act degrades in final layers
```

**Insight:** fc2 weights and activations have similar distributions (~40% zeros), but activations degrade in final layers while weights improve.

---

## Visual Distribution (ASCII)

```
WEIGHTS (MLP fc1+fc2 average):
0.0  [████████████████████████████████████████] 40.3%
0.5  [█████████████████████████████████████████████] 45.4%
1.0  [██████████████] 14.3%
1.5+ [                                        ] 0.0%

ACTIVATIONS (MLP fc1+fc2 average):
0.0  [██████████████████████████████████] 34.2%
0.5  [████████████████████████████████████████████████] 48.6%
1.0  [█████████████████] 17.1%
1.5+ [                                        ] 0.0%
```

---

## Key Findings

1. **Weights have worse underflow than activations** (40.3% vs 34.2%)
2. **Layer 0 fc2 weights are catastrophic:** 50.4% collapse to zero
3. **Upper dynamic range completely wasted:** 1.5-6.0 buckets unused (62.5% of range)
4. **fc2 layers consistently worse** than fc1 for both weights and activations
5. **Activations degrade in final layers** (28-31) while weights improve

---

## Recommendations

1. **Prioritize weight quantization improvements** — they're the bigger bottleneck
2. **Special handling for layer 0** — consider FP8 or BF16 for embedding-adjacent layers
3. **Custom quantization grid** — redistribute levels to [0, 1.0] range
4. **Mixed precision strategy:**
   - FP8 for layer 0 fc2 weights
   - FP8 for final layer (28-31) fc2 activations
   - NVFP4 for middle layers

---

## Raw Data

Full JSON data: `/mnt/fluxdata/happybot/nvfp4_analysis/nvfp4_weight_act_dist.json`

## Methodology

- **Model:** microsoft/phi-2 (2.7B parameters)
- **Weights:** Direct parameter tensors from MLP fc1/fc2 layers
- **Activations:** Post-layer outputs captured via forward hooks
- **Input:** "The quick brown fox jumps over the lazy dog. Machine learning models require careful quantization."
- **Quantization:** Block-scaled NVFP4 (block_size=16), per-block max-absolute scaling
