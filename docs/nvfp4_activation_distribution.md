# NVFP4 Activation Distribution Analysis

**Model:** microsoft/phi-2  
**Date:** 2026-03-04  
**Quantization:** NVFP4 (E2M1) with block-scaled quantization (block_size=16)

## Executive Summary

This report analyzes how NVFP4 (4-bit floating point) quantization affects activation distributions in the Phi-2 LLM. NVFP4 uses E2M1 format with only 8 representable magnitude levels: `{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`.

### Key Findings

| Metric | fc1 Layers | fc2 Layers | Overall |
|--------|------------|------------|---------|
| **0.0 bucket (underflow)** | 27.1% | 41.2% | 34.2% |
| **0.5 bucket** | 52.8% | 44.6% | 48.6% |
| **1.0 bucket** | 20.0% | 14.2% | 17.1% |
| **1.5+ buckets** | 0.0% | 0.0% | 0.0% |

**Critical observation:** The upper half of NVFP4's representable range (1.5, 2.0, 3.0, 4.0, 6.0) is **completely unused** — all normalized activations fall within [0, 1.0].

## Per-Layer Distribution

### Layer 0 (Embedding-adjacent)

| Layer | 0.0 | 0.5 | 1.0 | 1.5+ |
|-------|-----|-----|-----|------|
| layers.0.mlp.fc1 | 38.9% | 29.6% | 31.6% | 0.0% |
| layers.0.mlp.fc2 | 45.1% | 41.6% | 13.3% | 0.0% |

Layer 0 shows the highest underflow rates — likely due to embedding layer outputs having different scale characteristics.

### Early Layers (1-4)

| Layer | 0.0 | 0.5 | 1.0 | 1.5+ |
|-------|-----|-----|-----|------|
| layers.1.mlp.fc1 | 23.0% | 48.0% | 29.0% | 0.0% |
| layers.1.mlp.fc2 | 42.2% | 43.9% | 13.9% | 0.0% |
| layers.2.mlp.fc1 | 23.3% | 48.6% | 28.2% | 0.0% |
| layers.2.mlp.fc2 | 41.1% | 44.6% | 14.3% | 0.0% |
| layers.3.mlp.fc1 | 20.2% | 54.8% | 25.0% | 0.0% |
| layers.3.mlp.fc2 | 40.5% | 45.0% | 14.6% | 0.0% |
| layers.4.mlp.fc1 | 22.2% | 54.9% | 22.9% | 0.0% |
| layers.4.mlp.fc2 | 40.3% | 45.3% | 14.5% | 0.0% |

### Middle Layers (5-15)

| Layer | 0.0 | 0.5 | 1.0 | 1.5+ |
|-------|-----|-----|-----|------|
| layers.5.mlp.fc1 | 27.2% | 52.0% | 20.8% | 0.0% |
| layers.5.mlp.fc2 | 40.8% | 44.8% | 14.4% | 0.0% |
| layers.6.mlp.fc1 | 27.2% | 52.2% | 20.5% | 0.0% |
| layers.6.mlp.fc2 | 40.2% | 45.3% | 14.5% | 0.0% |
| layers.7.mlp.fc1 | 25.4% | 54.2% | 20.5% | 0.0% |
| layers.7.mlp.fc2 | 40.2% | 45.4% | 14.4% | 0.0% |
| layers.8.mlp.fc1 | 26.1% | 53.8% | 20.1% | 0.0% |
| layers.8.mlp.fc2 | 40.6% | 44.8% | 14.6% | 0.0% |
| layers.9.mlp.fc1 | 26.6% | 53.5% | 19.9% | 0.0% |
| layers.9.mlp.fc2 | 40.3% | 45.5% | 14.2% | 0.0% |
| layers.10.mlp.fc1 | 28.4% | 52.9% | 18.7% | 0.0% |
| layers.10.mlp.fc2 | 40.1% | 45.4% | 14.5% | 0.0% |
| layers.11.mlp.fc1 | 29.5% | 52.5% | 18.0% | 0.0% |
| layers.11.mlp.fc2 | 40.3% | 45.2% | 14.5% | 0.0% |
| layers.12.mlp.fc1 | 28.8% | 52.7% | 18.5% | 0.0% |
| layers.12.mlp.fc2 | 39.6% | 45.7% | 14.7% | 0.0% |
| layers.13.mlp.fc1 | 27.9% | 53.4% | 18.7% | 0.0% |
| layers.13.mlp.fc2 | 39.6% | 45.8% | 14.6% | 0.0% |
| layers.14.mlp.fc1 | 28.3% | 53.2% | 18.6% | 0.0% |
| layers.14.mlp.fc2 | 40.3% | 45.6% | 14.1% | 0.0% |
| layers.15.mlp.fc1 | 29.5% | 52.6% | 17.9% | 0.0% |
| layers.15.mlp.fc2 | 39.8% | 45.6% | 14.6% | 0.0% |

### Late-Middle Layers (16-25)

| Layer | 0.0 | 0.5 | 1.0 | 1.5+ |
|-------|-----|-----|-----|------|
| layers.16.mlp.fc1 | 28.6% | 52.9% | 18.6% | 0.0% |
| layers.16.mlp.fc2 | 40.0% | 45.5% | 14.5% | 0.0% |
| layers.17.mlp.fc1 | 31.8% | 50.9% | 17.3% | 0.0% |
| layers.17.mlp.fc2 | 39.6% | 46.2% | 14.2% | 0.0% |
| layers.18.mlp.fc1 | 30.9% | 51.7% | 17.4% | 0.0% |
| layers.18.mlp.fc2 | 39.3% | 46.2% | 14.5% | 0.0% |
| layers.19.mlp.fc1 | 27.8% | 53.6% | 18.7% | 0.0% |
| layers.19.mlp.fc2 | 39.9% | 45.7% | 14.5% | 0.0% |
| layers.20.mlp.fc1 | 29.3% | 52.6% | 18.1% | 0.0% |
| layers.20.mlp.fc2 | 39.9% | 45.6% | 14.5% | 0.0% |
| layers.21.mlp.fc1 | 28.0% | 53.7% | 18.3% | 0.0% |
| layers.21.mlp.fc2 | 40.4% | 45.0% | 14.6% | 0.0% |
| layers.22.mlp.fc1 | 27.6% | 53.9% | 18.6% | 0.0% |
| layers.22.mlp.fc2 | 40.1% | 45.4% | 14.6% | 0.0% |
| layers.23.mlp.fc1 | 26.8% | 54.4% | 18.8% | 0.0% |
| layers.23.mlp.fc2 | 40.5% | 45.4% | 14.1% | 0.0% |
| layers.24.mlp.fc1 | 25.6% | 55.4% | 19.0% | 0.0% |
| layers.24.mlp.fc2 | 40.2% | 45.4% | 14.4% | 0.0% |
| layers.25.mlp.fc1 | 26.2% | 55.2% | 18.6% | 0.0% |
| layers.25.mlp.fc2 | 40.5% | 45.2% | 14.3% | 0.0% |

### Final Layers (26-31)

| Layer | 0.0 | 0.5 | 1.0 | 1.5+ |
|-------|-----|-----|-----|------|
| layers.26.mlp.fc1 | 26.4% | 55.0% | 18.6% | 0.0% |
| layers.26.mlp.fc2 | 40.9% | 44.8% | 14.3% | 0.0% |
| layers.27.mlp.fc1 | 26.9% | 54.8% | 18.3% | 0.0% |
| layers.27.mlp.fc2 | 41.9% | 44.0% | 14.2% | 0.0% |
| layers.28.mlp.fc1 | 27.3% | 54.9% | 17.8% | 0.0% |
| layers.28.mlp.fc2 | 44.4% | 42.3% | 13.4% | 0.0% |
| layers.29.mlp.fc1 | 26.2% | 56.2% | 17.6% | 0.0% |
| layers.29.mlp.fc2 | **46.1%** | 40.4% | 13.5% | 0.0% |
| layers.30.mlp.fc1 | 28.1% | 54.2% | 17.7% | 0.0% |
| layers.30.mlp.fc2 | 44.9% | 41.6% | 13.5% | 0.0% |
| layers.31.mlp.fc1 | 27.8% | 53.0% | 19.3% | 0.0% |
| layers.31.mlp.fc2 | 43.8% | 42.8% | 13.4% | 0.0% |

**Note:** Final layers (28-31) show increased fc2 underflow rates (44-46%), suggesting output activations become sparser near the model head.

## Distribution Visualization (ASCII)

```
fc1 Layers (avg):
0.0  [████████████████████████████] 27.1%
0.5  [█████████████████████████████████████████████████████] 52.8%
1.0  [████████████████████] 20.0%
1.5+ [                             ] 0.0%

fc2 Layers (avg):
0.0  [█████████████████████████████████████████] 41.2%
0.5  [████████████████████████████████████████████] 44.6%
1.0  [██████████████] 14.2%
1.5+ [                             ] 0.0%
```

## Quantization Error Statistics

From the earlier underflow analysis (different metric — measures values that become zero after quantization):

| Layer Type | Avg Underflow | Max Underflow | Avg MAE |
|------------|---------------|---------------|---------|
| fc1 | 5.3% | 20.5% (layer 0) | 0.096 |
| fc2 | 8.0% | 10.6% (layer 29) | 0.022 |
| **Overall** | **6.8%** | **20.5%** | **0.060** |

## Implications

1. **Wasted Dynamic Range:** 62.5% of NVFP4's representable values (1.5-6.0) are never used
2. **High Underflow in fc2:** Output projections lose ~41% of information to the zero bucket
3. **Layer 0 Vulnerability:** First layer after embeddings is most affected
4. **Late Layer Degradation:** Layers 28-31 show increasing precision loss

## Recommendations

1. **Custom Quantization Grid:** Consider a denser grid in [0, 1.0] range, e.g., `{0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0}`
2. **Layer-Specific Scaling:** Apply different scale factors to fc1 vs fc2 layers
3. **Mixed Precision:** Use higher precision (FP8/BF16) for layer 0 and final layers
4. **Activation Clipping:** Pre-clipping outliers before quantization may improve utilization

## Methodology

- **Model:** microsoft/phi-2 (2.7B parameters)
- **Input:** "The quick brown fox jumps over the lazy dog. Machine learning models require careful quantization."
- **Quantization:** Block-scaled NVFP4 with block_size=16
- **Hooks:** Captured post-activation tensors from all MLP fc1/fc2 layers
- **Normalization:** Per-block max-absolute scaling to [0, 1] before quantizing to NVFP4 grid

## Raw Data

Full JSON data available at: `/mnt/fluxdata/happybot/nvfp4_analysis/nvfp4_distribution.json`
