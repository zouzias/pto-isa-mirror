# PTO Tile Performance Regression Report

**Date:** 2026-04-18
**Branch:** `a5-performance-check`
**Commit:** `b0595c43`

## Summary

This report compares A5 (Ascend910_9599) vs A2A3 (Ascend910B1) performance for TADD, TEXP, and TADDS operations using the PTO tile_perf benchmark.

## Test Matrix

- **Operations:** TADD, TEXP, TADDS
- **Data types:** float32, float16
- **Element counts:** 4096 (16KB), 8192 (32KB), 16384 (64KB), 32768 (128KB for fp16)
- **Shapes:** 1D (1×N), 2D with width≥256 (H×256, H×512)
- **Total tests:** 54 per architecture

## EPC Extraction Methods

### A5 (VF-based architecture)
- Source: `core0.veccore0.instr_log.dump`
- Field: `vf_real_execute_time` (pure VF compute, excludes icache overhead)
- Skip first cold VF, average warm VFs
- Formula: `EPC = elements / vf_real_execute_time`

### A2A3 (Traditional architecture)
- Source: `core0.veccore0.instr_popped_log.dump` + `core0.veccore0.instr_log.dump`
- Method: First VEC pop → last VEC retire per iteration
- Skip first 2 cold iterations, average warm iterations
- Formula: `EPC = elements / (last_retire - first_pop)`

## Results: TADD float32

| Test | A5 EPC | A5 cy | A2A3 EPC | A2A3 cy | A2A3/A5 |
|------|--------|-------|----------|---------|---------|
| 1×4096 | 24.38 | 168 | 48.19 | 85 | 2.0x |
| 16×256 | 21.11 | 194 | 48.19 | 85 | 2.3x |
| 8×512 | 23.81 | 172 | 48.19 | 85 | 2.0x |
| 1×8192 | 27.68 | 296 | 54.98 | 149 | 2.0x |
| 32×256 | 24.24 | 338 | 54.98 | 149 | 2.3x |
| 16×512 | 27.31 | 300 | 54.98 | 149 | 2.0x |
| 1×16384 | 29.68 | 552 | 58.94 | 278 | 2.0x |
| 64×256 | 26.17 | 626 | 58.94 | 278 | 2.3x |
| 32×512 | 29.47 | 556 | 58.94 | 278 | 2.0x |

**Average:** A2A3 is **2.1x faster** than A5 for TADD float32.

## Results: TADDS float32 ⚠️ CRITICAL

| Test | A5 EPC | A5 cy | A2A3 EPC | A2A3 cy | A2A3/A5 |
|------|--------|-------|----------|---------|---------|
| 1×4096 | 1.29 | 3182 | 48.19 | 85 | **37x** |
| 16×256 | 1.01 | 4037 | 48.19 | 85 | **47x** |
| 8×512 | 1.14 | 3581 | 48.19 | 85 | **42x** |
| 1×8192 | 1.31 | 6254 | 54.98 | 149 | **42x** |
| 32×256 | 1.02 | 8021 | 54.98 | 149 | **54x** |
| 16×512 | 1.15 | 7109 | 54.98 | 149 | **48x** |
| 1×16384 | 1.32 | 12398 | 59.15 | 277 | **45x** |
| 64×256 | 1.02 | 15989 | 59.15 | 277 | **58x** |
| 32×512 | 1.16 | 14165 | 59.15 | 277 | **51x** |

**Average:** A2A3 is **47x faster** than A5 for TADDS float32!

### Root Cause Analysis

A5 TADDS has severe scalar broadcast overhead. From `core0.veccore0.rvec.ASU.dump`:
- `RV_SCMP`: 670 (loop bound check)
- `RV_SCBZI`: 670 (conditional branch)
- `RV_SZEROEXT`: 640 (scalar type conversion)
- `RV_SMUL`: 640 (scalar format conversion)

These scalar ops run **per VF iteration** instead of being hoisted, causing massive overhead.

## Results: TADD half (float16)

| Test | A5 EPC | A5 cy | 
|------|--------|-------|
| 1×8192 | 48.76 | 168 |
| 32×256 | 29.90 | 274 |
| 1×16384 | 55.35 | 296 |
| 64×256 | 32.90 | 498 |
| 1×32768 | 59.36 | 552 |
| 128×256 | 34.64 | 946 |

**Best:** 59.4 EPC for 1×32768 (1D shapes)

## Key Findings

1. **A2A3 TADD ~2x faster than A5** — consistent across all shapes
2. **A5 TADDS is broken** — 37-58x slower than A2A3
3. **A2A3 is shape-agnostic** — 1D and 2D have identical performance
4. **A5 1D slightly faster than 2D** — ~15% for TADD
5. **A5 TADD half scales well** — up to 59 EPC for large 1D shapes

## Recommendations

1. **TADDS on A5 needs urgent fix** — vmuls VF fusion issue
2. **Use 1D shapes on A5** when possible for better performance
3. **A2A3 is the reference** for expected scalar op performance

## Files

- `worker.py` — Regression runner with EPC extraction
- `schema.sql` — SQLite database schema
- `report.py` — Report generator
- `results.db` — Test results database

## Usage

```bash
# Run A5 regression
python3 worker.py --arch a5 --tests '*' --output /tmp/pto_regress

# Run A2A3 regression  
python3 worker.py --arch a2a3 --tests '*' --output /tmp/pto_regress

# Run specific tests
python3 worker.py --arch a5 --tests 'TADD_float_*' --debug
```
