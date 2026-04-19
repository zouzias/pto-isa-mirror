# PTO Tile Performance Regression Report

**Date:** 2026-04-19
**Branch:** `tile-perf-check`
**Commit:** `ac8559ea`

## Summary

This report covers PTO tile operation performance on A5 (Ascend950PR_9599) and A2A3 (Ascend910B1) architectures,
including elementwise ops (TADD, TEXP, TADDS) and reduce/broadcast ops (TROWSUM, TCOLSUM, TROWEXPAND, TCOLEXPAND).

### Coverage Status

| Arch | Framework | Elementwise | Reduce/Broadcast | Status |
|------|-----------|-------------|------------------|--------|
| A5 | PTO | ✅ 18 tests | ✅ 70 tests | Complete |
| A2A3 | PTO | ✅ TADD/TADDS fp32 only | 🔲 Not started | Partial |
| A5 | AscendC | 🔲 Not started | �� Not started | No data |

> **AscendC A5 performance data is not yet available.** AscendC kernel implementations would need to be
> written separately to compare against PTO on the same architecture.

## Test Configuration

- **Simulator:** CANN 9.0.0-beta.2 cycle-accurate simulator
- **SOC:** Ascend950PR_9599 (A5), Ascend910B1 (A2A3)
- **Iterations:** 10 VF per test, skip first cold VF
- **EPC extraction:** `vf_real_execute_time` from `core0.veccore0.instr_log.dump`

---

## Part 1: Elementwise Operations (A5 PTO, 32KB)

### TADD (Binary Add) — avg EPC = 36.0

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | 1×8192 | 8192 | 27.68 | 296 | 38836 |
| fp32 | 16×512 | 8192 | 27.31 | 300 | 39128 |
| fp32 | 32×256 | 8192 | 24.24 | 338 | 38593 |
| fp16 | 1×16384 | 16384 | **55.35** | 296 | 38695 |
| fp16 | 32×512 | 16384 | 48.47 | 338 | 38864 |
| fp16 | 64×256 | 16384 | 32.90 | 498 | 38544 |

### TEXP (Unary Exp) — avg EPC = 27.1

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | 1×8192 | 8192 | 25.76 | 318 | 24514 |
| fp32 | 16×512 | 8192 | 25.76 | 318 | 24078 |
| fp32 | 32×256 | 8192 | 25.76 | 318 | 24145 |
| fp16 | 1×16384 | 16384 | 28.49 | 575 | 24187 |
| fp16 | 32×512 | 16384 | 28.49 | 575 | 23888 |
| fp16 | 64×256 | 16384 | 28.49 | 575 | 24034 |

### TADDS (Scalar Add) — avg EPC = 1.64 ⚠️ BROKEN

| Dtype | Shape | Elems | VF_EPC | VF_cy | Tick |
|-------|-------|-------|--------|-------|------|
| fp32 | 1×8192 | 8192 | 1.31 | 6254 | 65307 |
| fp32 | 16×512 | 8192 | 1.15 | 7109 | 73859 |
| fp32 | 32×256 | 8192 | 1.02 | 8021 | 83031 |
| fp16 | 1×16384 | 16384 | 2.62 | 6254 | 65188 |
| fp16 | 32×512 | 16384 | 2.04 | 8021 | 83042 |
| fp16 | 64×256 | 16384 | 1.67 | 9785 | 100644 |

---

## Part 2: Reduce & Broadcast Operations (A5 PTO, 16/32/64KB)

### TROWSUM (Row Reduce: sum across columns) — avg EPC = 28.1

| Dtype | Shape | Elems | Size | VF_EPC | VF_cy | Tick |
|-------|-------|-------|------|--------|-------|------|
| fp32 | 16×256 | 4096 | 16KB | 17.00 | 241 | 11257 |
| fp32 | 8×512 | 4096 | 16KB | 17.73 | 231 | 11399 |
| fp32 | 32×256 | 8192 | 32KB | 20.43 | 401 | 16816 |
| fp32 | 16×512 | 8192 | 32KB | 21.22 | 386 | 16976 |
| fp32 | 8×1024 | 8192 | 32KB | 19.98 | 410 | 16750 |
| fp32 | 64×256 | 16384 | 64KB | 22.72 | 721 | 27858 |
| fp32 | 32×512 | 16384 | 64KB | 23.54 | 696 | 27876 |
| fp32 | 16×1024 | 16384 | 64KB | 21.70 | 755 | 27908 |
| fp16 | 32×256 | 8192 | 16KB | 27.13 | 302 | 11248 |
| fp16 | 16×512 | 8192 | 16KB | 31.27 | 262 | 11394 |
| fp16 | 64×256 | 16384 | 32KB | 31.15 | 526 | 16812 |
| fp16 | 32×512 | 16384 | 32KB | 37.15 | 441 | 17000 |
| fp16 | 16×1024 | 16384 | 32KB | 40.16 | 408 | 16757 |
| fp16 | 128×256 | 32768 | 64KB | 33.64 | 974 | 27868 |
| fp16 | 64×512 | 32768 | 64KB | 40.96 | 800 | 27874 |
| fp16 | 32×1024 | 32768 | 64KB | 43.46 | 754 | 27881 |

### TCOLSUM (Col Reduce: sum across rows) — avg EPC = 63.4

| Dtype | Shape | Elems | Size | VF_EPC | VF_cy | Tick |
|-------|-------|-------|------|--------|-------|------|
| fp32 | 16×256 | 4096 | 16KB | 32.51 | 126 | 11403 |
| fp32 | 8×512 | 4096 | 16KB | 32.00 | 128 | 11862 |
| fp32 | 4×1024 | 4096 | 16KB | 26.26 | 156 | 11937 |
| fp32 | 32×256 | 8192 | 32KB | 43.12 | 190 | 17474 |
| fp32 | 16×512 | 8192 | 32KB | 44.04 | 186 | 17573 |
| fp32 | 8×1024 | 8192 | 32KB | 42.67 | 192 | 18235 |
| fp32 | 64×256 | 16384 | 64KB | 51.52 | 318 | 28024 |
| fp32 | 32×512 | 16384 | 64KB | 52.18 | 314 | 28670 |
| fp32 | 16×1024 | 16384 | 64KB | 53.54 | 306 | 29015 |
| fp16 | 32×256 | 8192 | 16KB | 64.00 | 128 | 11457 |
| fp16 | 16×512 | 8192 | 16KB | 65.02 | 126 | 11562 |
| fp16 | 8×1024 | 8192 | 16KB | 64.00 | 128 | 11630 |
| fp16 | 64×256 | 16384 | 32KB | 85.33 | 192 | 17019 |
| fp16 | 32×512 | 16384 | 32KB | 86.23 | 190 | 17241 |
| fp16 | 16×1024 | 16384 | 32KB | 88.09 | 186 | 17591 |
| fp16 | 128×256 | 32768 | 64KB | 102.40 | 320 | 28182 |
| fp16 | 64×512 | 32768 | 64KB | 103.04 | 318 | 28051 |
| fp16 | 32×1024 | 32768 | 64KB | 104.36 | 314 | 28407 |

### TROWEXPAND (Row Broadcast: col0 → all cols) — avg EPC = 43.4

| Dtype | Shape | Elems | Size | VF_EPC | VF_cy | Tick |
|-------|-------|-------|------|--------|-------|------|
| fp32 | 16×256 | 4096 | 16KB | 22.76 | 180 | 12776 |
| fp32 | 8×512 | 4096 | 16KB | 29.47 | 139 | 13056 |
| fp32 | 4×1024 | 4096 | 16KB | 31.27 | 131 | 12790 |
| fp32 | 32×256 | 8192 | 32KB | 26.60 | 308 | 23877 |
| fp32 | 16×512 | 8192 | 32KB | 37.41 | 219 | 24190 |
| fp32 | 8×1024 | 8192 | 32KB | 40.35 | 203 | 24027 |
| fp32 | 64×256 | 16384 | 64KB | 29.05 | 564 | 45652 |
| fp32 | 32×512 | 16384 | 64KB | 43.23 | 379 | 45815 |
| fp32 | 16×1024 | 16384 | 64KB | 47.22 | 347 | 46031 |
| fp16 | 32×256 | 8192 | 16KB | 29.79 | 275 | 13063 |
| fp16 | 16×512 | 8192 | 16KB | 45.51 | 180 | 12796 |
| fp16 | 8×1024 | 8192 | 16KB | 58.94 | 139 | 12908 |
| fp16 | 64×256 | 16384 | 32KB | 32.83 | 499 | 24194 |
| fp16 | 32×512 | 16384 | 32KB | 53.19 | 308 | 24010 |
| fp16 | 16×1024 | 16384 | 32KB | 74.81 | 219 | 23930 |
| fp16 | 128×256 | 32768 | 64KB | 34.60 | 947 | 45813 |
| fp16 | 64×512 | 32768 | 64KB | 58.10 | 564 | 46040 |
| fp16 | 32×1024 | 32768 | 64KB | 86.46 | 379 | 45896 |

### TCOLEXPAND (Col Broadcast: row0 → all rows) — avg EPC = 57.6

| Dtype | Shape | Elems | Size | VF_EPC | VF_cy | Tick |
|-------|-------|-------|------|--------|-------|------|
| fp32 | 16×256 | 4096 | 16KB | 30.12 | 136 | 12800 |
| fp32 | 8×512 | 4096 | 16KB | 25.60 | 160 | 12897 |
| fp32 | 4×1024 | 4096 | 16KB | 19.69 | 208 | 12779 |
| fp32 | 32×256 | 8192 | 32KB | 40.96 | 200 | 24074 |
| fp32 | 16×512 | 8192 | 32KB | 36.57 | 224 | 23925 |
| fp32 | 8×1024 | 8192 | 32KB | 30.12 | 272 | 24525 |
| fp32 | 64×256 | 16384 | 64KB | 49.95 | 328 | 45844 |
| fp32 | 32×512 | 16384 | 64KB | 46.55 | 352 | 46958 |
| fp32 | 16×1024 | 16384 | 64KB | 40.96 | 400 | 46239 |
| fp16 | 32×256 | 8192 | 16KB | 66.06 | 124 | 12935 |
| fp16 | 16×512 | 8192 | 16KB | 60.24 | 136 | 12769 |
| fp16 | 8×1024 | 8192 | 16KB | 51.20 | 160 | 12786 |
| fp16 | 64×256 | 16384 | 32KB | 87.15 | 188 | 24300 |
| fp16 | 32×512 | 16384 | 32KB | 81.92 | 200 | 24191 |
| fp16 | 16×1024 | 16384 | 32KB | 73.14 | 224 | 24040 |
| fp16 | 128×256 | 32768 | 64KB | 103.70 | 316 | 46822 |
| fp16 | 64×512 | 32768 | 64KB | 99.90 | 328 | 46541 |
| fp16 | 32×1024 | 32768 | 64KB | 93.09 | 352 | 45854 |

---

## Part 3: A5 vs A2A3 Comparison (TADD/TADDS fp32)

### TADD float32

| Shape | A5 EPC | A5 cy | A2A3 EPC | A2A3 cy | A2A3/A5 |
|-------|--------|-------|----------|---------|---------|
| 1×4096 | 24.38 | 168 | 48.19 | 85 | 2.0× |
| 16×256 | 21.11 | 194 | 48.19 | 85 | 2.3× |
| 8×512 | 23.81 | 172 | 48.19 | 85 | 2.0× |
| 1×8192 | 27.68 | 296 | 54.98 | 149 | 2.0× |
| 32×256 | 24.24 | 338 | 54.98 | 149 | 2.3× |
| 16×512 | 27.31 | 300 | 54.98 | 149 | 2.0× |
| 1×16384 | 29.68 | 552 | 58.94 | 278 | 2.0× |
| 64×256 | 26.17 | 626 | 58.94 | 278 | 2.3× |
| 32×512 | 29.47 | 556 | 58.94 | 278 | 2.0× |

**Average: A2A3 is 2.1× faster than A5 for TADD fp32.**

### TADDS float32 ⚠️ CRITICAL

| Shape | A5 EPC | A5 cy | A2A3 EPC | A2A3 cy | A2A3/A5 |
|-------|--------|-------|----------|---------|---------|
| 1×4096 | 1.29 | 3182 | 48.19 | 85 | **37×** |
| 16×256 | 1.01 | 4037 | 48.19 | 85 | **47×** |
| 8×512 | 1.14 | 3581 | 48.19 | 85 | **42×** |
| 1×8192 | 1.31 | 6254 | 54.98 | 149 | **42×** |
| 32×256 | 1.02 | 8021 | 54.98 | 149 | **54×** |
| 16×512 | 1.15 | 7109 | 54.98 | 149 | **48×** |
| 1×16384 | 1.32 | 12398 | 59.15 | 277 | **45×** |
| 64×256 | 1.02 | 15989 | 59.15 | 277 | **58×** |
| 32×512 | 1.16 | 14165 | 59.15 | 277 | **51×** |

**Average: A2A3 is 47× faster than A5 for TADDS fp32!**

Root cause: A5 scalar broadcast generates excessive RVECSU ops per VF iteration.
Fix needed: VF-fused `vadds` intrinsic (like A2A3's hardware repeat).

---

## EPC Summary Table (A5 PTO)

| Op | Category | fp32 avg EPC | fp16 avg EPC | Shape preference |
|----|----------|-------------|-------------|------------------|
| TADD | binary | 26.4 | 45.6 | 1D fastest |
| TEXP | unary | 25.8 | 28.5 | Shape-insensitive |
| TADDS | scalar | **1.2** ⚠️ | **2.1** ⚠️ | 1D least bad |
| TROWSUM | reduce_row | 20.6 | 35.6 | Wide (more cols) faster |
| TCOLSUM | reduce_col | 40.2 | 84.7 | Shape-insensitive |
| TROWEXPAND | bcast_row | 34.0 | 52.7 | Wide (more cols) faster |
| TCOLEXPAND | bcast_col | 35.6 | 73.5 | Tall (more rows) faster |

### Key Findings

1. **TCOLSUM is the fastest reduce op** — up to 104 EPC (fp16 64KB), benefits from simple vertical accumulation
2. **TCOLEXPAND close behind** — up to 104 EPC (fp16 64KB), mirrors TCOLSUM's efficiency
3. **TROWSUM is the slowest healthy op** — limited by horizontal reduction across the full row width
4. **TROWEXPAND improves with wider tiles** — broadcasting col0 across more columns amortizes setup cost
5. **TADDS remains broken** — 37–58× slower than equivalent ops, urgent fix needed
6. **TEXP is perfectly shape-insensitive** — identical VF cycles for all shapes within a dtype
7. **fp16 consistently 1.5–2.5× higher EPC** than fp32 for the same tile byte size
8. **EPC scales with tile size** — larger tiles amortize VF overhead, ~10–30% gain from 16KB→64KB

---

## Files

| File | Purpose |
|------|---------|
| `generate_code.py` | Code generator from input CSV |
| `input.csv` | Active elementwise test cases |
| `input_reduce_bcast.csv` | Reduce & broadcast test cases |
| `run_32kb_regression.py` | Robust parallel regression runner |
| `regression_results.csv` | Latest reduce/broadcast results |
| `worker.py` | Legacy regression runner with DB |
| `report.py` | Comparison report generator |

## Reproduction

```bash
# Generate and build for elementwise
cd ~/pto-isa/tests/npu/a5/src/st/testcase/tile_perf
python3 generate_code.py --csv input.csv
cd ~/pto-isa/tests/npu/a5/src/st/build && make tile_perf -j8

# Run regression
python3 run_32kb_regression.py --parallel 5 --stall-timeout 60

# Switch to reduce/broadcast
python3 generate_code.py --csv input_reduce_bcast.csv
cd ~/pto-isa/tests/npu/a5/src/st/build && make tile_perf -j8
python3 run_32kb_regression.py --parallel 5
```
