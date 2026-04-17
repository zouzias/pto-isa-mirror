# tile_perf - A5 Tile Performance Benchmarks

## Overview

Performance benchmarks for tile operations (TADD, TADDS, TEXP) on A5 simulator.

## Running Tests

```bash
cd ~/pto-isa/tests/npu/a5/src/st/build
make tile_perf -j8
./bin/tile_perf                          # Run all
./bin/tile_perf --gtest_filter="*TADD*"  # Filter
```

## EPC Extraction

Uses `vf_real_execute_time` from VF retire log for accurate throughput measurement.

### Source Files

| File | Purpose |
|------|---------|
| `core0.veccore0.instr_log.dump` | VF retire cycles + vf_real_execute_time |
| `core0.veccore0.instr_popped_log.dump` | VF dispatch (pop) cycles |

### Log Format

Retire log:
```
[00001700] (PC: 0x13b3a0d4) PUSHQ : ... VF ... vf_real_execute_time: 168
```

### Calculation

```python
# Skip first cold VF (icache miss), average warm VFs
warm_times = [168, 168, 168, ...]  # vf_real_execute_time for VF 2+
avg_warm = sum(warm_times) / len(warm_times)
vf_epc = elements / avg_warm
# Example: 4096 / 168 = 24.38 EPC
```

### Key Fields

| Field | Description |
|-------|-------------|
| vf_execute_time | Total pop to retire (includes icache prefetch) |
| vf_real_execute_time | Pure rvec compute time (USE THIS) |

## Regression System

### Running

```bash
cd tile_perf
python3 worker.py --arch a5 --tests "*" --output /tmp/pto_regress
```

### Results

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('results.db')
for r in conn.execute('SELECT test_name, vf_epc, vf_cycles FROM results ORDER BY vf_epc DESC'):
    print(f'{r[0]:30} | VF EPC: {r[1]:6.2f} | VF cy: {r[2]:.0f}')
"
```

## Test Matrix

| Op | Dtype | Sizes | Shapes |
|----|-------|-------|--------|
| TADD | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TEXP | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |
| TADDS | float/half | 16KB, 32KB, 64KB | 1D, Hx256, Hx512 |

Total: 54 tests (3 ops x 2 dtypes x 3 sizes x 3 shapes)

## Files

- `main.cpp` - GTest harness with test macros
- `tile_perf_kernel.cpp` - Kernel implementations
- `worker.py` - Regression runner with EPC extraction
- `schema.sql` - SQLite schema for results
- `report.py` - Report generator
