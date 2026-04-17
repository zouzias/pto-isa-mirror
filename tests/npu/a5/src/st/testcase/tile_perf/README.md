# Tile Performance Benchmark

Benchmark suite for comparing TADD/TADDS performance between A5 and A2A3 architectures.

## Quick Start

```bash
# Run both A5 and A2A3 benchmarks
python3 run_tile_bench.py --arch both

# Run A5 only
python3 run_tile_bench.py --arch a5

# Run A2A3 only  
python3 run_tile_bench.py --arch a2a3
```

## Output Files

| File | Description |
|------|-------------|
| `tile_perf_results.csv` | All benchmark results with raw data |
| `tile_perf_comparison.csv` | Side-by-side A5 vs A2A3 comparison |

## Command Line Options

```
python3 run_tile_bench.py [OPTIONS]

Options:
  --arch {a5,a2a3,both}   Architecture to benchmark (default: both)
  --input PATH            Input CSV with test cases (default: input.csv)
  --output PATH           Output CSV file (default: tile_perf_results.csv)
  --comparison PATH       Comparison CSV (default: tile_perf_comparison.csv)
  --cases OPS             Comma-separated ops to run (e.g., TADD,TADDS)
  --shapes SHAPES         Comma-separated shapes (e.g., 32x64,1x2048)
  --no-build              Skip build step (use existing binaries)
  --work-dir PATH         Working directory for dumps (default: /tmp/tile_perf_bench)
```

## Examples

```bash
# Run only TADD tests
python3 run_tile_bench.py --cases TADD

# Run only 1x2048 shape
python3 run_tile_bench.py --shapes 1x2048

# Run specific combination
python3 run_tile_bench.py --cases TADDS --shapes 32x64

# Custom output file
python3 run_tile_bench.py --output my_results.csv

# Skip rebuild (faster iteration)
python3 run_tile_bench.py --no-build
```

## Input CSV Format

`input.csv` defines test cases:

```csv
op,dtype,tile_h,tile_w,valid_h,valid_w[,scalar]
TADD,float,32,64,32,64
TADD,float,1,2048,1,2048
TADDS,float,32,64,32,64,1.5
TADDS,float,1,2048,1,2048,1.5
```

## Output CSV Format

### Results CSV (`tile_perf_results.csv`)

```csv
arch,op,dtype,shape,elements,total_ticks,warm_cycles,epc,theory_epc,efficiency_pct,raw_latencies
A5,TADD,float,1x2048,2048,12008,104.0,19.69,64.0,30.8,473;104;104;104;104
A2A3,TADD,float,1x2048,2048,6919,53.0,38.64,64.0,60.4,53;53;53;53;53
```

### Comparison CSV (`tile_perf_comparison.csv`)

```csv
op,dtype,shape,elements,a5_warm_cy,a5_epc,a2a3_warm_cy,a2a3_epc,speedup,notes
TADD,float,32x64,2048,242.0,8.46,53.0,38.64,4.57x,A2A3 faster
TADD,float,1x2048,2048,104.0,19.69,53.0,38.64,1.96x,
TADDS,float,32x64,2048,242.0,8.46,53.0,38.64,4.57x,A2A3 faster
TADDS,float,1x2048,2048,1646.0,1.24,53.0,38.64,31.13x,A2A3 faster
```

## EPC Calculation

**EPC (Elements Per Cycle)** measures compute throughput:

- **A5**: Uses VF (Vector Fusion) `vf_real_execute_time` from instruction logs
- **A2A3**: Uses VADD pop→retire latency from instruction logs

Formula: `EPC = elements / warm_cycles`

Theory maximum: 64 EPC (64 lanes processing 1 element each per cycle)

## UB Memory Layout (Bank Conflict Free)

The kernel uses optimized UB addresses to avoid bank conflicts:

```cpp
// TADD: src0 and src1 256B apart, dst 64KB from sources
TASSIGN(src0Tile, 0x0);
TASSIGN(src1Tile, 0x100);     // 256B offset
TASSIGN(dstTile, 0x10000);    // 64KB offset

// TADDS: src and dst 64KB apart
TASSIGN(srcTile, 0x0);
TASSIGN(dstTile, 0x10000);
```

## Build Requirements

- CANN 9.0.0-alpha.1 or later
- Source `set_env.sh` before running
- Simulator libraries for target SOC

```bash
source /usr/local/Ascend/cann/set_env.sh
```

## Architecture Differences

| Metric | A5 | A2A3 |
|--------|-----|------|
| SOC | Ascend910_9599 | Ascend910B1 |
| VF Fusion | Yes | No |
| VADD Latency | Inside VF block | 53 cycles (pop→retire) |
| Log Format | `vf_real_execute_time` | VADD Id matching |

## Known Issues

1. **A5 TADDS 1x2048**: VF scalar fusion issue causes 1.2 EPC (vs 38.6 on A2A3)
2. **Symlink dumps**: When using symlinks, run tests in separate directories to avoid dump file overwrites
