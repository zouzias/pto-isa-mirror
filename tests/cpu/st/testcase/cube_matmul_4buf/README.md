# PTOAS Small Tile ST Test Cases

This directory contains system test cases for PTOAS (PTO Assembler) with small tile configurations, specifically designed to validate the InsertSync pass and event ID allocation.

## Quick Start

```bash
# Run the full flow (compile + test)
./run_ptoas_st.sh

# Run with debug output
./run_ptoas_st.sh --debug

# Clean build
./run_ptoas_st.sh --clean

# Skip PTOAS compilation (use existing CCE C++)
./run_ptoas_st.sh --skip-compile
```

## Directory Structure

```
cube_matmul_4buf/
├── README.md                    # This file
├── run_ptoas_st.sh             # Main flow script
├── gen_data.py                  # Test data generator
├── cube_matmul_4buf_kernel.cpp  # Manual CCE C++ kernel (reference)
├── main.cpp                     # GTest harness
├── CMakeLists.txt              # Build config
│
├── ptoas_input/                 # PTOAS input files
│   └── cube_matmul_4buf.pto    # PTO IR for PTOAS
│
├── ptoas_output/                # PTOAS output (generated)
│   ├── cube_matmul_4buf_ptoas.cpp  # Generated CCE C++
│   └── compile.log              # Compilation log
│
└── CubeMatmul4BufTest.*/        # Test data (generated)
    ├── A_gm.bin                 # Input A tiles
    ├── B_gm.bin                 # Input B tiles
    └── golden.bin               # Expected output
```

## Test Case: cube_matmul_4buf

### Description

K-split matrix multiplication with 4-buffer software pipelining for B tiles.

```
C[32,256] = Σ_{k=0}^{63} (A_tile[k] @ B_tile[k])
```

### Matrix Dimensions

| Matrix | Shape | Type | Size |
|--------|-------|------|------|
| A | 64 tiles × (32, 16) | f16 | 65,536 bytes |
| B | 64 tiles × (16, 256) | f16 | 524,288 bytes |
| C | (32, 256) | f32 | 32,768 bytes |

### Memory Layout (SPLIT_K)

```
A_tile[i] at offset: i × 32 × 16 = i × 512 elements
B_tile[i] at offset: i × 16 × 256 = i × 4096 elements
```

### Event ID Analysis

With 4 buffers:
- **IDs needed:** 4 buffers × 2 IDs/buffer = 8 per pipeline pair
- **IDs available:** 8 per directed pair (hardware limit)
- **Result:** Exactly fits — no fallback to `pipe_barrier`

This is the **baseline case** that fits within the event ID budget.

## Full Flow

### Step 1: Compile PTO IR with PTOAS

```bash
PTOAS_BIN=~/PTOAS-official/build/tools/ptoas/ptoas

$PTOAS_BIN \
    --pto-arch=a5 \
    --pto-level=level3 \
    --enable-insert-sync \
    --pto-insert-sync-debug=1 \
    --emit-cce-cpp \
    ptoas_input/cube_matmul_4buf.pto \
    -o ptoas_output/cube_matmul_4buf_ptoas.cpp
```

### Step 2: Generate Test Data

```bash
python3 gen_data.py
```

This creates:
- `A_gm.bin`: 64 tiles of (32,16) f16 random data
- `B_gm.bin`: 64 tiles of (16,256) f16 random data
- `golden.bin`: Expected C = sum of all tile matmuls

### Step 3: Build CPU Simulator Test

```bash
cd ~/pto-isa-ptoas-st
python3 tests/run_cpu.py --testcase cube_matmul_4buf --clean --build-only
```

### Step 4: Run Regression Test

```bash
python3 tests/run_cpu.py --testcase cube_matmul_4buf
```

Expected output:
```
[  PASSED  ] CubeMatmul4BufTest.case_f16_32x1024_1024x256 (70 ms)
max diff: 1.14441e-05, diff threshold: 0.01
```

## PTOAS Options

| Option | Description |
|--------|-------------|
| `--pto-arch=a5` | Target A5 architecture |
| `--pto-level=level3` | Use level3 PTO dialect |
| `--enable-insert-sync` | Enable automatic sync insertion |
| `--pto-insert-sync-debug=N` | Debug verbosity (0-2) |
| `--emit-cce-cpp` | Output CCE C++ code |

## Files Description

### ptoas_input/cube_matmul_4buf.pto

PTO IR input for PTOAS. Contains:
- 4 L1 buffers for B tiles (`%b_l1_0` to `%b_l1_3`)
- 4 L0B buffers (`%b_l0_0` to `%b_l0_3`)
- K-loop with `scf.index_switch` for buffer rotation
- `pto.tload`, `pto.tmov`, `pto.tmatmul`, `pto.tmatmul_acc` operations

### cube_matmul_4buf_kernel.cpp

Manual reference kernel using PTO-ISA C++ API:
- Uses `Tile<>`, `GlobalTensor<>` types
- Manual `set_flag`/`wait_flag` synchronization
- Validates PTOAS-generated code correctness

### main.cpp

GTest harness that:
1. Loads binary test data
2. Calls kernel
3. Compares output against golden with tolerance

## Troubleshooting

### PTOAS not found
```bash
export PTOAS_BIN=~/PTOAS-official/build/tools/ptoas/ptoas
```

### Test data not found
```bash
cd tests/cpu/st/testcase/cube_matmul_4buf
python3 gen_data.py
```

### Build errors
```bash
# Clean rebuild
python3 tests/run_cpu.py --testcase cube_matmul_4buf --clean --verbose
```

## Related Work

- `npu_skills/pypto/ptoas-insert-sync-analysis.md` — InsertSync pass analysis
- `npu_skills/pypto/test_cases/` — Additional PTO IR test cases
- Section 8.3 of analysis doc: Event ID exhaustion with 5+ buffers
