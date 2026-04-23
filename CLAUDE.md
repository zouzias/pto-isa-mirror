# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build, Test, and Lint Commands

### Quick Start (CPU Simulator - Recommended First Step)
```bash
# Full CPU simulator test suite
python3 tests/run_cpu.py --clean --verbose

# Run specific demos
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose

# Run single test case
python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.case_float_64x64_64x64'
```

### NPU Testing (Requires Ascend CANN Environment)
```bash
# Build all NPU tests
python3 tests/script/build_st.py -r npu -v a3 -t all

# Run single NPU test
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64

# One-click scripts
./build.sh --run_all --a3 --sim    # Full ST on simulator
./build.sh --run_simple --a5 --npu # Simplified ST on hardware
./build.sh --pkg                    # Build package
```

### Code Formatting
```bash
# C++ code (Google style, 120 char limit)
clang-format -i -style=Google <file>

# Python code (Ruff formatter)
ruff format <file>
ruff check <file>
```

### Pre-commit Hooks
The project uses pre-commit hooks for formatting:
- `clang-format` for C/C++ files
- Standard pre-commit hooks for YAML, trailing whitespace, etc.

Install with: `pre-commit install`

## High-Level Architecture

### Multi-Backend Design
PTO Tile Lib implements a virtual ISA for tile-oriented programming on Ascend hardware with multiple backends:

1. **CPU Simulator** (`include/pto/cpu/`): Functional verification and development debugging
2. **CostModel** (`include/pto/costmodel/`): Performance simulation for A2/A3
3. **NPU Backends** (`include/pto/npu/`): Hardware implementations
   - A2/A3 (Ascend 910B/910C): `include/pto/npu/a2a3/`
   - A5 (Ascend 950): `include/pto/npu/a5/`
   - Kirin 9030: `include/pto/npu/kirin9030/`

### Tile Programming Model
The library uses a tile-based abstraction where computation operates on fixed-size tiles rather than individual elements:

- **Tile Types**: `Tile<TileType::Vec, ...>` for vector ops, `Tile<TileType::Cube, ...>` for matrix ops
- **Global Memory**: `GlobalTensor<T, Shape, Stride>` for memory access patterns
- **Static Tile Shapes**: Compile-time tile dimensions with runtime tile masks
- **Double Buffering**: Standard pattern using `BUFFER_NUM = 2` constant

### Instruction Categories
PTO instructions are organized into categories in `include/pto/`:
- **Compute**: Elementwise (TADD, TMUL), reduction (TROWSUM), matrix (TMATMUL)
- **Data Movement**: TLOAD, TSTORE, TGATHER, TSCATTER
- **Communication**: TPUT, TGET, TBROADCAST, TREDUCE (multi-device)
- **Control**: TSEL, TCMP, TSYNC
- **Special**: TTRANS, TRESHAPE, TIMG2COL

### Build System Structure
- **Top-level**: `CMakeLists.txt` includes `cmake/package.cmake` and `cmake/func.cmake`
- **Custom CMake Functions**: 
  - `pto_add_kernel()` for kernel builds
  - `pto_costmodel_sim_st()` for costmodel tests
  - Platform-specific build functions in `cmake/`
- **Test Structure**: Each test case has its own `CMakeLists.txt` with test-specific targets

### Code Organization Patterns

#### Header-Only Template API
Most PTO instructions are header-only templates:
```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

// Tile declaration with static shape
using TileData = Tile<TileType::Vec, T, kRows_, kCols_, BLayout::RowMajor, -1, -1>;
TileData srcTile(kRows_, kCols_);

// Instruction pattern
TLOAD(srcTile, srcGlobal);
TADD(dstTile, src0Tile, src1Tile);
TSTORE(dstGlobal, dstTile);
```

#### Kernel Implementation Pattern
```cpp
template <typename T, int... params>
AICORE void runTest(__gm__ T __out__ *out, __gm__ T __in__ *src) {
    // Kernel implementation using PTO instructions
}

template <typename T, int... params>
void LaunchTest(T *out, T **src, void *stream) {
    runTest<T, ...params>(out, src);
}

// Explicit instantiation
template void LaunchTest<float, ...params>(float*, float**, void*);
```

#### Event Synchronization
```cpp
// Set/wait flag pattern for pipeline synchronization
set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
```

## Key Implementation Details

### Platform Detection
- `__CPU_SIM`: CPU simulator backend
- `__COSTMODEL`: CostModel backend
- `AICORE` macro: Expands to `[aicore]` on NPU, empty on CPU
- `__gm__`, `__out__`, `__in__`: Memory space attributes for NPU compilation

### Type System
- **Scalar Types**: float, half, int32_t, uint8_t, etc.
- **Tile Types**: Vec vs Cube determines instruction selection
- **Layout**: RowMajor vs ColMajor for memory organization
- **BLayout**: Buffer layout for tile storage

### Memory Hierarchy
1. **Global Memory** (`__gm__`): DRAM, accessed via GlobalTensor
2. **Local Memory** (`__local__`): On-chip buffer (UB/L1/L0)
3. **Registers**: Tile storage for computation

### Testing Strategy
1. **CPU First**: Always validate on CPU simulator before NPU hardware
2. **Test Naming**: `T<Instruction>Test.case_<dtype>_<dimensions>`
3. **Test Organization**: Each instruction has dedicated test directory in `tests/cpu/st/testcase/` and `tests/npu/`

## Common Development Patterns

### Adding New PTO Instructions
1. Add instruction declaration in `include/pto/common/pto_instr.hpp`
2. Implement for each backend (cpu/, costmodel/, npu/)
3. Add test case in `tests/cpu/st/testcase/<instruction>/`
4. Update instruction status table in `include/README.md`

### Performance Optimization
- Use static tile shapes for compile-time optimization
- Leverage double buffering to hide memory latency
- Consider instruction pipeline dependencies (MTE1/MTE2/Vector)
- Profile with msprof tool on NPU hardware

### Debugging
- CPU simulator supports `PTO_CPU_ASSERT` for runtime checks
- Use `TPRINT` instruction for debugging on NPU
- CostModel backend provides cycle estimation without hardware

## Important Constraints

- **C++20 or later** required
- **bfloat16 support** requires GCC>=14 for CPU simulator
- **NPU builds** require `ASCEND_HOME_PATH` environment variable
- **PTO instructions are case-sensitive** and use `T` prefix
- **Tile shapes** must match hardware constraints (typically 16x16 or 32x32 blocks)
- **Communication tests** require MPI (MPICH or OpenMPI)
