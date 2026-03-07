# PTO Tile Library - Agent Development Guide

This guide provides essential information for agentic coding assistants working in the PTO Tile Library repository.

## Build, Lint, and Test Commands

### Build Commands

```bash
# Full build with CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# One-click build script (recommended)
./build.sh --build --a3  # Build for A2/A3 platform
./build.sh --build --a5  # Build for A5 platform

# Package build
./build.sh --pkg
```

### Lint Commands

```bash
# Format C++ code (Google style with modifications)
clang-format -i -style=file <file>

# Format Python code
ruff format <file>
```

### Test Commands

```bash
# CPU Simulator Tests (cross-platform, no hardware required)
python3 tests/run_cpu.py --verbose                    # Run all CPU tests
python3 tests/run_cpu.py --testcase tadd --verbose    # Run single testcase
python3 tests/run_cpu.py --gtest_filter TADDTest.case1 --verbose  # Run specific test
python3 tests/run_cpu.py --demo gemm --verbose        # Run GEMM demo
python3 tests/run_cpu.py --demo flash_attn --verbose  # Run Flash Attention demo

# ST Tests (requires Ascend CANN environment)
python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]

# Examples:
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1

# Run recommended test suites
./tests/run_st.sh a5 npu simple
./tests/run_st.sh a3 sim all
```

### Running a Single Test

```bash
# CPU simulator - single testcase
python3 tests/run_cpu.py --testcase tadd --verbose

# CPU simulator - specific gtest filter
python3 tests/run_cpu.py --gtest_filter TADDTest.case_float_64x64_64x64_64x64 --verbose

# ST tests - single testcase with filter
python3 tests/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case1
```

## Code Style Guidelines

### C++ Code Style

- **Formatting**: Google style with modifications (see `.clang-format`)
- **Line length**: 120 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Pointer alignment**: Right-aligned (`int* ptr` not `int *ptr`)
- **Braces**: Functions on new line, control statements on same line
- **Access modifiers**: Offset -4 (indented within class)

### Python Code Style

- **Formatting**: ruff format (see `pyproject.toml`)
- **Line length**: 120 characters maximum
- **Quotes**: Double quotes preferred
- **Indentation**: Spaces (no tabs)

### Import Order

1. System headers (e.g., `<cmath>`, `<cstdint>`)
2. Third-party headers (e.g., `<gtest/gtest.h>`)
3. Local headers (e.g., `#include "pto/pto-inst.hpp"`)

### Naming Conventions

- **Types/Classes**: PascalCase (`TileData`, `GlobalTensor`)
- **Functions**: PascalCase for PTO instructions (`TADD`, `TMATMUL`), camelCase for helpers (`runTAdd`)
- **Variables**: camelCase (`src0Tile`, `dstGlobal`)
- **Constants**: UPPER_CASE with underscores (`BUFFER_NUM`, `L0_PINGPONG_BYTES`)
- **Template parameters**: PascalCase with suffix (`DynShapeDim5`, `TileData`)
- **Macros**: UPPER_CASE with underscores (`PTO_ASSERT`, `AICORE`)

### Type System

- Use strong typing with templates for tile operations
- PTO tile types: `Tile<TileType::Vec, T, rows, cols, BLayout::RowMajor, -1, -1>`
- Global tensor types: `GlobalTensor<T, ShapeType, StrideType, Layout>`
- Supported types: `float`, `int32_t`, `int16_t`, `aclFloat16` (half), `uint8_t`
- Use `__gm__` and `__in__`/`__out__` address space qualifiers for NPU code

### Error Handling

- **Compile-time checks**: Use `PTO_STATIC_ASSERT(cond)` or `PTO_STATIC_ASSERT(cond, "message")`
- **Runtime checks (CPU sim)**: Use `PTO_CPU_ASSERT(cond)` or `PTO_CPU_ASSERT(cond, "message")`
- **GTest assertions**: Use `EXPECT_TRUE`, `ASSERT_TRUE` in test files
- **Error messages**: Include hint to `docs/coding/debug.md` for debugging guidance

### Template and Generic Code

- Use `constexpr` for compile-time constants
- Use `if constexpr` for compile-time branching in templates
- Template parameters: `template <typename T, int kRows_, int kCols_>`
- Use `static_assert` to validate template parameters

### Memory Management

- NPU code uses address space qualifiers: `__gm__` for global memory
- Manual buffer management in Manual Mode (allocate addresses, manage pipelining)
- Use `TLOAD`/`TSTORE` for memory transfers between GM and tiles
- Use `TEXTRACT` for slicing within memory hierarchy

### Synchronization and Pipelining

- Use `set_flag` and `wait_flag` for pipeline synchronization
- Pipeline types: `PIPE_MTE2`, `PIPE_MTE1`, `PIPE_V`, `PIPE_M`, `PIPE_FIX`
- Event IDs: `EVENT_ID0`, `EVENT_ID1`, etc.
- Follow double-buffering patterns for: `mte2DBFlag`, `mte1DBFlag`

### Platform-Specific Code

- Use `#ifdef __CPU_SIM` for CPU simulator specific code
- Use `#ifndef __CPU_SIM` for NPU specific code
- Platform directories: `include/pto/cpu/`, `include/pto/npu/a2a3/`, `include/pto/npu/a5/`
- AICORE functions marked with `AICORE` attribute

### Test Structure

- Test files: `tests/{platform}/st/testcase/{op_name}/`
- Required files: `{op_name}.cpp`, `main.cpp`, `gen_data.py`, `CMakeLists.txt`
- Use GTest framework with `TEST_F` for test cases
- Generate golden data with numpy in `gen_data.py`
- Binary data format: `.bin` files for inputs and golden outputs

### File Organization

- Headers: `include/pto/{platform}/{category}/{op_name}.hpp`
- Common headers: `include/pto/common/`
- Kernel implementations: `kernels/manual/{platform}/{op_name}/`
- Tests: `tests/{platform}/st/testcase/{op_name}/`
- Documentation: `docs/isa{op_name}.md`

### Performance Considerations

- Tile size tuning is critical for performance
- Use static tile shapes and dynamic masks for flexibility
- Pipeline stages: TLOAD (GM→L1), TEXTRACT (L1→L0), TMATMUL (Cube), TSTORE (L0→GM)
- Profile with msprof tool for NPU performance analysis
- Consider CUBE Bound, MTE Bound, Vector Bound when optimizing

### Adding New Operators

1. Create header in appropriate platform directory
2. Implement instruction in `include/pto/common/pto_instr_impl.hpp`
3. Add to `include/pto/common/pto_instr.hpp` if public API
4. Create test case with `gen_data.py`, kernel, and main.cpp
5. Add documentation in `docs/isa/{op_name}.md`
6. Update CMakeLists.txt to include new test

### License Headers

All source files must include the CANN Open Software License Agreement header at the top of the file. Use the existing format as a template.
