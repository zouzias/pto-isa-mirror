---
name: pto-isa-testing
description: PTO-ISA NPU kernel testing on Ascend hardware/simulator. Use when writing/running TFILLPAD, TLOAD, TSTORE, TASSIGN tests, custom PadValue handling, CPU SIM vs NPU SIM testing, or working with the pto-isa repository.
---

# PTO-ISA Testing Skill

Testing framework for PTO (Portable Tile Operations) ISA kernels on Ascend NPU hardware and simulator.

## Repository Structure

```
~/pto-isa/
├── include/pto/
│   ├── common/
│   │   ├── constants.hpp      # PadValue enum, GetPadValue<>, isCustomPadValue()
│   │   └── type.hpp           # TileType, Layout enums
│   ├── npu/a2a3/TLoad.hpp     # TLOAD implementation
│   ├── npu/a2a3/TFillPad.hpp  # TFILLPAD implementation
│   └── cpu/TLoad.hpp          # CPU SIM TLOAD
├── tests/
│   ├── cpu/st/testcase/       # CPU simulator tests
│   ├── npu/a2a3/src/st/       # A2A3 (Ascend910B) NPU tests
│   │   ├── testcase/<name>/   # Individual test directories
│   │   └── build/             # Build output + test output dirs
│   └── script/run_st.py       # Test runner
```

## Running Tests

### NPU Simulator (no hardware required)
```bash
source /usr/local/Ascend/cann/set_env.sh
cd ~/pto-isa
python3 tests/script/run_st.py -r sim -v a3 -t tfillpad
```

### NPU Hardware (requires Ascend device)
```bash
python3 tests/script/run_st.py -r npu -v a3 -t tfillpad
```

### CPU Simulator
```bash
cd ~/pto-isa/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest -R TFILLPADTest
```

## Test Case Structure (NPU)

Each test lives in `tests/npu/a2a3/src/st/testcase/<testname>/`:

```
tfillpad/
├── CMakeLists.txt
├── main.cpp              # GTest harness + test definitions
├── tfillpad_kernel.cpp   # AICORE kernel + golden generation
├── gen_data.py           # Creates output directories (REQUIRED!)
└── README.md             # Test documentation
```

### Critical: gen_data.py

**Every test case needs its directory listed in gen_data.py!**

```python
case_name_list = [
    "TFILLPADTest.case_float_GT_128_127_VT_128_128_BLK1_PADMAX_PADMAX",
    "TFILLPADTest.case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1",  # NEW!
    # ... all test cases
]
```

Without this, test will fail with:
```
[ERROR] Failed to get file. Path = ../TFILLPADTest.case_xxx/golden.bin
```

### main.cpp Pattern
```cpp
TEST_F(TFILLPADTest, case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1)
{
    tfillpad_test<12, float, 1>();  // testKey=12
}
```

### Golden Generation in Kernel

Golden data is generated **in-code** (not from files) using a template function:

```cpp
template <typename U, int Shape0, int Shape1, int Shape2, 
          int Shape3, int Shape4, int kTRows_, int kTCols_, auto PadVal_>
int get_input_golden_case(uint8_t *input, uint8_t *golden)
{
    // Generate random input
    // Compute expected output with padding
    // Copy to golden buffer
}
```

The `get_input_golden<testKey>()` switch maps each test to its generator:
```cpp
template <int testKey>
int get_input_golden(uint8_t *input, uint8_t *golden)
{
    if constexpr (testKey == 1) {
        return get_input_golden_case<float, 1, 1, 1, 128, 127, 128, 128, PadValue::Max>(...);
    } else if constexpr (testKey == 12) {
        return get_input_golden_case<float, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1>(...);
    }
    // ...
}
```

## Custom PadValue

### Encoding Format (CRITICAL!)

```
(floatBits << 32) | 0x00000001
       ^                ^
  upper 32 bits    lower 32 bits (marker)
```

| Float Value | Float Bits | Correct Encoding |
|-------------|------------|------------------|
| -1.0f | 0xBF800000 | **0xBF80000000000001** |
| 0.5f | 0x3F000000 | 0x3F00000000000001 |
| 1.0f | 0x3F800000 | 0x3F80000000000001 |

**WRONG:** `0xBF80000100000000` (bytes swapped!)
**CORRECT:** `0xBF80000000000001`

### NPU Side (constexpr required)
```cpp
// Can't call PadValueCustom() - not constexpr on NPU
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000000000001ULL);

// Use in template
runTFILLPAD<float, ..., PadCustomNeg1, PadCustomNeg1>(...);
```

### CPU/Host Side
```cpp
#include <pto/common/constants.hpp>

// Runtime helper (works on host)
PadValue pv = PadValueCustom(-1.0f);

// Or compile-time template
Tile<..., PadCustom<-1.0f>> tile;
```

### Helper Functions

In `constants.hpp`, marked `AICORE constexpr`:
```cpp
bool isCustomPadValue(PadValue pv) {
    return (static_cast<uint64_t>(pv) & 0xFFFFFFFFULL) == 0x00000001ULL;
}

uint32_t getCustomPadBits(PadValue pv) {
    return static_cast<uint32_t>(static_cast<uint64_t>(pv) >> 32);
}
```

**For host code (golden generation)**, inline the logic since AICORE functions aren't callable:
```cpp
if constexpr ((static_cast<uint64_t>(PadVal_) & 0xFFFFFFFFULL) == 0x00000001ULL) {
    uint32_t bits = static_cast<uint32_t>(static_cast<uint64_t>(PadVal_) >> 32);
    u_padVal[0] = *reinterpret_cast<const float *>(&bits);
}
```

### How TLOAD Uses Custom PadValue

In `TLoad.hpp`, the `TLoadGm2ubNd2nd` function:
```cpp
if constexpr (TileData::PadVal != PadValue::Null) {
    ubPad = ubGapElement % blockSizeElem;
    set_mov_pad_val(GetPadValue<TileData>());  // ← Uses GetPadValue
}
```

`GetPadValue<TileData>()` in `constants.hpp` handles custom values:
```cpp
if constexpr (isCustomPadValue(PadVal)) {
    constexpr uint32_t bits = getCustomPadBits(PadVal);
    return bits;  // For float, return raw bits
}
```

## Adding a New Test Case (Complete Checklist)

### 1. Add test in main.cpp
```cpp
TEST_F(TFILLPADTest, case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1)
{
    tfillpad_test<12, float, 1>();  // Use next available testKey
}
```

### 2. Add kernel in tfillpad_kernel.cpp
```cpp
extern "C" __global__ AICORE void launchTFILLPAD_12(
    __gm__ uint8_t *out, __gm__ uint8_t *src,
    int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 64, 128, 128, 1,
                PadValue::Null, PadCustomNeg1>(  // LoadPad, FillPad
        (__gm__ float *)out, (__gm__ float *)src,
        gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
```

### 3. Add to launchTFILLPAD switch
```cpp
} else if constexpr (testKey == 12) {
    launchTFILLPAD_12<<<1, nullptr, stream>>>(...);
}
```

### 4. Add golden generator
```cpp
} else if constexpr (testKey == 12) {
    return get_input_golden_case<float, 1, 1, 1, 128, 64, 128, 128, PadCustomNeg1>(...);
}
```

### 5. Add template instantiations
```cpp
template void launchTFILLPAD<12>(...);
template int get_input_golden<12>(...);
```

### 6. Update gen_data.py (CRITICAL!)
```python
case_name_list = [
    # ... existing cases
    "TFILLPADTest.case_float_GT_128_64_VT_128_128_PADCUSTOM_NEG1",  # ADD THIS!
]
```

## Naming Convention

`case_<type>_GT_<rows>_<cols>_VT_<vrows>_<vcols>_BLK<n>_<loadpad>_<fillpad>[_INPLACE|_EXPAND]`

- GT = Ground Truth (input) dimensions
- VT = Virtual Tile (output) dimensions  
- BLK = Block count
- PADMIN/PADMAX/PADCUSTOM_NEG1 = Pad value type

## Common Issues

### "Failed to get file" Error
```
[ERROR] Failed to get file. Path = ../TFILLPADTest.case_xxx/golden.bin
```
**Fix:** Add test case name to `gen_data.py`

### Clang-Format (CI Pipeline)
CI uses **clang-format 10.0.0**. Install matching version:
```bash
# On pto-b10
~/.local/bin/clang-format -style=file <file> | diff - <file>
```

Key rules:
- **No space before `{};`**: `AICORE Tile(){};` not `Tile() {};`
- **Multi-line empty structs**: `struct X {\n};` not `{};`

### 32B Alignment
Columns must align to 32 bytes. For float (4B): must be multiple of 8.
- 127 cols × 4B = 508B → NOT aligned (triggers TLOAD padding)
- 128 cols × 4B = 512B → aligned

### Test Output Verification
Even if you see `[ERROR] Failed to get file`, check:
```
max diff: 0  ← Test actually passed!
```
The error is from the comparison utility not finding files, but golden was generated in-memory.

### Missing libascend_hal.so
Use `-r sim` instead of `-r npu` when no hardware present.

## TLOAD + TFILLPAD Flow

```
Global Memory → TLOAD (with LoadPadVal) → UB → TFILLPAD (with FillPadVal) → TSTORE → Global Memory
```

- **LoadPadVal**: Padding applied during TLOAD when source is smaller than tile
- **FillPadVal**: Padding applied during TFILLPAD to expand valid region

Test case 13 validates both:
```cpp
runTFILLPAD<float, ..., PadCustomNeg1, PadCustomNeg1>  // Both use custom pad
```

## Memory Model (CPU SIM)

NPUMemoryModel singleton manages shared buffers:
- UB: 192KB (Vec tiles)
- L1: 512KB (Mat tiles)  
- L0A: 64KB (Left tiles)
- L0B: 64KB (Right tiles)
- L0C: 128KB (Acc tiles)

TASSIGN maps tiles to these regions by TileType.

## Thinking Process Template

When debugging PTO-ISA issues:

1. **Trace the code path**: TLOAD → TLoadGm2ubNd2nd → set_mov_pad_val
2. **Check template parameters**: Is PadVal correctly propagated?
3. **Verify encoding**: For custom values, is bit pattern correct?
4. **Check helper functions**: Are AICORE functions callable from host?
5. **Run test with logging**: Check `max diff` even if errors appear
6. **Update all files**: kernel + main + gen_data.py + instantiations
