---
name: pto-isa-testing
description: PTO-ISA NPU kernel testing on Ascend hardware/simulator. Use when writing/running TFILLPAD, TLOAD, TSTORE, TASSIGN tests, custom PadValue handling, CPU SIM vs NPU SIM testing, or working with the pto-isa repository.
---

# PTO-ISA Testing Skill

Testing framework for PTO (Portable Tile Operations) ISA kernels on Ascend NPU hardware and simulator.

## Repository Structure

```
~/pto-isa/
├── include/pto/           # Core headers
│   ├── common/constants.hpp   # PadValue enum, constants
│   └── pto-inst.hpp          # Main PTO instructions
├── tests/
│   ├── cpu/st/testcase/      # CPU simulator tests
│   ├── npu/a2a3/src/st/      # A2A3 (Ascend910B) NPU tests
│   └── script/run_st.py      # Test runner
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
├── main.cpp              # GTest harness + test case definitions
├── tfillpad_kernel.cpp   # AICORE kernel implementations
├── gen_data.py           # Optional input data generator
└── TFILLPADTest.<case>/  # Output directories (auto-created)
```

### main.cpp Pattern
```cpp
TEST_F(TFILLPADTest, case_float_GT_128_127_VT_128_160_BLK1_PADMIN_PADMAX)
{
    tfillpad_test<3, float, 1>();  // testKey=3, type=float, blocks=1
}
```

### Kernel Pattern
```cpp
extern "C" __global__ AICORE void launchTFILLPAD_3(
    __gm__ uint8_t *out, __gm__ uint8_t *src,
    int gShape0, int gShape1, int gShape2, int gRows, int gCols,
    __gm__ uint64_t *gLog)
{
    runTFILLPAD<float, 1, 1, 1, 128, 127, 128, 160, 1,
                PadValue::Min, PadValue::Max>(
        (__gm__ float *)out, (__gm__ float *)src,
        gShape0, gShape1, gShape2, gRows, gCols, gLog);
}
```

## Custom PadValue

### CPU Side
```cpp
#include <pto/common/constants.hpp>
// Use PadValueCustom() helper
Tile<..., PadValueCustom(-1.0f)> tile;
```

### NPU Side (constexpr required)
```cpp
// -1.0f bit pattern: 0xBF800000
// Encoding: (bits << 32) | 0x00000001
constexpr PadValue PadCustomNeg1 = static_cast<PadValue>(0xBF80000100000000ULL);

// Use in template
runTFILLPAD<float, ..., PadCustomNeg1, PadCustomNeg1>(...);
```

### Helper Functions
```cpp
bool isCustomPadValue(PadValue pv);      // Check if custom
uint32_t getCustomPadBits(PadValue pv);  // Extract float bits
```

## Golden Data Generation

For standard Min/Max padding:
```cpp
template <typename U, int Shape0, ..., PadValue PadVal_>
int get_input_golden_case(uint8_t *input, uint8_t *golden);
```

For custom float padding:
```cpp
template <int Shape0, ..., int kTRows_, int kTCols_>
int get_input_golden_case_custom_float(uint8_t *input, uint8_t *golden, float customPadVal);
```

## Adding a New Test Case

1. Add test in `main.cpp`:
```cpp
TEST_F(TFILLPADTest, case_float_GT_128_127_VT_128_160_PADCUSTOM)
{
    tfillpad_test<13, float, 1>();
}
```

2. Add kernel in `tfillpad_kernel.cpp`:
```cpp
extern "C" __global__ AICORE void launchTFILLPAD_13(...) { ... }
```

3. Add to switch in `launchTFILLPAD<testKey>`:
```cpp
} else if constexpr (testKey == 13) {
    launchTFILLPAD_13<<<1, nullptr, stream>>>(...);
}
```

4. Add golden generator if needed in `get_input_golden<testKey>`

5. Add template instantiations:
```cpp
template void launchTFILLPAD<13>(...);
template int get_input_golden<13>(...);
```

6. Create output directory:
```bash
mkdir -p TFILLPADTest.case_float_GT_128_127_VT_128_160_PADCUSTOM
```

## Naming Convention

`case_<type>_GT_<rows>_<cols>_VT_<vrows>_<vcols>_BLK<n>_<loadpad>_<fillpad>[_INPLACE|_EXPAND]`

- GT = Ground Truth dimensions
- VT = Virtual Tile dimensions  
- BLK = Block count
- PADMIN/PADMAX/PADCUSTOM = Pad value type

## Common Issues

### 32B Alignment
Columns must align to 32 bytes. For float (4B): must be multiple of 8.
- 127 cols × 4B = 508B → NOT aligned (tests TLOAD padding)
- 128 cols × 4B = 512B → aligned

### Golden Size 0
Test harness quirk for custom pad tests — verify via `max diff: 0` in output.

### Missing libascend_hal.so
Use `-r sim` instead of `-r npu` when no hardware present.

## Memory Model (CPU SIM)

NPUMemoryModel singleton manages shared buffers:
- UB: 192KB (Vec tiles)
- L1: 512KB (Mat tiles)  
- L0A: 64KB (Left tiles)
- L0B: 64KB (Right tiles)
- L0C: 128KB (Acc tiles)

TASSIGN maps tiles to these regions by TileType.
