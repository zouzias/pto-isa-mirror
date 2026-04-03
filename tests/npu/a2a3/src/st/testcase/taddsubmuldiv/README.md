# TAddSubMulMul Kernel

## Overview

This kernel implements a fused arithmetic operation that combines Add, Sub, Mul and Div operations into a single kernel:

```
out = ((src0 + src1) - src2) * src3 / src4
```

## Features

- **Operation Fusion**: Combines Add, Sub, Mul, and Div into a single kernel
- **Reduced Memory Access**: 5 GM reads → 2 GM accesses (optimized with intermediate results in UB)
- **Event-based Synchronization**: Automatic dependency tracking between operations
- **Type Support**: Supports float and aclFloat16 (half) data types
- **Configurable Tile Size**: Template parameter for flexible tile dimensions

## Implementation Details

### Data Flow

```
gm → ub (TLOAD) → vector (TADD) → vector (TSUB) → vector (TMUL) → vector (TDIV) → ub → gm (TSTORE)
```

### Memory Layout

- **Input**: 5 source tensors (src0, src1, src2, src3, src4)
- **Output**: 1 destination tensor (out)
- **Temporary**: 1 intermediate tile for storing computation results

### Buffer Allocation

```cpp
TASSIGN(src0Tile, 0x0);                           // Buffer 0
TASSIGN(src1Tile, sizeof(T) * TileData::Numel);  // Buffer 1
TASSIGN(src2Tile, 2 * sizeof(T) * TileData::Numel);  // Buffer 2
TASSIGN(src3Tile, 3 * sizeof(T) * TileData::Numel);  // Buffer 3
TASSIGN(src4Tile, 4 * sizeof(T) * TileData::Numel);  // Buffer 4
TASSIGN(tmpTile, 5 * sizeof(T) * TileData::Numel);    // Buffer 5 (temporary)
TASSIGN(dstTile, 6 * sizeof(T) * TileData::Numel);   // Buffer 6
```

### Synchronization Strategy

Uses event-based synchronization for automatic dependency tracking:

```cpp
Event<Op::TLOAD, Op::TADD> event0;
Event<Op::TADD, Op::TSUB> event1;
Event<Op::TSUB, Op::TMUL> event2;
Event<Op::TMUL, Op::TDIV> event3;
Event<Op::TDIV, Op::TSTORE_VEC> event4;

event0 = TLOAD(src1Tile, src1Global);
event1 = TADD(tmpTile, src0Tile, src1Tile, event0);
event2 = TSUB(tmpTile, tmpTile, src2Tile, event1);
event3 = TMUL(tmpTile, tmpTile, src3Tile, event2);
event4 = TDIV(dstTile, tmpTile, src4Tile, event3);
TSTORE(dstGlobal, dstTile, event4);
```

## Usage

### API

```cpp
namespace TAddSubMulDiv {

template <typename T, int kTRows_, int kTCols_, int vRows, int vCols>
void launchTAddSubMulDiv(T *out, T *src0, T *src1, T *src2, T *src3, T *src4, void *stream);

}
```

### Example

```cpp
#include "taddsubmuldiv_kernel.hpp"

//cpp
// Prepare data
const int rows = 64;
const int cols = 64;
const int size = rows * cols;

std::vector<float> src0(size, 1.0f);
std::vector<float> src1(size, 2.0f);
std::vector<float> src2(size, 3.0f);
std::vector<float> src3(size, 4.0f);
std::vector<float> src4(size, 5.0f);
std::vector<float> dst(size);

// Launch kernel
TAddSubMulDiv::launchTAddSubMulDiv<float, 64, 64, 64, 64>(
    dst.data(), src0.data(), src1.data(), src2.data(), src3.data(), src4.data(), stream);
```

## Performance Benefits

1. **Reduced Memory Access**: 5 GM reads → 2 GM accesses (optimized with intermediate results in UB)
2. **Reduced Kernel Launch Overhead**: 4 kernels → 1 kernel
3. **Improved Data Locality**: Intermediate results kept in UB (Unified Buffer)
4. **Automatic Synchronization**: Event-based synchronization eliminates manual wait/set_flag calls

## Supported Configurations

### Float

```cpp
launchTAddSubMulDiv<float, 64, 64, 64, 64>(...);
```

### aclFloat16

```cpp
launchTAddSubMulDiv<aclFloat16, 16, 256, 16, 256>(...);
```

## Building

```bash
# Add to CMakeLists.txt
pto_add_kernel(taddsubmuldiv)

# Build
cmake --build build
```

## Testing

### NPU Hardware Tests

```bash
python3 tests/script/run_st.py -r npu -v a3 -t taddsubmuldiv
```

**Test Cases**:
- `case_float_64x64_64x64`: Float type, 64x64 tile
- `case_int32_64x64_64x64`: int32_t type, 64x64 tile
- `case_int16_64x64_64x64`: int16_t type, 64x64 tile
- `case_half_16x256_16x256`: Half type, 16x256 tile

### Golden Data Generation

```bash
cd tests/npu/a2a3/src/st/testcase/taddsubmuldiv
python3 gen_data.py
```

## Golden File Structure

Each test case requires to following golden files:

```
TAddSubMulDivTest.TestCaseName/
├── input1.bin      # First input data (src0)
├── input2.bin      # Second input data (src1)
├── input3.bin      # Third input data (src2)
├── input4.bin      # Fourth input data (src3)
├── input5.bin      # Fifth input data (src4)
├── golden.bin      # Expected output result
└── output.bin      # Actual output result (generated during test)
```

## License

CANN Open Software License Agreement Version 2.0