# AscendC TADD ST (enqueue/dequeue + copyin/out)

This example mirrors the **WholeReduceSum** “完整样例” pattern (TPipe + TQue + CopyIn/Compute/CopyOut)
using **elementwise Add** instead of reduce.

File:
- `tadd_ascendc_kernel.cpp`

Notes:
- Uses AscendC kernel API (`kernel_operator.h`).
- Assumes **float32** inputs and output.
- Uses fixed shape **64x64** with `rowStride=64` (easy to adapt).
- Repeat/stride are derived from `rows/stride` following AscendC vector semantics.

This is a kernel‑side example. Host launch / build steps depend on your AscendC toolchain.
