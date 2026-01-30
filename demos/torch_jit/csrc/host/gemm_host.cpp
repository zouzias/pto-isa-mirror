#include <cstdint>
#include "gemm_kernel/aclrtlaunch_gemm_kernel_entry.h"

extern "C" void call_kernel(
    uint32_t blockDim,
    void* stream,
    void* x,
    void* y,
    void* z,
    int /*N*/)
{
    // out=z, src0=x, src1=y
    (void)aclrtlaunch_gemm_kernel_entry(blockDim, stream, z, x, y);
}