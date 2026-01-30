#include <cstdint>
#include "gemm_kernel/aclrtlaunch_gemm_kernel_entry.h"

extern "C" void call_kernel(
    uint32_t blockDim,
    void* stream,
    void* out,
    void* src0,
    void* src1
)
{
    (void)aclrtlaunch_gemm_kernel_entry(blockDim, stream, out, src0, src1);
}