#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"
using namespace pto;

void launch_weighted_sum(uint8_t *out, uint8_t *C, uint8_t *S, uint64_t m, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
}
