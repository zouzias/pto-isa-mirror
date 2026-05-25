#pragma once

#include <cstdint>

#include "task_plan.hpp"

struct PtoTpushTsyncPilotPlan {
    uint32_t activePathEnabled = 0;
    uint32_t requiresFusedCubeVecKernel = 1;
    uint32_t candidateBoundary = 0;
    uint32_t fifoDepth = 2;
    uint32_t syncPeriod = 1;
};

static constexpr uint32_t kPtoPilotBoundaryGmm1ToSwiGlu = 1;
static constexpr uint32_t kPtoPilotBoundarySwiGluToGmm2 = 2;
static constexpr uint32_t kPtoPilotCrossKernelUnsupported = 0;
static constexpr uint32_t kPtoPilotFusedKernelRequired = 1;

inline PtoTpushTsyncPilotPlan BuildGmm1ToSwiGluPilotPlanHost()
{
    PtoTpushTsyncPilotPlan plan{};
    plan.activePathEnabled = kPtoPilotCrossKernelUnsupported;
    plan.requiresFusedCubeVecKernel = kPtoPilotFusedKernelRequired;
    plan.candidateBoundary = kPtoPilotBoundaryGmm1ToSwiGlu;
    plan.fifoDepth = 2;
    plan.syncPeriod = 1;
    return plan;
}

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"

#ifndef V4_PTO_PILOT_INLINE
#define V4_PTO_PILOT_INLINE inline __attribute__((always_inline)) __aicore__
#endif

namespace v2_pto_pilot {

V4_PTO_PILOT_INLINE bool IsActiveTwoKernelPathReplacementAllowed(const PtoTpushTsyncPilotPlan& plan)
{
    return plan.activePathEnabled != 0 && plan.requiresFusedCubeVecKernel == 0;
}

}  // namespace v2_pto_pilot
#endif
