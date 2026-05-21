#ifndef DISPATCH_FFN_COMBINE_V3_STAGES_KERNEL_CONTEXT_HPP
#define DISPATCH_FFN_COMBINE_V3_STAGES_KERNEL_CONTEXT_HPP

#include "../utils/hccl_window.hpp"
#include "../utils/layout3d.hpp"

namespace pto_ext::Gemm::Kernel::stages {

template <typename Kernel>
using KernelParams = typename Kernel::Params;

template <typename Kernel>
PTO_DEVICE void RunAicMain(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.DebugStage(0, 0, params);
    kernel.RunGmm1Stage(params);
    kernel.DebugStage(0, 1, params);
    kernel.RunGmmInterlockStage(params);
    kernel.DebugStage(0, 2, params);
    kernel.RunGmm2Stage(params);
    kernel.DebugStage(0, 3, params);
}

template <typename Kernel>
PTO_DEVICE void RunAivMain(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.DebugStage(1, 0, params);
    kernel.RunRoutingStage(params);
    kernel.DebugStage(1, 1, params);
    kernel.RunDispatchGatherStage(params);
    kernel.DebugStage(1, 2, params);
    kernel.RunSwigluStage(params);
    kernel.DebugStage(1, 3, params);
    kernel.RunCombineStage(params);
    kernel.DebugStage(1, 4, params);
    kernel.RunRestoreStage(params);
    kernel.DebugStage(1, 5, params);
}

}  // namespace pto_ext::Gemm::Kernel::stages

#endif  // DISPATCH_FFN_COMBINE_V3_STAGES_KERNEL_CONTEXT_HPP
