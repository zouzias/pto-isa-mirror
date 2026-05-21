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
    kernel.RunGmm1Stage(params);
    kernel.RunGmmInterlockStage(params);
    kernel.RunGmm2Stage(params);
}

template <typename Kernel>
PTO_DEVICE void RunAivMain(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunRoutingStage(params);
    kernel.RunDispatchGatherStage(params);
    kernel.RunSwigluStage(params);
    kernel.RunCombineStage(params);
    kernel.RunRestoreStage(params);
}

}  // namespace pto_ext::Gemm::Kernel::stages

#endif  // DISPATCH_FFN_COMBINE_V3_STAGES_KERNEL_CONTEXT_HPP
