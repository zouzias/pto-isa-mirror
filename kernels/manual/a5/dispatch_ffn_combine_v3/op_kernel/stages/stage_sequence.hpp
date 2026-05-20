#ifndef DISPATCH_FFN_COMBINE_V3_STAGES_STAGE_SEQUENCE_HPP
#define DISPATCH_FFN_COMBINE_V3_STAGES_STAGE_SEQUENCE_HPP

namespace pto_ext::Gemm::Kernel::stages {

template <typename Kernel>
using KernelParams = typename Kernel::Params;

template <typename Kernel>
PTO_DEVICE void RunGmm1Stage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunGmm1Impl(params);
}

template <typename Kernel>
PTO_DEVICE void RunGmmInterlockStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunGmmInterlockImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunGmm2Stage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunGmm2Impl(params);
}

template <typename Kernel>
PTO_DEVICE void RunRoutingStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunRoutingImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunDispatchGatherStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunDispatchGatherImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunSwigluStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunSwigluImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunCombineStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunCombineImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunRestoreStage(Kernel &kernel, KernelParams<Kernel> const &params)
{
    kernel.RunRestoreImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunAicMain(Kernel &kernel, KernelParams<Kernel> const &params)
{
    RunGmm1Stage(kernel, params);
    RunGmmInterlockStage(kernel, params);
    RunGmm2Stage(kernel, params);
}

template <typename Kernel>
PTO_DEVICE void RunAivMain(Kernel &kernel, KernelParams<Kernel> const &params)
{
    RunRoutingStage(kernel, params);
    RunDispatchGatherStage(kernel, params);
    RunSwigluStage(kernel, params);
    RunCombineStage(kernel, params);
    RunRestoreStage(kernel, params);
}

}  // namespace pto_ext::Gemm::Kernel::stages

#endif  // DISPATCH_FFN_COMBINE_V3_STAGES_STAGE_SEQUENCE_HPP
