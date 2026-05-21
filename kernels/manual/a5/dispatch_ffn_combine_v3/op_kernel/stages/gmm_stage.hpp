#ifndef DISPATCH_FFN_COMBINE_V3_STAGES_GMM_STAGE_HPP
#define DISPATCH_FFN_COMBINE_V3_STAGES_GMM_STAGE_HPP

namespace pto_ext::Gemm::Kernel::stages {

template <typename Kernel>
PTO_DEVICE void RunGmm1Stage(Kernel &kernel, typename Kernel::Params const &params)
{
    kernel.RunGmm1Impl(params);
}

template <typename Kernel>
PTO_DEVICE void RunGmmInterlockStage(Kernel &kernel, typename Kernel::Params const &params)
{
    kernel.RunGmmInterlockImpl(params);
}

template <typename Kernel>
PTO_DEVICE void RunGmm2Stage(Kernel &kernel, typename Kernel::Params const &params)
{
    kernel.RunGmm2Impl(params);
}

} // namespace pto_ext::Gemm::Kernel::stages

#endif // DISPATCH_FFN_COMBINE_V3_STAGES_GMM_STAGE_HPP
