#ifndef DISPATCH_FFN_COMBINE_V3_STAGES_COMBINE_STAGE_HPP
#define DISPATCH_FFN_COMBINE_V3_STAGES_COMBINE_STAGE_HPP

namespace pto_ext::Gemm::Kernel::stages {

template <typename Kernel>
PTO_DEVICE void RunCombineStage(Kernel &kernel, typename Kernel::Params const &params)
{
    kernel.RunCombineImpl(params);
}

} // namespace pto_ext::Gemm::Kernel::stages

#endif // DISPATCH_FFN_COMBINE_V3_STAGES_COMBINE_STAGE_HPP
