#ifndef DISPATCH_FFN_COMBINE_V3_STAGES_ROUTING_STAGE_HPP
#define DISPATCH_FFN_COMBINE_V3_STAGES_ROUTING_STAGE_HPP

namespace pto_ext::Gemm::Kernel::stages {

template <typename Kernel>
PTO_DEVICE void RunRoutingStage(Kernel &kernel, typename Kernel::Params const &params)
{
    kernel.RunRoutingImpl(params);
}

} // namespace pto_ext::Gemm::Kernel::stages

#endif // DISPATCH_FFN_COMBINE_V3_STAGES_ROUTING_STAGE_HPP
