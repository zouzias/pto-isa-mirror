#ifndef PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_GLOBAL_VIEW_HPP
#define PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_GLOBAL_VIEW_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>

namespace pto_ext::dispatch_ffn_combine_v3::pto_bridge {

using PtoShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
using PtoStrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;

template <typename Element>
using PtoGlobalNd = pto::GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, pto::Layout::ND>;

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeContiguousGlobalFromPtr(__gm__ Element *ptr, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeGlobalFromPtr(__gm__ Element *ptr,
                                                 int64_t validRow,
                                                 int64_t validCol,
                                                 int64_t leadingDim)
{
    PtoShapeDyn shape(1, 1, 1, validRow, validCol);
    PtoStrideDyn stride(validRow * leadingDim, validRow * leadingDim, validRow * leadingDim, leadingDim, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeContiguousGlobal(AscendC::GlobalTensor<Element> const &tensor, uint32_t elemNum)
{
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return MakeContiguousGlobalFromPtr(ptr, elemNum);
}

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeGlobal(AscendC::GlobalTensor<Element> const &tensor,
                                          int64_t validRow,
                                          int64_t validCol,
                                          int64_t leadingDim)
{
    auto *ptr = const_cast<__gm__ Element *>(tensor.GetPhyAddr());
    return MakeGlobalFromPtr(ptr, validRow, validCol, leadingDim);
}

}  // namespace pto_ext::dispatch_ffn_combine_v3::pto_bridge

#endif  // PTO_EXT_DISPATCH_FFN_COMBINE_V3_PTO_GLOBAL_VIEW_HPP
