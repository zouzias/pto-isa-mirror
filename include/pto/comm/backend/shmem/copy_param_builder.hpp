#ifndef PTO_COMM_DETAIL_COPY_PARAM_BUILDER_HPP
#define PTO_COMM_DETAIL_COPY_PARAM_BUILDER_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/common/constants.hpp"
#include "pto/common/debug.h"

namespace pto {
namespace comm {
namespace detail {

template <typename GlobalData>
PTO_INST Copy2DParams BuildParamsNd2nd(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride3,
                                       int validRow, int validCol)
{
    PTO_ASSERT(validCol == gShape4, "TPut/TGet ND: validCol must equal Shape4");
    PTO_ASSERT(validRow == gShape0 * gShape1 * gShape2 * gShape3,
               "TPut/TGet ND: validRow must equal Shape0*Shape1*Shape2*Shape3");
    Copy2DParams p;
    p.repeat = static_cast<uint32_t>(validRow);
    p.lenElems = static_cast<uint32_t>(validCol);
    p.srcStrideElems = static_cast<uint32_t>(gStride3);
    p.dstStrideElems = static_cast<uint32_t>(gStride3);
    return p;
}

template <typename GlobalData>
PTO_INST Copy2DParams BuildParamsDn2dn(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride4,
                                       int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape3, "TPut/TGet DN: validRow must equal Shape3");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape2 * gShape4,
               "TPut/TGet DN: validCol must equal Shape0*Shape1*Shape2*Shape4");
    Copy2DParams p;
    p.repeat = static_cast<uint32_t>(validCol);
    p.lenElems = static_cast<uint32_t>(validRow);
    p.srcStrideElems = static_cast<uint32_t>(gStride4);
    p.dstStrideElems = static_cast<uint32_t>(gStride4);
    return p;
}

template <typename GlobalData>
PTO_INST Copy2DParams BuildParamsNz2nz(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride1,
                                       int validRow, int validCol)
{
    PTO_ASSERT(validRow == gShape2 * gShape3, "TPut/TGet NZ: validRow must equal Shape2*Shape3");
    PTO_ASSERT(validCol == gShape0 * gShape1 * gShape4, "TPut/TGet NZ: validCol must equal Shape0*Shape1*Shape4");
    constexpr uint32_t c0Elems = C0_SIZE_BYTE / sizeof(typename GlobalData::DType);
    Copy2DParams p;
    p.lenElems = static_cast<uint32_t>(validRow) * c0Elems;
    p.repeat = static_cast<uint32_t>(gShape0 * gShape1);
    p.srcStrideElems = static_cast<uint32_t>(gStride1);
    p.dstStrideElems = static_cast<uint32_t>(gStride1);
    return p;
}

template <typename GlobalData>
PTO_INST Copy2DParams BuildCopy2DParams(int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gStride0,
                                        int gStride1, int gStride2, int gStride3, int gStride4, int validRow,
                                        int validCol)
{
    (void)gStride0;
    (void)gStride2;
    if constexpr (GetTileLayoutCustom<GlobalData>() == TileLayoutCustom::ND) {
        return BuildParamsNd2nd<GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, gStride3, validRow, validCol);
    } else if constexpr (GetTileLayoutCustom<GlobalData>() == TileLayoutCustom::DN) {
        return BuildParamsDn2dn<GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, gStride4, validRow, validCol);
    } else { // NZ
        return BuildParamsNz2nz<GlobalData>(gShape0, gShape1, gShape2, gShape3, gShape4, gStride1, validRow, validCol);
    }
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DETAIL_COPY_PARAM_BUILDER_HPP





