/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef PTO_TEXTRACT_CONVTILE_CHECKS_HPP
#define PTO_TEXTRACT_CONVTILE_CHECKS_HPP

#include <type_traits>

namespace pto {

template <typename DstTileData, typename SrcTileData>
struct TExtractConvTileChecks {
    static_assert(SrcTileData::Loc == pto::TileType::Mat, "Fix: Src TileType must be Mat!");
    static_assert(DstTileData::Loc == pto::TileType::Right, "Fix: Dst TileType must be Right!");
    static_assert(sizeof(typename DstTileData::DType) == sizeof(typename SrcTileData::DType),
                  "Fix: Source dtype must be same with dst dtype!");

    static_assert((SrcTileData::layout == Layout::FRACTAL_Z) || (SrcTileData::layout == Layout::FRACTAL_Z_3D),
                  "TExtract: Source layout only support FRACTAL_Z or FRACTAL_Z_3D.");
    static_assert(DstTileData::SFractal == SLayout::ColMajor && DstTileData::isRowMajor,
                  "TExtract: Destination layout only support SLayout is ColMajor "
                  "ang BLayout is RowMajor.");
    static_assert(std::is_same<typename DstTileData::DType, int8_t>::value ||
                      std::is_same<typename DstTileData::DType, half>::value ||
                      std::is_same<typename DstTileData::DType, bfloat16_t>::value ||
                      std::is_same<typename DstTileData::DType, float>::value,
                  "TExtract: Invalid data type.");

    static constexpr bool validated = true;
};

} // namespace pto

#endif // PTO_TEXTRACT_CONVTILE_CHECKS_HPP
