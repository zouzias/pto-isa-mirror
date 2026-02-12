/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_GU_HPP
#define PTO_MACRO_FA_GU_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/pto_tile.hpp>
#include <type_traits>

namespace pto {

#ifndef FA4_SKIP_RESCALE_ENABLE
#define FA4_SKIP_RESCALE_ENABLE 1
#endif

#ifndef FA4_SKIP_RESCALE_EPS
#define FA4_SKIP_RESCALE_EPS 1.0e-5f
#endif

#ifndef FA4_SKIP_RESCALE_FORCE
#define FA4_SKIP_RESCALE_FORCE 0
#endif

template <typename ReduceTileData>
AICORE inline bool fa4_should_skip_rescale(ReduceTileData __in__ exp_max, float eps)
{
    using T = typename ReduceTileData::DType;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                  "fa4_should_skip_rescale only supports float/half reduce tiles.");

    __ubuf__ const T *ptr = (__ubuf__ const T *)exp_max.data();
    constexpr uint32_t rows = ReduceTileData::Rows;
    constexpr uint32_t stride = ReduceTileData::RowStride;

    for (uint32_t r = 0; r < rows; ++r) {
        const float scale = static_cast<float>(ptr[r * stride]);
        float diff = scale - 1.0f;
        if (diff < 0.0f) {
            diff = -diff;
        }
        if (diff > eps) {
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// FlashAttention "GU" (running update) macro
//
// This implements the numerically-stable streaming update:
//   O = O * exp(max_prev - max_new) + PV_tile
// and on the last tile:
//   O = O / global_sum
//
// Performance notes:
// - Keep O resident in UB across tiles to avoid extra TLOAD/TSTORE.
// - exp_max and global_sum are per-row reduced tiles (shape [S0, 1]) that get broadcast over columns.
// -----------------------------------------------------------------------------

template <typename reducedTileData, typename svTileData>
AICORE inline bool pto_macro_fa_gu(svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile,
                                   reducedTileData __in__ exp_max)
{
    bool skip_rescale = false;
    if constexpr (FA4_SKIP_RESCALE_FORCE) {
        skip_rescale = true;
    } else if constexpr (FA4_SKIP_RESCALE_ENABLE) {
        skip_rescale = fa4_should_skip_rescale(exp_max, FA4_SKIP_RESCALE_EPS);
        if (!skip_rescale) {
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
        }
    } else {
        pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
    }
    pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
    return skip_rescale;
}

template <typename reducedTileData, typename svTileData>
AICORE inline bool pto_macro_fa_gu_last(svTileData __out__ prev_sv_tile, svTileData __in__ est_sv_tile,
                                        reducedTileData __in__ exp_max, reducedTileData __in__ new_global_sum)
{
    bool skip_rescale = false;
    if constexpr (FA4_SKIP_RESCALE_FORCE) {
        skip_rescale = true;
    } else if constexpr (FA4_SKIP_RESCALE_ENABLE) {
        skip_rescale = fa4_should_skip_rescale(exp_max, FA4_SKIP_RESCALE_EPS);
        if (!skip_rescale) {
            pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
        }
    } else {
        pto::TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max);
    }
    pto::TADD(prev_sv_tile, prev_sv_tile, est_sv_tile);
    pto::TROWEXPANDDIV(prev_sv_tile, prev_sv_tile, new_global_sum);
    // pto::TCVT(prev_sv_nd_tile, prev_sv_tile, RoundMode::CAST_RINT);
    return skip_rescale;
}

template <typename reducedTileData, typename svTileData>
AICORE inline void pto_macro_fa_gu_single_and_last_tile(svTileData __out__ sv_tile,
                                                        reducedTileData __in__ new_global_sum)
{
    pto::TROWEXPANDDIV(sv_tile, sv_tile, new_global_sum);
}

} // namespace pto
#endif // TGU_PTO_H
