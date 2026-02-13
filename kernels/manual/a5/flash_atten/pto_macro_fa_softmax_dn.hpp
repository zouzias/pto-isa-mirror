/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_FA_SOFTMAX_DN_HPP
#define PTO_MACRO_FA_SOFTMAX_DN_HPP

#include <pto/pto-inst.hpp>

namespace pto {
    
template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_dn_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                            ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                            ReduceTileD1 __out__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                            ReduceTileD1 __out__ exp_max, TileDataS1 __out__ tmp_float,
                                            TileDataS1 __out__ p_tile_f32, TileDataS1 triu, int s0_index, int s1_index)
{
    (void)local_max;
    (void)exp_max;
    (void)local_sum;

    constexpr float scale = constexpr_inv_sqrt(HEAD_SIZE);

    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Cols == s1_index / TileDataS1::Cols) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 1>(triu, 1 + (s0_index % TileDataS1::Cols));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }

    TCOLMAX(new_global_max, input_x);
    TCOLEXPANDSUB(input_x, input_x, new_global_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(new_global_sum, input_x, tmp_float, false);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
}

template <int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2, typename TileDataS1>
AICORE inline void softmax_opt_fa_dn_not_init_impl(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                                ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                                ReduceTileD1 __out__ new_global_max,
                                                ReduceTileD1 __out__ new_global_sum, ReduceTileD1 __out__ exp_max,
                                                TileDataS1 __out__ tmp_float, TileDataS1 __out__ p_tile_f32,
                                                TileDataS1 triu, int s0_index, int s1_index)
{
    constexpr float scale  = constexpr_inv_sqrt(HEAD_SIZE);

    if constexpr (CAUSAL_MASK) {
        if (s0_index / TileDataS1::Cols == s1_index / TileDataS1::Cols) {
            constexpr float negInf = -3.40282e+38;
            TTRI<TileDataS1, 1>(triu, 1 + (s0_index % TileDataS1::Cols));
            TMULS(triu, triu, negInf);
            TADD(input_x, input_x, triu);
        }
    }

    // FA2.0 streaming mode (not first tile): update (global_max, global_sum) and rescale old sums.

    TCOLMAX(local_max, input_x);
    TMAX(local_max, local_max, new_global_max);
    TSUB(exp_max, new_global_max, local_max);
    TMULS(new_global_max, local_max, 1.0f);  //just copy
    TMULS(exp_max, exp_max, scale);
    TEXP(exp_max, exp_max);
    TCOLEXPANDSUB(input_x, input_x, local_max);
    TMULS(input_x, input_x, scale);
    TEXP(input_x, input_x);
    TCOLSUM(local_sum, input_x, tmp_float, false);
    TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
    TMUL(new_global_sum, exp_max, new_global_sum);
    TADD(new_global_sum, new_global_sum, local_sum);
}

template <bool init = false, int HEAD_SIZE, bool CAUSAL_MASK, typename ReduceTileD1, typename TileDataD2,
          typename TileDataS1>
AICORE inline void pto_macro_fa_softmax_dn(TileDataD2 __out__ x_exp, TileDataS1 __in__ input_x,
                                        ReduceTileD1 __out__ local_max, ReduceTileD1 __out__ local_sum,
                                        ReduceTileD1 __in__ new_global_max, ReduceTileD1 __out__ new_global_sum,
                                        ReduceTileD1 __out__ exp_max, TileDataS1 __out__ input_reduce_tmp,
                                        TileDataS1 __out__ p_tile_fp32, TileDataS1 triu, int s0_index, int s1_index)
{
    if (s1_index <= s0_index || !CAUSAL_MASK) {
        if constexpr (init) {
            softmax_opt_fa_dn_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        } else {
            softmax_opt_fa_dn_not_init_impl<HEAD_SIZE, CAUSAL_MASK, ReduceTileD1, TileDataD2, TileDataS1>(
                x_exp, input_x, local_max, local_sum, new_global_max, new_global_sum, exp_max, input_reduce_tmp,
                p_tile_fp32, triu, s0_index, s1_index);
        }
    } else if constexpr (CAUSAL_MASK) {
        TMULS(x_exp, x_exp, 0.0);
        TMULS(exp_max, exp_max, 0.0);
        TADDS(exp_max, exp_max, 1.0);
    }
}

} // namespace pto

#endif