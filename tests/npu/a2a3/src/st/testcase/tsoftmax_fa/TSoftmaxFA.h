/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSOFTMAXFA_HPP
#define TSOFTMAXFA_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {

template <typename ReduceTileD1, typename TileDataD1, typename TileDataS1, typename TileDataS2, int init>
    AICORE void TSOFTMAX_DN_NOFUSION(TileDataD1 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             ReduceTileD1 &local_max, ReduceTileD1 &local_sum,
                             ReduceTileD1 &new_global_max, ReduceTileD1 &new_global_sum,
                             ReduceTileD1 &exp_max,
                             TileDataS1 &tmp0,
                             ReduceTileD1 &tmp1,
                             ReduceTileD1 &tmp2) {

            if(!init){
                /*
                TCOLMAX(local_max, input_x);
                TMULS(tmp2, local_max, 0.8f);
                TMAX(tmp1, tmp2, new_global_max);
                TSUB(tmp1, new_global_max, tmp1);
                TEXP(exp_max, tmp1);
                TMULS(input_x, input_x, 0.8f);
                TCOLEXPAND(tmp0, tmp2);
                TSUB(input_x, input_x, tmp0);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp0, false);
                TMULS(input_x, input_x, 1.0f);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                TMUL(new_global_sum, exp_max, new_global_sum);
                TADD(new_global_sum, new_global_sum, local_sum);
                */
                
                TCOLMAX(local_max, input_x);
                TMAX(local_max, local_max, new_global_max);
                TSUB(exp_max, new_global_max, local_max);
                TMULS(new_global_max, local_max, 1.0f);
                TMULS(exp_max, exp_max, 0.8f);
                TEXP(exp_max, exp_max);
                TCOLEXPANDSUB(input_x, input_x, local_max);
                TMULS(input_x, input_x, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp1, false);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                TMUL(new_global_sum, exp_max, new_global_sum);
                TADD(new_global_sum, new_global_sum, local_sum);
                
            }
            else {
                /*
                TCOLMAX(local_max, input_x);
                TCOLEXPAND(tmp0, local_max);
                TSUB(input_x, input_x, tmp0);
                TMULS(input_x, input_x, 0.8f);
                // TMULS(local_max, local_max, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(local_sum, input_x, tmp0, false);
                TCOLSUM(new_global_sum, input_x, tmp0, false);
                TMULS(input_x, input_x, 1.0f);  //KeepProb, compiler cannot optimize
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);
                */
                
                TCOLMAX(new_global_max, input_x);
                TCOLEXPANDSUB(input_x, input_x, new_global_max);
                TMULS(input_x, input_x, 0.8f);
                TEXP(input_x, input_x);
                TCOLSUM(new_global_sum, input_x, tmp1, false);
                TMULS(input_x, input_x, 1.0f);
                TCVT(x_exp, input_x, RoundMode::CAST_ROUND);    //1111 ND output
                
            }

    }

template <typename ReduceTileD1, typename TileDataD1, typename TileDataS1, typename TileDataS2, int init>
    AICORE void TSOFTMAX_ND_NOFUSION(TileDataD1 &x_exp, TileDataS1 &input_x, TileDataS2 &bit_mask, 
                             ReduceTileD1 &local_max, ReduceTileD1 &local_sum,
                             ReduceTileD1 &new_global_max, ReduceTileD1 &new_global_sum,
                             ReduceTileD1 &exp_max,
                             TileDataS1 &tmp_float,
                             TileDataS1 &p_tile_f32) {

            if(!init){
                using ReduceTileD2 = Tile<TileType::Vec, float, 1, ReduceTileD1::Rows, BLayout::RowMajor, 1, ReduceTileD1::Rows>;
                using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                using Tile1D_out = Tile<TileType::Vec, typename TileDataD1::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;

                ReduceTileD2 tmp_shw_local_max;
                ReduceTileD2 tmp_shw_new_global_max;
                ReduceTileD2 tmp_shw_exp_max;
                ReduceTileD2 tmp_shw_new_global_sum;
                ReduceTileD2 tmp_shw_local_sum;
                Tile1D_fp32 p_tile_f32_1d;
                Tile1D_out x_exp_1d;

                TROWMAX(local_max, input_x, tmp_float);
                TRESHAPE(tmp_shw_local_max, local_max);
                TRESHAPE(tmp_shw_new_global_max, new_global_max);
                TMAX(tmp_shw_local_max, tmp_shw_local_max, tmp_shw_new_global_max);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TSUB(tmp_shw_exp_max, tmp_shw_new_global_max, tmp_shw_local_max);
                TMULS(tmp_shw_new_global_max, tmp_shw_local_max, 1.0f); // just copy
                TROWEXPANDSUB(tmp_float, input_x, local_max);
                TMULS(tmp_shw_exp_max, tmp_shw_exp_max, 0.8f);
                TMULS(tmp_float, tmp_float, 0.8f);
                TEXP(tmp_shw_exp_max, tmp_shw_exp_max);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TEXP(p_tile_f32, tmp_float);
                TRESHAPE(tmp_shw_exp_max, exp_max);
                TRESHAPE(p_tile_f32_1d, p_tile_f32);
                TRESHAPE(x_exp_1d, x_exp);    
                TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);  //TODO: check if 1d block the vf fusion?
                TRESHAPE(tmp_shw_new_global_sum, new_global_sum);
                TMUL(tmp_shw_new_global_sum, tmp_shw_exp_max, tmp_shw_new_global_sum);
                TROWSUM(local_sum, p_tile_f32, tmp_float);
                TRESHAPE(tmp_shw_local_sum, local_sum);
                TADD(tmp_shw_new_global_sum, tmp_shw_new_global_sum, tmp_shw_local_sum);

            }
            else {
                using Tile1D_fp32 = Tile<TileType::Vec, float, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                using Tile1D_out = Tile<TileType::Vec, typename TileDataD1::DType, 1, TileDataS1::Rows*TileDataS1::Cols, BLayout::RowMajor, 1, TileDataS1::Rows*TileDataS1::Cols>;
                Tile1D_fp32 p_tile_f32_1d;
                Tile1D_out x_exp_1d;

                TROWMAX(new_global_max, input_x, tmp_float);
                TROWEXPANDSUB(tmp_float, input_x, new_global_max);
                TMULS(tmp_float, tmp_float, 0.8f);
                TEXP(p_tile_f32, tmp_float);
                TROWSUM(new_global_sum, input_x, tmp_float);
                TMULS(input_x, input_x, 1.0f);
                TRESHAPE(p_tile_f32_1d, p_tile_f32);
                TRESHAPE(x_exp_1d, x_exp);
                TCVT(x_exp_1d, p_tile_f32_1d, RoundMode::CAST_ROUND);
            }

    }
}

#endif