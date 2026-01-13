/*
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MACRO_MATMUL_HPP
#define PTO_MACRO_MATMUL_HPP

#include <pto/pto-inst.hpp>

namespace pto{

    /**
     * Layout type for matrix multiplication operations.
     * First letter represents the layout of matrix A, second letter represents matrix B.
     * N = Normal (Row-major), T = Transposed (Column-major)
     */
    enum class layout_t {
        NN,  // Matrix A: Normal, Matrix B: Normal
        NT,  // Matrix A: Normal, Matrix B: Transposed
        TN,  // Matrix A: Transposed, Matrix B: Normal
        TT,   // Matrix A: Transposed, Matrix B: Transposed
        NONE
    };

    #define L0A_BUF0 ((__ca__ half*) (__ca__ char*) 0x0)
    #define L0A_BUF1 ((__ca__ half*) (__ca__ char*) 0x8000)
    #define L0B_BUF0 ((__ca__ half*) (__ca__ char*) 0x0)
    #define L0B_BUF1 ((__ca__ half*) (__ca__ char*) 0x8000)
    #define L0C_BUF0 ((__ca__ half*) (__ca__ char*) 0x0)
    #define L0C_BUF1 ((__ca__ half*) (__ca__ char*) 0x20000)

    #define LAST_LOOP(x, n) ((x) == ((n)-1))
    #define UNIT_FLAG_ENABLE(i, n) (LAST_LOOP(i, n) ? 3 : 2)


    // Memory constraints
    constexpr uint32_t MEM_BUFFER_SIZE_BYTES = 64 * 1024/2;  // 64KB per buffer with pingpong (32KB)
    constexpr uint32_t HALF_SIZE_BYTES = 2;                // sizeof(half) = 2 bytes
    constexpr uint32_t CANDIDATE_CUBE_K[] = {32, 64, 128, 256};
    constexpr uint32_t NUM_CANDIDATES = 4;

    /**
    * Calculate the largest Cube_K value that fits in the 64KB memory buffer.
    * Checks if both Cube_M * Cube_K (left matrix) and Cube_K * Cube_N (right matrix) 
    * can fit within the 64KB buffer.
    * 
    * @param Cube_M - The tile dimension M
    * @param Cube_N - The tile dimension N
    * @return - Largest Cube_K value (32, 64, 128, or 256) that fits in memory
    */
    [aicore] inline constexpr uint32_t calculateFittingCubeK(uint32_t Cube_M, uint32_t Cube_N) {
        uint32_t bestCubeK = 32;  // Default to smallest value
        
        // Test candidates from largest to smallest to find the largest that fits
        if (Cube_M * 256 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES && 
            256 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
            bestCubeK = 256;
        } else if (Cube_M * 128 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES && 
                128 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
            bestCubeK = 128;
        } else if (Cube_M * 64 * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES && 
                64 * Cube_N * HALF_SIZE_BYTES <= MEM_BUFFER_SIZE_BYTES) {
            bestCubeK = 64;
        }
        
        return bestCubeK;
    }


    // Deduce layout_t from SLayouts
    template<typename TileDataA, typename TileDataB>
    [aicore] inline constexpr layout_t deduce_layout() {
        if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::RowMajor) return layout_t::NN;
        if constexpr (TileDataA::SFractal == SLayout::RowMajor && TileDataB::SFractal == SLayout::ColMajor) return layout_t::NT;
        if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::RowMajor) return layout_t::TN;
        if constexpr (TileDataA::SFractal == SLayout::ColMajor && TileDataB::SFractal == SLayout::ColMajor) return layout_t::TT;
        return layout_t::NONE;
    }

    template <unsigned Cube_M, unsigned Tile_K, unsigned Cube_N, layout_t LAYOUT = layout_t::NONE, typename TileDataA,
        typename TileDataB, typename TileDataC>
    [aicore] inline RecordEvent pto_macro_matmul(TileDataA &aMatTile, TileDataB &bMatTile, TileDataC &cAccTile,
        bool accumulate = false) {
        constexpr layout_t layout = deduce_layout<TileDataA, TileDataB>();

        static_assert(layout != layout_t::NONE, "Deduced layout is NONE, check tile SLayouts");
        if constexpr (LAYOUT != layout_t::NONE) {
            static_assert(LAYOUT == layout,
                "Layout mismatch: template LAYOUT does not match deduced layout from tile SLayouts.");
        }

        const uint64_t Cube_K = calculateFittingCubeK(Cube_M, Cube_N);
        static_assert(Tile_K % Cube_K == 0, "Tile_K must be divisible by chosen Cube_K");

        // Fast path: single K-slice matmul needs no ping-pong dependency events.
        // This also avoids leaving unmatched TMATMUL->TEXTRACT events "recorded" without a corresponding wait,
        // which can exhaust limited hardware event resources over multiple tiles on real NPU hardware.
        if constexpr (Tile_K == Cube_K) {
            using LeftTile = TileLeft<half, Cube_M, Cube_K, Cube_M, Cube_K>;
            using RightTile = TileRight<half, Cube_K, Cube_N, Cube_K, Cube_N>;
            LeftTile al0;
            RightTile bl0;

            TASSIGN(al0, (uint64_t)L0A_BUF0);
            TASSIGN(bl0, (uint64_t)L0B_BUF0);

            // For NT layout, the input tiles may represent a larger K dimension; K-slice offset is 0 here.
            if (layout == layout_t::NT) {
                TASSIGN(aMatTile, (uint64_t)aMatTile.data());
                TASSIGN(bMatTile, (uint64_t)bMatTile.data());
            }

            TEXTRACT(al0, aMatTile, 0, 0);
            auto recordExtractB = TEXTRACT(bl0, bMatTile, 0, 0);
            Event<Op::TEXTRACT_M2LR, Op::TMATMUL> evExtractToMatmul;
            evExtractToMatmul = recordExtractB;

            if (!accumulate) {
                return TMATMUL(cAccTile, al0, bl0, evExtractToMatmul);
            }
            return TMATMUL_ACC(cAccTile, cAccTile, al0, bl0, evExtractToMatmul);
        }

        uint64_t pingpong = 0;
        bool matmulRecorded[2] = {false, false};
        Event<Op::TMATMUL, Op::TEXTRACT_M2LR> evMatmulToExtract[2];

        RecordEvent lastMatmul = {};
        for (uint64_t k = 0; k < (static_cast<uint64_t>(Tile_K) / Cube_K); k++) {
            using LeftTile = TileLeft<half, Cube_M, Cube_K, Cube_M, Cube_K>;
            using RightTile = TileRight<half, Cube_K, Cube_N, Cube_K, Cube_N>;
            LeftTile al0Tiles[2] = {LeftTile(), LeftTile()};
            RightTile bl0Tiles[2] = {RightTile(), RightTile()};

            TASSIGN(al0Tiles[0], (uint64_t)L0A_BUF0);
            TASSIGN(al0Tiles[1], (uint64_t)L0A_BUF1);
            TASSIGN(bl0Tiles[0], (uint64_t)L0B_BUF0);
            TASSIGN(bl0Tiles[1], (uint64_t)L0B_BUF1);

            if (layout == layout_t::NT) {
                TASSIGN(aMatTile,
                    (uint64_t)aMatTile.data() + k * Cube_K * Cube_M * sizeof(typename TileDataA::DType));
                TASSIGN(bMatTile,
                    (uint64_t)bMatTile.data() + k * Cube_K * Cube_N * sizeof(typename TileDataB::DType));
            }

            if (matmulRecorded[pingpong]) {
                TEXTRACT(al0Tiles[pingpong], aMatTile, 0, 0, evMatmulToExtract[pingpong]);
            } else {
                TEXTRACT(al0Tiles[pingpong], aMatTile, 0, 0);
            }
            auto recordExtractB = TEXTRACT(bl0Tiles[pingpong], bMatTile, 0, 0);
            Event<Op::TEXTRACT_M2LR, Op::TMATMUL> evExtractToMatmul;
            evExtractToMatmul = recordExtractB;

            if (k == 0 && !accumulate) {
                lastMatmul = TMATMUL(cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong], evExtractToMatmul);
            } else {
                lastMatmul =
                    TMATMUL_ACC(cAccTile, cAccTile, al0Tiles[pingpong], bl0Tiles[pingpong], evExtractToMatmul);
            }

            evMatmulToExtract[pingpong] = lastMatmul;
            matmulRecorded[pingpong] = true;
            pingpong ^= 1;
        }

        // Drain any outstanding TMATMUL->TEXTRACT flags that were recorded but won't be consumed
        // by a subsequent TEXTRACT within this call.
        if (matmulRecorded[0]) {
            TSYNC(evMatmulToExtract[0]);
        }
        if (matmulRecorded[1]) {
            TSYNC(evMatmulToExtract[1]);
        }

        return lastMatmul;
    }

}

#endif // PTO_MACRO_MATMUL_H
