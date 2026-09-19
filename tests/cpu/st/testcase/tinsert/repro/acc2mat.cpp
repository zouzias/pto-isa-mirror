/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cstdint>
#include <iostream>
#include <vector>

#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef A_TYPE
#define A_TYPE half
#endif
#ifndef B_TYPE
#define B_TYPE A_TYPE
#endif
#ifndef M_DIM
#define M_DIM 16
#endif

int main()
{
    NPU_MEMORY_INIT(NPUArch::A5);
    using A = A_TYPE;
    using B = B_TYPE;
    constexpr int M = M_DIM, K = 32, N = 32;
    using GlobalA = GlobalTensor<A, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<B, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    std::vector<A> src0(M * K, A(1.0f));
    std::vector<B> src1(K * N, B(1.0f));
    GlobalA src0Global(src0.data());
    GlobalB src1Global(src1.data());
    Tile<TileType::Mat, A, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor> src0Mat;
    Tile<TileType::Mat, B, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor> src1Mat;
    TileLeft<A, M, K, M, K> left;
    TileRight<B, K, N, K, N> right;
    TileAcc<float, M, N, M, N> acc;
    Tile<TileType::Mat, float, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor> dst;
    TASSIGN(src0Mat, 0);
    TASSIGN(src1Mat, 0x10000);
    TASSIGN(left, 0);
    TASSIGN(right, 0);
    TASSIGN(acc, 0);
    TASSIGN(dst, 0);
    TLOAD(src0Mat, src0Global);
    TLOAD(src1Mat, src1Global);
    TMOV(left, src0Mat);
    TMOV(right, src1Mat);
    TMATMUL(acc, left, right);
    TINSERT(dst, acc, 0, 0);
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            if (dst.GetElement(row, col) != K) {
                std::cerr << "Acc-to-Mat mismatch at row=" << row << " col=" << col << '\n';
                return 1;
            }
        }
    }
    std::cout << "TLOAD/TMOV/TMATMUL/TINSERT " << M << "x" << K << "x" << N << " PASS\n";
    return 0;
}
