/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

template <typename T, int RowsSrc0, int ColsSrc0, int RowsSrc1, int ColsSrc1, int RowsOut, int ColsOut>
AICORE void runTPARTMUL(__gm__ T __out__* out, __gm__ T __in__* src0, __gm__ T __in__* src1)
{
    static_assert(
        (RowsSrc0 == RowsOut && ColsSrc0 == ColsOut) || (RowsSrc1 == RowsOut && ColsSrc1 == ColsOut),
        "At least one source must match the destination valid region");
    static_assert(RowsSrc0 <= RowsOut && ColsSrc0 <= ColsOut);
    static_assert(RowsSrc1 <= RowsOut && ColsSrc1 <= ColsOut);

    GlobalTensor<T, Shape<1, 1, 1, RowsSrc0, ColsSrc0>, Stride<1, 1, 1, ColsSrc0, 1>> src0Global(src0);
    GlobalTensor<T, Shape<1, 1, 1, RowsSrc1, ColsSrc1>, Stride<1, 1, 1, ColsSrc1, 1>> src1Global(src1);
    GlobalTensor<T, Shape<1, 1, 1, RowsOut, ColsOut>, Stride<1, 1, 1, ColsOut, 1>> outGlobal(out);

    using TileDataSrc0 = Tile<TileType::Vec, T, RowsSrc0, ColsSrc0, BLayout::RowMajor, -1, -1>;

    using TileDataSrc1 = Tile<TileType::Vec, T, RowsSrc1, ColsSrc1, BLayout::RowMajor, -1, -1>;

    using TileDataDst = Tile<TileType::Vec, T, RowsOut, ColsOut, BLayout::RowMajor, -1, -1>;

    TileDataSrc0 src0Tile(RowsSrc0, ColsSrc0);
    TileDataSrc1 src1Tile(RowsSrc1, ColsSrc1);
    TileDataDst outTile(RowsOut, ColsOut);

    TASSIGN(src0Tile, 0);
    TASSIGN(src1Tile, TileDataSrc0::Numel * sizeof(T));
    TASSIGN(outTile, (TileDataSrc0::Numel + TileDataSrc1::Numel) * sizeof(T));

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    TPARTMUL(outTile, src0Tile, src1Tile);
    TSTORE(outGlobal, outTile);
}

template <typename T, int RowsSrc0, int ColsSrc0, int RowsSrc1, int ColsSrc1, int RowsOut, int ColsOut>
void LaunchTPARTMUL(T* out, T* src0, T* src1, void* stream)
{
    (void)stream;
    if constexpr (std::is_same_v<T, aclFloat16>) {
        runTPARTMUL<half, RowsSrc0, ColsSrc0, RowsSrc1, ColsSrc1, RowsOut, ColsOut>(
            reinterpret_cast<half*>(out), reinterpret_cast<half*>(src0), reinterpret_cast<half*>(src1));
    } else {
        runTPARTMUL<T, RowsSrc0, ColsSrc0, RowsSrc1, ColsSrc1, RowsOut, ColsOut>(out, src0, src1);
    }
}

#define INSTANTIATE_TPARTMUL(T, RS0, CS0, RS1, CS1, RO, CO) \
    template void LaunchTPARTMUL<T, RS0, CS0, RS1, CS1, RO, CO>(T * out, T * src0, T * src1, void* stream)

INSTANTIATE_TPARTMUL(float, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(float, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(float, 32, 32, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(float, 64, 64, 32, 64, 64, 64);
INSTANTIATE_TPARTMUL(float, 64, 64, 64, 32, 64, 64);

INSTANTIATE_TPARTMUL(int8_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(int8_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(uint8_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(uint8_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(int16_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(int16_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(uint16_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(uint16_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(int32_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(int32_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(uint32_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(uint32_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(int64_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(int64_t, 64, 64, 32, 32, 64, 64);
INSTANTIATE_TPARTMUL(uint64_t, 64, 64, 64, 64, 64, 64);
INSTANTIATE_TPARTMUL(uint64_t, 64, 64, 32, 32, 64, 64);

INSTANTIATE_TPARTMUL(aclFloat16, 16, 256, 16, 256, 16, 256);

#ifdef CPU_SIM_BFLOAT_ENABLED
INSTANTIATE_TPARTMUL(bfloat16_t, 16, 256, 16, 256, 16, 256);
#endif

#undef INSTANTIATE_TPARTMUL
