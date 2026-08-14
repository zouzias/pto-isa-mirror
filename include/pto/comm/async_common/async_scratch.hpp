/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_ASYNC_COMMON_ASYNC_SCRATCH_HPP
#define PTO_COMM_ASYNC_COMMON_ASYNC_SCRATCH_HPP

#include "pto/comm/async_common/async_types.hpp"

namespace pto {
namespace comm {
namespace detail {

PTO_INTERNAL bool IsValidAsyncTmpBuffer(const AsyncTmpBuffer& tmpBuf)
{
    return tmpBuf.addr != nullptr && tmpBuf.size >= sizeof(uint64_t);
}

template <typename ScratchTile>
PTO_INTERNAL bool MakeAsyncTmpBufferFromTile(ScratchTile& scratchTile, AsyncTmpBuffer& tmpBuf)
{
    static_assert(is_tile_data_v<ScratchTile>, "scratchTile must be a pto::Tile type");
    static_assert(ScratchTile::Loc == TileType::Vec, "scratchTile must be in Vec(UB) memory");
    tmpBuf.addr = reinterpret_cast<__ubuf__ uint8_t*>(scratchTile.data());
    tmpBuf.size = static_cast<uint32_t>(ScratchTile::Numel * sizeof(typename ScratchTile::DType));
    return IsValidAsyncTmpBuffer(tmpBuf);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_ASYNC_COMMON_ASYNC_SCRATCH_HPP
