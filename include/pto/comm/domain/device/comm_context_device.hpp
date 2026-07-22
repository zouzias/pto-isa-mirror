/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP
#define PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP

// Public device-side API for the PTO communication domain.
// Requires CCE/bisheng (AICORE, __gm__, __ubuf__, get_block_idx).
// Do NOT include from host-only translation units.

#include <cstdint>

#include "pto/comm/async_common/async_types.hpp"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/domain/comm_device_context.hpp"
#include "pto/common/pto_tile.hpp"

namespace pto {
namespace comm {
namespace domain {

/**
 * @brief Map a local symmetric pointer to the same offset on a peer rank.
 *
 * @tparam T  Element type of the pointed-to buffer.
 * @param[in] ctx       Device CommDeviceContext from GetDeviceContext (host).
 * @param[in] localPtr  Pointer into this rank's symmetric window.
 * @param[in] peer      Target rank id.
 * @return Pointer into peer's symmetric window at the same byte offset.
 *
 * @pre Symmetric memory: windowsIn[r] == windowsOut[r] for all ranks r.
 * @pre localPtr lies within this rank's windowsIn[rankId] region.
 */
template <typename T>
AICORE inline __gm__ T* RemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* localPtr, int peer)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(localPtr) - localBase;
    return reinterpret_cast<__gm__ T*>(ctx->windowsIn[peer] + offset);
}

/**
 * @brief Build a GlobalTensor view over a peer's symmetric buffer.
 *
 * Equivalent to RemotePtr followed by constructing GlobalT with the given
 * shape/stride arguments.
 *
 * @tparam GlobalT     GlobalTensor (or compatible) type.
 * @tparam ShapeArgs   Constructor arguments after the data pointer (shape, stride, …).
 * @param[in] ctx       Device CommDeviceContext.
 * @param[in] localPtr  Local buffer used to compute the peer offset.
 * @param[in] peer      Target rank id.
 * @param[in] shapeArgs Forwarded to GlobalT's constructor after the remote pointer.
 * @return GlobalT viewing the peer buffer.
 */
template <typename GlobalT, typename... ShapeArgs>
AICORE inline GlobalT RemoteTensor(
    __gm__ CommDeviceContext* ctx, typename GlobalT::DType* localPtr, int peer, ShapeArgs&&... shapeArgs)
{
    auto* remote = RemotePtr(ctx, localPtr, peer);
    return GlobalT(remote, static_cast<ShapeArgs&&>(shapeArgs)...);
}

/**
 * @brief Build a ParallelGroup covering all ranks for collective instructions.
 *
 * Fills @p tensorsStorage[0 .. rankNum-1] with RemoteTensor views of @p localBuf
 * on each peer, then returns ParallelGroup::Create(...).
 *
 * @tparam GlobalT  GlobalTensor type stored in the group.
 * @tparam ShapeT   Shape type passed to RemoteTensor.
 * @tparam StrideT  Stride type passed to RemoteTensor.
 * @param[in]  ctx             Device CommDeviceContext.
 * @param[out] tensorsStorage  Caller-owned array of size at least ctx->rankNum.
 * @param[in]  localBuf        Local symmetric buffer base for offset calculation.
 * @param[in]  root            Root index for the ParallelGroup (same on all ranks).
 * @param[in]  shape           Tensor shape.
 * @param[in]  stride          Tensor stride.
 * @return ParallelGroup ready for TGather / TScatter / TBroadcast / TReduce, etc.
 */
template <typename GlobalT, typename ShapeT, typename StrideT>
AICORE inline ParallelGroup<GlobalT> MakeParallelGroup(
    __gm__ CommDeviceContext* ctx, GlobalT* tensorsStorage, typename GlobalT::DType* localBuf, int root,
    const ShapeT& shape, const StrideT& stride)
{
    const int nranks = static_cast<int>(ctx->rankNum);
    for (int peer = 0; peer < nranks; ++peer) {
        tensorsStorage[peer] = RemoteTensor<GlobalT>(ctx, localBuf, peer, shape, stride);
    }
    return ParallelGroup<GlobalT>::Create(tensorsStorage, nranks, root);
}

/**
 * @brief Allocate a per-core SDMA sync / event slot id.
 *
 * Uses get_block_idx() % 8 so concurrent cores claim distinct hardware event
 * slots (8 == kSdmaEventSlotCount).
 *
 * @return syncId in [0, 7].
 */
AICORE inline uint32_t AcquireSdmaSyncId() { return static_cast<uint32_t>(get_block_idx()) % 8u; }

/**
 * @brief Bind UB scratch and fill per-core SDMA fields on an AsyncSession.
 *
 * For SDMA: writes tmpBuf from @p scratchTile, syncId via AcquireSdmaSyncId,
 * and channelGroupIdx from get_block_idx() into both exec and event contexts.
 * For URMA: no-op that returns session.valid (peer is supplied at TPUT_ASYNC).
 *
 * Call on a by-value (stack) copy of the session received from host so each
 * core modifies only its own session.
 *
 * @tparam ScratchTile  pto::Tile in Vec (UB) memory used as SDMA staging.
 * @param[in,out] session      AsyncSession from host BuildAsyncSession.
 * @param[in]     scratchTile  UB tile; must hold at least sizeof(uint64_t) bytes.
 * @return true if the session is ready for async DMA; false on invalid engine,
 *         invalid session, or insufficient scratch.
 */
template <typename ScratchTile>
AICORE inline bool ModifyAsyncSession(AsyncSession& session, ScratchTile& scratchTile)
{
    if (session.engine == DmaEngine::URMA) {
        return session.valid;
    }
    if (session.engine != DmaEngine::SDMA || !session.valid) {
        return false;
    }
    static_assert(pto::is_tile_data_v<ScratchTile>, "scratchTile must be a pto::Tile type");
    static_assert(ScratchTile::Loc == pto::TileType::Vec, "scratchTile must be in Vec(UB) memory");

    sdma::TmpBuffer tmpBuf{};
    tmpBuf.addr = reinterpret_cast<__ubuf__ uint8_t*>(scratchTile.data());
    tmpBuf.size = static_cast<uint32_t>(ScratchTile::Numel * sizeof(typename ScratchTile::DType));
    if (tmpBuf.addr == nullptr || tmpBuf.size < sizeof(uint64_t)) {
        return false;
    }

    const uint32_t syncId = AcquireSdmaSyncId();
    const uint32_t channelGroupIdx = static_cast<uint32_t>(get_block_idx());
    session.sdmaSession.execCtx.tmpBuf = tmpBuf;
    session.sdmaSession.execCtx.syncId = syncId;
    session.sdmaSession.execCtx.channelGroupIdx = channelGroupIdx;
    session.sdmaSession.eventCtx.tmpBuf = tmpBuf;
    session.sdmaSession.eventCtx.syncId = syncId;
    session.sdmaSession.valid = true;
    session.valid = true;
    return true;
}

} // namespace domain
} // namespace comm
} // namespace pto

#endif // PTO_COMM_DOMAIN_DEVICE_COMM_CONTEXT_DEVICE_HPP
