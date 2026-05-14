/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef PTO_COMM_A5_TBROADCAST_HPP
#define PTO_COMM_A5_TBROADCAST_HPP

// AIV path - shared with a2a3, forwarded to avoid code duplication.
#include "pto/comm/a2a3/TBroadCast.hpp"

// CCU path - A5-only implementation used by TBROADCAST<CollEngine::CCU>.
#include "pto/comm/comm_types.hpp"
#include "pto/comm/async_common/ccu_trigger.hpp"

namespace pto {
namespace comm {

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INTERNAL void TBROADCAST_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                      TileData &stagingTileData, const CcuTriggerContext &ctx,
                                      WaitEvents &... events)
{
    WaitAllEvents(events...);
    pto::comm::ccu::CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, stagingTileData);
}

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INTERNAL void TBROADCAST_CCU_IMPL(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                      TileData &pingTile, TileData &pongTile,
                                      const CcuTriggerContext &ctx, WaitEvents &... events)
{
    WaitAllEvents(events...);
    pto::comm::ccu::CkeTriggerFromTile(ctx.ckeSlotVA, ctx.mask, pingTile);
}

} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_TBROADCAST_HPP
