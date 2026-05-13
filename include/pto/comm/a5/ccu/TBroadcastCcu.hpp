/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_COMM_A5_CCU_TBROADCAST_CCU_HPP
#define PTO_COMM_A5_CCU_TBROADCAST_CCU_HPP

#include "pto/comm/comm_types.hpp"
#include "pto/comm/a5/ccu/ccu_trigger_pto.hpp"

namespace pto {
namespace comm {
namespace detail {

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INTERNAL void TbroadcastCcuDispatch(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                        TileData &stagingTileData, const CcuTriggerContext &ctx,
                                        WaitEvents &... events)
{
    WaitAllEvents(events...);
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(stagingTileData.data());
    pto::comm::ccu::CkeTrigger(ctx.ckeSlotVA, ctx.mask, ub);
}

template <typename ParallelGroupType, typename GlobalSrcData, typename TileData, typename... WaitEvents>
PTO_INTERNAL void TbroadcastCcuDispatchPP(ParallelGroupType &parallelGroup, GlobalSrcData &srcGlobalData,
                                          TileData &pingTile, TileData &pongTile,
                                          const CcuTriggerContext &ctx, WaitEvents &... events)
{
    WaitAllEvents(events...);
    __ubuf__ uint8_t *ub = reinterpret_cast<__ubuf__ uint8_t *>(pingTile.data());
    pto::comm::ccu::CkeTrigger(ctx.ckeSlotVA, ctx.mask, ub);
}

} // namespace detail
} // namespace comm
} // namespace pto

#endif // PTO_COMM_A5_CCU_TBROADCAST_CCU_HPP
