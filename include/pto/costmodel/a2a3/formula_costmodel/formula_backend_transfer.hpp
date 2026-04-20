/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_FORMULA_BACKEND_TRANSFER_HPP
#define PTO_MOCKER_FORMULA_BACKEND_TRANSFER_HPP

#include <cstdint>

#include <pto/costmodel/arch_config.hpp>

namespace pto::mocker::fit {

enum class TransferOp : uint8_t {
    TLoad,
    TStore,
    TMov,
};

enum class TransferTileType : uint8_t {
    Unknown,
    VecTile,
    MatTile,
    AccTile,
    LeftTile,
    RightTile,
    BiasTile,
    ScalingTile,
};

inline bool TryResolveTransferPipe(TransferOp op, TransferTileType tile_type, evaluator::PipeKey &pipe)
{
    switch (op) {
        case TransferOp::TLoad:
            if (tile_type == TransferTileType::VecTile) {
                pipe = evaluator::PipeKey::GM_TO_UB;
                return true;
            }
            if (tile_type == TransferTileType::MatTile) {
                pipe = evaluator::PipeKey::GM_TO_L1;
                return true;
            }
            return false;
        case TransferOp::TStore:
            if (tile_type == TransferTileType::VecTile) {
                pipe = evaluator::PipeKey::UB_TO_GM;
                return true;
            }
            if (tile_type == TransferTileType::MatTile) {
                pipe = evaluator::PipeKey::L1_TO_GM;
                return true;
            }
            if (tile_type == TransferTileType::AccTile) {
                pipe = evaluator::PipeKey::L0C_TO_GM;
                return true;
            }
            return false;
        case TransferOp::TMov:
            if (tile_type == TransferTileType::VecTile) {
                pipe = evaluator::PipeKey::UB_TO_UB;
                return true;
            }
            if (tile_type == TransferTileType::MatTile) {
                pipe = evaluator::PipeKey::L0C_TO_L1;
                return true;
            }
            if (tile_type == TransferTileType::LeftTile) {
                pipe = evaluator::PipeKey::L1_TO_L0A;
                return true;
            }
            if (tile_type == TransferTileType::RightTile) {
                pipe = evaluator::PipeKey::L1_TO_L0B;
                return true;
            }
            if (tile_type == TransferTileType::BiasTile) {
                pipe = evaluator::PipeKey::L1_TO_BT;
                return true;
            }
            if (tile_type == TransferTileType::ScalingTile) {
                pipe = evaluator::PipeKey::L1_TO_FB;
                return true;
            }
            return false;
    }
    return false;
}

inline bool TryEstimateTransferLatencyUs(TransferOp op, TransferTileType tile_type, uint64_t bytes,
                                         const evaluator::BandwidthTable &bandwidth, long double &latency_us)
{
    evaluator::PipeKey pipe = evaluator::PipeKey::COUNT;
    if (!TryResolveTransferPipe(op, tile_type, pipe)) {
        return false;
    }
    const double bw = bandwidth[pipe];
    if (bw <= 0.0) {
        return false;
    }
    latency_us = evaluator::TransferBytesToUs(bytes, bw);
    return true;
}

} // namespace pto::mocker::fit

#endif
