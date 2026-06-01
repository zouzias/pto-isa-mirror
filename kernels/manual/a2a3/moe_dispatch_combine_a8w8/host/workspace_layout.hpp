/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_HOST_WORKSPACE_LAYOUT_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_HOST_WORKSPACE_LAYOUT_HPP_

#include "moe_dispatch_combine_a8w8_layout.hpp"

#include <cstdint>
#include <iostream>
#include <string>

namespace moe_dispatch_combine_a8w8 {

inline void PrintField(std::ostream &os, const char *name, const FieldLayout &field)
{
    os << "  " << name << ": offset=" << field.offset << " bytes=" << field.bytes
       << " alignment=" << field.alignment << "\n";
}

inline void PrintWorkspaceLayout(std::ostream &os, const WorkspaceLayout &layout)
{
    os << "[WorkspaceLayout]\n";
    PrintField(os, "tokenPerExpertMatrix", layout.tokenPerExpertMatrix);
    PrintField(os, "blockTokenPerExpert", layout.blockTokenPerExpert);
    PrintField(os, "blockPrefixPerExpert", layout.blockPrefixPerExpert);
    PrintField(os, "expandedRowIdx", layout.expandedRowIdx);
    PrintField(os, "dispatchOffset", layout.dispatchOffset);
    PrintField(os, "cumsumMM", layout.cumsumMM);
    PrintField(os, "preSumBeforeRank", layout.preSumBeforeRank);
    PrintField(os, "expertTokenNums", layout.expertTokenNums);
    PrintField(os, "tokenOwnerRankOffsets", layout.tokenOwnerRankOffsets);
    PrintField(os, "dispatchedA", layout.dispatchedA);
    PrintField(os, "dispatchedScale", layout.dispatchedScale);
    PrintField(os, "gmm1InputInt8", layout.gmm1InputInt8);
    PrintField(os, "routingPerTokenScale", layout.routingPerTokenScale);
    PrintField(os, "gmm1WeightInt8", layout.gmm1WeightInt8);
    PrintField(os, "scale1Uint64", layout.scale1Uint64);
    PrintField(os, "gmm1AccInt32", layout.gmm1AccInt32);
    PrintField(os, "gmm1Out", layout.gmm1Out);
    PrintField(os, "swigluOut", layout.swigluOut);
    PrintField(os, "gmm2InputInt8", layout.gmm2InputInt8);
    PrintField(os, "gmm2PerTokenScale", layout.gmm2PerTokenScale);
    PrintField(os, "gmm2WeightInt8", layout.gmm2WeightInt8);
    PrintField(os, "scale2Uint64", layout.scale2Uint64);
    PrintField(os, "gmm2AccInt32", layout.gmm2AccInt32);
    PrintField(os, "gmm2Out", layout.gmm2Out);
    PrintField(os, "returnSegmentStaging", layout.returnSegmentStaging);
    PrintField(os, "readyCounters", layout.readyCounters);
    PrintField(os, "dispatchGroupReady", layout.dispatchGroupReady);
    PrintField(os, "gmm1SyncGroupReady", layout.gmm1SyncGroupReady);
    PrintField(os, "activationSyncGroupReady", layout.activationSyncGroupReady);
    PrintField(os, "gmm2GroupReady", layout.gmm2GroupReady);
    PrintField(os, "stageStatus", layout.stageStatus);
    PrintField(os, "swigluSyncGroups", layout.swigluSyncGroups);
    PrintField(os, "dequantSum", layout.dequantSum);
    PrintField(os, "swigluGroupDesc", layout.swigluGroupDesc);
    PrintField(os, "gmm1TileTaskPlan", layout.gmm1TileTaskPlan);
    PrintField(os, "gmm2TileTaskPlan", layout.gmm2TileTaskPlan);
    PrintField(os, "timeoutDump", layout.timeoutDump);
    PrintField(os, "subTileReturnPlan", layout.subTileReturnPlan);
    PrintField(os, "subTileOwnerSegments", layout.subTileOwnerSegments);
    PrintField(os, "subTileReady", layout.subTileReady);
    PrintField(os, "timelineScratch", layout.timelineScratch);
    os << "  totalBytes=" << layout.totalBytes << "\n";
}

inline void PrintPeerWindowLayout(std::ostream &os, const PeerWindowLayout &layout)
{
    os << "[PeerWindowLayout]\n";
    PrintField(os, "header", layout.header);
    PrintField(os, "tokenPerExpertMatrix", layout.tokenPerExpertMatrix);
    PrintField(os, "countReadySignal", layout.countReadySignal);
    PrintField(os, "dispatchPayload", layout.dispatchPayload);
    PrintField(os, "dispatchScale", layout.dispatchScale);
    PrintField(os, "returnPayload/offsetD", layout.returnPayload);
    PrintField(os, "combineDoneSignal", layout.combineDoneSignal);
    PrintField(os, "returnSegmentCounters", layout.returnSegmentCounters);
    PrintField(os, "debugCounters", layout.debugCounters);
    PrintField(os, "timeline", layout.timeline);
    os << "  dispatchPayloadRowBytes=" << layout.dispatchPayloadRowBytes << "\n";
    os << "  returnPayloadRowBytes=" << layout.returnPayloadRowBytes << "\n";
    os << "  totalBytes=" << layout.totalBytes << "\n";
}

inline void PrintLayoutDump(std::ostream &os, const ShapeConfig &shape, const RankConfig &rank,
                            const WorkspaceLayout &workspaceLayout, const PeerWindowLayout &peerWindowLayout)
{
    os << "[ShapeConfig]\n";
    os << "  rankNum=" << shape.rankNum << " rankId=" << rank.rankId << " expertPerRank="
       << shape.expertPerRank << " topK=" << shape.topK << " M=" << shape.m << "\n";
    os << "  hiddenSize=" << shape.hiddenSize << " intermediateSize=" << shape.intermediateSize
       << " maxTokensPerExpert=" << shape.maxTokensPerExpert << "\n";
    os << "  payloadTileCols=" << shape.payloadTileCols << " gmmBlockM=" << shape.gmmBlockM
       << " gmmBlockN=" << shape.gmmBlockN << " gmmBlockK=" << shape.gmmBlockK << "\n";
    os << "  dtype_in=" << shape.dtypeIn << " dtype_out=" << shape.dtypeOut << "\n";
    PrintWorkspaceLayout(os, workspaceLayout);
    PrintPeerWindowLayout(os, peerWindowLayout);
}

inline uint64_t EstimateHcclBuffSizeMb(uint64_t peerWindowBytes)
{
    constexpr uint64_t kMiB = 1024ULL * 1024ULL;
    uint64_t withGuard = peerWindowBytes + 64ULL * kMiB;
    return AlignUp(withGuard, kMiB) / kMiB;
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_HOST_WORKSPACE_LAYOUT_HPP_
