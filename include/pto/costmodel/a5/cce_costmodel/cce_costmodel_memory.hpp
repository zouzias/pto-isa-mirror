/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#pragma once

namespace pto {

// GM transfers are represented by the enclosing PTO instruction record. The
// kernel body is executed only to collect costmodel metadata, so these
// implementations intentionally perform no Host memory access and emit no VF
// instruction. RecordInstr assigns the transfer to the appropriate perf_sim
// pipe and supplies its cycle estimate exactly once.
template <typename TileData, typename GlobalData>
inline void TLOAD_IMPL(TileData&, GlobalData&)
{}

template <typename TileData, typename GlobalData>
inline void TPREFETCH_IMPL(TileData&, GlobalData&)
{}

template <typename TileData, typename GlobalData, AtomicType = AtomicType::AtomicNone, STPhase = STPhase::Unspecified>
inline void TSTORE_IMPL(GlobalData&, TileData&)
{}

template <
    typename TileData, typename GlobalData, AtomicType = AtomicType::AtomicNone, ReluPreMode,
    STPhase = STPhase::Unspecified>
inline void TSTORE_IMPL(GlobalData&, TileData&)
{}

template <
    typename TileData, typename GlobalData, AtomicType = AtomicType::AtomicNone, ReluPreMode = ReluPreMode::NoRelu,
    STPhase = STPhase::Unspecified>
inline void TSTORE_IMPL(GlobalData&, TileData&, uint64_t)
{}

template <
    typename TileData, typename GlobalData, typename FpTileData, AtomicType = AtomicType::AtomicNone,
    ReluPreMode = ReluPreMode::NoRelu, STPhase = STPhase::Unspecified>
inline void TSTORE_IMPL(GlobalData&, TileData&, FpTileData&)
{}

} // namespace pto
