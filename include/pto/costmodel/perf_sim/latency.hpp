/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_PERF_SIM_LATENCY_HPP
#define PTO_PERF_SIM_LATENCY_HPP

#include "recorder.hpp"
#include <pto/common/type.hpp>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace pto::perf_sim {

// ── Compile-time TileType extraction (returns Ctrl for non-tiles) ──

template <typename T>
constexpr pto::TileType SafeGetTileLoc()
{
    if constexpr (requires { T::Loc; })
        return T::Loc;
    else
        return pto::TileType::Ctrl;
}

// ── Static opcode → PipeStage lookup (for non-polymorphic instructions) ──

inline PipeStage StaticPipeStageLookup(const std::string &opcode)
{
    static const std::unordered_map<std::string, PipeStage> lut = {
        {"TSYNC", PipeStage::Scalar},
        // Matrix ops
        {"TMATMUL", PipeStage::Matrix},
        {"TMATMUL_ACC", PipeStage::Matrix},
        {"TMATMUL_BIAS", PipeStage::Matrix},
        {"TGEMV", PipeStage::Matrix},
        {"TGEMV_ACC", PipeStage::Matrix},
        {"TGEMV_BIAS", PipeStage::Matrix},
        // Cube-side data transform (L1→L0 im2col)
        {"TIMG2COL", PipeStage::MTE1},
        // Vector ops (default)
        {"TLOAD", PipeStage::MTE2_AIV},     // TLOAD default; ResolvePipeStage overrides by tile type
        {"TPREFETCH", PipeStage::MTE2_AIV}, // TPREFETCH: GM→UB only
        {"TSTORE", PipeStage::MTE3},
        {"TADD", PipeStage::Vector},
        {"TADDS", PipeStage::Vector},
        {"TSUB", PipeStage::Vector},
        {"TSUBS", PipeStage::Vector},
        {"TMUL", PipeStage::Vector},
        {"TMULS", PipeStage::Vector},
        {"TDIV", PipeStage::Vector},
        {"TDIVS", PipeStage::Vector},
        {"TEXP", PipeStage::Vector},
        {"TLOG", PipeStage::Vector},
        {"TSQRT", PipeStage::Vector},
        {"TRSQRT", PipeStage::Vector},
        {"TABS", PipeStage::Vector},
        {"TNEG", PipeStage::Vector},
        {"TRELU", PipeStage::Vector},
        {"TLRELU", PipeStage::Vector},
        {"TPRELU", PipeStage::Vector},
        {"TCAST", PipeStage::Vector},
        {"TCVT", PipeStage::Vector},
        {"TNOT", PipeStage::Vector},
        {"TAND", PipeStage::Vector},
        {"TOR", PipeStage::Vector},
        {"TXOR", PipeStage::Vector},
        {"TANDS", PipeStage::Vector},
        {"TORS", PipeStage::Vector},
        {"TXORS", PipeStage::Vector},
        {"TSHL", PipeStage::Vector},
        {"TSHR", PipeStage::Vector},
        {"TSHLS", PipeStage::Vector},
        {"TSHRS", PipeStage::Vector},
        {"TCMP", PipeStage::Vector},
        {"TCMPS", PipeStage::Vector},
        {"TMIN", PipeStage::Vector},
        {"TMINS", PipeStage::Vector},
        {"TMAX", PipeStage::Vector},
        {"TMAXS", PipeStage::Vector},
        {"TROWMAX", PipeStage::Vector},
        {"TROWMIN", PipeStage::Vector},
        {"TROWARGMAX", PipeStage::Vector},
        {"TROWARGMIN", PipeStage::Vector},
        {"TROWSUM", PipeStage::Vector},
        {"TROWPROD", PipeStage::Vector},
        {"TCOLSUM", PipeStage::Vector},
        {"TCOLPROD", PipeStage::Vector},
        {"TCOLMAX", PipeStage::Vector},
        {"TCOLMIN", PipeStage::Vector},
        {"TROWEXPAND", PipeStage::Vector},
        {"TROWEXPANDDIV", PipeStage::Vector},
        {"TROWEXPANDMUL", PipeStage::Vector},
        {"TROWEXPANDSUB", PipeStage::Vector},
        {"TROWEXPANDADD", PipeStage::Vector},
        {"TROWEXPANDMAX", PipeStage::Vector},
        {"TROWEXPANDMIN", PipeStage::Vector},
        {"TROWEXPANDEXPDIF", PipeStage::Vector},
        {"TCOLEXPAND", PipeStage::Vector},
        {"TCOLEXPANDDIV", PipeStage::Vector},
        {"TCOLEXPANDMUL", PipeStage::Vector},
        {"TCOLEXPANDSUB", PipeStage::Vector},
        {"TCOLEXPANDADD", PipeStage::Vector},
        {"TCOLEXPANDMAX", PipeStage::Vector},
        {"TCOLEXPANDMIN", PipeStage::Vector},
        {"TCOLEXPANDEXPDIF", PipeStage::Vector},
        {"TREMS", PipeStage::Vector},
        {"TFMODS", PipeStage::Vector},
        {"TREM", PipeStage::Vector},
        {"TFMOD", PipeStage::Vector},
        {"TCONCAT", PipeStage::Vector},
        {"TINSERT", PipeStage::Vector},
        {"TSEL", PipeStage::Vector},
        {"TSELS", PipeStage::Vector},
        {"TGATHER", PipeStage::Vector},
        {"TGATHERB", PipeStage::Vector},
        {"TSCATTER", PipeStage::Vector},
        {"TPARTADD", PipeStage::Vector},
        {"TPARTMUL", PipeStage::Vector},
        {"TPARTMAX", PipeStage::Vector},
        {"TPARTMIN", PipeStage::Vector},
        {"TSORT32", PipeStage::Vector},
        {"TMRGSORT", PipeStage::Vector},
        {"TAXPY", PipeStage::Vector},
        {"TADDC", PipeStage::Vector},
        {"TSUBC", PipeStage::Vector},
        {"TADDSC", PipeStage::Vector},
        {"TSUBSC", PipeStage::Vector},
        {"TDEQUANT", PipeStage::Vector},
        {"TEXPANDS", PipeStage::Vector},
        {"TFILLPAD_INPLACE", PipeStage::Vector},
        {"TFILLPAD_EXPAND", PipeStage::Vector},
        {"TRESHAPE", PipeStage::Scalar},
        {"TASSIGN", PipeStage::Scalar},
        {"TPRINT", PipeStage::Scalar},
        {"TGET_SCALE_ADDR", PipeStage::Scalar},
        {"TSUBVIEW", PipeStage::Scalar},
        {"TALLOC", PipeStage::Scalar},
        {"TFREE", PipeStage::Scalar},
        {"TPUSH", PipeStage::Scalar},
        {"TPOP", PipeStage::Scalar},
        {"MGATHER", PipeStage::Vector},
        {"MSCATTER", PipeStage::Vector},
    };
    auto it = lut.find(opcode);
    return it != lut.end() ? it->second : PipeStage::Vector;
}

// ── Dynamic resolution: opcode + TileTypes → PipeStage ──

inline PipeStage ResolvePipeStage(const std::string &opcode, pto::TileType dst_loc, pto::TileType src_loc)
{
    // TLOAD: GM/L2 → tile. dst tile determines MTE2_AIV (→UB) or MTE2_AIC (→L1)
    if (opcode == "TLOAD") {
        return (dst_loc == pto::TileType::Vec) ? PipeStage::MTE2_AIV : PipeStage::MTE2_AIC;
    }
    // TPREFETCH: always GM→UB (AIV side)
    if (opcode == "TPREFETCH") {
        return PipeStage::MTE2_AIV;
    }
    // TSTORE: tile → GM/L2. src tile determines component:
    //   Vec(UB)→GM: MTE3, Mat(L1)→GM: MTE3, Acc(L0C)→GM: Fixpipe
    if (opcode == "TSTORE") {
        if (src_loc == pto::TileType::Acc)
            return PipeStage::Fixpipe;
        return PipeStage::MTE3;
    }
    // TMOV: on-chip transfer, routing depends on dst+src buffer types
    //   dst=Vec(UB): src=Vec→Vector, src=Mat(L1)→MTE1, src=Acc(L0C)→Fixpipe
    //   dst=Mat(L1): src=Acc(L0C)→Fixpipe, src=Mat→MTE1, src=Vec(UB)→MTE3
    if (opcode == "TMOV") {
        if (dst_loc == pto::TileType::Vec) {
            if (src_loc == pto::TileType::Acc)
                return PipeStage::Fixpipe;
            if (src_loc == pto::TileType::Mat)
                return PipeStage::MTE1;
            return PipeStage::Vector; // Vec→Vec
        }
        // dst is L1-side (Mat, Left, Right, Bias, Scaling)
        if (src_loc == pto::TileType::Acc)
            return PipeStage::Fixpipe; // L0C→L1
        if (src_loc == pto::TileType::Vec)
            return PipeStage::MTE3; // UB→L1
        return PipeStage::MTE1;     // L1→L1 or similar on-chip
    }
    // TEXTRACT: sub-tile extract, same routing as TMOV (data movement
    // depends on src/dst buffer types)
    //   e.g. L1→L0A/L0B = MTE1, L0C→UB = Fixpipe, UB→UB = Vector
    if (opcode == "TEXTRACT") {
        if (dst_loc == pto::TileType::Vec) {
            if (src_loc == pto::TileType::Acc)
                return PipeStage::Fixpipe;
            if (src_loc == pto::TileType::Mat)
                return PipeStage::MTE1;
            return PipeStage::Vector;
        }
        if (src_loc == pto::TileType::Acc)
            return PipeStage::Fixpipe;
        if (src_loc == pto::TileType::Vec)
            return PipeStage::MTE3;
        return PipeStage::MTE1;
    }
    // TTRANS: L1→L1 transpose → MTE1
    if (opcode == "TTRANS")
        return PipeStage::MTE1;

    return StaticPipeStageLookup(opcode);
}

// ── Variadic dispatch: extract TileTypes from macro arguments ──

inline PipeStage ResolvePipeStageArgs(const char *opcode)
{
    return ResolvePipeStage(opcode, pto::TileType::Ctrl, pto::TileType::Ctrl);
}

template <typename First, typename... Rest>
inline PipeStage ResolvePipeStageArgs(const char *opcode, First &, const Rest &...)
{
    using DstType = std::remove_cv_t<std::remove_reference_t<First>>;
    constexpr auto dst_loc = SafeGetTileLoc<DstType>();
    if constexpr (sizeof...(Rest) > 0) {
        using SrcType = std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<0, std::tuple<Rest...>>>>;
        constexpr auto src_loc = SafeGetTileLoc<SrcType>();
        return ResolvePipeStage(opcode, dst_loc, src_loc);
    } else {
        return ResolvePipeStage(opcode, dst_loc, pto::TileType::Ctrl);
    }
}

// ── Instructions that need CPU simulation (iterative convergence depends on values) ──

inline bool NeedsCpuSim(const std::string &opcode)
{
    static const std::unordered_map<std::string, bool> lut = {
        {"TEXP", true}, {"TLOG", true}, {"TSQRT", true}, {"TRSQRT", true}, {"TDIV", true}, {"TDIVS", true},
    };
    auto it = lut.find(opcode);
    return it != lut.end() ? it->second : false;
}

// ── Backward-compatible alias ──
inline PipeStage OpToPipeStage(const std::string &opcode) __attribute__((deprecated));
inline PipeStage OpToPipeStage(const std::string &opcode)
{
    return StaticPipeStageLookup(opcode);
}

} // namespace pto::perf_sim

#endif
