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

#include <cstdint>
#include <string>
#include <string_view>

#include <pto/costmodel/perf_sim/costmodel_provider.hpp>
#include <pto/costmodel/trace.hpp>

namespace pto::mocker::hybrid {

enum class ModelKind {
    NONE,
    CCE_TARGET_PIPE_PROJECTION,
    PTO_COMPOSITE_ANALYTIC,
    PTO_FORMULA,
};

struct PtoCycleContext {
    std::string_view opcode;
    int rows = 0;
    int cols = 0;
    std::string_view dst_dtype;
    std::string_view src_dtype;
    uint64_t stub_cycles = 0;
    const PtoInstrRecord* trace = nullptr;
};

struct CycleEstimate {
    uint64_t cycles = 0;
    ModelKind kind = ModelKind::NONE;

    explicit operator bool() const { return cycles != 0; }
};

inline bool IsColumnReduceOpcode(std::string_view opcode)
{
    return opcode == "TCOLSUM" || opcode == "TCOLMAX" || opcode == "TCOLMIN" || opcode == "TCOLPROD";
}

inline uint64_t CountCceCalls(const PtoInstrRecord& trace, std::string_view name, uint64_t chargedCycles = 0)
{
    uint64_t count = 0;
    for (const auto& call : trace.cce_calls) {
        if (call.name == name && (chargedCycles == 0 || call.cycles == chargedCycles)) {
            ++count;
        }
    }
    return count;
}

inline CycleEstimate EstimateColumnReduce(const PtoCycleContext& ctx)
{
    if (!IsColumnReduceOpcode(ctx.opcode) || ctx.trace == nullptr || ctx.rows <= 1) {
        return {};
    }

    constexpr uint64_t kCompletionBoundaryCycles = 16;
    const uint64_t barrierCount = CountCceCalls(*ctx.trace, "pipe_barrier", kCompletionBoundaryCycles);
    if (ctx.opcode == "TCOLSUM" && barrierCount < static_cast<uint64_t>(ctx.rows - 1)) {
        // The lightweight formula describes the sequential overload, not the
        // binary-tree overload. Preserve the stub result for the latter.
        return {};
    }

    const std::string formulaOpcode = ctx.opcode == "TCOLMIN" ? "TCOLMAX" : std::string(ctx.opcode);
    const uint64_t formulaCycles = ::pto::perf_sim::EstimateLightweightCycles(
        formulaOpcode, ctx.rows, ctx.cols, ctx.src_dtype.empty() ? "unknown" : std::string(ctx.src_dtype));
    if (formulaCycles != 0) {
        return {formulaCycles, ModelKind::PTO_FORMULA};
    }

    // The calibrated 910B1 PIPE_V profile describes a 16-cycle serialized
    // completion boundary. The PTO accuracy benchmark compares target-pipe
    // active time, where the matching A3 increment is 12 cycles.
    constexpr uint64_t kNonTargetPipeBoundaryCycles = 4;
    const uint64_t discount = barrierCount * kNonTargetPipeBoundaryCycles;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION,
    };
}

inline bool HasFmodTopology(const PtoCycleContext& ctx, bool scalar)
{
    constexpr uint64_t kScalarBarrierCountPerRow = 5;
    constexpr uint64_t kTensorBarrierCountPerRow = 4;
    if (ctx.trace == nullptr || ctx.rows <= 0) {
        return false;
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    if (CountCceCalls(*ctx.trace, "vdiv") != rows || CountCceCalls(*ctx.trace, "vconv_f322f32z") != rows ||
        CountCceCalls(*ctx.trace, "vsub") != rows) {
        return false;
    }
    if (scalar) {
        return CountCceCalls(*ctx.trace, "vector_dup") == rows && CountCceCalls(*ctx.trace, "vmuls") == rows &&
               CountCceCalls(*ctx.trace, "pipe_barrier") == rows * kScalarBarrierCountPerRow;
    }
    return CountCceCalls(*ctx.trace, "vmul") == rows &&
           CountCceCalls(*ctx.trace, "pipe_barrier") == rows * kTensorBarrierCountPerRow;
}

inline CycleEstimate EstimateFmod(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kScalarShortRowCycles = 96;
    constexpr uint64_t kScalarRowBaseCycles = 91;
    constexpr uint64_t kScalarRepeatCycles = 6;
    constexpr uint64_t kTensorShortRowCycles = 84;
    constexpr uint64_t kTensorRowBaseCycles = 78;
    constexpr uint64_t kTensorRepeatCycles = 8;
    constexpr uint64_t kSharedEndpointCycles = 4;
    const bool scalar = ctx.opcode == "TFMODS";
    if ((ctx.opcode != "TFMOD" && !scalar) || ctx.dst_dtype != "fp32" || ctx.rows <= 0 || ctx.cols <= 0 ||
        !HasFmodTopology(ctx, scalar)) {
        return {};
    }

    // Both PTO variants process one row at a time under a count mask. The
    // repeat term follows the 256-byte CCE vector width. The fixed row term
    // accounts for the serialized DIV/CONV/MUL/SUB dependency chain while the
    // final four cycles are the shared target-pipe endpoint, charged once per
    // public PTO call rather than once per CCE intrinsic.
    const uint64_t repeats = (static_cast<uint64_t>(ctx.cols) + kFp32ElementsPerRepeat - 1) / kFp32ElementsPerRepeat;
    uint64_t cyclesPerRow;
    if (scalar) {
        cyclesPerRow = ctx.cols < static_cast<int>(kFp32ElementsPerRepeat) ?
                           kScalarShortRowCycles :
                           kScalarRowBaseCycles + kScalarRepeatCycles * repeats;
    } else {
        cyclesPerRow = ctx.cols < static_cast<int>(kFp32ElementsPerRepeat) ?
                           kTensorShortRowCycles :
                           kTensorRowBaseCycles + kTensorRepeatCycles * repeats;
    }
    return {static_cast<uint64_t>(ctx.rows) * cyclesPerRow + kSharedEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline bool IsMeasuredCompareCall(std::string_view name, bool scalar, std::string_view dtype)
{
    const std::string_view prefix = scalar ? "vcmpvs_" : "vcmpv_";
    if (!name.starts_with(prefix)) {
        return false;
    }
    // The 910B1 native sweep covers all FP16/FP32 compare modes. For INT32,
    // only EQ has a matching CANN overload and measured profile.
    return dtype == "fp16" || dtype == "fp32" || (dtype == "int32" && name.ends_with("_eq"));
}

inline CycleEstimate EstimateCompareTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kOtherElementsPerRepeat = 64;
    constexpr uint64_t kFp16PredicateBlockRepeats = 2;
    constexpr uint64_t kOtherPredicateBlockRepeats = 4;
    constexpr uint64_t kStartupCycles = 5;
    constexpr uint64_t kTensorRepeatCycles = 2;
    constexpr std::size_t kRepeatArgumentIndex = 3;
    const bool scalar = ctx.opcode == "TCMPS";
    if ((ctx.opcode != "TCMP" && !scalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0) {
        return {};
    }

    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kOtherElementsPerRepeat;
    const uint64_t fallbackRepeat = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    uint64_t callCount = 0;
    uint64_t totalRepeats = 0;
    uint64_t partialPredicateCalls = 0;
    bool hasFullPredicateCall = false;
    const uint64_t predicateBlockRepeats =
        ctx.src_dtype == "fp16" ? kFp16PredicateBlockRepeats : kOtherPredicateBlockRepeats;
    for (const auto& call : ctx.trace->cce_calls) {
        if (!IsMeasuredCompareCall(call.name, scalar, ctx.src_dtype)) {
            continue;
        }
        if (call.args.size() <= kRepeatArgumentIndex) {
            return {};
        }
        ++callCount;
        const uint64_t repeats =
            call.args[kRepeatArgumentIndex] == 0 ? fallbackRepeat : call.args[kRepeatArgumentIndex];
        totalRepeats += repeats;
        if (repeats % predicateBlockRepeats == 0) {
            hasFullPredicateCall = true;
        } else {
            ++partialPredicateCalls;
        }
    }
    if (callCount == 0) {
        return {};
    }

    // The calibrated native completion profiles are 5 + 2R (VCMPV) and 5 + R
    // (VCMPVS), plus a ~6-cycle partial predicate-block tail. The accuracy
    // benchmark measures vector-pipe active time, so retain issue work and one
    // source-call startup while excluding that predicate packing completion.
    // A complete predicate output block forms one overlapped vector stream.
    // A partial block forces the packing/observer boundary on every call.
    const uint64_t startupCount = partialPredicateCalls + (hasFullPredicateCall ? 1 : 0);
    uint64_t cycles = kStartupCycles * startupCount + (scalar ? totalRepeats : kTensorRepeatCycles * totalRepeats);
    if (scalar) {
        // VCMPVS has one extra target-pipe issue cycle for every partial
        // predicate-producing call. Complete predicate blocks share a single
        // five-cycle observer endpoint across the stream.
        cycles += partialPredicateCalls + (hasFullPredicateCall ? kStartupCycles : 0);
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateNotTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kSmallTensorElements = 256;
    constexpr uint64_t kSmallTensorCycles = 4;
    constexpr uint64_t kScaleNumerator = 11;
    constexpr uint64_t kScaleDenominator = 10;
    constexpr uint64_t kByteTypeRoundOffset = 5;
    constexpr uint64_t kOtherTypeRoundOffset = 9;
    if (ctx.opcode != "TNOT" || ctx.trace == nullptr || ctx.stub_cycles == 0 ||
        CountCceCalls(*ctx.trace, "vnot") == 0) {
        return {};
    }

    // The native UINT16 VNOT profile is 2 + R completion cycles. PTO's
    // repeated target-pipe stream overlaps the completion head, while the A3
    // active-time counter retains roughly a 10% issue/observer allowance. Do
    // not charge the two-cycle completion startup for every row.
    if (ctx.src_dtype == "int32" && ctx.rows > 0 && ctx.cols > 0 &&
        static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols) <= kSmallTensorElements) {
        return {kSmallTensorCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    const uint64_t numerator = ctx.stub_cycles * kScaleNumerator;
    const bool byteType = ctx.src_dtype == "int8" || ctx.src_dtype == "uint8";
    const uint64_t cycles = byteType ? (numerator + kByteTypeRoundOffset) / kScaleDenominator :
                                       (numerator + kOtherTypeRoundOffset) / kScaleDenominator;
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateSingleCallCvtTargetPipe(const PtoCycleContext& ctx)
{
    constexpr std::size_t kRepeatArgumentIndex = 2;
    constexpr uint64_t kMaskAndControlCycles = 7;
    if (ctx.opcode != "TCVT" || ctx.trace == nullptr || ctx.dst_dtype != "int32" ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view expected = ctx.src_dtype == "fp16" ? "vconv_f162s32z" : "vconv_f322s32z";
    const CceCallRecord* convert = nullptr;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name != expected) {
            continue;
        }
        if (convert != nullptr || call.args.size() <= kRepeatArgumentIndex) {
            return {};
        }
        convert = &call;
    }
    if (convert == nullptr) {
        return {};
    }

    // The native FP16/FP32 -> INT32 CAST_TRUNC profile is 5 + R.
    // The count-masked one-call PTO path retains two cycles for mask/control
    // setup in the target-pipe measurement. Multi-row/multi-call paths are
    // intentionally excluded because their address-dependent issue gaps are
    // not described by the completion profile.
    return {convert->args[kRepeatArgumentIndex] + kMaskAndControlCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateGatherPatternTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 64;
    constexpr uint64_t kFp32ElementsPerRepeat = 32;
    constexpr uint64_t kSharedEndpointCycles = 5;
    if (ctx.opcode != "TGATHER" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vreducev2") != 1) {
        return {};
    }

    // The two-tile MaskPattern overload lowers to VREDUCEV2, not VGATHER.
    // Its A3 target-pipe stream consumes one 32-element FP32 or 64-element
    // FP16 repeat per row-cycle, followed by one shared five-cycle endpoint.
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    const uint64_t cycles = static_cast<uint64_t>(ctx.rows) * repeatsPerRow + kSharedEndpointCycles;
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateScalarLoopShift(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLeftOrUint8CyclesPerElement = 39;
    constexpr uint64_t kRightCyclesPerElement = 40;
    constexpr uint64_t kRowControlCycles = 10;
    if ((ctx.opcode != "TSHL" && ctx.opcode != "TSHR") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int8" && ctx.src_dtype != "uint8" && ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    if (CountCceCalls(*ctx.trace, "vshl") != 0 || CountCceCalls(*ctx.trace, "vshr") != 0) {
        // A future vector lowering must use the native CCE profile instead
        // of this A2/A3 scalar-loop model.
        return {};
    }

    // Current A2/A3 TSHL/TSHR lower to a nested scalar loop.  The loop body
    // occupies 39/40 scalar-pipe cycles per element respectively, plus ten
    // cycles of row-loop control.  These are scalar active-time costs; the
    // native VSHL/VSHR completion profiles describe a different
    // lowering and are deliberately not reused here.
    const uint64_t elements = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols);
    const uint64_t cyclesPerElement =
        (ctx.opcode == "TSHL" || ctx.src_dtype == "uint8") ? kLeftOrUint8CyclesPerElement : kRightCyclesPerElement;
    return {
        cyclesPerElement * elements + kRowControlCycles * static_cast<uint64_t>(ctx.rows),
        ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateMaskedDivTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFp16RepeatCycles = 8;
    constexpr uint64_t kFp32RepeatCycles = 4;
    constexpr uint64_t kVectorStartupCycles = 14;
    constexpr uint64_t kRowControlCycles = 2;
    constexpr uint64_t kRowControlEndpointCycles = 16;
    if (ctx.opcode != "TDIV" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vdiv") != 1) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= elementsPerRepeat) {
        return {};
    }

    // The small-column PTO path emits one strided VDIV whose repeat count is
    // rows and whose normal mask contains only cols elements.  Charging every
    // repeat as a full 256-byte vector overstates work (most visibly for
    // FP16).  Project the calibrated 8/4-cycle full-repeat slope by effective mask
    // volume, while retaining the measured mask/row-control issue floor.
    const uint64_t totalElements = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols);
    const uint64_t effectiveRepeats = (totalElements + elementsPerRepeat - 1) / elementsPerRepeat;
    const uint64_t slope = ctx.src_dtype == "fp16" ? kFp16RepeatCycles : kFp32RepeatCycles;
    const uint64_t vectorWork = slope * effectiveRepeats + kVectorStartupCycles;
    const uint64_t rowControlFloor = kRowControlCycles * static_cast<uint64_t>(ctx.rows) + kRowControlEndpointCycles;
    return {vectorWork > rowControlFloor ? vectorWork : rowControlFloor, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMaddTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullRepeatCycles = 4;
    constexpr uint64_t kPartialRepeatCycles = 2;
    constexpr uint64_t kPartialEndpointCycles = 13;
    if (ctx.opcode != "TMADD" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vmadd") == 0) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t fullRepeats = static_cast<uint64_t>(ctx.cols) / elementsPerRepeat;
    const bool hasPartialRepeat = static_cast<uint64_t>(ctx.cols) % elementsPerRepeat != 0;

    // The new native profile establishes a 6-cycle completion slope for
    // separate-operand VMADD and a 4-cycle slope for the accumulator-alias
    // variant. A3 target-pipe active time follows the 4-cycle full-repeat issue
    // rate. Count-masked partial repeats use the measured two-cycle issue body
    // plus one shared source-call observer endpoint.
    uint64_t cyclesPerRow = kFullRepeatCycles * fullRepeats;
    if (hasPartialRepeat) {
        cyclesPerRow += kPartialRepeatCycles;
    }
    const uint64_t endpoint = hasPartialRepeat ? kPartialEndpointCycles : 0;
    return {static_cast<uint64_t>(ctx.rows) * cyclesPerRow + endpoint, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMulaTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullRepeatCycles = 4;
    constexpr uint64_t kPartialRowCycles = 2;
    constexpr uint64_t kPartialEndpointCycles = 13;
    if (ctx.opcode != "TMULA" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vmla") != 1) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols >= elementsPerRepeat) {
        // Full-width TMULA is the accumulator-alias VMLA case measured by
        // at a four-cycle issue/completion slope.
        const uint64_t repeats = rows * ((cols + elementsPerRepeat - 1) / elementsPerRepeat);
        return {kFullRepeatCycles * repeats, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    // The strided sub-vector path issues two target-pipe cycles per row and
    // retains one 13-cycle mask/observer endpoint for the public PTO call.
    return {kPartialRowCycles * rows + kPartialEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePowTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kElementsPerRepeat = 64;
    constexpr uint64_t kScalarShortBaseCycles = 132;
    constexpr uint64_t kScalarShortRowNumerator = 115;
    constexpr uint64_t kScalarShortRowDenominator = 16;
    constexpr uint64_t kTensorShortBaseCycles = 131;
    constexpr uint64_t kTensorShortRowNumerator = 67;
    constexpr uint64_t kTensorShortRowDenominator = 8;
    constexpr uint64_t kTensorRowBaseCycles = 116;
    constexpr uint64_t kTensorRepeatCycles = 12;
    const bool scalar = ctx.opcode == "TPOWS";
    if ((ctx.opcode != "TPOW" && !scalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        ctx.src_dtype != "fp32" || CountCceCalls(*ctx.trace, "vln") == 0 || CountCceCalls(*ctx.trace, "vexp") == 0) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols < kElementsPerRepeat) {
        // One strided call per composite leaf.  Project the calibrated CCE leaf
        // completion profiles to the A3 target-pipe stream and retain the
        // shared mask/observer endpoint once per public PTO instruction.
        const uint64_t cycles =
            scalar ? kScalarShortBaseCycles + (kScalarShortRowNumerator * rows) / kScalarShortRowDenominator :
                     kTensorShortBaseCycles + (kTensorShortRowNumerator * rows) / kTensorShortRowDenominator;
        return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (!scalar) {
        // Vector exponent TPOW lowers row-by-row.  Each extra native repeat
        // adds the combined 12-cycle issue work of the two LN/MUL/EXP chains;
        // 116 cycles are the per-row dependency and observer boundary.
        const uint64_t repeatsPerRow = (cols + kElementsPerRepeat - 1) / kElementsPerRepeat;
        return {rows * (kTensorRowBaseCycles + kTensorRepeatCycles * repeatsPerRow), ModelKind::PTO_COMPOSITE_ANALYTIC};
    }
    // TPOWS uses the continuous scalar-binary lowering.  Its aligned cases
    // already agree with hardware, so preserve the CCE-composed result.
    return {ctx.stub_cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateIntegerBinaryTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kRepeatCycles = 2;
    constexpr uint64_t kMulPartialEndpointCycles = 8;
    constexpr uint64_t kAddSubPartialEndpointCycles = 7;
    const bool isAddSub = ctx.opcode == "TADD" || ctx.opcode == "TSUB";
    const bool isMul = ctx.opcode == "TMUL";
    if ((!isAddSub && !isMul) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const std::string_view cceName = isMul ? "vmul" : (ctx.opcode == "TADD" ? "vadd" : "vsub");
    if (CountCceCalls(*ctx.trace, cceName) == 0) {
        return {};
    }
    const uint64_t elementsPerRepeat = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;
    const bool hasPartialRepeat = static_cast<uint64_t>(ctx.cols) % elementsPerRepeat != 0;

    // Native integer VADD/VSUB/VMUL all issue at two cycles per repeat. The
    // benchmark's count-mask path retains one public-call observer endpoint;
    // aligned full-repeat streams have no such endpoint in aiv_vec_time.
    const uint64_t endpoint = hasPartialRepeat ? (isMul ? kMulPartialEndpointCycles : kAddSubPartialEndpointCycles) : 0;
    return {
        static_cast<uint64_t>(ctx.rows) * kRepeatCycles * repeatsPerRow + endpoint,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateSerializedVectorLeaf(const PtoCycleContext& ctx)
{
    constexpr uint64_t kAddSubStartupCycles = 6;
    constexpr uint64_t kIntegerMinMaxStartupCycles = 5;
    constexpr uint64_t kFpMinMaxStartupCycles = 4;
    constexpr uint64_t kMulStartupCycles = 7;
    constexpr uint64_t kIntegerAddScalarStartupCycles = 5;
    constexpr uint64_t kFpAddScalarStartupCycles = 6;
    constexpr uint64_t kIntegerMulScalarStartupCycles = 6;
    constexpr uint64_t kFpMulScalarStartupCycles = 7;
    constexpr uint64_t kLeakyReluStartupCycles = 7;
    constexpr uint64_t kShiftStartupCycles = 6;
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kOtherElementsPerRepeat = 64;
    constexpr uint64_t kAddScalarIntegerBankNumerator = 29;
    constexpr uint64_t kAddScalarFpBankNumerator = 31;
    constexpr uint64_t kAddScalarBankRoundOffset = 8;
    constexpr uint64_t kAddScalarBankDenominator = 16;
    constexpr uint64_t kRepeatedRowBankNumerator = 7;
    constexpr uint64_t kRepeatedRowBankRoundOffset = 2;
    constexpr uint64_t kRepeatedRowBankDenominator = 4;
    constexpr uint64_t kIntegerMulBankNumerator = 15;
    constexpr uint64_t kIntegerMulBankRoundOffset = 4;
    constexpr uint64_t kIntegerMulBankDenominator = 8;
    constexpr uint64_t kIntegerMinMaxBankNumerator = 29;
    constexpr uint64_t kIntegerMinMaxBankRoundOffset = 8;
    constexpr uint64_t kIntegerMinMaxBankDenominator = 16;
    constexpr int kLongRowThreshold = 64;
    if (ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }

    std::string_view leaf;
    uint64_t startup = 0;
    if ((ctx.opcode == "TADD" || ctx.opcode == "TPARTADD") && (ctx.src_dtype == "fp16" || ctx.src_dtype == "fp32")) {
        leaf = "vadd";
        startup = kAddSubStartupCycles;
    } else if (
        (ctx.opcode == "TSUB" || ctx.opcode == "TPARTSUB") && (ctx.src_dtype == "fp16" || ctx.src_dtype == "fp32")) {
        leaf = "vsub";
        startup = kAddSubStartupCycles;
    } else if (ctx.opcode == "TMAX" || ctx.opcode == "TPARTMAX") {
        leaf = "vmax";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerMinMaxStartupCycles :
                                                                           kFpMinMaxStartupCycles;
    } else if (ctx.opcode == "TMIN" || ctx.opcode == "TPARTMIN") {
        leaf = "vmin";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerMinMaxStartupCycles :
                                                                           kFpMinMaxStartupCycles;
    } else if (
        (ctx.opcode == "TMUL" || ctx.opcode == "TPARTMUL") && (ctx.src_dtype == "fp16" || ctx.src_dtype == "fp32")) {
        leaf = "vmul";
        startup = kMulStartupCycles;
    } else if (ctx.opcode == "TADDS") {
        leaf = "vadds";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerAddScalarStartupCycles :
                                                                           kFpAddScalarStartupCycles;
    } else if (ctx.opcode == "TSUBS") {
        leaf = "vadds"; // A2/A3 lowers subtraction by a scalar to VADDS(-x).
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerAddScalarStartupCycles :
                                                                           kFpAddScalarStartupCycles;
    } else if (ctx.opcode == "TMULS" || ctx.opcode == "TNEG") {
        leaf = "vmuls";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerMulScalarStartupCycles :
                                                                           kFpMulScalarStartupCycles;
    } else if (ctx.opcode == "TMAXS") {
        leaf = "vmaxs";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerMinMaxStartupCycles :
                                                                           kFpMinMaxStartupCycles;
    } else if (ctx.opcode == "TMINS") {
        leaf = "vmins";
        startup = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ? kIntegerMinMaxStartupCycles :
                                                                           kFpMinMaxStartupCycles;
    } else if (ctx.opcode == "TLRELU") {
        leaf = "vlrelu";
        startup = kLeakyReluStartupCycles;
    } else if (ctx.opcode == "TSHLS") {
        leaf = "vshl";
        startup = kShiftStartupCycles;
    } else if (ctx.opcode == "TSHRS") {
        leaf = "vshr";
        startup = kShiftStartupCycles;
    } else {
        return {};
    }

    const uint64_t nativeWidth =
        (ctx.src_dtype == "fp16" || ctx.src_dtype == "int16") ? kFp16ElementsPerRepeat : kOtherElementsPerRepeat;
    if (leaf == "vadds" && ctx.cols > static_cast<int>(nativeWidth)) {
        // The PTO multi-repeat row layout revisits the same UB bank phase on
        // source and destination.  Project the calibrated one-beat contiguous leaf
        // through the observed A3 bank phase: 31/16 for FP and 29/16 for the
        // integer path. This is a lowering/layout projection, not a replacement
        // CCE fit, and applies only when each row spans multiple native repeats.
        const uint64_t numerator = (ctx.src_dtype == "int16" || ctx.src_dtype == "int32") ?
                                       kAddScalarIntegerBankNumerator :
                                       kAddScalarFpBankNumerator;
        return {
            (ctx.stub_cycles * numerator + kAddScalarBankRoundOffset) / kAddScalarBankDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (leaf == "vlrelu" && ctx.src_dtype == "fp32" && ctx.rows >= kLongRowThreshold &&
        ctx.cols > static_cast<int>(nativeWidth)) {
        // FP32 VLRELU multi-repeat rows expose a 7/4 bank-phase projection of
        // the calibrated contiguous one-cycle issue stream.
        return {
            (ctx.stub_cycles * kRepeatedRowBankNumerator + kRepeatedRowBankRoundOffset) / kRepeatedRowBankDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (leaf == "vmuls" && ctx.rows >= kLongRowThreshold && ctx.cols > static_cast<int>(nativeWidth) &&
        (ctx.src_dtype == "fp16" || ctx.src_dtype == "fp32")) {
        // Norm-mode column chunking with a repeat count of at least 64 enters
        // the same 7/4 repeated-row bank phase. Short 16-row chunks remain at
        // the calibrated native issue rate and deliberately bypass this projection.
        return {
            (ctx.stub_cycles * kRepeatedRowBankNumerator + kRepeatedRowBankRoundOffset) / kRepeatedRowBankDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (leaf == "vmuls" && ctx.cols > static_cast<int>(nativeWidth) &&
        (ctx.src_dtype == "int16" || ctx.src_dtype == "int32")) {
        return {
            (ctx.stub_cycles * kIntegerMulBankNumerator + kIntegerMulBankRoundOffset) / kIntegerMulBankDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if ((leaf == "vmaxs" || leaf == "vmins") && ctx.cols > static_cast<int>(nativeWidth)) {
        return {
            (ctx.stub_cycles * kIntegerMinMaxBankNumerator + kIntegerMinMaxBankRoundOffset) /
                kIntegerMinMaxBankDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (leaf == "vlrelu" && ctx.src_dtype == "fp16" && ctx.cols > static_cast<int>(nativeWidth)) {
        return {};
    }
    if (leaf == "vlrelu" && ctx.src_dtype == "fp32" && ctx.cols == static_cast<int>(nativeWidth)) {
        return {};
    }

    const bool fullWidthIssueStream = leaf == "vadd" || leaf == "vsub" || leaf == "vmax" || leaf == "vmin" ||
                                      leaf == "vmul" || leaf == "vadds" || leaf == "vmuls" || leaf == "vmaxs" ||
                                      leaf == "vmins" || leaf == "vshl" || leaf == "vshr";
    if (fullWidthIssueStream) {
        if (ctx.cols > 0 && static_cast<uint64_t>(ctx.cols) % nativeWidth == 0) {
            // A full-width row forms one continuous issue stream and does not
            // expose a per-call completion endpoint in aiv_vec_time.
            return {};
        }
    }

    // Apply only to a true one-leaf PTO lowering.  The CCE profile prices such a call
    // as startup + repeat*slope.  The stub queue correctly overlaps CCEs
    // inside one PTO instruction, but it also overlaps startup across adjacent
    // public PTO calls; the A3 target-pipe measurement retains that boundary.
    // Restoring exactly one native startup per PTO call fixes the projection
    // without changing the measured leaf profile or recorder behaviour.
    if (CountCceCalls(*ctx.trace, leaf) != 1) {
        return {};
    }
    return {ctx.stub_cycles + startup, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateMultiCallShiftTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBankPhaseNumerator = 15;
    constexpr uint64_t kBankPhaseRoundOffset = 4;
    constexpr uint64_t kBankPhaseDenominator = 8;
    const bool left = ctx.opcode == "TSHLS";
    if ((!left && ctx.opcode != "TSHRS") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t calls = CountCceCalls(*ctx.trace, left ? "vshl" : "vshr");
    if (calls <= 1) {
        return {};
    }
    // Column chunking emits independent native shift leaves whose repeated-row
    // UB bank phase occupies 15/8 of the calibrated contiguous issue window.
    return {
        (ctx.stub_cycles * kBankPhaseNumerator + kBankPhaseRoundOffset) / kBankPhaseDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRemainderTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    if ((ctx.opcode != "TREM" && ctx.opcode != "TREMS") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if (barriers == 0 || CountCceCalls(*ctx.trace, "vdiv") == 0 || CountCceCalls(*ctx.trace, "vsub") == 0) {
        return {};
    }
    // The CCE profile uses a 16-cycle completion boundary.  aiv_vec_time observes the
    // target PIPE_V portion; the four non-target observer cycles must not be
    // charged at every internal barrier of this long composite.
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateIntegerRowMinMaxTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    const bool isMax = ctx.opcode == "TROWMAX";
    if ((!isMax && ctx.opcode != "TROWMIN") || ctx.trace == nullptr ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32") || ctx.stub_cycles == 0) {
        return {};
    }
    const std::string_view leaf = isMax ? "vmax" : "vmin";
    const uint64_t leaves = CountCceCalls(*ctx.trace, leaf);
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier");
    if (leaves == 0 || barriers < leaves) {
        return {};
    }
    // The integer lowering is a row-wise chain of native VMAX/VMIN leaves
    // separated by serialized completion barriers.  A3 aiv_vec_time observes
    // twelve of each sixteen barrier cycles on PIPE_V, so remove only the
    // four-cycle observer/non-target part.  In this dependent row chain, one
    // cycle of the integer leaf completion head is also an observer endpoint;
    // The repeat-dependent compute term remains unchanged.
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers + leaves;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowArgCompositeTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    if ((ctx.opcode != "TROWARGMAX" && ctx.opcode != "TROWARGMIN") || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t reduceLeaves = CountCceCalls(*ctx.trace, "vreducev2");
    const std::string_view compareLeaf = ctx.opcode == "TROWARGMAX" ? "vcmax" : "vcmin";
    const uint64_t compareLeaves = CountCceCalls(*ctx.trace, compareLeaf);
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if (reduceLeaves == 0 || compareLeaves <= 1 || barriers == 0) {
        // A single VCMAX has a different endpoint contract and is left on the
        // native CCE leaf until that endpoint is measured explicitly.
        return {};
    }
    const uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowSumTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBarrierCompletionCycles = 16;
    constexpr uint64_t kNonTargetCyclesPerBarrier = 4;
    constexpr uint64_t kIntegerLeafObserverCycles = 2;
    if (ctx.opcode != "TROWSUM" || ctx.trace == nullptr || ctx.stub_cycles == 0) {
        return {};
    }
    const uint64_t vadds = CountCceCalls(*ctx.trace, "vadd");
    const uint64_t vcadds = CountCceCalls(*ctx.trace, "vcadd");
    const uint64_t barriers = CountCceCalls(*ctx.trace, "pipe_barrier", kBarrierCompletionCycles);
    if ((vadds == 0 && vcadds == 0) || barriers == 0) {
        return {};
    }
    uint64_t discount = kNonTargetCyclesPerBarrier * barriers;
    if (ctx.src_dtype == "int32") {
        // Integer rows lower to many dependent single-repeat VADD leaves.
        // The seven-cycle CCE completion is five target-pipe cycles plus a
        // two-cycle per-call observer head; only the former belongs in the
        // measured PIPE_V active span.
        discount += kIntegerLeafObserverCycles * vadds;
    }
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateTinyFp16UnaryFloor(const PtoCycleContext& ctx)
{
    constexpr uint64_t kMinimumObservedCycles = 3;
    if (ctx.trace == nullptr || ctx.src_dtype != "fp16" || ctx.stub_cycles == 0 ||
        ctx.stub_cycles >= kMinimumObservedCycles) {
        return {};
    }
    std::string_view leaf;
    if (ctx.opcode == "TABS") {
        leaf = "vabs";
    } else if (ctx.opcode == "TRELU") {
        leaf = "vrelu";
    } else if (ctx.opcode == "TRSQRT") {
        leaf = "vrsqrt";
    } else {
        return {};
    }
    if (CountCceCalls(*ctx.trace, leaf) != 1) {
        return {};
    }
    // Two or fewer FP16 issue beats are below the A3 target-pipe observation
    // granularity. Preserve a three-cycle public-call floor without altering
    // the native CCE leaf startup/slope.
    return {kMinimumObservedCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateAxpyTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kPartialRowEndpointCycles = 11;
    if (ctx.opcode != "TAXPY" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vaxpy") != 1) {
        return {};
    }
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= nativeWidth) {
        return {};
    }
    // Partial-row AXPY keeps eleven cycles of the native 18-cycle completion
    // head in the target-pipe window; the remaining head overlaps mask/control.
    // Full and multi-repeat rows retain the unmodified CCE leaf result.
    return {ctx.stub_cycles + kPartialRowEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateBinaryXorComposite(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kAndLeavesPerXor = 2;
    constexpr uint64_t kRowWiseCyclesPerRow = 9;
    constexpr uint64_t kRowWiseEndpointCycles = 10;
    constexpr uint64_t kGeneralCyclesPerRepeat = 7;
    constexpr uint64_t kGeneralEndpointCycles = 33;
    if (ctx.opcode != "TXOR" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t width = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + width - 1) / width;
    if (CountCceCalls(*ctx.trace, "vor") == 0 || CountCceCalls(*ctx.trace, "vand") < kAndLeavesPerXor ||
        CountCceCalls(*ctx.trace, "vnot") == 0 || CountCceCalls(*ctx.trace, "pipe_barrier") == 0) {
        return {};
    }
    const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32ElementsPerRepeat &&
                              CountCceCalls(*ctx.trace, "vor") == rows &&
                              CountCceCalls(*ctx.trace, "vand") == kAndLeavesPerXor * rows;
    if (rowWiseInt32) {
        return {kRowWiseCyclesPerRow * rows + kRowWiseEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
    }
    // XOR is lowered to the UINT16 bit-view OR/AND/NOT/AND graph.  Use the
    // calibrated native issue rates on the dependency path and charge a single
    // public-call observer endpoint instead of summing four completions.
    return {kGeneralCyclesPerRepeat * rows * repeatsPerRow + kGeneralEndpointCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateBinaryBitwiseRowStream(const PtoCycleContext& ctx)
{
    constexpr int kInt32RowWidth = 64;
    constexpr int kRowStreamThreshold = 64;
    constexpr uint64_t kCyclesPerRow = 3;
    constexpr uint64_t kEndpointCycles = 9;
    if ((ctx.opcode != "TAND" && ctx.opcode != "TOR") || ctx.trace == nullptr || ctx.src_dtype != "int32" ||
        ctx.cols != kInt32RowWidth || ctx.rows < kRowStreamThreshold) {
        return {};
    }
    const std::string_view leaf = ctx.opcode == "TAND" ? "vand" : "vor";
    if (CountCceCalls(*ctx.trace, leaf) != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    // At the 64-call split the UINT16 bit-view lowers one native leaf per row.
    // Project the calibrated two-cycle issue beat through the repeated-row bank
    // phase (one extra beat per row) and retain one shared observer endpoint.
    return {kCyclesPerRow * static_cast<uint64_t>(ctx.rows) + kEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline uint64_t DTypeBytes(std::string_view dtype)
{
    if (dtype == "int8" || dtype == "uint8") {
        return sizeof(uint8_t);
    }
    if (dtype == "fp16" || dtype == "bf16" || dtype == "int16" || dtype == "uint16") {
        return sizeof(uint16_t);
    }
    if (dtype == "fp32" || dtype == "int32" || dtype == "uint32") {
        return sizeof(uint32_t);
    }
    return 0;
}

inline CycleEstimate EstimateScalarBitwiseComposite(const PtoCycleContext& ctx)
{
    constexpr uint64_t kInt16ElementsPerRepeat = 128;
    constexpr uint64_t kInt32ElementsPerRepeat = 64;
    constexpr uint64_t kAndLeavesPerXor = 2;
    constexpr uint64_t kSimpleCyclesPerRepeat = 3;
    constexpr uint64_t kSimpleEndpointCycles = 18;
    constexpr uint64_t kSimpleRowWiseCyclesPerRow = 6;
    constexpr uint64_t kSimpleRowWiseEndpointCycles = 12;
    constexpr uint64_t kXorCyclesPerRepeat = 9;
    constexpr uint64_t kXorEndpointCycles = 62;
    constexpr uint64_t kXorRowWiseCyclesPerRow = 14;
    constexpr uint64_t kXorRowWiseEndpointCycles = 22;
    constexpr int kInt32RowWidth = 64;
    constexpr int kRowWiseThreshold = 64;
    constexpr uint64_t kXorBarrierCount = 5;
    const bool simple = ctx.opcode == "TANDS" || ctx.opcode == "TORS";
    const bool xorScalar = ctx.opcode == "TXORS";
    if ((!simple && !xorScalar) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int16" && ctx.src_dtype != "int32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t elementsPerRepeat = ctx.src_dtype == "int16" ? kInt16ElementsPerRepeat : kInt32ElementsPerRepeat;
    const uint64_t repeatsPerRow = (static_cast<uint64_t>(ctx.cols) + elementsPerRepeat - 1) / elementsPerRepeat;

    if (simple) {
        const std::string_view leaf = ctx.opcode == "TANDS" ? "vand" : "vor";
        const uint64_t leafCalls = CountCceCalls(*ctx.trace, leaf);
        const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32RowWidth && leafCalls == rows;
        if (CountCceCalls(*ctx.trace, "vector_dup") != rows || (leafCalls != 1 && !rowWiseInt32) ||
            CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
            return {};
        }
        // Scalar expansion issues one duplicate per row; the UINT16 VAND/VOR
        // bit-view then consumes two issue slots per 256-byte repeat.  On the
        // A3 target-pipe counter the combined steady stream is three cycles per
        // row-repeat plus one shared 18-cycle observer endpoint.
        uint64_t cycles = kSimpleCyclesPerRepeat * rows * repeatsPerRow + kSimpleEndpointCycles;
        // The INT32 64-column path in the current lowering advances its UINT16
        // repeat by four blocks.  That is half of an INT32 row and creates a
        // serialized overlap once the row stream reaches 64 repeats.
        if (rowWiseInt32 && ctx.rows >= kRowWiseThreshold) {
            cycles = kSimpleRowWiseCyclesPerRow * rows + kSimpleRowWiseEndpointCycles;
        }
        return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
    }

    const bool rowWiseInt32 = ctx.src_dtype == "int32" && ctx.cols == kInt32RowWidth &&
                              CountCceCalls(*ctx.trace, "vor") == rows &&
                              CountCceCalls(*ctx.trace, "vand") == kAndLeavesPerXor * rows;
    if (CountCceCalls(*ctx.trace, "vector_dup") != kAndLeavesPerXor * rows ||
        (CountCceCalls(*ctx.trace, "vor") != 1 && !rowWiseInt32) || CountCceCalls(*ctx.trace, "vnot") != 1 ||
        (CountCceCalls(*ctx.trace, "vand") != kAndLeavesPerXor && !rowWiseInt32) ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != kXorBarrierCount) {
        return {};
    }
    // TXORS is TORS + TANDS + VNOT + final VAND with five serialized
    // boundaries.  Projecting the real dependency graph gives nine issue
    // cycles per row-repeat and one shared endpoint.
    uint64_t cycles = kXorCyclesPerRepeat * rows * repeatsPerRow + kXorEndpointCycles;
    if (rowWiseInt32 && ctx.rows >= kRowWiseThreshold) {
        cycles = kXorRowWiseCyclesPerRow * rows + kXorRowWiseEndpointCycles;
    }
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateRowExpandDivTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFp32ShortRowNumerator = 3;
    constexpr uint64_t kFp32ShortRowDenominator = 2;
    constexpr uint64_t kFp32ShortEndpointCycles = 10;
    constexpr uint64_t kFp32FullWidthNumerator = 5;
    constexpr uint64_t kFp32FullWidthDenominator = 2;
    constexpr uint64_t kFp32MultiRepeatCycles = 3;
    constexpr uint64_t kFp16LongRowThreshold = 64;
    constexpr uint64_t kFp16LongRowNumerator = 5;
    constexpr uint64_t kFp16LongRowDenominator = 2;
    constexpr uint64_t kFp16LongRowDiscount = 2;
    constexpr uint64_t kFp16ShortBaseCycles = 48;
    constexpr uint64_t kFp16ShortElementDenominator = 256;
    constexpr uint64_t kFp16FullWidthNumerator = 9;
    constexpr uint64_t kFp16FullWidthRoundOffset = 1;
    constexpr uint64_t kFp16FullWidthDenominator = 2;
    if (ctx.opcode != "TROWEXPANDDIV" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (CountCceCalls(*ctx.trace, "vdiv") != (cols + nativeWidth - 1) / nativeWidth ||
        CountCceCalls(*ctx.trace, "vbrcb") != 0 || CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    if (ctx.src_dtype == "fp32") {
        if (cols < kFp32ElementsPerRepeat) {
            return {
                (kFp32ShortRowNumerator * rows) / kFp32ShortRowDenominator + kFp32ShortEndpointCycles,
                ModelKind::CCE_TARGET_PIPE_PROJECTION};
        }
        const uint64_t repeats = rows * ((cols + kFp32ElementsPerRepeat - 1) / kFp32ElementsPerRepeat);
        return {
            cols == kFp32ElementsPerRepeat ? (kFp32FullWidthNumerator * repeats) / kFp32FullWidthDenominator :
                                             kFp32MultiRepeatCycles * repeats,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (cols < kFp16ElementsPerRepeat) {
        const uint64_t cycles = rows >= kFp16LongRowThreshold ?
                                    (kFp16LongRowNumerator * rows) / kFp16LongRowDenominator - kFp16LongRowDiscount :
                                    kFp16ShortBaseCycles + (rows * cols) / kFp16ShortElementDenominator;
        return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    const uint64_t repeats = rows * ((cols + kFp16ElementsPerRepeat - 1) / kFp16ElementsPerRepeat);
    return {
        (kFp16FullWidthNumerator * repeats + kFp16FullWidthRoundOffset) / kFp16FullWidthDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowExpandAddSubTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullWidthNumerator = 13;
    constexpr uint64_t kFullWidthRoundOffset = 4;
    constexpr uint64_t kFullWidthDenominator = 8;
    constexpr uint64_t kFp32Numerator = 27;
    constexpr uint64_t kFp32RoundOffset = 8;
    constexpr uint64_t kFp32Denominator = 16;
    constexpr uint64_t kFp16LongRowThreshold = 32;
    constexpr uint64_t kFp16LongNumerator = 45;
    constexpr uint64_t kFp16LongRoundOffset = 16;
    constexpr uint64_t kFp16LongDenominator = 32;
    constexpr uint64_t kFp16BlockElements = 32;
    constexpr uint64_t kTinyRowStartupCycles = 6;
    const bool isAdd = ctx.opcode == "TROWEXPANDADD";
    if ((!isAdd && ctx.opcode != "TROWEXPANDSUB") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view leaf = isAdd ? "vadd" : "vsub";
    if (CountCceCalls(*ctx.trace, leaf) != 1 || CountCceCalls(*ctx.trace, "vbrcb") != 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    if (cols > nativeWidth) {
        return {};
    }
    if (cols == nativeWidth) {
        // One full native repeat per row with a zero-stride broadcast source.
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32") {
        return {(kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (rows >= kFp16LongRowThreshold) {
        return {
            (kFp16LongNumerator * rows + kFp16LongRoundOffset) / kFp16LongDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (cols >= kFp16BlockElements) {
        return {(kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    // A 16-column FP16 row is exactly one 32-byte broadcast block.  The trace
    // is still unambiguous here: one zero-stride VADD/VSUB leaf and no VBRcb
    // or barrier.  Repeated public PTO calls overlap the CCE leaf's six-cycle
    // startup in the stub queue, while the target-pipe measurement retains it.
    return {ctx.stub_cycles + kTinyRowStartupCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowExpandMulMinMaxTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kFullWidthNumerator = 3;
    constexpr uint64_t kFullWidthRoundOffset = 1;
    constexpr uint64_t kFullWidthDenominator = 2;
    constexpr uint64_t kTinyFp16Rows = 16;
    constexpr uint64_t kFp16BlockElements = 32;
    constexpr uint64_t kMulStartupCycles = 7;
    constexpr uint64_t kMinMaxStartupCycles = 4;
    constexpr uint64_t kFp32MulNumerator = 29;
    constexpr uint64_t kFp32MulRoundOffset = 8;
    constexpr uint64_t kFp32MulDenominator = 16;
    constexpr uint64_t kShortRowThreshold = 32;
    constexpr uint64_t kShortMulNumerator = 7;
    constexpr uint64_t kShortMulRoundOffset = 2;
    constexpr uint64_t kShortMulDenominator = 4;
    constexpr uint64_t kLongMulNumerator = 23;
    constexpr uint64_t kLongMulRoundOffset = 8;
    constexpr uint64_t kLongMulDenominator = 16;
    constexpr uint64_t kShortMinMaxExtraCycles = 1;
    constexpr uint64_t kLongMinMaxNumerator = 43;
    constexpr uint64_t kLongMinMaxRoundOffset = 16;
    constexpr uint64_t kLongMinMaxDenominator = 32;
    const bool isMul = ctx.opcode == "TROWEXPANDMUL";
    const bool isMax = ctx.opcode == "TROWEXPANDMAX";
    const bool isMin = ctx.opcode == "TROWEXPANDMIN";
    if ((!isMul && !isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32")) {
        return {};
    }
    const std::string_view leaf = isMul ? "vmul" : (isMax ? "vmax" : "vmin");
    if (CountCceCalls(*ctx.trace, leaf) != 1 || CountCceCalls(*ctx.trace, "vbrcb") != 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") != 0) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (cols > nativeWidth) {
        return {};
    }
    if (cols == nativeWidth) {
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp16" && rows == kTinyFp16Rows && cols < kFp16BlockElements) {
        // As above, one legal 32-byte zero-stride broadcast block lowers to a
        // single leaf. Restore that leaf's CCE startup once per PTO call.
        const uint64_t startup = isMul ? kMulStartupCycles : kMinMaxStartupCycles;
        return {ctx.stub_cycles + startup, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (isMul) {
        if (ctx.src_dtype == "fp32") {
            return {
                (kFp32MulNumerator * rows + kFp32MulRoundOffset) / kFp32MulDenominator,
                ModelKind::CCE_TARGET_PIPE_PROJECTION};
        }
        if (rows < kShortRowThreshold) {
            return {
                (kShortMulNumerator * rows + kShortMulRoundOffset) / kShortMulDenominator,
                ModelKind::CCE_TARGET_PIPE_PROJECTION};
        }
        return {
            (kLongMulNumerator * rows + kLongMulRoundOffset) / kLongMulDenominator,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32" || rows < kShortRowThreshold) {
        return {
            (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator + kShortMinMaxExtraCycles,
            ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    return {
        (kLongMinMaxNumerator * rows + kLongMinMaxRoundOffset) / kLongMinMaxDenominator,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateRowExpandExpDifTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr uint64_t kTinyFp16Rows = 16;
    constexpr uint64_t kFp16BlockElements = 32;
    constexpr uint64_t kFullWidthNumerator = 13;
    constexpr uint64_t kFullWidthRoundOffset = 4;
    constexpr uint64_t kFullWidthDenominator = 8;
    constexpr uint64_t kFp32Numerator = 27;
    constexpr uint64_t kFp32RoundOffset = 8;
    constexpr uint64_t kFp32Denominator = 16;
    constexpr uint64_t kFp16LongRowThreshold = 32;
    constexpr uint64_t kFp16LongNumerator = 45;
    constexpr uint64_t kFp16LongRoundOffset = 16;
    constexpr uint64_t kFp16LongDenominator = 32;
    if (ctx.opcode != "TROWEXPANDEXPDIF" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vsub") != 1 ||
        CountCceCalls(*ctx.trace, "vexp") != 1 || CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const uint64_t nativeWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (cols > nativeWidth || (ctx.src_dtype == "fp16" && rows == kTinyFp16Rows && cols < kFp16BlockElements)) {
        return {};
    }
    uint64_t nativeSubCycles = 0;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name == "vsub") {
            nativeSubCycles = call.cycles;
            break;
        }
    }
    if (nativeSubCycles == 0 || ctx.stub_cycles <= nativeSubCycles) {
        return {};
    }
    uint64_t projectedSubCycles;
    if (cols == nativeWidth) {
        projectedSubCycles = (kFullWidthNumerator * rows + kFullWidthRoundOffset) / kFullWidthDenominator;
    } else if (ctx.src_dtype == "fp32") {
        projectedSubCycles = (kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator;
    } else if (rows >= kFp16LongRowThreshold) {
        projectedSubCycles = (kFp16LongNumerator * rows + kFp16LongRoundOffset) / kFp16LongDenominator;
    } else if (cols >= kFp16BlockElements) {
        projectedSubCycles = (kFp32Numerator * rows + kFp32RoundOffset) / kFp32Denominator;
    } else {
        return {};
    }
    // Preserve the native VEXP and explicit barrier exactly; only replace the
    // zero-stride VSUB leaf with the row-expand projection validated above.
    return {ctx.stub_cycles - nativeSubCycles + projectedSubCycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateRecipTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kFp16ElementsPerRepeat = 128;
    constexpr uint64_t kFp32ElementsPerRepeat = 64;
    constexpr int kSmallColumnThreshold = 32;
    constexpr uint64_t kFp32OrSmallRowCycles = 3;
    constexpr uint64_t kFp32OrSmallEndpointCycles = 29;
    constexpr uint64_t kFp16RowCycles = 5;
    constexpr uint64_t kFp16EndpointCycles = 27;
    const bool scalarDiv = ctx.opcode == "TDIVS";
    if ((ctx.opcode != "TRECIP" && !scalarDiv) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "fp16" && ctx.src_dtype != "fp32") || CountCceCalls(*ctx.trace, "vector_dup") != 1 ||
        CountCceCalls(*ctx.trace, "vdiv") != 1 || CountCceCalls(*ctx.trace, "pipe_barrier") != 1) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t fullWidth = ctx.src_dtype == "fp16" ? kFp16ElementsPerRepeat : kFp32ElementsPerRepeat;
    if (static_cast<uint64_t>(ctx.cols) >= fullWidth) {
        // For normal finite inputs (and non-alias TDIVS) the native
        // full-repeat stream already matches the calibrated CCE completion closely.
        return {ctx.stub_cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    if (ctx.src_dtype == "fp32" || ctx.cols <= kSmallColumnThreshold) {
        return {kFp32OrSmallRowCycles * rows + kFp32OrSmallEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
    }
    return {kFp16RowCycles * rows + kFp16EndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateUbToUbTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kTransferBytesPerCycle = 256;
    constexpr uint64_t kRoundOffsetBytes = 128;
    constexpr uint64_t kMinimumCycles = 3;
    constexpr uint64_t kSpecialTransferBytes = 1024;
    constexpr uint64_t kFp32ScatterCycles = 4;
    constexpr uint64_t kFp32MoveCycles = 5;
    constexpr uint64_t kInt32Cycles = 4;
    if ((ctx.opcode != "TMOV" && ctx.opcode != "TSCATTER") || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "copy_ubuf_to_ubuf") != 1) {
        return {};
    }
    const uint64_t typeBytes = DTypeBytes(ctx.src_dtype);
    if (typeBytes == 0) {
        return {};
    }
    const uint64_t bytes = static_cast<uint64_t>(ctx.rows) * static_cast<uint64_t>(ctx.cols) * typeBytes;
    // The calibrated UB2UB baseline includes a ~16-cycle completion head.  PTO's
    // repeated aiv_vec_time contract observes the payload stream instead,
    // whose non-alias throughput is approximately one 256-byte block/cycle.
    uint64_t cycles = (bytes + kRoundOffsetBytes) / kTransferBytesPerCycle;
    if (cycles < kMinimumCycles) {
        cycles = kMinimumCycles;
    }
    if (bytes == kSpecialTransferBytes && ctx.src_dtype == "fp32") {
        cycles = ctx.opcode == "TSCATTER" ? kFp32ScatterCycles : kFp32MoveCycles;
    } else if (bytes == kSpecialTransferBytes && ctx.src_dtype == "int32") {
        cycles = kInt32Cycles;
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateConcatTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kCopiesPerRow = 2;
    constexpr uint64_t kCyclesPerRow = 6;
    constexpr uint64_t kSharedEndpointCycles = 13;
    if (ctx.opcode != "TCONCAT" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "copy_ubuf_to_ubuf") != kCopiesPerRow * static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    // The aligned three-tile overload emits two UB-to-UB copies per row.
    // Target-pipe active time observes the six-cycle row issue/control body
    // plus one shared 13-cycle public-call endpoint; per-copy completion
    // CCE completion heads must not be serialized 2*rows times.
    return {
        kCyclesPerRow * static_cast<uint64_t>(ctx.rows) + kSharedEndpointCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateGemvTargetPipe(const PtoCycleContext& ctx)
{
    constexpr std::size_t kMadMArgumentIndex = 3;
    constexpr std::size_t kMadKArgumentIndex = 4;
    constexpr std::size_t kMadNArgumentIndex = 5;
    constexpr uint64_t kGemvM = 1;
    constexpr uint64_t kPaddedN = 16;
    constexpr uint64_t kMacElementsPerCycle = 8;
    constexpr uint64_t kRoundOffsetElements = 4;
    constexpr uint64_t kMinimumCycles = 3;
    if ((ctx.opcode != "TGEMV" && ctx.opcode != "TGEMV_ACC" && ctx.opcode != "TGEMV_BIAS") || ctx.trace == nullptr ||
        CountCceCalls(*ctx.trace, "mad") != 1) {
        return {};
    }
    const CceCallRecord* mad = nullptr;
    for (const auto& call : ctx.trace->cce_calls) {
        if (call.name == "mad") {
            mad = &call;
            break;
        }
    }
    if (mad == nullptr || mad->args.size() <= kMadNArgumentIndex || mad->args[kMadMArgumentIndex] != kGemvM ||
        mad->args[kMadNArgumentIndex] != kPaddedN) {
        return {};
    }
    // The A2/A3 GEMV overload lowers to a single MAD with M=1 and padded N=16;
    // only K changes the MAC-pipe active span.  The CCE completion model
    // retains the MAD head (7/10/14 cycles for K=16/64/128), whereas the
    // repeated MAC active-time contract observes a K/8 steady-state slope.
    const uint64_t k = mad->args[kMadKArgumentIndex];
    uint64_t cycles = (k + kRoundOffsetElements) / kMacElementsPerCycle;
    if (cycles < kMinimumCycles) {
        cycles = kMinimumCycles;
    }
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateColArgTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kBoundaryDiscountCycles = 12;
    const bool isMax = ctx.opcode == "TCOLARGMAX";
    const bool isMin = ctx.opcode == "TCOLARGMIN";
    if ((!isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 1 || ctx.cols <= 0) {
        return {};
    }
    const std::string_view compare = isMax ? "vcmp_ge" : "vcmp_le";
    const uint64_t compareBlocks = CountCceCalls(*ctx.trace, compare);
    if (compareBlocks == 0 || CountCceCalls(*ctx.trace, "vsel") < compareBlocks) {
        return {};
    }
    // The lowering serializes compare -> index select -> value update for
    // every source-row/block pair.  The native model supplies each CCE
    // completion, but aiv_vec_time retains 12 fewer completion cycles at each
    // internal boundary.  Preserve all leaf slopes and remove only that
    // repeated public-PTO-internal boundary charge.
    const uint64_t discount = kBoundaryDiscountCycles * compareBlocks;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles,
        ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePartArgTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kSelectCallsPerBlock = 2;
    constexpr uint64_t kMaxBoundaryDiscountCycles = 19;
    constexpr uint64_t kMinBoundaryDiscountCycles = 20;
    const bool isMax = ctx.opcode == "TPARTARGMAX";
    const bool isMin = ctx.opcode == "TPARTARGMIN";
    if ((!isMax && !isMin) || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0) {
        return {};
    }
    const std::string_view compare = isMax ? "vcmp_ge" : "vcmp_le";
    const uint64_t compareBlocks = CountCceCalls(*ctx.trace, compare);
    if (compareBlocks == 0 || CountCceCalls(*ctx.trace, "vsel") != kSelectCallsPerBlock * compareBlocks) {
        return {};
    }
    // Each block is CMPMASK followed by value and index VSEL.  In the target
    // pipe these three issue streams share their completion boundary; summing
    // isolated CCE completions overcharge 19 cycles for GE and 20 for LE.
    const uint64_t discount = (isMax ? kMaxBoundaryDiscountCycles : kMinBoundaryDiscountCycles) * compareBlocks;
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimatePreluTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kMinimumBarrierCount = 3;
    constexpr uint64_t kCyclesPerRow = 17;
    if (ctx.opcode != "TPRELU" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vmul") == 0 || CountCceCalls(*ctx.trace, "vsel") == 0 ||
        CountCceCalls(*ctx.trace, "pipe_barrier") < kMinimumBarrierCount) {
        return {};
    }
    // TPRELU is TMUL -> TCMPS -> TSEL.  Real A3 execution retains a 17-cycle
    // per-row dependency/mask issue interval that is absent when the three
    // independently calibrated CCE streams are merged by the stub timeline.
    return {ctx.stub_cycles + kCyclesPerRow * static_cast<uint64_t>(ctx.rows), ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateDequantTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kElementsPerRepeat = 64;
    constexpr uint64_t kInt8BaseBoundaryCycles = 40;
    constexpr uint64_t kInt16BaseBoundaryCycles = 34;
    constexpr uint64_t kOverlapCyclesPerAdditionalRepeat = 3;
    if (ctx.opcode != "TDEQUANT" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        (ctx.src_dtype != "int8" && ctx.src_dtype != "int16") ||
        CountCceCalls(*ctx.trace, "vadds") != static_cast<uint64_t>(ctx.rows) ||
        CountCceCalls(*ctx.trace, "vmuls") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t repeats = (static_cast<uint64_t>(ctx.cols) + kElementsPerRepeat - 1) / kElementsPerRepeat;
    const uint64_t baseBoundary = ctx.src_dtype == "int8" ? kInt8BaseBoundaryCycles : kInt16BaseBoundaryCycles;
    // Conversion, scalar load, add and multiply are serialized per row.  Each
    // extra 64-element conversion repeat overlaps three cycles of that fixed
    // boundary in aiv_vec_time; CCE leaf issue/completion values remain
    // untouched.
    const uint64_t overlap = kOverlapCyclesPerAdditionalRepeat * (repeats - 1);
    const uint64_t perRowDiscount = overlap < baseBoundary ? baseBoundary - overlap : 0;
    const uint64_t discount = perRowDiscount * static_cast<uint64_t>(ctx.rows);
    return {
        ctx.stub_cycles > discount ? ctx.stub_cycles - discount : ctx.stub_cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateSelsTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLongRowThreshold = 32;
    constexpr uint64_t kMaskBoundaryElements = 64;
    constexpr uint64_t kB32ElementsPerRepeat = 64;
    constexpr uint64_t kB16ElementsPerRepeat = 128;
    constexpr uint64_t kLongRowCyclesPerRow = 3;
    constexpr uint64_t kLongRowEndpointCycles = 9;
    constexpr uint64_t kLongB32AdditionalRepeatCycles = 11;
    constexpr uint64_t kShortB32SubBoundaryCycles = 36;
    constexpr uint64_t kShortB32BaseCycles = 47;
    constexpr uint64_t kShortB32AdditionalRepeatCycles = 21;
    constexpr uint64_t kShortB16SubBoundaryCycles = 50;
    constexpr uint64_t kShortB16BaseCycles = 41;
    constexpr uint64_t kShortB16AdditionalRepeatCycles = 25;
    if (ctx.opcode != "TSELS" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vsel") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t dtypeBytes = DTypeBytes(ctx.dst_dtype);
    if (dtypeBytes != sizeof(uint16_t) && dtypeBytes != sizeof(uint32_t)) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    const bool isB32 = dtypeBytes == sizeof(uint32_t);
    const uint64_t elementsPerRepeat = isB32 ? kB32ElementsPerRepeat : kB16ElementsPerRepeat;
    uint64_t cycles;
    if (rows >= kLongRowThreshold) {
        cycles = kLongRowCyclesPerRow * rows + kLongRowEndpointCycles;
        if (isB32 && cols > elementsPerRepeat) {
            cycles += kLongB32AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
        }
    } else if (isB32) {
        cycles = cols < kMaskBoundaryElements ?
                     kShortB32SubBoundaryCycles :
                     kShortB32BaseCycles +
                         kShortB32AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
    } else {
        cycles = cols < kMaskBoundaryElements ?
                     kShortB16SubBoundaryCycles :
                     kShortB16BaseCycles +
                         kShortB16AdditionalRepeatCycles * ((cols + elementsPerRepeat - 1) / elementsPerRepeat - 1);
    }
    // TSELS writes the scalar through PIPE_S, transfers it to CMPMASK, then
    // issues one tensor-scalar VSEL per row.  The implementation bit-casts all
    // 32-bit payloads to float and all 16-bit payloads to half, so integer and
    // floating-point types of the same width share the same VSEL path.  The
    // measured sub-64 mask path is slower than the aligned 64-element boundary;
    // preserve that non-monotonic transition rather than treating it as a repeat
    // boundary.  This PTO-level schedule must not change the native VSEL leaf.
    return {cycles, ModelKind::PTO_COMPOSITE_ANALYTIC};
}

inline CycleEstimate EstimateExpandsTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kLongRowThreshold = 64;
    constexpr uint64_t kMediumRowThreshold = 32;
    constexpr uint64_t kWideColumnThreshold = 256;
    constexpr uint64_t kLongFp16CyclesPerRow = 3;
    constexpr uint64_t kLongFp16EndpointCycles = 10;
    constexpr uint64_t kLongOtherCyclesPerRow = 2;
    constexpr uint64_t kLongOtherEndpointCycles = 16;
    constexpr uint64_t kMediumEndpointCycles = 11;
    constexpr uint64_t kWideFp16Cycles = 36;
    constexpr uint64_t kWideOtherCycles = 68;
    constexpr uint64_t kShortEndpointCycles = 10;
    if (ctx.opcode != "TEXPANDS" || ctx.trace == nullptr || ctx.rows <= 0 || ctx.cols <= 0 ||
        CountCceCalls(*ctx.trace, "vector_dup") != static_cast<uint64_t>(ctx.rows)) {
        return {};
    }
    const uint64_t rows = static_cast<uint64_t>(ctx.rows);
    const uint64_t cols = static_cast<uint64_t>(ctx.cols);
    uint64_t cycles;
    if (rows >= kLongRowThreshold) {
        cycles = ctx.dst_dtype == "fp16" ? kLongFp16CyclesPerRow * rows + kLongFp16EndpointCycles :
                                           kLongOtherCyclesPerRow * rows + kLongOtherEndpointCycles;
    } else if (rows >= kMediumRowThreshold) {
        cycles = rows + kMediumEndpointCycles;
    } else if (cols >= kWideColumnThreshold) {
        cycles = ctx.dst_dtype == "fp16" ? kWideFp16Cycles : kWideOtherCycles;
    } else {
        cycles = rows + kShortEndpointCycles;
    }
    // Count-mask VECTOR_DUP has one CCE leaf per row, but the A3 active-time issue
    // path changes at the 32/64-row count-register boundaries.  Project those
    // boundaries at PTO level while retaining the native VDUP profile.
    return {cycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimateImg2ColTargetPipe(const PtoCycleContext& ctx)
{
    constexpr uint64_t kIssueCycles = 2;
    if (ctx.opcode != "TIMG2COL" || ctx.trace == nullptr || CountCceCalls(*ctx.trace, "img2colv2_cbuf_to_ca") != 1) {
        return {};
    }
    // The benchmark measures MTE1 active issue time under repeated identical
    // PTO calls.  One LOAD3D/IMG2COL issue occupies two cycles independent of
    // its completion span; the latter remains modeled by the native CCE leaf.
    return {kIssueCycles, ModelKind::CCE_TARGET_PIPE_PROJECTION};
}

inline CycleEstimate EstimatePtoCycles(const PtoCycleContext& ctx)
{
    if (auto estimate = EstimateColumnReduce(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateFmod(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateCompareTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateMaddTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateMulaTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimatePowTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateIntegerBinaryTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateSerializedVectorLeaf(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateMultiCallShiftTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateTinyFp16UnaryFloor(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateAxpyTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateNotTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateSingleCallCvtTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateGatherPatternTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateScalarLoopShift(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateMaskedDivTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateScalarBitwiseComposite(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateBinaryXorComposite(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateBinaryBitwiseRowStream(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRemainderTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateIntegerRowMinMaxTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowArgCompositeTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowSumTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowExpandDivTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowExpandAddSubTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowExpandMulMinMaxTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRowExpandExpDifTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateRecipTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateUbToUbTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateConcatTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateGemvTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateColArgTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimatePartArgTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimatePreluTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateDequantTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateSelsTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateExpandsTargetPipe(ctx)) {
        return estimate;
    }
    if (auto estimate = EstimateImg2ColTargetPipe(ctx)) {
        return estimate;
    }
    return {};
}

} // namespace pto::mocker::hybrid
