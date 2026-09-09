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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <source_location>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <pto/common/type.hpp>
#include <ostream>
#include <pto/cpu/MXTypes.hpp>
#include "vf_trace.hpp"
#include "pto/costmodel/trace.hpp"

// A5 vector register stand-ins used by the raw VF intrinsic mocks below.
struct vector_stub {
    vector_stub() = default;
    template <class T>
    vector_stub(const T&)
    {}
};
struct vector_f32 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_f16 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_u8 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_u16 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_s8 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_s16 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_s32 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_u32 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_s64 : vector_stub {
    using vector_stub::vector_stub;
};
struct vector_u64 : vector_stub {
    using vector_stub::vector_stub;
};

inline uintptr_t PtoPredicateCallsiteKey(const std::source_location& location)
{
    uintptr_t key = std::hash<std::string_view>{}(location.file_name());
    key ^= static_cast<uintptr_t>(location.line()) << 16;
    key ^= static_cast<uintptr_t>(location.column());
    return key == 0 ? 1 : key;
}

struct vector_bool {
    vector_bool() : traceKey(reinterpret_cast<uintptr_t>(this)) {}
    explicit vector_bool(uintptr_t key) : traceKey(key) {}

    uintptr_t traceKey;
};
struct vector_address {};
struct vector_align {};
enum class Spr : uint8_t { AR = 74 };

namespace pto {
template <typename T>
struct RegTensor;
}

// Device attributes and ISA constants unavailable in the host costmodel build.
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef __out__
#define __out__
#endif
#ifndef __in__
#define __in__
#endif
#ifndef __tf__
#define __tf__
#endif
#ifndef __cbuf__
#define __cbuf__
#endif
#ifndef __ca__
#define __ca__
#endif
#ifndef __cb__
#define __cb__
#endif
#ifndef __cc__
#define __cc__
#endif
#ifndef __fbuf__
#define __fbuf__
#endif
#ifndef set_vector_mask
#define set_vector_mask(...)
#endif
#ifndef set_mask_norm
#define set_mask_norm(...)
#endif
#ifndef __cce_get_tile_ptr
#define __cce_get_tile_ptr(x) (x)
#endif
#ifndef __simt_callee__
#define __simt_callee__
#endif
#ifndef __simt_vf__
#define __simt_vf__
#endif
#ifndef LAUNCH_BOUND
#define LAUNCH_BOUND(x)
#endif
#include <pto/common/fifo.hpp>

#ifndef __VEC_SCOPE__
extern "C" __attribute__((used)) void __pto_vf_scope_enter();
extern "C" __attribute__((used)) void __pto_vf_scope_exit();
// VF scope and trace capture state.
namespace pto::mocker::vf::capture {
void ResetOperands();
}
namespace pto::mocker::vf {
inline uint64_t FallbackCyclesForCurrentTraceEvents()
{
    uint64_t cycles = 0;
    for (const trace::Event& event : trace::Events()) {
        if (event.kind == trace::EvKind::Op) {
            cycles += FallbackVecCycle(event.inst.opName);
        } else if (event.kind == trace::EvKind::MemBar) {
            cycles += kMemBarPenaltyPlaceholder;
        }
    }
    return cycles == 0 ? kUnknownInstructionFallbackCycles : cycles;
}

struct ScopeSentinel {
    ScopeSentinel()
    {
        __pto_vf_scope_enter();
        trace::Arm(true);
        capture::ResetOperands();
    }
    ~ScopeSentinel()
    {
        __pto_vf_scope_exit();
        trace::Arm(false);
        auto& ts = ::pto::mocker::g_trace_state;
        const std::string_view op =
            ts.active_pto_stack.empty() ? std::string_view{} : ts.executed_pto[ts.active_pto_stack.back()].name;
        trace::BuildResult br = trace::BuildVfInfo(op, "");
        if (br.ok) {
            if (!ts.active_pto_stack.empty())
                ts.executed_pto[ts.active_pto_stack.back()].vf_infos.push_back(std::move(br.info));
        } else {
            ::pto::mocker::RecordVfFallbackCycles(FallbackCyclesForCurrentTraceEvents());
        }
        trace::Reset();
    }
    explicit operator bool() const { return true; }
};
} // namespace pto::mocker::vf
#define __VEC_SCOPE__ if (::pto::mocker::vf::ScopeSentinel _pto_vf_scope_{}; _pto_vf_scope_)
#endif

#ifndef NORM
#define NORM 0
#endif
#ifndef POST_UPDATE
#define POST_UPDATE 1
#endif
#ifndef MODE_ZEROING
#define MODE_ZEROING 0
#endif
#ifndef MODE_NORM
#define MODE_NORM 1
#endif
#ifndef PAT_ALL
#define PAT_ALL 0
#endif
#ifndef ROUND_R
#define ROUND_R 0
#endif
#ifndef ROUND_Z
#define ROUND_Z 1
#endif
#ifndef ROUND_F
#define ROUND_F 2
#endif
#ifndef RS_DISABLE
#define RS_DISABLE 0
#endif
#ifndef RS_ENABLE
#define RS_ENABLE 1
#endif
#ifndef PART_EVEN
#define PART_EVEN 0
#endif
#ifndef PART_ODD
#define PART_ODD 1
#endif
#ifndef NORM_B32
#define NORM_B32 0
#endif
#ifndef NORM_B16
#define NORM_B16 1
#endif
#ifndef PK_B32
#define PK_B32 2
#endif
#ifndef PK_B16
#define PK_B16 3
#endif
#ifndef UNPK_B16
#define UNPK_B16 4
#endif
#ifndef UNPK_B32
#define UNPK_B32 5
#endif
#ifndef INC_ORDER
#define INC_ORDER 0
#endif
#ifndef POS_LOWEST
#define POS_LOWEST 1
#endif
#ifndef PAT_ALLF
#define PAT_ALLF 1
#endif
#ifndef MODE_STORED
#define MODE_STORED 2
#endif
#ifndef MODE_MERGING
#define MODE_MERGING 3
#endif
#ifndef PAT_H
#define PAT_H 2
#endif
#ifndef UNPK_B8
#define UNPK_B8 6
#endif
#ifndef PART_P0
#define PART_P0 2
#endif
#ifndef ROUND_A
#define ROUND_A 3
#endif
#ifndef BRC_B16
#define BRC_B16 10
#endif
#ifndef BRC_B8
#define BRC_B8 14
#endif
#ifndef BRC_B32
#define BRC_B32 11
#endif
#ifndef BLK
#define BLK 15
#endif
#ifndef SPR_AR
#define SPR_AR Spr::AR
#endif
#ifndef DINTLV_B32
#define DINTLV_B32 0
#endif
#ifndef INTLV_B32
#define INTLV_B32 0
#endif
#ifndef LOWER
#define LOWER 0
#endif
#ifndef US
#define US 0
#endif
#ifndef CCE_VL
#define CCE_VL 256
#endif

// Small compatibility helpers required by A5 PTO instruction implementations.
namespace pto {
template <class T, class U>
inline constexpr uint32_t CeilDivision(T a, U b)
{
    return (b == 0) ? 0 : static_cast<uint32_t>((static_cast<uint64_t>(a) + static_cast<uint64_t>(b) - 1) / b);
}

} // namespace pto

// Raw VF intrinsic capture helpers.
namespace pto::mocker::vf::capture {
struct Ctx {
    std::vector<std::string> seq;
    std::unordered_map<uintptr_t, std::string> registers;
    std::unordered_map<uintptr_t, std::string> predicates;
    std::unordered_map<uintptr_t, std::string> ubAddresses;
    uint64_t nextRegister = 0;
    uint64_t nextPredicate = 0;
    uint64_t nextUbAddress = 0;
    bool on = false;
};
inline Ctx& cur()
{
    thread_local Ctx c;
    return c;
}
inline void ResetOperands()
{
    Ctx& c = cur();
    c.registers.clear();
    c.predicates.clear();
    c.ubAddresses.clear();
    c.nextRegister = 0;
    c.nextPredicate = 0;
    c.nextUbAddress = 0;
}

inline void rec(VfInst inst)
{
    Ctx& c = cur();
    if (c.on) {
        c.seq.emplace_back(inst.opName);
    }
    ::pto::mocker::vf::trace::RecordOp(std::move(inst));
}
inline void rec(const char* name) { rec(VfInst{std::string{name}, {}, {}}); }
inline void clear() { cur().seq.clear(); }
inline void start()
{
    cur().seq.clear();
    cur().on = true;
}
inline void stop() { cur().on = false; }

template <typename T>
struct RegTensorTraits {
    static constexpr bool value = false;
};

template <typename T>
struct RegTensorTraits<::pto::RegTensor<T>> {
    static constexpr bool value = true;
    using DType = T;
};

template <typename T>
inline std::string DTypeName()
{
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float> || std::is_same_v<U, vector_f32>)
        return "fp32";
    if constexpr (std::is_same_v<U, half> || std::is_same_v<U, vector_f16>)
        return "fp16";
    if constexpr (std::is_same_v<U, bfloat16_t>)
        return "bf16";
    if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, vector_s8>)
        return "int8";
    if constexpr (std::is_same_v<U, uint8_t> || std::is_same_v<U, vector_u8>)
        return "uint8";
    if constexpr (std::is_same_v<U, int16_t> || std::is_same_v<U, vector_s16>)
        return "int16";
    if constexpr (std::is_same_v<U, uint16_t> || std::is_same_v<U, vector_u16>)
        return "uint16";
    if constexpr (std::is_same_v<U, int32_t> || std::is_same_v<U, vector_s32>)
        return "int32";
    if constexpr (std::is_same_v<U, uint32_t> || std::is_same_v<U, vector_u32>)
        return "uint32";
    if constexpr (std::is_same_v<U, vector_bool>)
        return "bool";
    return "unknown";
}

inline std::string NameFor(
    std::unordered_map<uintptr_t, std::string>& names, uintptr_t key, uint64_t& next, const char* prefix)
{
    const auto [it, inserted] = names.try_emplace(key);
    if (inserted)
        it->second = std::string(prefix) + std::to_string(next++);
    return it->second;
}

template <typename T>
inline std::optional<MemInfo> Operand(T&& value)
{
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    Ctx& c = cur();
    if constexpr (RegTensorTraits<U>::value) {
        using DType = typename RegTensorTraits<U>::DType;
        const auto key = reinterpret_cast<uintptr_t>(std::addressof(value));
        return MemInfo{NameFor(c.registers, key, c.nextRegister, "reg"), MemLocation::PhyRegister, DTypeName<DType>()};
    } else if constexpr (std::is_pointer_v<U>) {
        using DType = std::remove_cv_t<std::remove_pointer_t<U>>;
        const auto key = reinterpret_cast<uintptr_t>(value);
        return MemInfo{NameFor(c.ubAddresses, key, c.nextUbAddress, "ub"), MemLocation::UB, DTypeName<DType>()};
    } else if constexpr (std::is_same_v<U, vector_bool>) {
        return MemInfo{
            NameFor(c.predicates, value.traceKey, c.nextPredicate, "predicate"), MemLocation::PredicateRegister,
            DTypeName<U>()};
    } else if constexpr (
        std::is_same_v<U, vector_f32> || std::is_same_v<U, vector_f16> || std::is_same_v<U, vector_s8> ||
        std::is_same_v<U, vector_u8> || std::is_same_v<U, vector_s16> || std::is_same_v<U, vector_u16> ||
        std::is_same_v<U, vector_s32> || std::is_same_v<U, vector_u32>) {
        const auto key = reinterpret_cast<uintptr_t>(std::addressof(value));
        return MemInfo{NameFor(c.registers, key, c.nextRegister, "reg"), MemLocation::PhyRegister, DTypeName<U>()};
    }
    return std::nullopt;
}

template <typename T>
inline void AppendOperand(std::vector<MemInfo>& operands, T&& value)
{
    if (auto operand = Operand(std::forward<T>(value)))
        operands.push_back(std::move(*operand));
}

template <typename Tuple, std::size_t... DstIndices, std::size_t... SrcIndices>
inline VfInst MakeInstruction(
    const char* name, Tuple& operands, std::index_sequence<DstIndices...>, std::index_sequence<SrcIndices...>)
{
    VfInst inst{std::string{name}, {}, {}, {}};
    (AppendOperand(inst.dst, std::get<DstIndices>(operands)), ...);
    (AppendOperand(inst.src, std::get<SrcIndices>(operands)), ...);
    return inst;
}

template <typename T>
inline std::string ArgumentValue(T&& value)
{
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (requires { U::value; }) {
        return ArgumentValue(U::value);
    } else if constexpr (std::is_enum_v<U>) {
        return std::to_string(static_cast<std::underlying_type_t<U>>(value));
    } else if constexpr (std::is_integral_v<U>) {
        return std::to_string(value);
    } else if constexpr (std::is_floating_point_v<U>) {
        std::ostringstream output;
        output << value;
        return output.str();
    }
    return "<opaque>";
}

template <typename T>
inline void AddArgument(VfInst& inst, uint32_t argumentIndex, VfArgKind kind, const char* name, T&& value)
{
    inst.arguments.push_back(VfArgInfo{argumentIndex, kind, name, ArgumentValue(std::forward<T>(value))});
}

template <typename... A>
inline void RecordCompute(const char* name, A&&... args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto dst = Operand(std::get<0>(operands)))
            inst.dst.push_back(std::move(*dst));
    }
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (
            [&] {
                if (auto src = Operand(std::get<I + 1>(operands)))
                    inst.src.push_back(std::move(*src));
            }(),
            ...);
    }(std::make_index_sequence<(sizeof...(A) > 0 ? sizeof...(A) - 1 : 0)>{});
    rec(std::move(inst));
}

template <typename... A>
inline void RecordLoad(const char* name, A&&... args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto dst = Operand(std::get<0>(operands)))
            inst.dst.push_back(std::move(*dst));
    }
    if constexpr (sizeof...(A) > 1) {
        if (auto src = Operand(std::get<1>(operands)))
            inst.src.push_back(std::move(*src));
    }
    rec(std::move(inst));
}

template <typename... A>
inline void RecordStore(const char* name, A&&... args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto src = Operand(std::get<0>(operands)))
            inst.src.push_back(std::move(*src));
    }
    if constexpr (sizeof...(A) > 1) {
        if (auto dst = Operand(std::get<1>(operands)))
            inst.dst.push_back(std::move(*dst));
    }
    rec(std::move(inst));
}
} // namespace pto::mocker::vf::capture

#define PTO_VF_RECORD_VOID(NAME)                                                    \
    template <class... A>                                                           \
    inline void NAME(A&&... args)                                                   \
    {                                                                               \
        ::pto::mocker::vf::capture::RecordCompute(#NAME, std::forward<A>(args)...); \
    }
#define PTO_VF_RECORD_MASK(NAME)                                                                                    \
    template <class Pattern>                                                                                        \
    inline ::vector_bool NAME(                                                                                      \
        Pattern&& pattern, const std::source_location& location = std::source_location::current())                  \
    {                                                                                                               \
        ::vector_bool result(PtoPredicateCallsiteKey(location));                                                    \
        auto operands = std::forward_as_tuple(result);                                                              \
        auto inst = ::pto::mocker::vf::capture::MakeInstruction(                                                    \
            #NAME, operands, std::index_sequence<0>{}, std::index_sequence<>{});                                    \
        ::pto::mocker::vf::capture::AddArgument(inst, 0, ::pto::mocker::vf::VfArgKind::Config, "pattern", pattern); \
        ::pto::mocker::vf::capture::rec(std::move(inst));                                                           \
        return result;                                                                                              \
    }

// Raw A5 VF intrinsic mocks. Each mock records a canonical VF instruction.
template <class... A>
inline void vlds(A&&... args)
{
    ::pto::mocker::vf::capture::RecordLoad("vlds", std::forward<A>(args)...);
}
template <class... A>
inline void vsts(A&&... args)
{
    ::pto::mocker::vf::capture::RecordStore("vsts", std::forward<A>(args)...);
}

template <class Dst, class Src, class Index, class Mask>
inline void vgather2(Dst&& dst, Src&& src, Index&& index, Mask&& mask)
{
    auto operands = std::forward_as_tuple(dst, src, index, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vgather2", operands, std::index_sequence<0>{}, std::index_sequence<1, 2, 3>{}));
}

template <class Src, class Dst, class Index, class Mask>
inline void vscatter(Src&& src, Dst&& dst, Index&& index, Mask&& mask)
{
    auto operands = std::forward_as_tuple(src, dst, index, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vscatter", operands, std::index_sequence<1>{}, std::index_sequence<0, 2, 3>{}));
}

template <class Dst, class Scalar, class Order>
inline void vci(Dst&& dst, Scalar&& scalar, Order&& order)
{
    auto operands = std::forward_as_tuple(dst);
    auto inst =
        ::pto::mocker::vf::capture::MakeInstruction("vci", operands, std::index_sequence<0>{}, std::index_sequence<>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 1, ::pto::mocker::vf::VfArgKind::Immediate, "scalar", scalar);
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Config, "order", order);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Dst, class Src, class Mask, class Mode>
inline void vsqz(Dst&& dst, Src&& src, Mask&& mask, Mode&& mode)
{
    auto operands = std::forward_as_tuple(dst, src, mask);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "vsqz", operands, std::index_sequence<0>{}, std::index_sequence<1, 2>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 3, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class... A>
inline void sprclr(A&&...)
{}

template <class Align, class Src, class Dst, class Update>
inline void vstur(Align&& align, Src&& src, Dst&& dst, Update&& update)
{
    auto operands = std::forward_as_tuple(align, src, dst);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "vstur", operands, std::index_sequence<2>{}, std::index_sequence<0, 1>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 3, ::pto::mocker::vf::VfArgKind::Config, "update", update);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Align, class Dst>
inline void vstar(Align&& align, Dst&& dst)
{
    auto operands = std::forward_as_tuple(align, dst);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vstar", operands, std::index_sequence<1>{}, std::index_sequence<0>{}));
}

template <class SpecialRegister, class Dst, class Offset>
inline void sprsts(SpecialRegister&& specialRegister, Dst&& dst, Offset&& offset)
{
    auto operands = std::forward_as_tuple(specialRegister, dst);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "sprsts", operands, std::index_sequence<1>{}, std::index_sequence<>{});
    ::pto::mocker::vf::capture::AddArgument(
        inst, 0, ::pto::mocker::vf::VfArgKind::Config, "special_register", specialRegister);
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Immediate, "offset", offset);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Dst0, class Dst1, class Src0, class Src1>
inline void RecordPredicateInterleave(const char* name, Dst0&& dst0, Dst1&& dst1, Src0&& src0, Src1&& src1)
{
    auto operands = std::forward_as_tuple(dst0, dst1, src0, src1);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        name, operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3>{}));
}

template <class... A>
inline void pintlv_b8(A&&... args)
{
    RecordPredicateInterleave("pintlv_b8", std::forward<A>(args)...);
}
template <class... A>
inline void pintlv_b16(A&&... args)
{
    RecordPredicateInterleave("pintlv_b16", std::forward<A>(args)...);
}
template <class... A>
inline void pintlv_b32(A&&... args)
{
    RecordPredicateInterleave("pintlv_b32", std::forward<A>(args)...);
}

template <class Dst, class Src, class Predicate, class Mode>
inline void vdup(Dst&& dst, Src&& src, Predicate&& predicate, Mode&& mode)
{
    auto operands = std::forward_as_tuple(dst, src, predicate);
    auto instruction = ::pto::mocker::vf::capture::MakeInstruction(
        "vdup", operands, std::index_sequence<0>{}, std::index_sequence<1, 2>{});
    if (!::pto::mocker::vf::capture::Operand(src)) {
        ::pto::mocker::vf::capture::AddArgument(instruction, 1, ::pto::mocker::vf::VfArgKind::Immediate, "scalar", src);
    }
    ::pto::mocker::vf::capture::AddArgument(instruction, 3, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::rec(std::move(instruction));
}

template <class Dst, class Src, class Predicate, class Mode, class ExtendedMode>
inline void vdup(Dst&& dst, Src&& src, Predicate&& predicate, Mode&& mode, ExtendedMode&& extendedMode)
{
    auto operands = std::forward_as_tuple(dst, src, predicate);
    auto instruction = ::pto::mocker::vf::capture::MakeInstruction(
        "vdup", operands, std::index_sequence<0>{}, std::index_sequence<1, 2>{});
    if (!::pto::mocker::vf::capture::Operand(src)) {
        ::pto::mocker::vf::capture::AddArgument(instruction, 1, ::pto::mocker::vf::VfArgKind::Immediate, "scalar", src);
    }
    ::pto::mocker::vf::capture::AddArgument(instruction, 3, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::AddArgument(
        instruction, 4, ::pto::mocker::vf::VfArgKind::Config, "extended_mode", extendedMode);
    ::pto::mocker::vf::capture::rec(std::move(instruction));
}

PTO_VF_RECORD_VOID(vadd)
PTO_VF_RECORD_VOID(vsub)
PTO_VF_RECORD_VOID(vmul)
PTO_VF_RECORD_VOID(vdiv)
PTO_VF_RECORD_VOID(vmod)
PTO_VF_RECORD_VOID(vpack)
PTO_VF_RECORD_VOID(vaxpy)
PTO_VF_RECORD_VOID(vmax)
PTO_VF_RECORD_VOID(vmin)
PTO_VF_RECORD_VOID(vcvt)
PTO_VF_RECORD_VOID(vexp)
PTO_VF_RECORD_VOID(vsqrt)
PTO_VF_RECORD_VOID(vsel)
PTO_VF_RECORD_VOID(vand)
PTO_VF_RECORD_VOID(vor)
PTO_VF_RECORD_VOID(vxor)
PTO_VF_RECORD_VOID(vshl)
PTO_VF_RECORD_VOID(vshr)
PTO_VF_RECORD_VOID(vmadd)
PTO_VF_RECORD_VOID(vabs)
PTO_VF_RECORD_VOID(vln)
PTO_VF_RECORD_VOID(vrelu)
PTO_VF_RECORD_VOID(vlrelu)
PTO_VF_RECORD_VOID(vnot)
PTO_VF_RECORD_VOID(vadds)
PTO_VF_RECORD_VOID(vmins)
PTO_VF_RECORD_VOID(vmov)
PTO_VF_RECORD_VOID(vmula)
PTO_VF_RECORD_VOID(vmuls)
PTO_VF_RECORD_VOID(vneg)
PTO_VF_RECORD_VOID(vshls)
PTO_VF_RECORD_VOID(vshrs)
PTO_VF_RECORD_VOID(vtrc)
PTO_VF_RECORD_VOID(vcmp_eq)
PTO_VF_RECORD_VOID(vcmp_gt)
PTO_VF_RECORD_VOID(vcmp_le)
PTO_VF_RECORD_VOID(vcmp_lt)
PTO_VF_RECORD_VOID(vcmp_ne)
PTO_VF_RECORD_VOID(vcmps_eq)
PTO_VF_RECORD_VOID(vcmps_ge)
PTO_VF_RECORD_VOID(vcmps_gt)
PTO_VF_RECORD_VOID(vcmps_le)
PTO_VF_RECORD_VOID(vcmps_lt)
PTO_VF_RECORD_VOID(vcmps_ne)
PTO_VF_RECORD_VOID(pand)
PTO_VF_RECORD_VOID(por)
PTO_VF_RECORD_VOID(pnot)

template <class Dst, class Src0, class Src1, class Mask>
inline void pxor(Dst&& dst, Src0&& src0, Src1&& src1, Mask&& mask)
{
    auto operands = std::forward_as_tuple(dst, src0, src1, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "pxor", operands, std::index_sequence<0>{}, std::index_sequence<1, 2, 3>{}));
}

template <class Dst, class Src0, class Src1, class Src2>
inline void psel(Dst&& dst, Src0&& src0, Src1&& src1, Src2&& src2)
{
    auto operands = std::forward_as_tuple(dst, src0, src1, src2);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "psel", operands, std::index_sequence<0>{}, std::index_sequence<1, 2, 3>{}));
}

template <class Dst, class Src, class Offset, class Mode>
inline void plds(Dst&& dst, Src&& src, Offset&& offset, Mode&& mode)
{
    auto operands = std::forward_as_tuple(dst, src);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "plds", operands, std::index_sequence<0>{}, std::index_sequence<1>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Immediate, "offset", offset);
    ::pto::mocker::vf::capture::AddArgument(inst, 3, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Dst, class Src, class Part>
inline void ppack(Dst&& dst, Src&& src, Part&& part)
{
    auto operands = std::forward_as_tuple(dst, src);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "ppack", operands, std::index_sequence<0>{}, std::index_sequence<1>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Config, "part", part);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Src, class Dst, class Offset, class Mode>
inline void psts(Src&& src, Dst&& dst, Offset&& offset, Mode&& mode)
{
    auto operands = std::forward_as_tuple(src, dst);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "psts", operands, std::index_sequence<1>{}, std::index_sequence<0>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Immediate, "offset", offset);
    ::pto::mocker::vf::capture::AddArgument(inst, 3, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Dst, class Src, class Part>
inline void punpack(Dst&& dst, Src&& src, Part&& part)
{
    auto operands = std::forward_as_tuple(dst, src);
    auto inst = ::pto::mocker::vf::capture::MakeInstruction(
        "punpack", operands, std::index_sequence<0>{}, std::index_sequence<1>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 2, ::pto::mocker::vf::VfArgKind::Config, "part", part);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class CarryOut, class Dst, class Src0, class Src1, class Mask>
inline void vaddc(CarryOut&& carryOut, Dst&& dst, Src0&& src0, Src1&& src1, Mask&& mask)
{
    auto operands = std::forward_as_tuple(carryOut, dst, src0, src1, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vaddc", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3, 4>{}));
}

template <class CarryOut, class Dst, class Src0, class Src1, class CarryIn, class Mask>
inline void vaddcs(CarryOut&& carryOut, Dst&& dst, Src0&& src0, Src1&& src1, CarryIn&& carryIn, Mask&& mask)
{
    auto operands = std::forward_as_tuple(carryOut, dst, src0, src1, carryIn, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vaddcs", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3, 4, 5>{}));
}

template <class CarryOut, class Dst, class Src0, class Src1, class Mask>
inline void vsubc(CarryOut&& carryOut, Dst&& dst, Src0&& src0, Src1&& src1, Mask&& mask)
{
    auto operands = std::forward_as_tuple(carryOut, dst, src0, src1, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vsubc", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3, 4>{}));
}

template <class CarryOut, class Dst, class Src0, class Src1, class CarryIn, class Mask>
inline void vsubcs(CarryOut&& carryOut, Dst&& dst, Src0&& src0, Src1&& src1, CarryIn&& carryIn, Mask&& mask)
{
    auto operands = std::forward_as_tuple(carryOut, dst, src0, src1, carryIn, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vsubcs", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3, 4, 5>{}));
}

template <class Dst0, class Dst1, class Src0, class Src1, class Mask>
inline void vmull(Dst0&& dst0, Dst1&& dst1, Src0&& src0, Src1&& src1, Mask&& mask)
{
    auto operands = std::forward_as_tuple(dst0, dst1, src0, src1, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vmull", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3, 4>{}));
}

template <class Dst, class Src0, class Src1, class Mask>
inline void vcmp_ge(Dst&& dst, Src0&& src0, Src1&& src1, Mask&& mask)
{
    auto operands = std::forward_as_tuple(dst, src0, src1, mask);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vcmp_ge", operands, std::index_sequence<0>{}, std::index_sequence<1, 2, 3>{}));
}

template <class Dst, class Immediate>
inline void vbr(Dst&& dst, Immediate&& immediate)
{
    auto operands = std::forward_as_tuple(dst);
    auto inst =
        ::pto::mocker::vf::capture::MakeInstruction("vbr", operands, std::index_sequence<0>{}, std::index_sequence<>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 1, ::pto::mocker::vf::VfArgKind::Immediate, "value", immediate);
    ::pto::mocker::vf::capture::rec(std::move(inst));
}

template <class Dst0, class Dst1, class Src0, class Src1>
inline void vintlv(Dst0&& dst0, Dst1&& dst1, Src0&& src0, Src1&& src1)
{
    auto operands = std::forward_as_tuple(dst0, dst1, src0, src1);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vintlv", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3>{}));
}

template <class Dst0, class Dst1, class Src0, class Src1>
inline void vdintlv(Dst0&& dst0, Dst1&& dst1, Src0&& src0, Src1&& src1)
{
    auto operands = std::forward_as_tuple(dst0, dst1, src0, src1);
    ::pto::mocker::vf::capture::rec(::pto::mocker::vf::capture::MakeInstruction(
        "vdintlv", operands, std::index_sequence<0, 1>{}, std::index_sequence<2, 3>{}));
}

template <class Count, class Mode>
inline ::vector_bool RecordPlt(const char* name, Count&& count, Mode&& mode, const std::source_location& location)
{
    ::vector_bool result(PtoPredicateCallsiteKey(location));
    auto operands = std::forward_as_tuple(result);
    auto inst =
        ::pto::mocker::vf::capture::MakeInstruction(name, operands, std::index_sequence<0>{}, std::index_sequence<>{});
    ::pto::mocker::vf::capture::AddArgument(inst, 0, ::pto::mocker::vf::VfArgKind::Immediate, "count", count);
    ::pto::mocker::vf::capture::AddArgument(inst, 1, ::pto::mocker::vf::VfArgKind::Config, "mode", mode);
    ::pto::mocker::vf::capture::rec(std::move(inst));
    return result;
}

template <class Count, class Mode>
inline ::vector_bool plt_b8(
    Count&& count, Mode&& mode, const std::source_location& location = std::source_location::current())
{
    return RecordPlt("plt_b8", std::forward<Count>(count), std::forward<Mode>(mode), location);
}
template <class Count, class Mode>
inline ::vector_bool plt_b16(
    Count&& count, Mode&& mode, const std::source_location& location = std::source_location::current())
{
    return RecordPlt("plt_b16", std::forward<Count>(count), std::forward<Mode>(mode), location);
}
template <class Count, class Mode>
inline ::vector_bool plt_b32(
    Count&& count, Mode&& mode, const std::source_location& location = std::source_location::current())
{
    return RecordPlt("plt_b32", std::forward<Count>(count), std::forward<Mode>(mode), location);
}
PTO_VF_RECORD_MASK(pset_b8)
PTO_VF_RECORD_MASK(pset_b16)
PTO_VF_RECORD_MASK(pset_b32)

#undef PTO_VF_RECORD_VOID
#undef PTO_VF_RECORD_MASK

// A5 instruction-header compatibility types.
namespace pto {
enum QuantMode_t {
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 16,
    DEQF16 = 5,
    VDEQF16 = 4,
    QF322B8_PRE = 24,
    QF322HIF8_PRE = 25,
    QF322FP8_PRE = 26,
    QF322F32_PRE = 27,
    QF322F16_PRE = 32,
    QF322BF16_PRE = 34,
    QS322BF16_PRE = 35,
    VQF322B8_PRE = 23,
    VQF322HIF8_PRE = 28,
    VQF322F16_PRE = 33,
    VQF322BF16_PRE = 36,
    VQF322FP8_PRE = 37,
    VQF322F32_PRE = 38,
    REQ8 = 3,
    VREQ8 = 2,
    VQS322BF16_PRE = 39
};
enum class atomic_type_t { ATOMIC_NONE };
} // namespace pto

namespace pto {
template <>
union FloatIntUnion<float> {
    uint32_t i;
    float f;
    constexpr FloatIntUnion() : f(0.0f) {}
    constexpr FloatIntUnion(uint32_t v) : i(v) {}
};
template <>
union FloatIntUnion<half> {
    uint16_t i;
    half f;
    constexpr FloatIntUnion() : f(0) {}
    constexpr FloatIntUnion(uint16_t v) : i(v) {}
};
using HalfUnion = FloatIntUnion<half>;
} // namespace pto
