/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the full text of the License in the software repository.
*/
// a5_vf_stub.hpp — A5 VF 循环体捕获的 CPU-mock 桩层。让 __VEC_SCOPE__ 内的 for+body 在 CPU 上
// 编译执行一次,变参记录型 intrinsic 桩只记指令名(不算 cycle),vf_trace 据事件流折叠成 VfInfo 树。
// 桩须在真实 VF 头之前定义(非依赖名查找),include 顺序见文末。
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
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

// 1) CCE 寄存器类型(内建,仓库无定义):空结构,桩只记名字不读写。放全局(datatype.hpp 在 namespace pto 内 using 引用)。
struct vector_f32 {};
struct vector_f16 {};
struct vector_u8 {};
struct vector_u16 {};
struct vector_s8 {};
struct vector_s16 {};
struct vector_s32 {};
struct vector_u32 {};
struct vector_s64 {};
struct vector_u64 {};
struct vector_bool {};
struct vector_address {};
struct vector_align {};

namespace pto {
template <typename T>
struct RegTensor;
}

// 2) 地址/作用域宏,guarded 防与真实头/type.hpp 冲突。
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
#include <pto/common/fifo.hpp>  // TileSplitAxis(pto_instr.hpp 的 TPUSH 等声明引用);须在 __gm__ 等 define 后

// __VEC_SCOPE__ 展开为带 RAII 哨兵的 C++17 if-init:哨兵构造发 scope_enter、析构发 scope_exit,
// pass 据此 marker 只给 scope 内的 for 插桩(scope 外的 host 标量预处理 for 不算 VF)。
// 用 if-init 而非裸哨兵:哨兵生命周期严格 == scope 块,析构紧随块尾,区间精确。
#ifndef __VEC_SCOPE__
extern "C" __attribute__((used)) void __pto_vf_scope_enter();
extern "C" __attribute__((used)) void __pto_vf_scope_exit();
namespace pto::mocker::vf::capture {
void ResetOperands();
}
namespace pto::mocker::vf {
struct ScopeSentinel {
    ScopeSentinel() {
        __pto_vf_scope_enter();
        trace::Arm(true);
        capture::ResetOperands();
    }
    ~ScopeSentinel() {
        __pto_vf_scope_exit();
        trace::Arm(false);
        // 本 VF(__VEC_SCOPE__)结束:事件流折叠成一个 VfInfo,push 进当前 PTO 的 vf_infos。
        // 一个 PTO 理论上可含多个 __VEC_SCOPE__,各自折叠攒着。
        auto &ts = ::pto::mocker::g_trace_state;
        const std::string_view op = ts.active_pto_stack.empty()
                                        ? std::string_view{}
                                        : ts.executed_pto[ts.active_pto_stack.back()].name;
        trace::BuildResult br = trace::BuildVfInfo(op, "");
        if (br.ok) {
            if (!ts.active_pto_stack.empty())
                ts.executed_pto[ts.active_pto_stack.back()].vf_infos.push_back(std::move(br.info));
        }
        trace::Reset();
    }
    explicit operator bool() const { return true; }
};
}  // namespace pto::mocker::vf
#define __VEC_SCOPE__ if (::pto::mocker::vf::ScopeSentinel _pto_vf_scope_{}; _pto_vf_scope_)
#endif

// 3) 模式/常量(内建值,guarded 宏;数值无关,桩不解释,取互异小整数即可)。
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
// CCE_VL:CCE 向量长度(字节),须与 REPEAT_BYTE 一致(cost 侧按此算 loop 次数)。
#ifndef CCE_VL
#define CCE_VL 256
#endif

namespace pto {
// CeilDivision:循环计数上取整,仓库 npu/common 内无定义,在此补。
template <class T, class U>
inline constexpr uint32_t CeilDivision(T a, U b)
{
    return (b == 0) ? 0 : static_cast<uint32_t>((static_cast<uint64_t>(a) + static_cast<uint64_t>(b) - 1) / b);
}
// MrgSortExecutedNumList 真定义在 common/type.hpp(TADD 链必 include),此处不占位,否则 redefinition。
}  // namespace pto

// 4) 捕获上下文:thread_local 指令序列缓冲 + 开关。
namespace pto::mocker::vf::capture {
struct Ctx {
    std::vector<std::string> seq;
    std::unordered_map<uintptr_t, std::string> registers;
    std::unordered_map<uintptr_t, std::string> ubAddresses;
    uint64_t nextRegister = 0;
    uint64_t nextUbAddress = 0;
    bool on = false;
};
inline Ctx &cur()
{
    thread_local Ctx c;
    return c;
}
inline void ResetOperands()
{
    Ctx &c = cur();
    c.registers.clear();
    c.ubAddresses.clear();
    c.nextRegister = 0;
    c.nextUbAddress = 0;
}

inline void rec(VfInst inst)
{
    Ctx &c = cur();
    if (c.on) {
        c.seq.emplace_back(inst.opName);
    }
    // op 事件挂到 vf_trace 事件流(由 Arm() 门控),与 pass 的 loop 事件交织 → BuildVfInfo 建树。
    ::pto::mocker::vf::trace::RecordOp(std::move(inst));
}
inline void rec(const char *name)
{
    rec(VfInst{std::string{name}, {}, {}});
}
inline void clear()
{
    cur().seq.clear();
}
inline void start()
{
    cur().seq.clear();
    cur().on = true;
}
inline void stop()
{
    cur().on = false;
}

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
    if constexpr (std::is_same_v<U, float> || std::is_same_v<U, vector_f32>) return "fp32";
    if constexpr (std::is_same_v<U, half> || std::is_same_v<U, vector_f16>) return "fp16";
    if constexpr (std::is_same_v<U, bfloat16_t>) return "bf16";
    if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, vector_s8>) return "int8";
    if constexpr (std::is_same_v<U, uint8_t> || std::is_same_v<U, vector_u8>) return "uint8";
    if constexpr (std::is_same_v<U, int16_t> || std::is_same_v<U, vector_s16>) return "int16";
    if constexpr (std::is_same_v<U, uint16_t> || std::is_same_v<U, vector_u16>) return "uint16";
    if constexpr (std::is_same_v<U, int32_t> || std::is_same_v<U, vector_s32>) return "int32";
    if constexpr (std::is_same_v<U, uint32_t> || std::is_same_v<U, vector_u32>) return "uint32";
    if constexpr (std::is_same_v<U, vector_bool>) return "bool";
    return "unknown";
}

inline std::string NameFor(std::unordered_map<uintptr_t, std::string> &names, uintptr_t key,
                           uint64_t &next, const char *prefix)
{
    const auto [it, inserted] = names.try_emplace(key);
    if (inserted) it->second = std::string(prefix) + std::to_string(next++);
    return it->second;
}

template <typename T>
inline std::optional<MemInfo> Operand(T &&value)
{
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    Ctx &c = cur();
    if constexpr (RegTensorTraits<U>::value) {
        using DType = typename RegTensorTraits<U>::DType;
        const auto key = reinterpret_cast<uintptr_t>(std::addressof(value));
        return MemInfo{NameFor(c.registers, key, c.nextRegister, "reg"), MemLocation::PhyRegister,
                       DTypeName<DType>()};
    } else if constexpr (std::is_pointer_v<U>) {
        using DType = std::remove_cv_t<std::remove_pointer_t<U>>;
        const auto key = reinterpret_cast<uintptr_t>(value);
        return MemInfo{NameFor(c.ubAddresses, key, c.nextUbAddress, "ub"), MemLocation::UB,
                       DTypeName<DType>()};
    } else if constexpr (std::is_same_v<U, vector_f32> || std::is_same_v<U, vector_f16> ||
                         std::is_same_v<U, vector_s8> || std::is_same_v<U, vector_u8> ||
                         std::is_same_v<U, vector_s16> || std::is_same_v<U, vector_u16> ||
                         std::is_same_v<U, vector_s32> || std::is_same_v<U, vector_u32>) {
        const auto key = reinterpret_cast<uintptr_t>(std::addressof(value));
        return MemInfo{NameFor(c.registers, key, c.nextRegister, "reg"), MemLocation::PhyRegister, DTypeName<U>()};
    }
    return std::nullopt;
}

template <typename... A>
inline void RecordCompute(const char *name, A &&...args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto dst = Operand(std::get<0>(operands))) inst.dst.push_back(std::move(*dst));
    }
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        ([&] {
            if (auto src = Operand(std::get<I + 1>(operands))) inst.src.push_back(std::move(*src));
        }(), ...);
    }(std::make_index_sequence<(sizeof...(A) > 0 ? sizeof...(A) - 1 : 0)>{});
    rec(std::move(inst));
}

template <typename... A>
inline void RecordLoad(const char *name, A &&...args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto dst = Operand(std::get<0>(operands))) inst.dst.push_back(std::move(*dst));
    }
    if constexpr (sizeof...(A) > 1) {
        if (auto src = Operand(std::get<1>(operands))) inst.src.push_back(std::move(*src));
    }
    rec(std::move(inst));
}

template <typename... A>
inline void RecordStore(const char *name, A &&...args)
{
    VfInst inst{std::string{name}, {}, {}};
    auto operands = std::forward_as_tuple(std::forward<A>(args)...);
    if constexpr (sizeof...(A) > 0) {
        if (auto src = Operand(std::get<0>(operands))) inst.src.push_back(std::move(*src));
    }
    if constexpr (sizeof...(A) > 1) {
        if (auto dst = Operand(std::get<1>(operands))) inst.dst.push_back(std::move(*dst));
    }
    rec(std::move(inst));
}
}  // namespace pto::mocker::vf::capture

// 5) 变参记录型 intrinsic 桩(全局命名空间,匹配 CCE 内建查找)。VOID 返回 void,MASK 返回 vector_bool。
//    slice 仅用 vlds/vsts/vadd/plt_b*/PSetWithType；plt_b* 只生成 predicate，不记 micro-op。
#define PTO_VF_RECORD_VOID(NAME)                                  \
    template <class... A> inline void NAME(A &&...args)           \
    {                                                             \
        ::pto::mocker::vf::capture::RecordCompute(#NAME, std::forward<A>(args)...); \
    }
#define PTO_VF_RECORD_MASK(NAME)                                  \
    template <class... A> inline ::vector_bool NAME(A &&...)      \
    {                                                             \
        ::pto::mocker::vf::capture::rec(#NAME);                   \
        return ::vector_bool{};                                   \
    }

template <class... A> inline void vlds(A &&...args)
{
    ::pto::mocker::vf::capture::RecordLoad("vlds", std::forward<A>(args)...);
}
template <class... A> inline void vsts(A &&...args)
{
    ::pto::mocker::vf::capture::RecordStore("vsts", std::forward<A>(args)...);
}
PTO_VF_RECORD_VOID(vdup)
PTO_VF_RECORD_VOID(vadd)
PTO_VF_RECORD_VOID(vsub)
PTO_VF_RECORD_VOID(vmul)
PTO_VF_RECORD_VOID(vdiv)
PTO_VF_RECORD_VOID(vmax)
PTO_VF_RECORD_VOID(vmin)
PTO_VF_RECORD_VOID(vcvt)
PTO_VF_RECORD_VOID(vexp)
PTO_VF_RECORD_VOID(vsqrt)
PTO_VF_RECORD_VOID(vsel)
// binop 扩展(整数/融合)
PTO_VF_RECORD_VOID(vand)
PTO_VF_RECORD_VOID(vor)
PTO_VF_RECORD_VOID(vxor)
PTO_VF_RECORD_VOID(vshl)
PTO_VF_RECORD_VOID(vshr)
PTO_VF_RECORD_VOID(vmadd)
// unary 扩展
PTO_VF_RECORD_VOID(vabs)
PTO_VF_RECORD_VOID(vln)
PTO_VF_RECORD_VOID(vrelu)
PTO_VF_RECORD_VOID(vlrelu)
PTO_VF_RECORD_VOID(vnot)
// custom-impl(HIGH_PRECISION exp/sqrt/log、Div754 等)展开的额外内建,void-by-ref
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
PTO_VF_RECORD_VOID(vcmps_le)
PTO_VF_RECORD_VOID(vcmps_lt)
PTO_VF_RECORD_VOID(vcmps_ne)
PTO_VF_RECORD_VOID(pand)
PTO_VF_RECORD_VOID(por)
PTO_VF_RECORD_VOID(pnot)
// plt_b* 只设置 predicate，不是 VF micro-op，不进入 VfInfo。
template <class... A> inline ::vector_bool plt_b8(A &&...)
{
    return ::vector_bool{};
}
template <class... A> inline ::vector_bool plt_b16(A &&...)
{
    return ::vector_bool{};
}
template <class... A> inline ::vector_bool plt_b32(A &&...)
{
    return ::vector_bool{};
}
// pset_b*:PSetWithType<T>(utils.hpp)内部调用,桩这几个即可;PSetWithType 本身复用真实定义。
PTO_VF_RECORD_MASK(pset_b8)
PTO_VF_RECORD_MASK(pset_b16)
PTO_VF_RECORD_MASK(pset_b32)
// pipe_barrier:同步屏障,记一条 MemBar。
template <class... A> inline void pipe_barrier(A &&...)
{
    ::pto::mocker::vf::capture::cur().on
        ? (void)::pto::mocker::vf::capture::cur().seq.emplace_back("pipe_barrier")
        : (void)0;
    ::pto::mocker::vf::trace::RecordMemBar("pipe_barrier");
}

#undef PTO_VF_RECORD_VOID
#undef PTO_VF_RECORD_MASK

// 5c) PTO 入口链 VF 外内建的 no-op 兜底:搬运/同步/padding,不在 scope 内,不参与 VF 建树。
template <class... A> inline void set_flag(A &&...) {}
template <class... A> inline void wait_flag(A &&...) {}
template <class... A> inline void dsb(A &&...) {}
template <class... A> inline void copy_ubuf_to_ubuf(A &&...) {}
template <class... A> inline void set_mov_pad_val(A &&...) {}
template <class... A> inline void set_pad_val_outtol1(A &&...) {}
template <class... A> inline void set_pad_val_outtoub(A &&...) {}
template <class... A> inline void set_intra_block(A &&...) {}
template <class... A> inline void wait_intra_block(A &&...) {}

// 5b) CCE 异构浮点类型 + QuantMode_t:common.hpp 的 is_same/quant 分支引用(仅类型标记,空结构足矣)。
struct hifloat8_t {};

namespace pto {
enum class QuantMode_t {
    NoQuant,
    F322F16,
    F322BF16,
    QF322B8_PRE,
    QF322HIF8_PRE,
    QF322F16_PRE,
    QF322BF16_PRE,
    VQF322FP8_PRE,
    REQ8,
    DEQF16,
    QS322BF16_PRE,
    VQF322B8_PRE,
    VQF322HIF8_PRE,
    VQF322F16_PRE,
    VQF322BF16_PRE,
    VREQ8,
    VDEQF16,
    VQS322BF16_PRE
};
}  // namespace pto

// FloatIntUnion<float/half> 的 .f 与 HalfUnion 在 type.hpp 被 __CCE_AICORE__ 挡住,custom 实现
// (Div754/TSqrtHp…,经 TDiv/TUnaryOp 引入)需用 → 在此显式特化补出(对齐 type.hpp 的 AICORE 版)。
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
}  // namespace pto

// 6) 复用真实 VF 头。顺序严格:TypeGet → utils → common → 全局 DistVST 别名 → VF 驱动/Op 头。
//    各 intrinsic/NORM/MODE_ZEROING/CeilDivision 均已上文可见,满足非依赖名查找。
#include <pto/npu/a5/datatype.hpp>
#include <pto/npu/a5/common.hpp>
#include <pto/npu/a5/utils.hpp>
// TBinOp.hpp 写 `::DistVST`(全局),utils.hpp 把 DistVST 放 namespace pto,补全局别名。
using DistVST = ::pto::DistVST;
#include <pto/npu/a5/TBinOp.hpp>
#include <pto/npu/a5/TAdd.hpp>
