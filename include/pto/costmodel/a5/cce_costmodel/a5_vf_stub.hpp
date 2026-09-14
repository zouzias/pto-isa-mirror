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

#include <cstddef>
#include <cstdint>

#include <pto/common/type.hpp>
#include <pto/cpu/MXTypes.hpp>

// Host-only stand-ins for A5 vector register types. They make the device implementation headers
// parsable by a standard C++ compiler; they intentionally do not record or execute VF instructions.
struct vector_stub {
    vector_stub() = default;
    template <class T>
    vector_stub(const T&)
    {}
};

#define PTO_A5_VECTOR_STUB(TYPE)       \
    struct TYPE : vector_stub {         \
        using vector_stub::vector_stub; \
    }

PTO_A5_VECTOR_STUB(vector_f32);
PTO_A5_VECTOR_STUB(vector_f16);
PTO_A5_VECTOR_STUB(vector_u8);
PTO_A5_VECTOR_STUB(vector_u16);
PTO_A5_VECTOR_STUB(vector_s8);
PTO_A5_VECTOR_STUB(vector_s16);
PTO_A5_VECTOR_STUB(vector_s32);
PTO_A5_VECTOR_STUB(vector_u32);
PTO_A5_VECTOR_STUB(vector_s64);
PTO_A5_VECTOR_STUB(vector_u64);
PTO_A5_VECTOR_STUB(vector_bool);

#undef PTO_A5_VECTOR_STUB

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

namespace pto::mocker::a5::host {

// A5 implementations contain synchronization operations inside __VEC_SCOPE__. The host costmodel
// must ignore those device-internal operations while retaining synchronization around PTO calls.
inline thread_local uint32_t gVfScopeDepth = 0;

class VfScopeSentinel {
public:
    VfScopeSentinel() { ++gVfScopeDepth; }
    ~VfScopeSentinel() { --gVfScopeDepth; }
    explicit operator bool() const { return true; }
};

inline bool InVfScope() { return gVfScopeDepth != 0; }

} // namespace pto::mocker::a5::host

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__                                                                                              \
    if (::pto::mocker::a5::host::VfScopeSentinel _ptoA5VfScope{}; static_cast<bool>(_ptoA5VfScope))
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

namespace pto {
template <class T, class U>
inline constexpr uint32_t CeilDivision(T a, U b)
{
    return (b == 0) ? 0 : static_cast<uint32_t>((static_cast<uint64_t>(a) + static_cast<uint64_t>(b) - 1) / b);
}
} // namespace pto

#define PTO_A5_VF_STUB_VOID(NAME) \
    template <class... Args>       \
    inline void NAME(Args&&...)    \
    {}

#define PTO_A5_VF_STUB_MASK(NAME)       \
    template <class... Args>             \
    inline ::vector_bool NAME(Args&&...) \
    {                                    \
        return {};                       \
    }

PTO_A5_VF_STUB_VOID(vlds)
PTO_A5_VF_STUB_VOID(vsts)
PTO_A5_VF_STUB_VOID(vgather2)
PTO_A5_VF_STUB_VOID(vscatter)
PTO_A5_VF_STUB_VOID(vci)
PTO_A5_VF_STUB_VOID(vsqz)
PTO_A5_VF_STUB_VOID(sprclr)
PTO_A5_VF_STUB_VOID(vstur)
PTO_A5_VF_STUB_VOID(vstar)
PTO_A5_VF_STUB_VOID(sprsts)
PTO_A5_VF_STUB_VOID(pintlv_b8)
PTO_A5_VF_STUB_VOID(pintlv_b16)
PTO_A5_VF_STUB_VOID(pintlv_b32)
PTO_A5_VF_STUB_VOID(vdup)
PTO_A5_VF_STUB_VOID(vadd)
PTO_A5_VF_STUB_VOID(vsub)
PTO_A5_VF_STUB_VOID(vmul)
PTO_A5_VF_STUB_VOID(vdiv)
PTO_A5_VF_STUB_VOID(vmod)
PTO_A5_VF_STUB_VOID(vpack)
PTO_A5_VF_STUB_VOID(vaxpy)
PTO_A5_VF_STUB_VOID(vmax)
PTO_A5_VF_STUB_VOID(vmin)
PTO_A5_VF_STUB_VOID(vcvt)
PTO_A5_VF_STUB_VOID(vexp)
PTO_A5_VF_STUB_VOID(vsqrt)
PTO_A5_VF_STUB_VOID(vsel)
PTO_A5_VF_STUB_VOID(vand)
PTO_A5_VF_STUB_VOID(vor)
PTO_A5_VF_STUB_VOID(vxor)
PTO_A5_VF_STUB_VOID(vshl)
PTO_A5_VF_STUB_VOID(vshr)
PTO_A5_VF_STUB_VOID(vmadd)
PTO_A5_VF_STUB_VOID(vabs)
PTO_A5_VF_STUB_VOID(vln)
PTO_A5_VF_STUB_VOID(vrelu)
PTO_A5_VF_STUB_VOID(vlrelu)
PTO_A5_VF_STUB_VOID(vnot)
PTO_A5_VF_STUB_VOID(vadds)
PTO_A5_VF_STUB_VOID(vmins)
PTO_A5_VF_STUB_VOID(vmov)
PTO_A5_VF_STUB_VOID(vmula)
PTO_A5_VF_STUB_VOID(vmuls)
PTO_A5_VF_STUB_VOID(vneg)
PTO_A5_VF_STUB_VOID(vshls)
PTO_A5_VF_STUB_VOID(vshrs)
PTO_A5_VF_STUB_VOID(vtrc)
PTO_A5_VF_STUB_VOID(vcmp_eq)
PTO_A5_VF_STUB_VOID(vcmp_ge)
PTO_A5_VF_STUB_VOID(vcmp_gt)
PTO_A5_VF_STUB_VOID(vcmp_le)
PTO_A5_VF_STUB_VOID(vcmp_lt)
PTO_A5_VF_STUB_VOID(vcmp_ne)
PTO_A5_VF_STUB_VOID(vcmps_eq)
PTO_A5_VF_STUB_VOID(vcmps_ge)
PTO_A5_VF_STUB_VOID(vcmps_gt)
PTO_A5_VF_STUB_VOID(vcmps_le)
PTO_A5_VF_STUB_VOID(vcmps_lt)
PTO_A5_VF_STUB_VOID(vcmps_ne)
PTO_A5_VF_STUB_VOID(pand)
PTO_A5_VF_STUB_VOID(por)
PTO_A5_VF_STUB_VOID(pnot)
PTO_A5_VF_STUB_VOID(pxor)
PTO_A5_VF_STUB_VOID(psel)
PTO_A5_VF_STUB_VOID(plds)
PTO_A5_VF_STUB_VOID(ppack)
PTO_A5_VF_STUB_VOID(psts)
PTO_A5_VF_STUB_VOID(punpack)
PTO_A5_VF_STUB_VOID(vaddc)
PTO_A5_VF_STUB_VOID(vaddcs)
PTO_A5_VF_STUB_VOID(vsubc)
PTO_A5_VF_STUB_VOID(vsubcs)
PTO_A5_VF_STUB_VOID(vmull)
PTO_A5_VF_STUB_VOID(vbr)
PTO_A5_VF_STUB_VOID(vintlv)
PTO_A5_VF_STUB_VOID(vdintlv)

PTO_A5_VF_STUB_MASK(plt_b8)
PTO_A5_VF_STUB_MASK(plt_b16)
PTO_A5_VF_STUB_MASK(plt_b32)
PTO_A5_VF_STUB_MASK(pset_b8)
PTO_A5_VF_STUB_MASK(pset_b16)
PTO_A5_VF_STUB_MASK(pset_b32)

#undef PTO_A5_VF_STUB_VOID
#undef PTO_A5_VF_STUB_MASK

namespace pto {
enum class atomic_type_t { ATOMIC_NONE };

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
