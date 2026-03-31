#pragma once

#include <cstdint>

#ifndef STRAIGHT_INTRINSICS_IMPL
#define STRAIGHT_INTRINSICS_IMPL
#endif

struct float4_e1m2x2_t {
    std::uint8_t raw = 0;
};

struct float4_e2m1x2_t {
    std::uint8_t raw = 0;
};

struct float8_e8m0_t {
    std::uint8_t raw = 0;
};

struct float8_e4m3_t {
    std::uint8_t raw = 0;
};

struct float8_e5m2_t {
    std::uint8_t raw = 0;
};

struct hifloat8_t {
    std::uint8_t raw = 0;
};

#define PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(Name) \
    struct Name {                              \
        std::uint8_t bytes[32]{};             \
    }

PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_u64);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_s64);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_u32);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_s32);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_f32);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_u16);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_f16);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_s16);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_u8);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_s8);
PTO_MOCKER_A5_DEFINE_VECTOR_TYPE(vector_bf16);

#undef PTO_MOCKER_A5_DEFINE_VECTOR_TYPE

struct vector_bool {
    std::uint32_t count = 0;
};

struct vector_align {
    std::uint64_t value = 0;
};

struct vector_address {
    std::uint64_t value = 0;
};

enum class atomic_type_t : std::uint8_t
{
    ATOMIC_NONE = 0,
    ATOMIC_ADD = 1,
};

namespace pto {

struct MrgSortExecutedNumList {
};

enum class TInsertMode : int
{
    Default = 0,
};

enum class GatherOOB : int
{
    Default = 0,
};

enum class ScatterAtomicOp : int
{
    None = 0,
};

enum class ScatterOOB : int
{
    Default = 0,
};

enum class TileSplitAxis : int
{
    Row = 0,
    Col = 1,
};

} // namespace pto

constexpr int MODE_ZEROING = 0;
constexpr int MODE_MERGING = 1;
constexpr int POST_UPDATE = 1;
constexpr int NORM = 0;
constexpr int UNPK_B16 = 1;
constexpr int ROUND_R = 0;
constexpr int ROUND_Z = 1;
constexpr int RS_ENABLE = 1;
constexpr int PART_EVEN = 0;
constexpr int CCE_VL = 256;

#define __VEC_SCOPE__ if (true)

inline vector_bool plt_b8(std::uint32_t scalar, auto... /*unused*/)
{
    return vector_bool{scalar};
}

inline vector_bool plt_b16(std::uint32_t scalar, auto... /*unused*/)
{
    return vector_bool{scalar};
}

inline vector_bool plt_b32(std::uint32_t scalar, auto... /*unused*/)
{
    return vector_bool{scalar};
}
