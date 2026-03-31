/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_MOCKER_ARCH_CONFIG_HPP
#define PTO_MOCKER_ARCH_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <pto/common/arch_macro.hpp>

namespace pto::mocker::evaluator {

enum class CoreType
{
    Unknown,
    Scalar,
    Vector,
    Mte1,
    Mte2,
    Mte3,
    Matrix,
    Fix,
};

inline constexpr std::string_view GetCoreTypeName(CoreType core_type)
{
    switch (core_type) {
        case CoreType::Scalar:
            return "scalar";
        case CoreType::Vector:
            return "vector";
        case CoreType::Mte1:
            return "mte1";
        case CoreType::Mte2:
            return "mte2";
        case CoreType::Mte3:
            return "mte3";
        case CoreType::Matrix:
            return "matrix";
        case CoreType::Fix:
            return "fix";
        case CoreType::Unknown:
        default:
            return "unknown";
    }
}

enum class LatencyRuleKind
{
    Fixed,
    Repeat,
    BurstLen,
    Nd2Nz,
    Mad,
};

struct CceLatencyRule {
    std::string_view cce_name;
    CoreType core_type = CoreType::Unknown;
    LatencyRuleKind kind = LatencyRuleKind::Fixed;
    uint64_t fixed_cycles = 0;
    uint64_t startup_cycles = 0;
    uint64_t per_repeat_cycles = 0;
    uint64_t per_burst_cycles = 0;
    uint64_t per_len_burst_cycles = 0;
    std::string_view note;
};

inline constexpr CceLatencyRule FixedRule(std::string_view cce_name, CoreType core_type, uint64_t fixed_cycles,
                                          std::string_view note = {})
{
    return {cce_name, core_type, LatencyRuleKind::Fixed, fixed_cycles, 0, 0, 0, 0, note};
}

inline constexpr CceLatencyRule RepeatRule(std::string_view cce_name, CoreType core_type, uint64_t startup_cycles,
                                           uint64_t per_repeat_cycles, std::string_view note = {})
{
    return {cce_name, core_type, LatencyRuleKind::Repeat, 0, startup_cycles, per_repeat_cycles, 0, 0, note};
}

inline constexpr CceLatencyRule BurstLenRule(std::string_view cce_name, CoreType core_type, uint64_t startup_cycles,
                                             uint64_t per_burst_cycles, uint64_t per_len_burst_cycles,
                                             std::string_view note = {})
{
    return {cce_name,
            core_type,
            LatencyRuleKind::BurstLen,
            0,
            startup_cycles,
            0,
            per_burst_cycles,
            per_len_burst_cycles,
            note};
}

inline constexpr CceLatencyRule Nd2NzRule(std::string_view cce_name, CoreType core_type, uint64_t startup_cycles,
                                          uint64_t per_n_value_cycles, uint64_t per_nd_matrix_cycles,
                                          uint64_t per_d_value_cycles, std::string_view note = {})
{
    return {cce_name,
            core_type,
            LatencyRuleKind::Nd2Nz,
            0,
            startup_cycles,
            per_n_value_cycles,
            per_nd_matrix_cycles,
            per_d_value_cycles,
            note};
}

inline constexpr CceLatencyRule MadRule(std::string_view cce_name, CoreType core_type, uint64_t startup_cycles,
                                        uint64_t per_tile_cycles, std::string_view note = {})
{
    return {cce_name, core_type, LatencyRuleKind::Mad, 0, startup_cycles, per_tile_cycles, 0, 0, note};
}

struct ArchConfig {
    std::string_view arch_name;
    const CceLatencyRule *rules = nullptr;
    std::size_t rule_count = 0;
};

inline const CceLatencyRule *FindRule(const ArchConfig &arch, std::string_view cce_name)
{
    for (std::size_t i = 0; i < arch.rule_count; ++i) {
        if (arch.rules[i].cce_name == cce_name) {
            return &arch.rules[i];
        }
    }
    return nullptr;
}

inline constexpr CceLatencyRule kA2A3Rules[] = {
    FixedRule("set_flag", CoreType::Scalar, 1),
    FixedRule("wait_flag", CoreType::Scalar, 1),
    FixedRule("wait_flag_dev", CoreType::Scalar, 1),
    FixedRule("set_vector_mask", CoreType::Scalar, 1),
    FixedRule("set_cmpmask", CoreType::Scalar, 1),
    FixedRule("set_mask_count", CoreType::Scalar, 1),
    FixedRule("set_mask_norm", CoreType::Scalar, 1),
    FixedRule("set_mov_pad_val", CoreType::Scalar, 1),
    FixedRule("set_padding", CoreType::Scalar, 1),
    FixedRule("set_l3d_rpt", CoreType::Scalar, 1),
    FixedRule("pipe_barrier", CoreType::Scalar, 1),
    RepeatRule("vadd", CoreType::Vector, 4, 2),
    RepeatRule("vsub", CoreType::Vector, 4, 2),
    RepeatRule("vmul", CoreType::Vector, 4, 2),
    RepeatRule("vdiv", CoreType::Vector, 6, 4),
    RepeatRule("vmin", CoreType::Vector, 4, 2),
    RepeatRule("vmax", CoreType::Vector, 4, 2),
    RepeatRule("vand", CoreType::Vector, 4, 2),
    RepeatRule("vor", CoreType::Vector, 4, 2),
    RepeatRule("vector_dup", CoreType::Vector, 4, 1),
    RepeatRule("vadds", CoreType::Vector, 4, 1),
    RepeatRule("vmuls", CoreType::Vector, 4, 1),
    RepeatRule("vmins", CoreType::Vector, 4, 1),
    RepeatRule("vmaxs", CoreType::Vector, 4, 1),
    RepeatRule("vabs", CoreType::Vector, 4, 1),
    RepeatRule("vaxpy", CoreType::Vector, 4, 1),
    RepeatRule("vexp", CoreType::Vector, 6, 3),
    RepeatRule("vln", CoreType::Vector, 6, 3),
    RepeatRule("vlrelu", CoreType::Vector, 4, 1),
    RepeatRule("vnot", CoreType::Vector, 4, 1),
    RepeatRule("vrelu", CoreType::Vector, 4, 1),
    RepeatRule("vrsqrt", CoreType::Vector, 6, 3),
    RepeatRule("vsqrt", CoreType::Vector, 6, 3),
    RepeatRule("vcopy", CoreType::Vector, 4, 1),
    RepeatRule("vshl", CoreType::Vector, 4, 1),
    RepeatRule("vshr", CoreType::Vector, 4, 1),
    RepeatRule("vsel", CoreType::Vector, 4, 2),
    RepeatRule("vcmpv_eq", CoreType::Vector, 4, 1),
    RepeatRule("vcmpv_ge", CoreType::Vector, 4, 1),
    RepeatRule("vcmpv_gt", CoreType::Vector, 4, 1),
    RepeatRule("vcmpv_le", CoreType::Vector, 4, 1),
    RepeatRule("vcmpv_lt", CoreType::Vector, 4, 1),
    RepeatRule("vcmpv_ne", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_eq", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_ge", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_gt", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_le", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_lt", CoreType::Vector, 4, 1),
    RepeatRule("vcmpvs_ne", CoreType::Vector, 4, 1),
    RepeatRule("vconv_f322f32f", CoreType::Vector, 6, 2),
    RepeatRule("vconv_f322f32z", CoreType::Vector, 6, 2),
    RepeatRule("vconv_f322s32r", CoreType::Vector, 6, 2),
    RepeatRule("vconv_s322f32", CoreType::Vector, 6, 2),
    RepeatRule("scatter_vnchwconv_b8", CoreType::Vector, 8, 2),
    RepeatRule("scatter_vnchwconv_b16", CoreType::Vector, 6, 2),
    RepeatRule("scatter_vnchwconv_b32", CoreType::Vector, 6, 2),
    BurstLenRule("copy_gm_to_ubuf_align_b8", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b16", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b32", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b8", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b16", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b32", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_cbuf_to_gm", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_gm_to_cbuf", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_ubuf", CoreType::Mte2, 4, 1, 1),
    Nd2NzRule("copy_gm_to_cbuf_multi_nd2nz_b8", CoreType::Mte2, 8, 1, 2, 1, "approx nd2nz"),
    Nd2NzRule("copy_gm_to_cbuf_multi_nd2nz_b16", CoreType::Mte2, 8, 1, 2, 1, "approx nd2nz"),
    Nd2NzRule("copy_gm_to_cbuf_multi_nd2nz_b32s", CoreType::Mte2, 8, 1, 2, 1, "approx nd2nz"),
    FixedRule("copy_matrix_cc_to_gm", CoreType::Fix, 12),
    MadRule("mad", CoreType::Matrix, 16, 16, "approx 16x16x16 mmad blocks"),
};

inline constexpr ArchConfig kA2A3ArchConfig{
    "a2a3",
    kA2A3Rules,
    sizeof(kA2A3Rules) / sizeof(kA2A3Rules[0]),
};

inline constexpr CceLatencyRule kA5Rules[] = {
    FixedRule("set_flag", CoreType::Scalar, 1),
    FixedRule("wait_flag", CoreType::Scalar, 1),
    FixedRule("wait_flag_dev", CoreType::Scalar, 1),
    FixedRule("set_vector_mask", CoreType::Scalar, 1),
    FixedRule("set_cmpmask", CoreType::Scalar, 1),
    FixedRule("set_mask_count", CoreType::Scalar, 1),
    FixedRule("set_mask_norm", CoreType::Scalar, 1),
    FixedRule("set_mov_pad_val", CoreType::Scalar, 1),
    FixedRule("set_padding", CoreType::Scalar, 1),
    FixedRule("pipe_barrier", CoreType::Scalar, 1),
    RepeatRule("vadd", CoreType::Vector, 4, 2),
    RepeatRule("vsub", CoreType::Vector, 4, 2),
    RepeatRule("vmul", CoreType::Vector, 4, 2),
    RepeatRule("vdiv", CoreType::Vector, 6, 4),
    RepeatRule("vmin", CoreType::Vector, 4, 2),
    RepeatRule("vmax", CoreType::Vector, 4, 2),
    RepeatRule("vand", CoreType::Vector, 4, 2),
    RepeatRule("vor", CoreType::Vector, 4, 2),
    RepeatRule("vector_dup", CoreType::Vector, 4, 1),
    RepeatRule("vadds", CoreType::Vector, 4, 1),
    RepeatRule("vmuls", CoreType::Vector, 4, 1),
    RepeatRule("vmins", CoreType::Vector, 4, 1),
    RepeatRule("vmaxs", CoreType::Vector, 4, 1),
    RepeatRule("vabs", CoreType::Vector, 4, 1),
    RepeatRule("vaxpy", CoreType::Vector, 4, 1),
    RepeatRule("vexp", CoreType::Vector, 6, 3),
    RepeatRule("vln", CoreType::Vector, 6, 3),
    RepeatRule("vlrelu", CoreType::Vector, 4, 1),
    RepeatRule("vnot", CoreType::Vector, 4, 1),
    RepeatRule("vrelu", CoreType::Vector, 4, 1),
    RepeatRule("vrsqrt", CoreType::Vector, 6, 3),
    RepeatRule("vsqrt", CoreType::Vector, 6, 3),
    BurstLenRule("copy_gm_to_ubuf_align_b8", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b16", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_gm_to_ubuf_align_b32", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b8", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b16", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_gm_align_b32", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_cbuf_to_gm", CoreType::Mte3, 8, 2, 1),
    BurstLenRule("copy_gm_to_cbuf", CoreType::Mte2, 8, 2, 1),
    BurstLenRule("copy_ubuf_to_ubuf", CoreType::Mte2, 4, 1, 1),
    FixedRule("copy_matrix_cc_to_gm", CoreType::Fix, 12),
};

inline constexpr ArchConfig kA5ArchConfig{
    "a5",
    kA5Rules,
    sizeof(kA5Rules) / sizeof(kA5Rules[0]),
};

inline const ArchConfig &GetDefaultArchConfig()
{
#if defined(PTO_NPU_ARCH_A5)
    return kA5ArchConfig;
#else
    return kA2A3ArchConfig;
#endif
}

} // namespace pto::mocker::evaluator

#endif
