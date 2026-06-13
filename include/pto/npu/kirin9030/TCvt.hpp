/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
*/

/**
 * @file TCvt.hpp
 * @brief Type Conversion (TCVT) Implementation for NPU Kirin9030 Architecture
 */

#ifndef TCVT_HPP
#define TCVT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <array>
#include "common.hpp"
#include "utils.hpp"

namespace pto {

// Define architecture (uses RS_ENABLE workaround)
#define ARCH_KIRIN9030

// Include common implementation
#include <pto/common/arch/register/tcvt_common.hpp>

// Undefine architecture macros
#undef ARCH_KIRIN9030
#undef ARCH_RS_SAT
#undef ARCH_HAS_FP8
#undef ARCH_HAS_BFLOAT16

} // namespace pto

#endif // TCVT_HPP
