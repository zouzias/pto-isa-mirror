/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_NPU_A2A3_T_ALL_TEMPLATES_HPP
#define PTO_NPU_A2A3_T_ALL_TEMPLATES_HPP

/**
 * A2/A3 consolidated template entrypoint.
 *
 * This header exists to keep the public op headers small and to avoid scattering
 * core template implementations across many translation units.
 *
 * It collects:
 * - `TOpTemplates.hpp`  : BinOp/BinS/Unary (+ plus variants) template helpers
 * - `TReduceOps.hpp`    : row/col reduce templates
 * - `TPartOp.hpp`       : partial (shape-mismatch) binary op templates
 * - `TRowExpandBinOp.hpp`: row-expand binary templates
 * - `TExtraOps.hpp`     : extra elementwise helpers (TF loops)
 */

#include "pto/npu/a2a3/TOpTemplates.hpp"
#include "pto/npu/a2a3/TReduceOps.hpp"
#include "pto/npu/a2a3/TPartOp.hpp"
#include "pto/npu/a2a3/TRowExpandBinOp.hpp"
#include "pto/npu/a2a3/TExtraOps.hpp"

#endif // PTO_NPU_A2A3_T_ALL_TEMPLATES_HPP

