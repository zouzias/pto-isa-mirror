/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef TEXTRACT_OP_HPP
#define TEXTRACT_OP_HPP

#include <vector>
#include "pto/costmodel/costmodel_types.hpp"

namespace pto {

// TEXTRACT: img2colv2_cbuf_to_ca / load_cbuf_to_ca_transpose / load_cbuf_to_cb
// All are cube DMA (PIPE_M) calls → PIPE_V placeholder.
PTO_INTERNAL std::vector<CostModelStats> runExtractOp()
{
    std::vector<CostModelStats> stats;
    stats.emplace_back("PIPE_V"); // img2colv2_cbuf_to_ca / load_cbuf_to_ca_transpose / load_cbuf_to_cb (PIPE_M)
    return stats;
}

} // namespace pto
#endif // TEXTRACT_OP_HPP
