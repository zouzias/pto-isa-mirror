/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cassert>
#include "acl/acl.h"
#include "aclrtlaunch_bpa_custom.h"
#include "tiling/platform/platform_ascendc.h"
#include "../kernel/bpa_custom.h"
#include "utils.h"

namespace ascendc_path {

#define CEIL_ALIGN(x, y) ((((x) + (y)-1) / (y)) * (y))

at::Tensor run_bpa_custom(const at::Tensor &query, const at::Tensor &key_cache, const at::Tensor &value_cache,
                          const at::Tensor &block_table, const at::Tensor &context_lens)
{
    // Validate inputs
    TORCH_CHECK(query.device().type() == DEVICE_TYPE, "query must be on NPU");
    TORCH_CHECK(key_cache.device().type() == DEVICE_TYPE, "key_cache must be on NPU");
    TORCH_CHECK(value_cache.device().type() == DEVICE_TYPE, "value_cache must be on NPU");
    TORCH_CHECK(block_table.device().type() == DEVICE_TYPE, "block_table must be on NPU");
    TORCH_CHECK(context_lens.device().type() == DEVICE_TYPE, "context_lens must be on NPU");

    TORCH_CHECK(query.scalar_type() == at::kHalf, "query must be float16");
    TORCH_CHECK(key_cache.scalar_type() == at::kHalf, "key_cache must be float16");
    TORCH_CHECK(value_cache.scalar_type() == at::kHalf, "value_cache must be float16");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");

    // Extract dimensions
    // query: (batch * num_heads, head_dim) flattened
    // We expect query to have shape (batch * kBpaNumHeads * kBpaHeadDim,) or (batch, kBpaNumHeads * kBpaHeadDim)
    uint32_t batch = context_lens.sizes()[0];
    uint32_t num_heads = kBpaNumHeads;
    uint32_t head_dim = kBpaHeadDim;
    uint32_t block_size = kBpaBlockSize;
    uint32_t max_num_blocks = kBpaMaxNumBlocks;

    // Compute max_bn: maximum number of blocks across all batches
    // Copy context_lens to CPU for computation
    auto ctx_cpu = context_lens.cpu();
    auto ctx_ptr = ctx_cpu.data_ptr<int32_t>();
    uint32_t max_bn = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        uint32_t bn_b = (static_cast<uint32_t>(ctx_ptr[b]) + block_size - 1) / block_size;
        if (bn_b > max_bn) {
            max_bn = bn_b;
        }
    }

    // Allocate output tensor: (batch * num_heads, head_dim) float32
    at::Tensor out = at::empty({static_cast<long>(batch * num_heads), static_cast<long>(head_dim)},
                               at::TensorOptions().dtype(at::kFloat).device(query.options().device()));

    // Compute workspace size
    size_t user_workspace_size = kBpaPerBlockWorkspace + kBpaCvCommSlotBytes;
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t system_workspace_size = static_cast<size_t>(ascendc_platform->GetLibApiWorkSpaceSize());
    size_t workspace_size = user_workspace_size + system_workspace_size;

    auto workspace_tensor =
        at::empty({static_cast<long>(workspace_size)},
                  at::TensorOptions().dtype(at::kByte).device(query.options().device()));

    // Launch kernel with blockDim = 1 (single AICore block processes all batches sequentially)
    uint32_t blockDim = 1;

    EXEC_KERNEL_CMD(bpa_custom, blockDim, query, key_cache, value_cache, block_table, context_lens, out, batch, max_bn,
                    workspace_tensor);

    return out;
}
} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    // Declare the custom operator schema
    m.def("my_bpa(Tensor query, Tensor key_cache, Tensor value_cache, "
          "Tensor block_table, Tensor context_lens) -> Tensor");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    // Register the custom operator implementation function
    m.impl("my_bpa", TORCH_FN(ascendc_path::run_bpa_custom));
}
} // namespace
