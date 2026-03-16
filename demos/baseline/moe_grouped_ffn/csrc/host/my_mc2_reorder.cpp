/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include "utils.h"

#include <vector>

namespace ascendc_path {

namespace {

at::Tensor NormalizeCpuLongVector(const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
    auto cpuTensor = tensor.device().is_cpu() ? tensor.contiguous() : tensor.cpu().contiguous();
    TORCH_CHECK(
        cpuTensor.scalar_type() == at::kLong || cpuTensor.scalar_type() == at::kInt ||
            cpuTensor.scalar_type() == at::kShort,
        name,
        " must use an integer dtype"
    );
    return cpuTensor.to(at::kLong);
}

at::Tensor BuildDeviceRowIndex(const at::Tensor &input, const at::Tensor &splitSizesCpu, const at::Tensor &sortedIdxsCpu)
{
    const auto numChunks = splitSizesCpu.numel();
    TORCH_CHECK(sortedIdxsCpu.numel() == numChunks, "sorted_idxs must have the same length as split_sizes");

    const auto *splitPtr = splitSizesCpu.data_ptr<int64_t>();
    const auto *sortedPtr = sortedIdxsCpu.data_ptr<int64_t>();

    std::vector<int64_t> offsets(static_cast<size_t>(numChunks), 0);
    int64_t totalRows = 0;
    for (int64_t idx = 0; idx < numChunks; ++idx) {
        TORCH_CHECK(splitPtr[idx] >= 0, "split_sizes must be non-negative");
        offsets[static_cast<size_t>(idx)] = totalRows;
        totalRows += splitPtr[idx];
    }
    TORCH_CHECK(totalRows == input.size(0), "sum(split_sizes) must equal input.size(0)");

    std::vector<int64_t> rowIdxVec;
    rowIdxVec.reserve(static_cast<size_t>(totalRows));
    for (int64_t idx = 0; idx < numChunks; ++idx) {
        const int64_t chunkIdx = sortedPtr[idx];
        TORCH_CHECK(chunkIdx >= 0 && chunkIdx < numChunks, "sorted_idxs contains an out-of-range chunk index");
        const int64_t chunkOffset = offsets[static_cast<size_t>(chunkIdx)];
        const int64_t chunkSize = splitPtr[chunkIdx];
        for (int64_t row = 0; row < chunkSize; ++row) {
            rowIdxVec.push_back(chunkOffset + row);
        }
    }

    at::Tensor rowIdxCpu;
    if (rowIdxVec.empty()) {
        rowIdxCpu = at::empty({0}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
    } else {
        rowIdxCpu = at::from_blob(
                           rowIdxVec.data(),
                           {static_cast<int64_t>(rowIdxVec.size())},
                           at::TensorOptions().dtype(at::kLong).device(at::kCPU)
        )
                           .clone();
    }
    return CopyTensorHostToDevice(rowIdxCpu);
}

} // namespace

std::tuple<at::Tensor, at::Tensor> run_pto_mc2_reorder_chunks(
    const at::Tensor &input,
    const at::Tensor &splitSizes,
    const at::Tensor &sortedIdxs
)
{
    TORCH_CHECK(input.device().type() == DEVICE_TYPE, "input must be a NPU tensor (PrivateUse1)");
    auto splitSizesCpu = NormalizeCpuLongVector(splitSizes, "split_sizes");
    auto sortedIdxsCpu = NormalizeCpuLongVector(sortedIdxs, "sorted_idxs");

    auto rowIdx = BuildDeviceRowIndex(input, splitSizesCpu, sortedIdxsCpu);
    auto output = input.index_select(0, rowIdx);
    return std::make_tuple(output, rowIdx);
}

} // namespace ascendc_path

namespace {
TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("pto_mc2_reorder_chunks(Tensor input, Tensor split_sizes, Tensor sorted_idxs) -> (Tensor, Tensor)");
}
} // namespace

namespace {
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("pto_mc2_reorder_chunks", TORCH_FN(ascendc_path::run_pto_mc2_reorder_chunks));
}
} // namespace
