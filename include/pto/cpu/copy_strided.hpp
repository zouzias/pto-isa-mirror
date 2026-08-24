/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_CPU_COPY_STRIDED_HPP
#define PTO_CPU_COPY_STRIDED_HPP

#include <cstddef>

namespace pto {
namespace comm {

template <typename DstType, typename SrcType>
void CopyStrided5D(DstType* dst, SrcType* src, long int shape[], long int stride[])
{
    const std::size_t total = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t rem = linear;
        const std::size_t m = rem % shape[4];
        rem /= shape[4];
        const std::size_t l = rem % shape[3];
        rem /= shape[3];
        const std::size_t k = rem % shape[2];
        rem /= shape[2];
        const std::size_t j = rem % shape[1];
        const std::size_t i = rem / shape[1];
        const std::size_t index = i * stride[0] + j * stride[1] + k * stride[2] + l * stride[3] + m * stride[4];
        dst[index] = src[index];
    }
}

} // namespace comm
} // namespace pto

#endif
