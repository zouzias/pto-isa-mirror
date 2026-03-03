/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef BPA_CUSTOM_KERNEL_H
#define BPA_CUSTOM_KERNEL_H

#include <cstddef>
#include <cstdint>

// Batch Paged Attention configuration constants (matching CPU demo)
constexpr int kBpaNumHeads = 16;
constexpr int kBpaHeadDim = 16;
constexpr int kBpaBlockSize = 16;
constexpr int kBpaMaxNumBlocks = 4;
constexpr int kBpaTotalBlocks = 8;

// Workspace layout: per-block offsets (aligned to 512 bytes)
// sij:    kBpaNumHeads * kBpaBlockSize * sizeof(float)  = 1024 bytes
// pij_f16: kBpaNumHeads * kBpaBlockSize * sizeof(half)  = 512 bytes
// oi_new: kBpaNumHeads * kBpaHeadDim * sizeof(float)    = 1024 bytes
constexpr std::size_t kBpaSijBytes = kBpaNumHeads * kBpaBlockSize * sizeof(float);
constexpr std::size_t kBpaPijBytes = kBpaNumHeads * kBpaBlockSize * sizeof(uint16_t);  // half
constexpr std::size_t kBpaOiNewBytes = kBpaNumHeads * kBpaHeadDim * sizeof(float);

// Align each section to 512 bytes for DMA efficiency
constexpr std::size_t kBpaAlign = 512;
constexpr std::size_t ALIGN_UP(std::size_t x, std::size_t a) { return ((x + a - 1) / a) * a; }

constexpr std::size_t kBpaSijOffset = 0;
constexpr std::size_t kBpaPijOffset = ALIGN_UP(kBpaSijBytes, kBpaAlign);
constexpr std::size_t kBpaOiNewOffset = ALIGN_UP(kBpaPijOffset + kBpaPijBytes, kBpaAlign);
constexpr std::size_t kBpaPerBlockWorkspace = ALIGN_UP(kBpaOiNewOffset + kBpaOiNewBytes, kBpaAlign);

// CV comm slot size for cross-core synchronization
constexpr std::size_t kBpaCvCommSlotBytes = 512U;

#endif // BPA_CUSTOM_KERNEL_H
