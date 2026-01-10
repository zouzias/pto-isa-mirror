/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef T_PUSH_HPP
#define T_PUSH_HPP

#include <pto/pto-inst.hpp>
#include <pto/common/pto_pipe.hpp>

namespace pto {

template <typename T, typename = void>
struct has_producer_alloc : std::false_type {};
template <typename T>
struct has_producer_alloc<T, std::void_t<decltype(T::alloc)>> : std::true_type {
};

template <typename T, typename = void>
struct has_producer_record : std::false_type {};
template <typename T>
struct has_producer_record<T, std::void_t<decltype(T::record)>>
    : std::true_type {};

template <typename T>
struct is_pipe_producer
    : std::conjunction<has_producer_alloc<T>, has_producer_record<T>> {};
template <typename T>
constexpr bool is_pipe_producer_v = is_pipe_producer<T>::value;

/**
 * TPUSH: Push Tile to Global Memory FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Calc]    Calculate GM address based on iter
 * 3. [Barrier] Wait for calculation to finish (Intra-Core RAW protection)
 * 4. [Store]   Write data to GM
 * 5. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename PipeProd, typename TileData, typename DType>
PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DType *fifo_base) {
  static_assert(is_pipe_producer_v<PipeProd>,
                "TPUSH: PipeProd must be a Pipe Producer.");
  // 1. Cross-Core: Wait for space
  prod.alloc();

  // 2. Address Calculation
  int buffer_idx = prod.iter % PipeProd::PipeDepth;
  auto *addr = fifo_base + buffer_idx * TileData::Numel;

  // Construct GlobalTensor helper
  using GShape = TileShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GStride =
      BaseShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GT = GlobalTensor<DType, GShape, GStride, Layout::ND>;
  GT global_tensor(addr);

  // 3. Intra-Core Sync & Store
  // Identify Tile location to enforce correct barriers
  if constexpr (TileData::Loc == TileType::Acc) {
    // === Case: CUBE Output (L0C -> GM) ===
    // Must ensure Matrix computation (PIPE_M) is done before FIX moves it.
    // CceEventIdType token = __pto_set_flag(PIPE_M, PIPE_FIX);
    // __pto_wait_flag(PIPE_M, PIPE_FIX, token);
    pipe_barrier(PIPE_ALL);
    TSTORE(global_tensor, tile);
  } else if constexpr (TileData::Loc == TileType::Vec) {
    // === Case: VECTOR Output (UB -> GM) ===
    // Must ensure Vector computation (PIPE_V) is done before MTE3 moves it.
    // CceEventIdType token = __pto_set_flag(PIPE_V, PIPE_MTE3);
    // __pto_wait_flag(PIPE_V, PIPE_MTE3, token);
    pipe_barrier(PIPE_ALL);
    TSTORE(global_tensor, tile);
  } else {
    static_assert(sizeof(TileData) == 0,
                  "Unsupported TileType for TPUSH operation.");
  }

  // 4. Cross-Core: Commit & Signal
  prod.record();
}

} // namespace pto

#endif