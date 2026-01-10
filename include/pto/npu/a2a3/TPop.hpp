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
struct has_consumer_wait : std::false_type {};
template <typename T>
struct has_consumer_wait<T, std::void_t<decltype(T::wait)>> : std::true_type {};

template <typename T, typename = void>
struct has_consumer_free : std::false_type {};
template <typename T>
struct has_consumer_free<T, std::void_t<decltype(T::free)>> : std::true_type {};

template <typename T>
struct is_pipe_consumer
    : std::conjunction<has_consumer_wait<T>, has_consumer_free<T>> {};

template <typename T>
constexpr bool is_pipe_consumer_v = is_pipe_consumer<T>::value;

/**
 * TPOP: Pop Tile from Global Memory FIFO
 * * Flow:
 * 1. [Wait]    Wait for data ready (Cross-Core)
 * 2. [Calc]    Calculate GM address
 * 3. [Barrier] Wait for Tile to be free (Intra-Core WAR protection)
 * 4. [Load]    Load data from GM
 * 5. [Barrier] Signal Tile is updated (Intra-Core RAW protection)
 * 6. [Free]    Release GM space (Cross-Core)
 */
template <typename PipeCons, typename TileData, typename DType>
PTO_INTERNAL void TPOP_IMPL(PipeCons &cons, TileData &tile, DType *fifo_base) {
  static_assert(is_pipe_consumer_v<PipeCons>,
                "TPOP: PipeCons must be a Pipe Consumer.");
  // 1. Cross-Core: Wait for Data
  cons.wait();

  // 2. Address Calculation
  int buffer_idx = cons.iter % PipeCons::PipeDepth;
  auto *addr = fifo_base + buffer_idx * TileData::Numel;

  using GShape = TileShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GStride =
      BaseShape2D<DType, TileData::Rows, TileData::Cols, Layout::ND>;
  using GT = GlobalTensor<DType, GShape, GStride, Layout::ND>;
  GT global_tensor(addr);

  // 3. Intra-Core Sync & Load
  if constexpr (TileData::Loc == TileType::Vec) {
    // === Case: VECTOR Input (GM -> UB) ===

    // A. WAR Protection: Ensure Vector finished using old data in this Tile
    // CceEventIdType token = __pto_set_flat(PIPE_V, PIPE_MTE2);
    // __pto_wait_flag(PIPE_V, PIPE_MTE2, token);
    pipe_barrier(PIPE_ALL);

    // B. Data Transfer
    TLOAD(tile, global_tensor);

    // C. RAW Protection: Signal Vector that new data is ready
    // token = __pto_set_flat(PIPE_MTE2, PIPE_V);
    // __pto_wait_flag(PIPE_MTE2, PIPE_V, token);
    pipe_barrier(PIPE_ALL);
  } else if constexpr (TileData::Loc == TileType::Mat) {
    // === Case: CUBE Input (GM -> L1) ===
    // Assuming Cube reads from L1 via M/V pipe, L1 write is MTE2

    // A. WAR Protection
    // CceEventIdType token = __pto_set_flat(PIPE_M, PIPE_MTE2);
    // __pto_wait_flag(PIPE_M, PIPE_MTE2, token);
    pipe_barrier(PIPE_ALL);
    // B. Data Transfer
    TLOAD(tile, global_tensor);

    // C. RAW Protection
    // CceEventIdType token = __pto_set_flat(PIPE_MTE2, PIPE_MTE1);
    // __pto_wait_flag(PIPE_MTE2, PIPE_MTE1, token);
    pipe_barrier(PIPE_ALL);
  } else {
    static_assert(sizeof(TileData) == 0, "TPOP: Unsupported TileType.");
  }

  // 4. Cross-Core: Free Space
  cons.free();
}

} // namespace pto

#endif