/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TPUSH_HPP
#define TPUSH_HPP

#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/fifo.hpp>

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class TSyncOpType : uint8_t
{
    TSTORE_C2GM, // Store (Cube core operation)
    TSTORE_V2GM, // Store (Vector core operation)
    TLOAD        // Load operation (consumer operation)
};

// Compile-time direction inference based on producer/consumer ops
// TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector
// TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube
template <TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSyncTraits {
    // Direction is inferred from producer operation:
    // TSTORE_C2GM -> Cube produces (C2V)
    // TSTORE_V2GM -> Vector produces (V2C)
    static constexpr bool is_cube_to_vec = (ProducerOp == TSyncOpType::TSTORE_C2GM);
    static constexpr bool is_vec_to_cube = (ProducerOp == TSyncOpType::TSTORE_V2GM);

    static_assert(ConsumerOp == TSyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(is_cube_to_vec || is_vec_to_cube,
                  "Producer must be either TSTORE_C2GM (Cube) or TSTORE_V2GM (Vector)");
};

enum TSyncCVMode : uint8_t
{
    CUBE_ALL_CORE_SYNC = 0,
    VEC_ALL_CORE_SYNC = 0,
    VEC_SUBCORES_SYNC = 1,
    CV_CORES_SYNC = 2
};

/**
 * Pipe: Manages Cross-Core FIFO Synchronization
 * @tparam ReadyFlag    Signal from Producer to Consumer (Data Ready)
 * @tparam ConsumedFlag Signal from Consumer to Producer (Space Released)
 * @tparam Depth        FIFO Depth (e.g., 2 for Double Buffering)
 * @tparam Period       Sync Period (Sync once every N tiles)
 * @tparam ProdRole     Logic role of Producer (CUBE/VECTOR) -> Deduce signal pipe
 * @tparam ConsRole     Logic role of Consumer (CUBE/VECTOR) -> Deduce signal pipe
 */
template <uint16_t FlagID, int Depth, int Period, TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSync {
    using Traits = TSyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;

    static constexpr uint16_t flag_id = FlagID; // FFTS flag ID for cross-core synchronization

    static inline uint16_t _getFFTSMsg(TSyncCVMode mode, uint16_t flag_id, uint16_t base_const = 0x1)
    {
        return ((base_const & 0xf) + ((mode & 0x3) << 4) + ((flag_id & 0xf) << 8));
    }

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int iter = 0; // Private iterator for Producer

        /**
         * alloc: Request space in FIFO
         * Logic:
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate()
        {
            if (iter >= Depth && (iter % Period) == 0) {
                if constexpr (is_c2v) {
                    // Cube waits for Vector to free buffer
                    wait_flag_dev(flag_id + 1);
                } else { // is_v2c
                         // Vector waits for Cube to free buffer
                    wait_flag_dev(flag_id + 1);
                }
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record()
        {
            if constexpr (is_c2v) {
                // Cube produces, Vector consumes
                ffts_cross_core_sync(PIPE_FIX, _getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, flag_id));
            } else { // is_v2c
                // Vector produces, Cube consumes
                ffts_cross_core_sync(PIPE_MTE3, _getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, flag_id));
            }
            iter++;
        }
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int iter = 0; // Private iterator for Consumer

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait()
        {
            if constexpr (is_c2v) {
                // Vector waits for Cube
                wait_flag_dev(flag_id);
            } else { // is_v2c
                // Cube waits for Vector
                wait_flag_dev(flag_id);
            }
        }

        /**
         * free: Release space in FIFO
         * Logic:
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free()
        {
            bool is_sync_step = ((iter + 1) % Period) == 0;
            if constexpr (is_c2v) {
                if (iter >= Depth - Period && is_sync_step) {
                    // Vector frees buffer for Cube
                    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, flag_id + 1));
                }
            } else { // is_v2c
                if (iter >= Depth - Period && is_sync_step) {
                    // Cube frees buffer for Vector
                    ffts_cross_core_sync(PIPE_MTE2, _getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, flag_id + 1));
                }
            }
        }
    };
};

/**
 * TPUSH: Push Tile to Global Memory FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Calc]    Calculate GM address based on iter
 * 3. [Barrier] Wait for calculation to finish (Intra-Core RAW protection)
 * 4. [Store]   Write data to GM
 * 5. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename PipeProd, typename TileData, typename DataFifo>
PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DataFifo &fifo, int tileIdx)
{
    // 1. Cross-Core: Wait for space
    // 第一次 或者达到同步周期时需要进行等待
    prod.allocate();

    // 2. Address Calculation
    int buffer_idx = prod.iter % DataFifo::fifoDepth;
    using T = typename TileData::DType;
    __gm__ T *addr = (__gm__ T *)fifo.fifoBase + buffer_idx * TileData::Numel;

    // Construct GlobalTensor helper
    using GShape = TileShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GStride = BaseShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    GT global_tensor(addr);

    // 3. Intra-Core Sync & Store
    // Identify Tile location to enforce correct barriers
    if constexpr (TileData::Loc == TileType::Acc) {
        // === Case: CUBE Output (L0C -> GM) ===
        TSTORE(global_tensor, tile);
    } else {
        // === Case: VECTOR Output (UB -> GM) ===
        TSTORE(global_tensor, tile);
    }

    // 4. Cross-Core: Commit & Signal
    prod.record();
}

} // namespace pto

#endif