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

#include <pto/common/fifo.hpp>
#include <pto/npu/a2a3/TStore.hpp>
#include <pto/npu/a2a3/TLoad.hpp>

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
template <uint16_t FlagID, typename DataFiFo, typename TileDataProd, typename TileDataCons, TSyncOpType ProducerOp,
          TSyncOpType ConsumerOp>
struct TFiFoSync {
    using Traits = TSyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;
    static constexpr int FiFoSize = DataFiFo::fifoDepth;
    static constexpr int SyncPeriod = DataFiFo::fifoPeriod;

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        volatile int tile_id;
        volatile int sub_tile_id;

        PTO_INTERNAL Producer()
        {
            tile_id = -1;
            sub_tile_id = -1;
        }

        PTO_INTERNAL void set_tile_id(int t_id, int sub_t_id)
        {
            tile_id = t_id;
            sub_tile_id = sub_t_id;
        }

        PTO_INTERNAL int get_tile_id()
        {
            return tile_id;
        }

        PTO_INTERNAL int get_sub_tile_id()
        {
            return sub_tile_id;
        }

        PTO_INTERNAL uint16_t getFFTSMsg(TSyncCVMode mode, uint16_t flag_id, uint16_t base_const = 0x1)
        {
            constexpr uint16_t FFTS_MODE_BIT_START = 4;
            constexpr uint16_t FFTS_FLAG_ID_BIT_START = 6;
            return ((base_const & 0xf) + ((mode & 0x3) << FFTS_MODE_BIT_START) +
                    ((flag_id & 0xf) << FFTS_FLAG_ID_BIT_START));
        }

        /**
         * alloc: Request space in FIFO
         * Logic:
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate()
        {
            if constexpr (is_c2v) {
                // Cube waits for Vector to free buffer
                wait_flag_dev(FlagID + 1);
            } else { // is_v2c
                // Vector waits for Cube to free buffer
                wait_flag_dev(FlagID + 1);
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
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            } else { // is_v2c
                // Vector produces, Cube consumes
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            }
        }

        // Decide whether to block or signal consumption flags for a given tile index.
        // Reverse dependency: notify one step before the corresponding wait within each sync period.
        PTO_INTERNAL bool should_wait_cons()
        {
            static_assert(FiFoSize >= 1, "CV FIFO size must be >= 1");
            constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
            static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
            if (tile_id < static_cast<int>(FiFoSize))
                return false;
            return (tile_id % period) == 0;
        }

        PTO_INTERNAL bool getAllocate()
        {
            return (sub_tile_id == 0) && (should_wait_cons());
        }

        template <typename T, int ProdM, int ProdN, int ConsN>
        PTO_INTERNAL void storeAccTile(DataFiFo &fifo, TileDataProd &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            const size_t base_elems = buf_idx * kTileFactor * ProdM * ProdN + sub_tile_id * ProdM * ProdN;
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor(fifo.fifoBase + base_elems);
            // store tile to GM FIFO, enable unit-flag one
            TSTORE_IMPL<TileDataProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN, int VEC_CORES = 2>
        PTO_INTERNAL void storeVecTile(DataFiFo &fifo, TileDataProd &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ProdN / ConsN;
            const size_t subblock_base_rows = ConsM / VEC_CORES * get_subblockid();
            const size_t row_offset = subblock_base_rows + sub_tile_id * ProdM;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_id % DataFiFo::fifoDepth);
            const size_t base_elems = buf_idx * kTileFactor * ConsM * ConsN;
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN>;
            TileDataSub subTile;
            __gm__ T *addr = fifo.fifoBase + base_elems + row_offset * ConsN;
            // store tile to GM FIFO in sub-tiles if needed (when Tile_S1 > Cube_S1)
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ConsM * ConsN;
                GlobalDataSub globalDataSub((__gm__ T *)(addrSub));

                const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * ConsN * sizeof(T));
                TASSIGN_IMPL(subTile, (uint64_t)tile.data() + col_byte_offset);
                TSTORE_IMPL(globalDataSub, subTile);
            }
        }

        PTO_INTERNAL bool store(DataFiFo &fifo, TileDataProd &tile)
        {
            // get tile shape and valid shape
            using T = typename TileDataProd::DType;
            constexpr int VEC_CORES = 2;
            constexpr int prodTileRows = TileDataProd::Rows;
            constexpr int prodTileCols = TileDataProd::Cols;
            constexpr int consTileRows = TileDataCons::Rows;
            constexpr int consTileCols = TileDataCons::Cols;
            int prodTileValidRows = tile.GetValidRow();
            int prodTileValidCols = tile.GetValidCol();

            // AccTile -> GM_FIFO
            if constexpr (TileDataProd::Loc == TileType::Acc) {
                constexpr int ProdM = prodTileRows; // Cube_S0
                constexpr int ProdN = prodTileCols; // Cube_S1
                constexpr int ConsN = consTileCols; // Tile_S1
                constexpr int kTileFactor = ConsN / ProdN;
                storeAccTile<T, ProdM, ProdN, ConsN>(fifo, tile);
                return (sub_tile_id == (kTileFactor - 1));
            } else if constexpr (TileDataProd::Loc == TileType::Vec) {
                constexpr int ProdM = prodTileRows; // Vec_S0
                constexpr int ProdN = prodTileCols; // Tile_S1
                constexpr int ConsM = consTileRows; // Cube_S0
                constexpr int ConsN = consTileCols; // Cube_S1
                constexpr int kTileFactor = ProdN / ConsN;

                storeVecTile<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
                return (sub_tile_id == (kTileFactor - 1));
            } // end of TileType::Vec
        } // end of store
    }; // end of Producer

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        volatile int tile_id;
        volatile int sub_tile_id;

        PTO_INTERNAL Consumer()
        {
            tile_id = -1;
            sub_tile_id = -1;
        }

        PTO_INTERNAL void set_tile_id(int tid, int sub_tid)
        {
            tile_id = tid;
            sub_tile_id = sub_tid;
        }

        PTO_INTERNAL int get_tile_id()
        {
            return tile_id;
        }

        PTO_INTERNAL int get_sub_tile_id()
        {
            return sub_tile_id;
        }

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait()
        {
            if constexpr (is_c2v) {
                // Vector waits for Cube
                wait_flag_dev(FlagID);
            } else { // is_v2c
                // Cube waits for Vector
                wait_flag_dev(FlagID);
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
            if constexpr (is_c2v) {
                // Vector frees buffer for Cube
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
            } else { // is_v2c
                // Cube frees buffer for Vector
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
            }
        }

        PTO_INTERNAL bool should_notify_cons()
        {
            static_assert(FiFoSize >= 1, "CV FIFO size must be >= 1");
            constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
            static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
            return ((tile_id + 1) % period) == 0; // notify one tile earlier than the wait check
        }

        PTO_INTERNAL bool getWaitStatus()
        {
            return (sub_tile_id == 0);
        }

        template <typename T, int ProdM, int ProdN, int ConsM, int ConsN, int VEC_CORES = 2>
        PTO_INTERNAL void loadVecTile(DataFiFo &fifo, TileDataCons &tile)
        {
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            const size_t subblock_base_rows = static_cast<size_t>(ProdM / VEC_CORES) * get_subblockid();
            const size_t row_offset = subblock_base_rows + sub_tile_id * ConsM;
            const uint32_t buf_idx = static_cast<uint32_t>(tile_id % fifo.fifoDepth);
            const size_t base_elems = static_cast<size_t>(buf_idx) * kTileFactor * ProdM * ProdN;

            __gm__ T *addr = fifo.fifoBase + base_elems + row_offset * ProdN;
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ProdN>;
            TileDataSub tileSub;
            for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ProdM * ProdN;
                const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * ProdN * sizeof(T));
                GlobalDataSub globalTensorSub(addrSub);
                TASSIGN_IMPL(tileSub, (uint64_t)tile.data() + col_byte_offset);
                TLOAD_IMPL(tileSub, globalTensorSub);
            }
        }

        template <typename T, int ConsM, int ConsN, int ProdN>
        PTO_INTERNAL void loadMatTile(DataFiFo &fifo, TileDataCons &tile)
        {
            const uint32_t buf_idx = static_cast<uint32_t>(tile_id % FiFoSize);
            const size_t base_elems = buf_idx * ConsM * ProdN + sub_tile_id * ConsM * ConsN;
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor(fifo.fifoBase + base_elems);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL bool load(DataFiFo &fifo, TileDataCons &tile)
        {
            using T = typename TileDataCons::DType;
            constexpr int VEC_CORES = 2;
            constexpr int consTileRows = TileDataCons::Rows;
            constexpr int consTileCols = TileDataCons::Cols;
            constexpr int prodTileRows = TileDataProd::Rows;
            constexpr int prodTileCols = TileDataProd::Cols;
            int consTileValidRows = tile.GetValidRow();
            int consTileValidCols = tile.GetValidCol();
            int kTileFactor = consTileCols / prodTileCols;

            int row_slice = get_sub_tile_id();
            if constexpr (TileDataCons::Loc == TileType::Vec) {
                constexpr int ProdM = prodTileRows; // CUBE_S0
                constexpr int ProdN = prodTileCols; // CUBE_S1
                constexpr int ConsM = consTileRows; // VEC_S0
                constexpr int ConsN = consTileCols; // TILE_S1

                loadVecTile<T, ProdM, ProdN, ConsM, ConsN, VEC_CORES>(fifo, tile);
            } else if constexpr (TileDataCons::Loc == TileType::Mat) {
                constexpr int ConsM = consTileRows;
                constexpr int ConsN = consTileCols;
                constexpr int ProdN = prodTileCols;
                loadMatTile<T, ConsM, ConsN, ProdN>(fifo, tile);
            }
            return (row_slice == (static_cast<int>(kTileFactor) - 1)) && (should_notify_cons());
        }
    };
};

/**
 * TPUSH: Push Tile to FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Store]   Write data to GM
 * 3. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename PipeProd, typename TileData, typename DataFiFo>
PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DataFiFo &fifo)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = prod.getAllocate();
    if (isAllocate) {
        prod.allocate();
    }

    // 2. Address Calculation
    bool isRecord = prod.store(fifo, tile);

    // 3； Cross-Core: Commit & Signal
    if (isRecord) {
        prod.record();
    }
}

} // namespace pto

#endif
