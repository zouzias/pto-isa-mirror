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

enum TSyncCVMode : uint8_t
{
    CUBE_ALL_CORE_SYNC = 0,
    VEC_ALL_CORE_SYNC = 0,
    VEC_SUBCORES_SYNC = 1,
    CV_CORES_SYNC = 2
};

template <uint8_t FlagID, uint8_t DirType, uint32_t SlotSize, uint32_t SlotNum, uint32_t LocalSlotNum = 2,
          bool EN_UNIT_FLAG = false>
struct TPipe {
    static constexpr uint8_t DIR_MASK = 0x7;
    static constexpr uint8_t DIR_TYPE = DIR_MASK & DirType;
    static constexpr bool is_c2v = (DIR_TYPE == Direction::DIR_C2V);
    static constexpr bool is_v2c = (DIR_TYPE == Direction::DIR_V2C);
    static constexpr bool is_v2c_ctrl = (DIR_TYPE == Direction::DIR_V2C_CTRL);

    // uint32_t AIV_INDEX = get_subblockid();

    struct DataFiFo {
        __gm__ void *GM_SLOT_BUFFER = nullptr; // Global memory
        uint32_t C2V_CONSUMER_BUF = 0x0;       // UB buffer
        uint32_t V2C_CONSUMER_BUF = 0x0;       // L1 buffer
        uint64_t V2C_CONTROL_BUF = 0x0;        // scalar buffer for control signals
        static constexpr uint32_t SLOT_NUM = SlotNum;
        static constexpr uint32_t SLOT_SIZE = SlotSize;
        static constexpr uint32_t SLOT_SYNCT = SLOT_NUM;
        static constexpr uint32_t LOCAL_SLOT_NUM = LocalSlotNum;

        PTO_INTERNAL DataFiFo(__gm__ void *gmSlotBuf, uint32_t c2vConsBuf, uint32_t v2cConBuf)
            : GM_SLOT_BUFFER(gmSlotBuf), C2V_CONSUMER_BUF(c2vConsBuf), V2C_CONSUMER_BUF(v2cConBuf)
        {}
    };

    PTO_INTERNAL static uint64_t getFFTSMsgCfg(TSyncCVMode mode, uint16_t flagID, uint16_t base_const = 0x1)
    {
        constexpr uint16_t FFTS_MODE_BIT_START = 4;
        constexpr uint16_t FFTS_FLAG_ID_BIT_START = 8;
        return ((base_const & 0xf) + ((mode & 0x3) << FFTS_MODE_BIT_START) +
                ((flagID & 0xf) << FFTS_FLAG_ID_BIT_START));
    }

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int tile_id = 0;
        int sub_tile_id = 0;
        bool isAllocate = true;
        bool isRecord = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int t_id, int sub_t_id)
        {
            tile_id = t_id;
            sub_tile_id = sub_t_id;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setAllocateStatus(bool allocate)
        {
            isAllocate = allocate;
        }

        PTO_INTERNAL bool getAllocateStatus() const
        {
            return isAllocate;
        }

        PTO_INTERNAL void setRecordStatus(bool record)
        {
            isRecord = record;
        }

        PTO_INTERNAL bool getRecordStatus() const
        {
            return isRecord;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        /**
         * alloc: Request space in FIFO
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate() const
        {
            // Cube waits for Vector to free buffer
            if constexpr (is_c2v) {
#ifdef __DAV_CUBE__
                wait_flag_dev(FlagID + 1);
#endif
            } else {
                // Vector waits for Cube to free buffer
#ifdef __DAV_VEC__
                wait_flag_dev(FlagID + 1);
#endif
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record() const
        {
            if constexpr (is_c2v) {
                // Cube produces, Vector consumes
                ffts_cross_core_sync(PIPE_FIX, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            } else { // is_v2c
                // Vector produces, Cube consumes
                ffts_cross_core_sync(PIPE_MTE3, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID));
            }
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ConsN / ProdN;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * DataFiFo::SLOT_SIZE;
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag one
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;

            // calculate base address in GM FIFO for this tile
            constexpr int kTileFactor = ProdN / ConsN;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * kTileFactor * ConsM * ConsN * sizeof(T);
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN>;
            TileDataSub subTile;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);
            // store tile to GM FIFO in sub-tiles if needed (when Tile_S1 > Cube_S1)
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ConsM * ConsN;
                GlobalDataSub globalDataSub((__gm__ T *)(addrSub));
                uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * ConsN * sizeof(T));
                TASSIGN_IMPL(subTile, (uint64_t)tile.data() + col_byte_offset);
                TSTORE_IMPL(globalDataSub, subTile);
            }
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void pushVec2CtrlFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            uint64_t entryBase = slotIndex * sizeof(uint32_t);
            __gm__ uint32_t *ctrlBuf = (__gm__ uint32_t *)(fifo.CTRL_SLOT_BUFFER + entryBase + entryOffset);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            uint32_t ctrlSignal = *(tile.data());
            *(ctrlBuf) = ctrlSignal;
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void push(DataFiFo &fifo, TileProd &tile)
        {
            // get tile shape and valid shape
            static_assert(TileProd::Loc == TileType::Acc || TileProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (TileProd::Loc == TileType::Acc) {
                pushAcc2GMFiFo<TileProd, TileCons>(fifo, tile);
            } else if constexpr (TileProd::Loc == TileType::Vec) {
                if constexpr (is_v2c) {
                    pushVec2GMFiFo<TileProd, TileCons>(fifo, tile);
                } else if constexpr (is_v2c_ctrl) {
                    pushVec2CtrlFiFo<TileProd, TileCons>(fifo, tile);
                }
            }
        } // end of store

        //--------------------------------------------------------------
        template <typename TileProd>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag one
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        }

        template <typename TileProd>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);
            GlobalData globalData((__gm__ T *)(addr));
            TSTORE_IMPL(globalData, tile);
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void push(DataFiFo &fifo, TileProd &tile)
        {
            static_assert(TileProd::Loc == TileType::Acc || TileProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (TileProd::Loc == TileType::Acc) {
                pushAcc2GMFiFo<TileProd>(fifo, tile);
            } else if constexpr (TileProd::Loc == TileType::Vec) {
                pushVec2GMFiFo<TileProd>(fifo, tile);
            }
        }
    }; // end of Producer

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int tile_id = 0;
        int sub_tile_id = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tile_id = tid;
            sub_tile_id = sub_tid;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tile_id;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return sub_tile_id;
        }

        PTO_INTERNAL void setEntryOffset(int offset)
        {
            entryOffset = offset;
        }

        PTO_INTERNAL void setWaitStatus(bool wait)
        {
            isWait = wait;
        }

        PTO_INTERNAL bool getWaitStatus() const
        {
            return isWait;
        }

        PTO_INTERNAL void setFreeStatus(bool free)
        {
            isFree = free;
        }

        PTO_INTERNAL bool getFreeStatus() const
        {
            return isFree;
        }

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait() const
        {
            // Vector waits for Cube
            // Or Cube waits for Vector
            wait_flag_dev(FlagID);
        }

        /**
         * free: Release space in FIFO
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free() const
        {
            // Vector frees buffer for Cube
            // Or Cube frees buffer for Vector
            if constexpr (is_c2v) {
#ifdef __DAV_VEC__
                // Vec consumer frees buffer for Cube
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
#endif
            } else { // is_v2c
                     // cube consumer frees buffer for vec
#ifdef __DAV_CUBE__
                ffts_cross_core_sync(PIPE_MTE2, getFFTSMsgCfg(TSyncCVMode::CV_CORES_SYNC, FlagID + 1));
#endif
            }
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;

            size_t slotIndex = static_cast<size_t>(tile_id) % DataFiFo::SLOT_NUM;
            constexpr int kTileFactor = ConsN / ProdN;
            size_t entryBase = static_cast<size_t>(slotIndex) * kTileFactor * ProdM * ProdN * sizeof(T);
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);

            uint64_t localTileBase =
                (uint64_t)fifo.C2V_SLOT_BUFFER +
                (static_cast<size_t>(tile_id) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);

            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ProdN>;
            TileDataSub tileSub;
            for (int sub_col = 0; sub_col < kTileFactor; ++sub_col) {
                __gm__ T *addrSub = addr + sub_col * ProdM * ProdN;
                uint64_t col_byte_offset = sub_col * ProdN * sizeof(T);
                GlobalDataSub globalTensorSub(addrSub);
                TASSIGN_IMPL(tileSub, (uint64_t)tile.data() + col_byte_offset);
                TLOAD_IMPL(tileSub, globalTensorSub);
            }
        }

        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * ConsM * ProdN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                (uint64_t)fifo.V2C_CONSUMER_BUF +
                (static_cast<size_t>(tile_id) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(DataFiFo &fifo)
        {
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.CTRL_SLOT_BUFFER + entryBase + entryOffset;
            fifo.ctrlSignal = ((*(__gm__ uint32_t *)(ctrlTileBase)) == 1) ? true : false;
        }

        template <typename TileProd, typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void pop(DataFiFo &fifo, TileCons &tile)
        {
            static_assert(TileCons::Loc == TileType::Vec || TileCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (TileCons::Loc == TileType::Vec) {
                popVecTileFromGMFiFo<TileProd, TileCons>(fifo, tile);
            } else if constexpr (TileCons::Loc == TileType::Mat) {
                popMatTileFromGMFiFo<TileProd, TileCons>(fifo, tile);
            } else {
                popCtrlFromCtrlFiFo(fifo);
            }
        }

        //--------------------------------------------------------------------
        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            constexpr int splitNum = 2;
            constexpr int ProdM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ConsM / splitNum : ConsM;
            constexpr int ProdN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ConsN / splitNum : ConsN;

            // global tensor
            size_t slotIndex = static_cast<size_t>(tile_id) % DataFiFo::SLOT_NUM;
            size_t entryBase = static_cast<size_t>(slotIndex) * ProdM * ProdN * sizeof(T);
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor(addr);

            // local vector tile
            uint64_t localTileBase =
                (uint64_t)fifo.C2V_CONSUMER_BUF +
                (static_cast<size_t>(tile_id) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            uint32_t slotIndex = static_cast<uint32_t>(tile_id % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * ConsM * ConsN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                (uint64_t)fifo.V2C_CONSUMER_BUF +
                (static_cast<size_t>(tile_id) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void pop(DataFiFo &fifo, TileCons &tile)
        {
            static_assert(TileCons::Loc == TileType::Vec || TileCons::Loc == TileType::Mat,
                          "Fix: TPOP has unsupported tile type!");
            if constexpr (TileCons::Loc == TileType::Vec) {
                popVecTileFromGMFiFo<TileCons, Split>(fifo, tile);
            } else if constexpr (TileCons::Loc == TileType::Mat) {
                popMatTileFromGMFiFo<TileCons, Split>(fifo, tile);
            } else {
                popCtrlFromCtrlFiFo(fifo);
            }
        }
    };

    DataFiFo fifo;
    Producer prod;
    Consumer cons;

    // initialize_pipe(DIR_MASK, entry_SIZE, GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF)
    PTO_INTERNAL explicit TPipe(__gm__ void *GM_SLOT_BUFFER, uint32_t C2V_CONSUMER_BUF, uint32_t V2C_CONSUMER_BUF)
        : fifo(GM_SLOT_BUFFER, C2V_CONSUMER_BUF, V2C_CONSUMER_BUF), prod(), cons()
    {
        cons.free();
    }

    // Destructor for TPipe
    PTO_INTERNAL ~TPipe()
    {
        prod.allocate();
    }
};

/**
 * TPUSH: Push Tile to FIFO
 * * Flow:
 * 1. [Alloc]   Check GM space (Cross-Core)
 * 2. [Store]   Write data to GM
 * 3. [Commit]  Signal Consumer (Cross-Core)
 */
template <typename Pipe, typename TileProd, typename TileCons>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus();
    if (isAllocate) {
        pipe.prod.allocate();
    }

    // 2. Address Calculation
    pipe.prod.template push<TileProd, TileCons>(pipe.fifo, tile);
    pipe.prod.tile_id++;

    // 3； Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

template <typename Pipe, typename TileProd, TileSplitAxis Split>
PTO_INTERNAL void TPUSH_IMPL(Pipe &pipe, TileProd &tile)
{
    // 1. Cross-Core: Wait for space
    bool isAllocate = pipe.prod.getAllocateStatus();
    if (isAllocate) {
        pipe.prod.allocate();
    }

    // 2. Address Calculation
    pipe.prod.template push<TileProd, Split>(pipe.fifo, tile);
    pipe.prod.tile_id++;

    // 3； Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}
} // namespace pto

#endif
