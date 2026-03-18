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

    using DataFiFo = DataFIFO<SlotSize, SlotNum, LocalSlotNum>;

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
        uint32_t tileIndex = 0;
        uint32_t subTileIndex = 0;
        bool isAllocate = true;
        bool isRecord = true;
        int entryOffset = 0;

        PTO_INTERNAL Producer() = default;

        PTO_INTERNAL void setTileId(int tIndex, int subIndex)
        {
            tileIndex = tIndex;
            subTileIndex = subIndex;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
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

        template <typename TileProd>
        PTO_INTERNAL void pushAcc2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            size_t entryBase = (tileIndex % DataFiFo::SLOT_NUM) * DataFiFo::SLOT_SIZE; // ProdM * ProdN * sizeof(T);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));
            // store tile to GM FIFO, enable unit-flag one
            if constexpr (EN_UNIT_FLAG) {
                TSTORE_IMPL<TileProd, GlobalData, AtomicType::AtomicNone, STPhase::Final>(globalTensor, tile);
            } else { // disable unit flag
                TSTORE_IMPL(globalTensor, tile);
            }
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void pushVec2GMFiFo(DataFiFo &fifo, TileProd &tile)
        {
            using T = typename TileProd::DType;
            constexpr int splitNum = 2;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ProdM * splitNum : ProdM;
            constexpr int ConsN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ProdN * splitNum : ProdN;
            size_t entryBase = (tileIndex % DataFiFo::SLOT_NUM) * DataFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            size_t subAIVOffset =
                (Split == TileSplitAxis::TILE_NO_SPLIT) ? 0 : (get_subblockid() * ProdM * ProdN * sizeof(T));
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            GlobalData globalData(addr);
            TSTORE_IMPL(globalData, tile);
        }

        template <typename TileProd, TileSplitAxis Split>
        PTO_INTERNAL void push(DataFiFo &fifo, TileProd &tile)
        {
            static_assert(TileProd::Loc == TileType::Acc || TileProd::Loc == TileType::Vec,
                          "Fix: TPUSH has unsupported tile type!");
            if constexpr (is_c2v) {
                pushAcc2GMFiFo<TileProd>(fifo, tile);
            } else if constexpr (is_v2c) {
                pushVec2GMFiFo<TileProd, Split>(fifo, tile);
            } else if constexpr (is_v2c_ctrl) {
                pushVec2CtrlFiFo<TileProd>(fifo, tile);
            }
        }

        //--------------------------------------------------------------
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
            size_t slotIndex = (tileIndex % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * kTileFactor * ProdM * ProdN * sizeof(T);
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
            size_t slotIndex = (tileIndex % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * kTileFactor * ConsM * ConsN * sizeof(T);
            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ProdM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ProdM, ProdN, BLayout::RowMajor, ProdM, ConsN>;
            TileDataSub subTile;
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);
            // store tile to GM FIFO in sub-tiles if needed (when Tile_S1 > Cube_S1)
            for (int subCol = 0; subCol < kTileFactor; ++subCol) {
                __gm__ T *addrSub = addr + subCol * ConsM * ConsN;
                GlobalDataSub globalDataSub(addrSub);
                uint64_t colByteOffset = static_cast<uint64_t>(subCol * ConsN * sizeof(T));
                TASSIGN_IMPL(subTile, (uint64_t)tile.data() + colByteOffset);
                TSTORE_IMPL(globalDataSub, subTile);
            }
        }

        template <typename TileProd>
        PTO_INTERNAL void pushVec2CtrlFiFo(DataFiFo &fifo, TileProd &tile)
        {
            size_t slotIndex = (tileIndex % DataFiFo::SLOT_NUM);
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
            if constexpr (is_c2v) {
                pushAcc2GMFiFo<TileProd, TileCons>(fifo, tile);
            } else if constexpr (is_v2c) {
                pushVec2GMFiFo<TileProd, TileCons>(fifo, tile);
            } else if constexpr (is_v2c_ctrl) {
                pushVec2CtrlFiFo<TileProd>(fifo, tile);
            }
        } // end of store

    }; // end of Producer

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int tileIndex = 0;
        int subTileIndex = 0;
        bool isWait = true;
        bool isFree = true;
        int entryOffset = 0;

        PTO_INTERNAL Consumer() = default;

        PTO_INTERNAL void setTileId(int tid, int sub_tid)
        {
            tileIndex = tid;
            subTileIndex = sub_tid;
        }

        PTO_INTERNAL int getTileId() const
        {
            return tileIndex;
        }

        PTO_INTERNAL int getSubTileId() const
        {
            return subTileIndex;
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

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            constexpr int splitNum = 2;
            constexpr int ProdM = (Split == TileSplitAxis::TILE_UP_DOWN) ? ConsM * splitNum : ConsM;
            constexpr int ProdN = (Split == TileSplitAxis::TILE_LEFT_RIGHT) ? ConsN * splitNum : ConsN;

            // global tensor
            size_t entryBase = (static_cast<size_t>(tileIndex) % DataFiFo::SLOT_NUM) *
                               DataFiFo::SLOT_SIZE; // ProdM * ProdN * sizeof(T);
            size_t subAIVOffset =
                (Split == TileSplitAxis::TILE_NO_SPLIT) ? 0 : (get_subblockid() * ConsM * ConsN * sizeof(T));
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + subAIVOffset + entryOffset);
            using GlobalData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            GlobalData globalTensor(addr);

            // local vector tile
            uint64_t localTileBase =
                fifo.C2V_CONSUMER_BUF +
                (static_cast<size_t>(tileIndex) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        template <typename TileCons, TileSplitAxis Split>
        PTO_INTERNAL void popMatTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileCons::DType;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;
            size_t entryBase = (static_cast<size_t>(tileIndex) % DataFiFo::SLOT_NUM) *
                               DataFiFo::SLOT_SIZE; // ConsM * ConsN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                fifo.V2C_CONSUMER_BUF +
                (static_cast<size_t>(tileIndex) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
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

        //--------------------------------------------------------------------
        template <typename TileProd, typename TileCons>
        PTO_INTERNAL void popVecTileFromGMFiFo(DataFiFo &fifo, TileCons &tile)
        {
            using T = typename TileProd::DType;
            constexpr int ProdM = TileProd::Rows;
            constexpr int ProdN = TileProd::Cols;
            constexpr int ConsM = TileCons::Rows;
            constexpr int ConsN = TileCons::Cols;

            size_t slotIndex = static_cast<size_t>(tileIndex) % DataFiFo::SLOT_NUM;
            constexpr int kTileFactor = ConsN / ProdN;
            size_t entryBase = static_cast<size_t>(slotIndex) * kTileFactor * ProdM * ProdN * sizeof(T);
            __gm__ T *addr = (__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset);

            uint64_t localTileBase =
                (uint64_t)fifo.C2V_SLOT_BUFFER +
                (static_cast<size_t>(tileIndex) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);

            using GlobalDataSub = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ProdN>, pto::Stride<1, 1, 1, ProdN, 1>>;
            using TileDataSub = Tile<TileType::Vec, T, ConsM, ConsN, BLayout::RowMajor, ConsM, ProdN>;
            TileDataSub tileSub;
            for (int subCol = 0; subCol < kTileFactor; ++subCol) {
                __gm__ T *addrSub = addr + subCol * ProdM * ProdN;
                uint64_t colByteOffset = subCol * ProdN * sizeof(T);
                GlobalDataSub globalTensorSub(addrSub);
                TASSIGN_IMPL(tileSub, (uint64_t)tile.data() + colByteOffset);
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
            uint32_t slotIndex = (tileIndex % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * ConsM * ProdN * sizeof(T);
            using GlobaData = GlobalTensor<T, pto::Shape<1, 1, 1, ConsM, ConsN>, pto::Stride<1, 1, 1, ConsN, 1>>;
            GlobaData globalTensor((__gm__ T *)((uint64_t)fifo.GM_SLOT_BUFFER + entryBase + entryOffset));

            uint64_t localTileBase =
                (uint64_t)fifo.V2C_CONSUMER_BUF +
                (static_cast<size_t>(tileIndex) % DataFiFo::LOCAL_SLOT_NUM) * ConsM * ConsN * sizeof(T);
            TASSIGN_IMPL(tile, localTileBase);
            TLOAD_IMPL(tile, globalTensor);
        }

        PTO_INTERNAL void popCtrlFromCtrlFiFo(DataFiFo &fifo)
        {
            uint32_t slotIndex = (tileIndex % DataFiFo::SLOT_NUM);
            size_t entryBase = slotIndex * sizeof(uint32_t);
            uint64_t ctrlTileBase = fifo.CTRL_SLOT_BUFFER + entryBase + entryOffset;
            fifo.ctrlSignal = ((*(__gm__ uint32_t *)(ctrlTileBase)) == 1) ? true : false;
        }

        template <typename TileProd, typename TileCons>
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
    };

    DataFiFo fifo;
    Producer prod;
    Consumer cons;

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
    pipe.prod.tileIndex++;

    // 3； Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

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
    pipe.prod.tileIndex++;

    // 3； Cross-Core: Commit & Signal
    bool isRecord = pipe.prod.getRecordStatus();
    if (isRecord) {
        pipe.prod.record();
    }
}

} // namespace pto

#endif
