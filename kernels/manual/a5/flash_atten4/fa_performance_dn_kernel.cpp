/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <acl/acl.h>
#include <pto/pto-inst.hpp>

#include "fa_performance_kernel.h"
#include <pto/npu/kernels/Pto_prefetch.hpp>
#include <pto/npu/a5/custom/TSyncCVID.hpp>
#include <pto/npu/a5/custom/TSync_Custom.hpp>

#include "pto_macro_dn_matmul.hpp"

#if defined (SOFTMAX_S064_4VSSTB)
#include "pto_macro_dn_softmax_s064_4vsstb.hpp"
#elif defined (SOFTMAX_S064_2VSSTB)
#include "pto_macro_dn_softmax_s064_2vsstb.hpp"
#elif defined (SOFTMAX_S0128_1VSSTB)
#include "pto_macro_dn_softmax_s0128_1vsstb.hpp"
#else
#error "Must define one of SOFTMAX_S064_4VSSTB, SOFTMAX_S064_2VSSTB, or SOFTMAX_S0128_1VSSTB"
#endif

#include "pto_macro_fa_dn_gu.hpp"

using namespace std;
using namespace pto;


enum CoreEvtID : uint32_t
{
    QK_EVENT_ID0,
    QK_EVENT_ID1,
    PV_EVENT_ID0,
    PV_EVENT_ID1,
};

// -----------------------------------------------------------------------------
// Performance tuning knobs (high-level)
//
// The kernel is a cross-core pipeline (Cube + Vec) with explicit FIFOs:
//   QK (Cube):  compute_qk   -> qk_tile_fifo (fp32)
//   P  (Vec):   compute_p    -> p_tile_fifo  (fp16 x_exp) + l1_exp_max_ififo
//   PV (Cube):  compute_pv   -> pv_tile_fifo (fp32)
//   GU (Vec):   compute_gu   -> o_out (fp32) with running rescale/update
//
// Key knobs that impact throughput (see runTFA<> below):
// - CUBE_S0 / CUBE_S1: tile sizes for QK/PV cube matmuls (compute intensity vs. buffer pressure)
// - qkPreloadNum: pipeline warmup depth (more overlap vs. more L1 FIFO footprint)
// - *_TNBuffers: ping/pong depth for Mat tiles (overlap) and Vec tiles (latency hiding)
// - QKV_CV_FIFO / PV_CV_FIFO: FIFO depth between stages (avoid backpressure)
// -----------------------------------------------------------------------------

// Inline macro used for small, performance-sensitive functions
#ifndef PTO_INLINE
#define PTO_INLINE __attribute__((always_inline)) inline
#endif

// Detect build-time macros and expose as constexpr flags for clearer conditionals
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

constexpr std::size_t MAX_TILE_L1_BYTES = 512U * 1024U;
constexpr std::size_t MAX_VEC_UB_BYTES = 256U * 1024U;

template <typename DstTileData, typename SrcTileData>
AICORE inline void TMOVUB2L1(DstTileData &dst, const SrcTileData &src)
{
    uint16_t rows = src.GetValidRow(); // 256
    uint16_t cols = src.GetValidCol(); // 64
    copy_ubuf_to_cbuf(dst.data(), src.data(), 0, cols / 16, rows / 2, 1, rows / 2);
}

// Decide whether to block or signal consumption flags for a given tile index.
// Reverse dependency: notify one step before the corresponding wait within each sync period.
template <int FifoSize, int SyncPeriod>
AICORE inline bool should_wait_consumption(int sync_iter)
{
    static_assert(FifoSize >= 1, "CV FIFO size must be >= 1");
    constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
    static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
    if (sync_iter < static_cast<int>(FifoSize))
        return false;
    return (sync_iter % period) == 0;
}

template <int FifoSize, int SyncPeriod>
AICORE inline bool should_notify_consumption(int sync_iter)
{
    static_assert(FifoSize >= 1, "CV FIFO size must be >= 1");
    constexpr int period = (SyncPeriod > 0) ? SyncPeriod : 1;
    static_assert(period >= 1, "CV FIFO consume sync period must be >= 1");
    return ((sync_iter + 1) % period) == 0; // notify one tile earlier than the wait check
}

// Drains the outstanding backward credits for a fixed-size producer ring.
AICORE inline int pending_ring_events(int tiles_processed, int ring_size)
{
    if (tiles_processed <= 0 || ring_size <= 0)
        return 0;
    return (tiles_processed < ring_size) ? tiles_processed : ring_size;
}

AICORE inline void wait_mte3_to_v_pingpong(uint64_t event_id)
{
    if (event_id == EVENT_ID0) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    } else {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    }
}

AICORE inline void set_mte3_to_v_pingpong(uint64_t event_id)
{
    if (event_id == EVENT_ID0) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
    } else {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    }
}

// Compute how many consumption notifications have not been waited on yet so we can drain them at kernel tail.
AICORE inline int pending_consumption_events(int tiles_processed, int fifo_size, int sync_period)
{
    if (tiles_processed <= 0 || sync_period <= 0 || fifo_size <= 0)
        return 0;

    const int notify_count = tiles_processed / sync_period; // notifications fire every sync_period tiles

    int wait_count = 0;
    if (tiles_processed > fifo_size) {
        const int last_iter = tiles_processed - 1;
        wait_count = (last_iter / sync_period) - ((fifo_size - 1) / sync_period); // waits start after FIFO is filled
        if (wait_count < 0)
            wait_count = 0;
    }

    int pending = notify_count - wait_count;
    if (pending < 0)
        pending = 0;

    const int max_pending = (fifo_size + sync_period - 1) / sync_period; // ceil(fifo_size / sync_period)
    return (pending > max_pending) ? max_pending : pending;
}

template <int TilesProcessed, int FifoSize, int SyncPeriod>
AICORE inline constexpr int pending_consumption_events_const()
{
    static_assert(TilesProcessed >= 0, "TilesProcessed must be >= 0");
    static_assert(FifoSize >= 1, "FifoSize must be >= 1");
    static_assert(SyncPeriod >= 1, "SyncPeriod must be >= 1");

    constexpr int notify_count = TilesProcessed / SyncPeriod;
    constexpr int wait_count =
        (TilesProcessed > FifoSize) ? (((TilesProcessed - 1) / SyncPeriod) - ((FifoSize - 1) / SyncPeriod)) : 0;
    constexpr int pending = (notify_count > wait_count) ? (notify_count - wait_count) : 0;
    constexpr int max_pending = (FifoSize + SyncPeriod - 1) / SyncPeriod;
    return (pending > max_pending) ? max_pending : pending;
}

template <typename TileType>
constexpr AICORE std::size_t tile_storage_bytes()
{
    using ElementType = typename TileType::DType;
    return static_cast<std::size_t>(TileType::Rows * TileType::Cols) * sizeof(ElementType);
}

template <typename TileType, std::size_t NumBuffers>
constexpr AICORE std::size_t tile_buffer_total_bytes()
{
    return tile_storage_bytes<TileType>() * NumBuffers;
}

template <typename TileType, std::size_t NumBuffers>
AICORE inline uint32_t assign_tile_buffers(TileType (&tiles)[NumBuffers], uint32_t base_offset)
{
    if constexpr (NumBuffers == 0) {
        return base_offset;
    }

    constexpr std::size_t total_storage_bytes = tile_buffer_total_bytes<TileType, NumBuffers>();
    static_assert(total_storage_bytes <= MAX_TILE_L1_BYTES, "Tile buffer L1 allocation exceeds 512KB");

    for (std::size_t idx = 0; idx < NumBuffers; ++idx) {
        const uint32_t tile_offset = base_offset + static_cast<uint32_t>(idx * tile_storage_bytes<TileType>());
        TASSIGN(tiles[idx], tile_offset);
    }

    return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileA, std::size_t NumA, typename TileB, std::size_t NumB>
AICORE inline uint32_t assign_tile_buffers_union(TileA (&tilesA)[NumA], TileB (&tilesB)[NumB], uint32_t base_offset)
{
    static_assert(NumA == NumB, "Union assignment expects matching buffer counts");
    if constexpr (NumA == 0) {
        return base_offset;
    }

    constexpr std::size_t stride_bytes = (tile_storage_bytes<TileA>() > tile_storage_bytes<TileB>()) ?
                                             tile_storage_bytes<TileA>() :
                                             tile_storage_bytes<TileB>();
    constexpr std::size_t total_storage_bytes = stride_bytes * NumA;
    static_assert(total_storage_bytes <= MAX_VEC_UB_BYTES, "Union tile UB allocation exceeds 256KB");

    for (std::size_t idx = 0; idx < NumA; ++idx) {
        const uint32_t tile_offset = base_offset + static_cast<uint32_t>(idx * stride_bytes);
        TASSIGN(tilesA[idx], tile_offset);
        TASSIGN(tilesB[idx], tile_offset);
    }

    return base_offset + static_cast<uint32_t>(total_storage_bytes);
}

template <typename TileQType, std::size_t NumQ, typename TileKType, std::size_t NumK, typename TilePType,
          std::size_t NumP, typename TileVType, std::size_t NumV>
AICORE inline void allocate_cube_tile_buffers(TileQType (&qTiles)[NumQ], TileKType (&kTiles)[NumK],
                                              TilePType (&pTiles)[NumP], TileVType (&vTiles)[NumV])
{
    constexpr std::size_t total_bytes =
        tile_buffer_total_bytes<TileQType, NumQ>() + tile_buffer_total_bytes<TileKType, NumK>() +
        tile_buffer_total_bytes<TilePType, NumP>() + tile_buffer_total_bytes<TileVType, NumV>();
    static_assert(total_bytes <= MAX_TILE_L1_BYTES, "Total cube L1 allocation exceeds 512KB");

    uint32_t l1_offset = 0;
    l1_offset = assign_tile_buffers(qTiles, l1_offset);
    l1_offset = assign_tile_buffers(kTiles, l1_offset);
    l1_offset = assign_tile_buffers(pTiles, l1_offset);
    l1_offset = assign_tile_buffers(vTiles, l1_offset);
    (void)l1_offset;
}

template <typename TileDataF_T, typename ReduceTileF_T, typename TileDataH_T, typename TileOutT,
          typename TileDataH_NZ_T, std::size_t SrcBuffers, std::size_t XexpBuffers, std::size_t pvVecBuffers,
          std::size_t ExpMaxBuffers>
AICORE inline void allocate_vec_tile_buffers(TileDataF_T (&srcTiles)[SrcBuffers], ReduceTileF_T &m1_local_max,
                                             TileDataF_T &input_reduce_tmp, ReduceTileF_T &l1_local_sum,
                                             ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
                                             ReduceTileF_T (&l1_exp_max)[ExpMaxBuffers],
                                             TileDataH_T (&x_expT)[XexpBuffers], TileOutT (&pvTile)[pvVecBuffers],
                                             TileOutT &runningOTile, TileDataH_NZ_T (&nzConvBuffer)[XexpBuffers])
{
#if REUSE_QK_PV_BUFFERS
    constexpr std::size_t union_stride = (tile_storage_bytes<TileDataF_T>() > tile_storage_bytes<TileOutT>()) ?
                                             tile_storage_bytes<TileDataF_T>() :
                                             tile_storage_bytes<TileOutT>();
    constexpr std::size_t union_bytes = union_stride * SrcBuffers;
    static_assert(SrcBuffers == pvVecBuffers, "src/pv buffer counts must match");

    constexpr std::size_t p_nz_bytes = tile_buffer_total_bytes<TileDataH_NZ_T, XexpBuffers>();
    constexpr std::size_t reduce_tile_bytes = tile_storage_bytes<ReduceTileF_T>();
    constexpr std::size_t out_tile_bytes = tile_storage_bytes<TileOutT>();

    constexpr std::size_t total_bytes =
        union_bytes + p_nz_bytes + (reduce_tile_bytes * (4U + ExpMaxBuffers)) + out_tile_bytes;
    static_assert(total_bytes <= MAX_VEC_UB_BYTES, "Vec tile UB allocation exceeds 256KB");

    uint32_t offset = 0;
    offset = assign_tile_buffers_union(srcTiles, pvTile, offset);
#else
    constexpr std::size_t float_tile_bytes = tile_storage_bytes<TileDataF_T>();
    constexpr std::size_t src_bytes = tile_buffer_total_bytes<TileDataF_T, SrcBuffers>();
    constexpr std::size_t pv_bytes = tile_buffer_total_bytes<TileOutT, pvVecBuffers>();
    constexpr std::size_t p_nz_bytes = tile_buffer_total_bytes<TileDataH_NZ_T, XexpBuffers>();
    constexpr std::size_t reduce_tile_bytes = tile_storage_bytes<ReduceTileF_T>();
    constexpr std::size_t out_tile_bytes = tile_storage_bytes<TileOutT>();
    constexpr std::size_t total_bytes =
        src_bytes + pv_bytes + p_nz_bytes + (reduce_tile_bytes * (4U + ExpMaxBuffers)) + out_tile_bytes;
    static_assert(total_bytes <= MAX_VEC_UB_BYTES, "Vec tile UB allocation exceeds 256KB");

    uint32_t offset = 0;
    // Allocate qkVecTile (srcTiles) first
    offset = assign_tile_buffers(srcTiles, offset);
    offset = assign_tile_buffers(pvTile, offset);
    // Allocate pvVecTile separately (no union - avoids TLOAD overwriting QK data)
#endif
    TASSIGN(runningOTile, offset);
    offset += out_tile_bytes;

    TASSIGN(m1_local_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(m2_global_max, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(l1_local_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    TASSIGN(l2_global_sum, offset);
    offset += static_cast<uint32_t>(reduce_tile_bytes);

    offset = assign_tile_buffers(l1_exp_max, offset);
    uint32_t tail_offset = assign_tile_buffers(nzConvBuffer, offset);
}

// Helper to assign an accumulator tile to one of two ping-pong L0C addresses separated by slot_bytes.
// Keeps a per-type static running index that toggles on every call. Caller may pass
// `initial_id` (0 or 1) to set the starting buffer index on the first call for that tile type.
template <typename AccTileT>
AICORE inline int assign_running_acc_tile(AccTileT &accTile, uint32_t slot_bytes, int initial_id = -1)
{
    static int running_tile_buffer_idx = 0; // per-instantiation running buffer index: 0 -> base0, 1 -> base1
    if (initial_id == 0 || initial_id == 1) {
        running_tile_buffer_idx = initial_id;
    }
    const int id = running_tile_buffer_idx;
    const uint32_t base_addr = static_cast<uint32_t>(id) * slot_bytes;
    TASSIGN(accTile, base_addr);
    running_tile_buffer_idx ^= 1; // toggle for next call
    return id;
}

template <typename TSyncSM2PV>
struct Sm2PvFreeHook {
    TSyncSM2PV &sync;
    bool enable;

    AICORE inline void operator()() const
    {
        if (enable) {
            sync.free();
        }
    }
};

struct PreATExtOpReadyHook {
    AICORE inline void operator()() const
    {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    }
};

struct PreBTExtOpReadyHook {
    AICORE inline void operator()() const
    {
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    }
};

template <typename TSyncSM2PV>
struct PReadyHook {
    TSyncSM2PV &sync;
    bool enable;

    AICORE inline void operator()() const
    {
        if (enable) {
            sync.wait();
        }
    }
};

struct QReadyHook {
    bool enable;

    AICORE inline void operator()() const
    {
        if (enable) {
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }
    }
};

template <bool PrefetchK1, bool PrefetchK2, typename TileScratch, typename GlobalK, typename GlobalKNext,
          typename GlobalKNext2, typename GlobalV>
AICORE inline void prefetch_first_qkv_tiles(TileScratch &scratchTile, GlobalK &kGlobal, GlobalKNext &kNextGlobal,
                                            GlobalKNext2 &kNext2Global, GlobalV &vGlobal)
{
    if constexpr (DAV_VEC) {
        if (static_cast<size_t>(get_subblockid()) == 0U) {
            TPREFETCH(scratchTile, kGlobal);
            if constexpr (PrefetchK2) {
                TPREFETCH(scratchTile, kNext2Global);
            }
        } else {
            if constexpr (PrefetchK1) {
                TPREFETCH(scratchTile, kNextGlobal);
            }
            TPREFETCH(scratchTile, vGlobal);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO,
          int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int SRC_VEC_TN_BUFFERS,
          typename TileMatQData, typename TileMatKData, typename TileQKData, typename TileQKVecData,
          typename TSyncQK2SM>
AICORE inline void compute_qk(int tile_id, int sub_tile_id, int ub_buf_idx, __gm__ half *q, __gm__ half *k,
                              __gm__ float *qk_tile_fifo, TileMatQData &qMatTile, TileMatKData &kMatTile,
                              TileQKData &qkAccTile, TileQKVecData &qkVecTile, uint64_t qkMatTileEventId,
                              int accTileEvtID, TSyncQK2SM &qk2smSync, int blk_idx, int qkDrainCarryCount)
{
    if constexpr (DAV_CUBE) {
        constexpr uint32_t Cube_S0 = CUBE_S0;
        constexpr uint32_t Cube_S1 = CUBE_S1;
        constexpr uint32_t Tile_S1 = TILE_S1;
        constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
        constexpr uint32_t Cube_HEAD = HEAD_SIZE;
        static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
        static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

        const int s0_index = blk_idx * CUBE_S0;
        const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0 && tile_id >= static_cast<int>(SRC_VEC_TN_BUFFERS))
                    qk2smSync.allocate(); // wait for SM consume data
                if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
                    qk2smSync.record(); // notify for QK produce data
                return;
            }
        }
        using GlobalDataQ =
            GlobalTensor<half, pto::Shape<1, 1, 1, HEAD_SIZE, Cube_S0>, pto::Stride<1, 1, 1, 1, HEAD_SIZE>, Layout::DN>;
        using GlobalDataK =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

        GlobalDataQ qGlobal(q);
        GlobalDataK kGlobal(k + s1_index * HEAD_SIZE);

        wait_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);

        if (tile_id == 0 && sub_tile_id == 0) {
            TLOAD(qMatTile, qGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        }

        TLOAD(kMatTile, kGlobal);
#if defined MARK_STAMP_DATA_PIPE
        bisheng::cce::mark_stamp<PIPE_MTE2>(QK_DONE * 1000 + tile_id);
#endif

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        QReadyHook qReadyHook{tile_id == 0 && sub_tile_id == 0};
        PreATExtOpReadyHook preATExtOpReadyHook;

        wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);

        pto_macro_matmul<Cube_S1, Cube_HEAD, Cube_S0, true>(kMatTile, qMatTile, qkAccTile, AccMode::Init,
                                                            QK_DONE * 1000 + tile_id, preATExtOpReadyHook, qReadyHook);
#if defined MARK_STAMP
        bisheng::cce::mark_stamp<PIPE_M>(QK_DONE * 1000 + tile_id);
#endif

        set_flag(PIPE_MTE1, PIPE_MTE2, qkMatTileEventId);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        if constexpr (INTERMEDIATE_CHECK) {
            // When ND_LAYOUT=1: matmul is (Cube_S0, HEAD, Cube_S1) → L0C is Cube_S0×Cube_S1, store ND shape.
            // When ND_LAYOUT=0: matmul is (Cube_S1, HEAD, Cube_S0) → L0C is Cube_S1×Cube_S0, store DN shape.
            // Using the wrong shape with a square tile still writes the same byte count but transposes the data,
            // so we must match the actual L0C M×N dimensions.
            const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
            const size_t base_elems =
                static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) * static_cast<size_t>(Cube_S0) *
                    static_cast<size_t>(Cube_S1) +
                static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
            // DN matmul result is M=Cube_S1, N=Cube_S0 → use matching DN shape.
            using GlobalDataQK =
                GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S1, Cube_S0>, pto::Stride<1, 1, 1, Cube_S0, 1>>;
            GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);
            TSTORE(qkGlobalTile, qkAccTile);
            // set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
            // wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);
        }

        constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
        const uint64_t col_byte_offset = static_cast<uint64_t>(sub_tile_id * Cube_S1 * sizeof(float));
        // const uint64_t col_byte_offset = static_cast<uint64_t>(sub_tile_id * sizeof(float));
        // using TileDataF_Sub = Tile<TileType::Vec, float, Tile_S1, Vec_S0, BLayout::ColMajor, Tile_S1, Vec_S0>;
        using TileDataF_Sub = Tile<TileType::Vec, float, Tile_S1, Vec_S0, BLayout::RowMajor, Tile_S1, Vec_S0>;
        TileDataF_Sub qkVecTileSubDN;
        TASSIGN(qkVecTileSubDN, (uint64_t)qkVecTile.data() + col_byte_offset);

        // qk2smSync is the credit semaphore for qkVecTile[] reuse on the UB path.
        const bool should_wait_ub_reuse =
            sub_tile_id == 0 && (tile_id < qkDrainCarryCount || tile_id >= static_cast<int>(SRC_VEC_TN_BUFFERS));
        if (should_wait_ub_reuse) {
            qk2smSync.allocate();
        }

        TMOV<TileDataF_Sub, TileQKData, AccToVecMode::DualModeSplitN>(qkVecTileSubDN, qkAccTile);
#if defined MARK_STAMP_DATA_PIPE
        bisheng::cce::mark_stamp<PIPE_FIX>(QK_DONE * 1000 + tile_id);
#endif

        set_flag(PIPE_FIX, PIPE_M, accTileEvtID);

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1) {
            qk2smSync.record();
        }
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int PV_CV_FIFO,
          int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int OUT_O_TILE_NBUFFERS,
          typename TileMatPData, typename TileMatVData, typename TilePVData, typename TileOutT, typename TSyncSM2PV,
          typename TSyncPV2GU>
AICORE inline void compute_pv(int tile_id, int sub_tile_id, int pv_ub_buf_idx, __gm__ half *p_tile_fifo, __gm__ half *v,
                              __gm__ float *pv_tile_fifo, __gm__ float *pv_pend_tile_fifo, TileMatPData &pMatTile,
                              TileMatVData &vMatTile, TilePVData &pvAccTile, TilePVData &pvAccPendTile,
                              TileOutT &runningOTile, TileOutT &pvPendTile, TileOutT (&pvVecTile)[OUT_O_TILE_NBUFFERS],
                              uint64_t svMatTileEventId, int accTileEvtID, TSyncSM2PV &sm2pvSync, TSyncPV2GU &pv2guSync,
                              int blk_idx, int pvDrainCarryCount)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");

    const int s0_index = blk_idx * Cube_S0;
    const int s1_index = tile_id * static_cast<int>(Tile_S1) + sub_tile_id * static_cast<int>(Cube_S1);
    const int sync_iter = tile_id;
    const bool next_will_be_skipped = (s1_index + static_cast<int>(Cube_S1)) > s0_index && CAUSAL_MASK;

    if constexpr (DAV_CUBE) {
        if constexpr (CAUSAL_MASK) {
            if (s1_index > s0_index) {
                if (sub_tile_id == 0)
                    sm2pvSync.wait(); // wait for softmax produce data
                if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
                    sm2pvSync.free(); // notify SV consume data
                return;
            }
        }

#if skip_rescale
        constexpr int SKIP_SYNC_PERIOD = kFaCvFifoConsSyncPeriod;
        const bool should_notify_consumed =
            should_notify_consumption<SKIP_STATUS_FIFO_SIZE, SKIP_SYNC_PERIOD>(sync_iter);

        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY);
        wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 16);

        const bool prev_skip = (tile_id == 0) ? false :
                                                (read_skip_status_slot(tile_id - 1, false) != 0 &&
                                                 read_skip_status_slot(tile_id - 1, true) != 0);
        const bool curr_skip =
            (read_skip_status_slot(tile_id, false) != 0 && read_skip_status_slot(tile_id, true) != 0);

        if (should_notify_consumed) {
            set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
            set_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1 + 16);
        }
#else
        const bool prev_skip = false;
        const bool curr_skip = false;
#endif

        const bool cond2_skip_after_skip = curr_skip && prev_skip;
        const bool cond3_non_skip_after_non_skip = !curr_skip && !prev_skip;
        const bool cond4_non_skip_after_skip = !curr_skip && prev_skip;

#if skip_rescale
        TilePVData &dstTile = curr_skip ? pvAccPendTile : pvAccTile;
#endif

        using GlobalVT =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

        wait_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        GlobalVT vLoad((__gm__ half *)(v + s1_index * HEAD_SIZE));
        TLOAD(vMatTile, vLoad);
#if defined MARK_STAMP_DATA_PIPE
        bisheng::cce::mark_stamp<PIPE_MTE2>(PV_DONE * 1000 + tile_id);
#endif

        PReadyHook<TSyncSM2PV> pReadyHook{sm2pvSync, sub_tile_id == 0};

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

        wait_flag(PIPE_FIX, PIPE_M, accTileEvtID);

        PreBTExtOpReadyHook preBTExtOpReadyHook;

#if skip_rescale
        AccMode accMode;
        if (cond2_skip_after_skip) {
            accMode = AccMode::Acc;
        } else {
            accMode = (sub_tile_id == 0) ? AccMode::Init : AccMode::Acc;
        }
        Sm2PvFreeHook<TSyncSM2PV> sm2pvFreeHook{sm2pvSync, sub_tile_id == static_cast<int>(kTileFactor) - 1};
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD, true>(pMatTile, vMatTile, dstTile, accMode,
                                                            PV_DONE * 1000 + tile_id, pReadyHook, preBTExtOpReadyHook,
                                                            sm2pvFreeHook);
#else
        const AccMode accMode = (sub_tile_id == 0) ? AccMode::Init : AccMode::Acc;
        Sm2PvFreeHook<TSyncSM2PV> sm2pvFreeHook{sm2pvSync, sub_tile_id == static_cast<int>(kTileFactor) - 1};
        pto_macro_matmul<Cube_S0, Cube_S1, Cube_HEAD, true>(pMatTile, vMatTile, pvAccTile, accMode,
                                                            PV_DONE * 1000 + tile_id, pReadyHook, preBTExtOpReadyHook,
                                                            sm2pvFreeHook);
#endif
#if defined MARK_STAMP
        bisheng::cce::mark_stamp<PIPE_M>(PV_DONE * 1000 + tile_id);
#endif
        set_flag(PIPE_MTE1, PIPE_MTE2, svMatTileEventId);

        if (sub_tile_id == static_cast<int>(kTileFactor) - 1 || next_will_be_skipped) {
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

#if skip_rescale
            if (tile_id < pvDrainCarryCount || tile_id >= static_cast<int>(1)) {
                pv2guSync.allocate();
            }
            if (cond4_non_skip_after_skip) {
                TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(pvPendTile, pvAccPendTile);
                TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(pvVecTile[pv_ub_buf_idx], pvAccTile);

                if constexpr (INTERMEDIATE_CHECK) {
                    using GlobalDataPV = GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>,
                                                      pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
                    const uint32_t buf_idx_pv = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
                    const size_t base_elems_pv =
                        static_cast<size_t>(buf_idx_pv) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);
                    GlobalDataPV pvPendGlobalTile((__gm__ float *)(pv_pend_tile_fifo + base_elems_pv));
                    GlobalDataPV pvGlobalTile((__gm__ float *)(pv_tile_fifo + base_elems_pv));
                    TSTORE(pvPendGlobalTile, pvAccPendTile);
                    TSTORE(pvGlobalTile, pvAccTile);
                }
                set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
            } else if (cond3_non_skip_after_non_skip) {
                if (tile_id == 0) {
                    wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE);
                    wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE + 16);
                    TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(runningOTile, pvAccTile);
                } else {
                    TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(pvVecTile[pv_ub_buf_idx], pvAccTile);
                }
#if defined MARK_STAMP_DATA_PIPE
                bisheng::cce::mark_stamp<PIPE_FIX>(PV_DONE * 1000 + tile_id);
#endif

                if constexpr (INTERMEDIATE_CHECK) {
                    using GlobalDataPV = GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>,
                                                      pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
                    const uint32_t buf_idx_pv = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
                    const size_t base_elems_pv =
                        static_cast<size_t>(buf_idx_pv) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);
                    GlobalDataPV pvGlobalTile((__gm__ float *)(pv_tile_fifo + base_elems_pv));
                    TSTORE(pvGlobalTile, pvAccTile);
                }
                set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
            } else {
                set_flag(PIPE_FIX, PIPE_M, accTileEvtID);
            }

            pv2guSync.record();
#else

            if (tile_id < pvDrainCarryCount || tile_id >= static_cast<int>(OUT_O_TILE_NBUFFERS)) {
                pv2guSync.allocate();
            }

            if (tile_id == 0) {
                wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE);
                wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE + 16);
                TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(runningOTile, pvAccTile);
            } else {
                TMOV<TileOutT, TilePVData, AccToVecMode::DualModeSplitM>(pvVecTile[pv_ub_buf_idx], pvAccTile);
            }
#if defined MARK_STAMP_DATA_PIPE
            bisheng::cce::mark_stamp<PIPE_FIX>(PV_DONE * 1000 + tile_id);
#endif

            if constexpr (INTERMEDIATE_CHECK) {
                using GlobalDataPV =
                    GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
                const uint32_t buf_idx_pv = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
                const size_t base_elems_pv =
                    static_cast<size_t>(buf_idx_pv) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);
                GlobalDataPV pvGlobalTile((__gm__ float *)(pv_tile_fifo + base_elems_pv));
                TSTORE(pvGlobalTile, pvAccTile);
            }

            set_flag(PIPE_FIX, PIPE_M, accTileEvtID);

            pv2guSync.record();
#endif
        }
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO,
          int CV_FIFO_CONS_SYNC_PERIOD, bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int PMAT_TN_BUFFERS,
          typename TileDataF_T, typename TileDataH_T, typename TileDataH_NZ_T, typename ReduceTileF_T,
          typename TileMatPData, typename TSyncQK2SM, typename TSyncSM2PV>
AICORE inline void compute_p(int tile_id, int row_slice, __gm__ float *qk_tile_fifo, __gm__ half *p_tile_fifo,
                             __gm__ float *exp_max_ififo, TileDataF_T &qkVecTile, TileDataH_T &x_expT,
                             TileDataF_T &input_reduce_tmp, ReduceTileF_T &m1_local_max, ReduceTileF_T &l1_local_sum,
                             ReduceTileF_T &m2_global_max, ReduceTileF_T &l2_global_sum,
                             ReduceTileF_T &l1_exp_max_ififo, TileMatPData &pMatTile, TileDataH_NZ_T &nzConvBuffer,
                             uint64_t pTileEventId, TSyncQK2SM &qk2smSync, TSyncSM2PV sm2pvSync, int blk_idx,
                             int num_tiles)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    const bool initFlag = (tile_id == 0);
    static_assert(QKP_CV_FIFO >= 1, "QKP_CV_FIFO must be >= 1");
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");
    if constexpr (DAV_VEC) {
        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
        const int s0_index = blk_idx * Cube_S0 + row_offset;
        const int s1_index = tile_id * static_cast<int>(Tile_S1);
        const int sync_iter = tile_id;
        const bool last_tile = (tile_id == num_tiles - 1);

        if (row_slice == 0)
            qk2smSync.wait(); // wait for QK produce data

        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
        const size_t base_elems = static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
                                  static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);

        (void)base_elems;

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        using ReduceSliceTile = Tile<TileType::Vec, float, 1, Vec_S0, BLayout::RowMajor, 1, Vec_S0>;
        const size_t reduce_slice_rows = static_cast<size_t>(row_slice * Vec_S0);
        const uint64_t reduce_row_byte_offset = reduce_slice_rows * sizeof(float);

        ReduceSliceTile m1_local_max_slice;
        ReduceSliceTile l1_local_sum_slice;
        ReduceSliceTile m2_global_max_slice;
        ReduceSliceTile l2_global_sum_slice;
        ReduceSliceTile l1_exp_max_slice;

        TASSIGN(m1_local_max_slice, (uint64_t)m1_local_max.data() + reduce_row_byte_offset);
        TASSIGN(l1_local_sum_slice, (uint64_t)l1_local_sum.data() + reduce_row_byte_offset);
        TASSIGN(m2_global_max_slice, (uint64_t)m2_global_max.data() + reduce_row_byte_offset);
        TASSIGN(l2_global_sum_slice, (uint64_t)l2_global_sum.data() + reduce_row_byte_offset);
        TASSIGN(l1_exp_max_slice, (uint64_t)l1_exp_max_ififo.data() + reduce_row_byte_offset);

        wait_mte3_to_v_pingpong(pTileEventId);
        if (initFlag) {
            pto_macro_fa_softmax_dn<true, HEAD_SIZE, CAUSAL_MASK, ReduceTileF_T, TileDataH_T, TileDataF_T,
                                    TileDataH_NZ_T, FIFO_MODE>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, input_reduce_tmp, nzConvBuffer, s0_index, s1_index,
                tile_id, sync_iter, last_tile);
        } else {
            pto_macro_fa_softmax_dn<false, HEAD_SIZE, CAUSAL_MASK, ReduceTileF_T, TileDataH_T, TileDataF_T,
                                    TileDataH_NZ_T, FIFO_MODE>(
                x_expT, qkVecTile, m1_local_max_slice, l1_local_sum_slice, m2_global_max_slice, l2_global_sum_slice,
                l1_exp_max_slice, input_reduce_tmp, qkVecTile, input_reduce_tmp, nzConvBuffer, s0_index, s1_index,
                tile_id, sync_iter, last_tile);
        }
#if defined MARK_STAMP
        bisheng::cce::mark_stamp<PIPE_V>(P_DONE * 1000 + tile_id);
#endif

        if (row_slice == static_cast<int>(kTileFactor) - 1)
            qk2smSync.free();

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        if (row_slice == 0 && tile_id >= static_cast<int>(PMAT_TN_BUFFERS))
            sm2pvSync.allocate();

        using GlobalPTileHalfSub =
            GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, Vec_S0>, pto::Stride<1, 1, 1, Cube_S0, 1>>;
        using TileDataH_Sub = Tile<TileType::Vec, half, Tile_S1, Vec_S0, BLayout::RowMajor, Cube_S1, Vec_S0>;
        __gm__ half *p_ptr = p_tile_fifo + base_elems + row_offset;

        for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
            if constexpr (INTERMEDIATE_CHECK) {
                constexpr uint32_t NzBufRows = Cube_S1 + 1;
                constexpr uint32_t VecChunks = Vec_S0 / 16;
                __gm__ half *p_ptr_sub =
                    p_ptr + static_cast<size_t>(sub_col) * static_cast<size_t>(Cube_S1) * static_cast<size_t>(Cube_S0);
                for (uint32_t vec_chunk = 0; vec_chunk < VecChunks; ++vec_chunk) {
                    using GlobalPTileChunk =
                        GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S1, 16>, pto::Stride<1, 1, 1, 16, 1>>;
                    using TileChunkH = Tile<TileType::Vec, half, Cube_S1, 16, BLayout::RowMajor, Cube_S1, 16>;
                    size_t gm_offset = static_cast<size_t>(vec_chunk) * static_cast<size_t>(Cube_S1) * 16;
                    __gm__ half *p_chunk_gm = p_ptr_sub + gm_offset;
                    GlobalPTileChunk pTileChunk(p_chunk_gm);
                    TileChunkH chunkTile;
                    uint64_t nz_buf_byte_offset =
                        static_cast<uint64_t>(vec_chunk) * static_cast<uint64_t>(NzBufRows) * 16 * sizeof(half);
                    TASSIGN(chunkTile, (uint64_t)nzConvBuffer.data() + nz_buf_byte_offset);
                    TSTORE(pTileChunk, chunkTile);
                }
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            }

            // Softmax vsstb already filled nzConvBuffer (NZ+1); skip ND->NZ TMOV before TINSERT.
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
#if defined (SOFTMAX_S064_2VSSTB)
            using TileMatPSub = Tile<TileType::Mat, half, Cube_S0, Cube_S1, BLayout::RowMajor, Cube_S0, Cube_S1,
                                    SLayout::ColMajor, 512>;
            TileMatPSub pMatSub;
            TASSIGN(pMatSub, (uint64_t)(pMatTile.data() +
                                get_subblockid() * static_cast<uint64_t>(Cube_S1) * Vec_S0));
            TMOVUB2L1(pMatSub, nzConvBuffer);

            TASSIGN(pMatSub, (uint64_t)(pMatTile.data() + Cube_S1 / 2 * 16 +
                                get_subblockid() * static_cast<uint64_t>(Cube_S1) * Vec_S0 ));
            TASSIGN(nzConvBuffer, (uint64_t)(nzConvBuffer.data() + Vec_S0 * (Cube_S1 / 2 + 1)));
            TMOVUB2L1(pMatSub, nzConvBuffer);
#else            
            uint16_t row_offset = static_cast<uint16_t>(sub_col * Cube_S1);
            uint16_t col_offset = static_cast<uint16_t>(Vec_S0 * static_cast<size_t>(get_subblockid()));
            TINSERT(pMatTile, nzConvBuffer, row_offset, col_offset);
#endif
        }
#if defined MARK_STAMP_DATA_PIPE
        bisheng::cce::mark_stamp<PIPE_MTE3>(P_DONE * 1000 + tile_id);
#endif
        if constexpr (INTERMEDIATE_CHECK) {
            if (row_slice == static_cast<int>(kTileFactor) - 1) {
                constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
                using GlobalPMaxFloatSub =
                    GlobalTensor<float, pto::Shape<1, 1, 1, 1, SubblockRows>, pto::Stride<1, 1, 1, Cube_S0, 1>>;
                using ExpMaxSub = Tile<TileType::Vec, float, 1, SubblockRows, BLayout::RowMajor, 1, SubblockRows>;
                const size_t base_elems_pmax =
                    static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) + subblock_base_rows;
                __gm__ float *p_ptr_fp32 = exp_max_ififo + base_elems_pmax;
                GlobalPMaxFloatSub pMaxGlobal(p_ptr_fp32);
                ExpMaxSub l1_exp_max_rowmajor;
                TRESHAPE(l1_exp_max_rowmajor, l1_exp_max_ififo);
                TSTORE(pMaxGlobal, l1_exp_max_rowmajor);
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }

        if (row_slice == static_cast<int>(kTileFactor) - 1)
            sm2pvSync.record();

        set_mte3_to_v_pingpong(pTileEventId);
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int TILE_S1, int PV_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int SRC_VEC_TN_BUFFERS, int OUT_O_TILE_NBUFFERS, typename TileOutT,
          typename ReduceTileF_T, typename TSyncPV2GU>
AICORE inline void compute_gu(int tile_id, int num_tiles, __gm__ float *pv_tile_fifo, __gm__ float *pv_pend_tile_fifo,
                              __gm__ float *o_out, __gm__ float *o_parts_out, TileOutT &runningOTile,
                              TileOutT &pvVecTile, TileOutT &pvPendTile, ReduceTileF_T &l1_exp_max_ififo,
                              ReduceTileF_T &l2_global_sum, uint64_t guEventId, TSyncPV2GU &pv2guSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES;

    using GlobalDataPV_VEC =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;

    if constexpr (DAV_VEC) {
#if skip_rescale
        uint32_t vec0_cur_skip = read_skip_status_slot(tile_id, false);
        uint32_t vec1_cur_skip = read_skip_status_slot(tile_id, true);
        uint32_t vec0_prev_skip = (tile_id == 0) ? 0 : read_skip_status_slot(tile_id - 1, false);
        uint32_t vec1_prev_skip = (tile_id == 0) ? 0 : read_skip_status_slot(tile_id - 1, true);
        if (vec0_cur_skip && vec1_cur_skip) {
            pv2guSync.wait();
            pv2guSync.free();
            return;
        }

        const bool prev_skip = (vec0_prev_skip != 0 && vec1_prev_skip != 0);
        const bool curr_skip = (vec0_cur_skip != 0 && vec1_cur_skip != 0);
        const bool skip_cond = !curr_skip && prev_skip;
#else
        const bool skip_cond = false;
#endif

        const uint32_t buf_idx = static_cast<uint32_t>(tile_id % PV_CV_FIFO);
        const size_t base_elems =
            static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

        const size_t subblock_base_rows =
            static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
        __gm__ float *pv_out_ptr = pv_tile_fifo + base_elems + subblock_base_rows * HEAD_SIZE;
        GlobalDataPV_VEC pvGlobalVec(pv_out_ptr);

        pv2guSync.wait();

        if (tile_id > 0) {
            if (tile_id < num_tiles - 1) {
                pto_macro_fa_gu<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo, pvPendTile,
                                                         skip_cond);
            } else {
                pto_macro_fa_gu_last<ReduceTileF_T, TileOutT>(runningOTile, pvVecTile, l1_exp_max_ififo, l2_global_sum,
                                                              pvPendTile, skip_cond);
            }
        } else {
            if constexpr (CAUSAL_MASK) {
                if (tile_id == num_tiles - 1)
                    pto_macro_fa_gu_single_and_last_tile(runningOTile, l2_global_sum);
            }
        }
#if defined MARK_STAMP
        bisheng::cce::mark_stamp<PIPE_V>(GU_DONE * 1000 + tile_id);
#endif

        pv2guSync.free();

        if (tile_id == num_tiles - 1) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            using GlobalOutT =
                GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, HEAD_SIZE>, pto::Stride<1, 1, 1, HEAD_SIZE, 1>>;
            GlobalOutT outGlobal((__gm__ float *)(o_out + subblock_base_rows * HEAD_SIZE));
            TSTORE(outGlobal, runningOTile);
            set_intra_block(PIPE_MTE3, RUNNING_O_AVALIABLE);
#if defined MARK_STAMP_DATA_PIPE
            bisheng::cce::mark_stamp<PIPE_MTE3>(GU_DONE * 1000 + tile_id);
#endif
        }
    }
}

template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
__global__ AICORE void runTFA(__gm__ uint64_t *ffts_addr, __gm__ half *q, __gm__ half *k, __gm__ half *v,
                              __gm__ half *p_tile_fifo, __gm__ float *exp_max_ififo, __gm__ float *o_out,
                              __gm__ float *o_parts_out, __gm__ float *qk_tile_fifo, __gm__ float *pv_tile_fifo,
                              __gm__ float *pv_pend_tile_fifo, __gm__ uint8_t *cv_comm_buf, __gm__ uint8_t *profile_buf)
{
    set_ffts_base_addr((uint64_t)ffts_addr);

    // S0 (rows total), Cube_S0 (per-block rows), S1 (cols), HEAD_SIZE (inner)
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t logical_block_count = S0 / CUBE_S0;
    constexpr uint32_t launch_block_count = (logical_block_count < static_cast<uint32_t>(kFaLaunchCoreCount)) ?
                                                logical_block_count :
                                                static_cast<uint32_t>(kFaLaunchCoreCount);
    constexpr uint32_t Cube_S1 = CUBE_S1; // per-tile S1 chunk
    constexpr uint32_t Tile_S1 = TILE_S1; // logical tile along S1
    static_assert(Tile_S1 % Cube_S1 == 0, "TILE_S1 must be divisible by CUBE_S1");
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1; // sub-tiles per TILE_S1
    constexpr uint32_t Cube_HEAD = HEAD_SIZE;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    constexpr uint32_t VecGuRows = Cube_S0 / VEC_CORES;
    static_assert(Cube_S0 % (VEC_CORES * kTileFactor) == 0, "Vec rows must divide evenly across tile slices");

    if constexpr (DAV_VEC && VECTOR_PREFETCH_QKV_TILE) {
        constexpr uint32_t kPrefetchTileBytes = Cube_S0 * HEAD_SIZE * sizeof(half);
        using PrefetchScratchTile =
            Tile<TileType::Vec, uint8_t, 1, kPrefetchTileBytes, BLayout::RowMajor, 1, kPrefetchTileBytes>;
        using PrefetchGlobalBytes = GlobalTensor<uint8_t, pto::Shape<1, 1, 1, 1, kPrefetchTileBytes>,
                                                 pto::Stride<1, 1, 1, kPrefetchTileBytes, 1>>;

        PrefetchScratchTile startupScratch;
        TASSIGN(startupScratch, 0u);

        PrefetchGlobalBytes kPrefetch(reinterpret_cast<__gm__ uint8_t *>(k));
        PrefetchGlobalBytes kNextPrefetch(reinterpret_cast<__gm__ uint8_t *>(k + static_cast<size_t>(Tile_S1) * HEAD_SIZE));
        PrefetchGlobalBytes kNext2Prefetch(reinterpret_cast<__gm__ uint8_t *>(k + 2U * static_cast<size_t>(Tile_S1) * HEAD_SIZE));
        PrefetchGlobalBytes vPrefetch(reinterpret_cast<__gm__ uint8_t *>(v));
        prefetch_first_qkv_tiles<(S1 > Tile_S1), (S1 > 2U * Tile_S1)>(startupScratch, kPrefetch, kNextPrefetch,
                                                                       kNext2Prefetch, vPrefetch);
    }

    // ------------------------------------------------------------------------------
    // Tuning knobs (pipeline)
    //
    // qkPreloadNum controls how many (QK -> P) tiles we warm up before entering the steady-state loop.
    // - Larger preload improves overlap (Cube/VEC concurrency) for long S1.
    // - Larger preload increases FIFO footprint (qkGlobalTensorNBuffers / pvGlobalTensorNBuffers /
    // guGlobalTensorNBuffers).
    //
    // Buffer counts for optional double-buffering (default 1)
    // - srcVecTNBuffers/xexpVecTNBuffers: Vec ping-pong for QK load and x_exp output
    // - *MatTNBuffers: L1 ping-pong for Cube stage (K/P/V)
    // Keep these small (1-2) unless you have measured stall bubbles that require deeper buffering.
    // ------------------------------------------------------------------------------
    constexpr uint32_t qkPreloadNum = QK_PRELOAD;
    constexpr uint32_t srcVecTNBuffers = 2;
    constexpr uint32_t xexpVecTNBuffers = 2;
    constexpr uint32_t outOTileNBuffers = 2;
    constexpr uint32_t qMatTNBuffers = 1;
    constexpr uint32_t kMatTNBuffers = 2;
    constexpr uint32_t pMatTNBuffers = qkPreloadNum + 1;
    constexpr uint32_t vMatTNBuffers = 2;
    // These stay at CV_FIFO_SIZE so intermediate-check dumps and delayed GU reduce data keep the 8-slot layout.
    // QK/PV cross-core sync below uses the actual UB ring depths: srcVecTNBuffers and outOTileNBuffers.
    constexpr uint32_t qkp_tile_fifo_size = CV_FIFO_SIZE;
    constexpr uint32_t pv_tile_fifo_size = CV_FIFO_SIZE;
    static_assert(qkPreloadNum >= 1, "qkPreloadNum must be >= 1");
    static_assert(CV_FIFO_CONS_SYNC_PERIOD >= 1, "CV_FIFO_CONS_SYNC_PERIOD must be >= 1");
    static_assert((qkPreloadNum > 1) || (kTileFactor == 1), "qkPreloadNum must be > 1 unless kTileFactor == 1");

    static_assert(qkPreloadNum <= pMatTNBuffers,
                  "USE_UB_TO_L1_PATH requires qkPreloadNum <= pMatTNBuffers (2) to avoid buffer races. "
                  "Use --qk-preload 2 when running with UB mode enabled.");
    // Define tile types for first QK matmul
    using TileMatQData =
        Tile<TileType::Mat, half, HEAD_SIZE, Cube_S0, BLayout::RowMajor, HEAD_SIZE, Cube_S0, SLayout::ColMajor, 512>;
    using TileMatKData =
        Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1, HEAD_SIZE, SLayout::RowMajor, 512>;
    // Accumulator rows must match Cube_S0 (per-block rows), not logical S0
    using TileQKData = TileAcc<float, Cube_S1, Cube_S0, Cube_S1, Cube_S0>;

    TileMatQData qMatTile[qMatTNBuffers];
    TileMatKData kMatTile[kMatTNBuffers];
    TileQKData qkAccTile;

    // Define tile types for second PV matmul
    using TileMatPData =
        Tile<TileType::Mat, half, Cube_S0, Cube_S1, BLayout::RowMajor, Cube_S0, Cube_S1, SLayout::ColMajor, 512>;
    using TileMatVData =
        Tile<TileType::Mat, half, Cube_S1, HEAD_SIZE, BLayout::ColMajor, Cube_S1, HEAD_SIZE, SLayout::RowMajor, 512>;
    using TilePVData = TileAcc<float, Cube_S0, HEAD_SIZE, Cube_S0, HEAD_SIZE>;

    TileMatPData pMatTile[pMatTNBuffers];
    TileMatVData vMatTile[vMatTNBuffers];
    TilePVData pvAccPendTile;
    TilePVData pvAccCurrTile;
    TilePVData pvAccTailCurrTile;

    allocate_cube_tile_buffers(qMatTile, kMatTile, pMatTile, vMatTile);

    constexpr uint32_t l0cAlignBytes = 0x8000U;
    constexpr uint32_t qkAccBytes =
        ((static_cast<uint32_t>(Cube_S1 * Cube_S0 * sizeof(float)) + l0cAlignBytes - 1U) / l0cAlignBytes) *
        l0cAlignBytes;
    constexpr uint32_t pvAccBytes =
        ((static_cast<uint32_t>(Cube_S0 * HEAD_SIZE * sizeof(float)) + l0cAlignBytes - 1U) / l0cAlignBytes) *
        l0cAlignBytes;
    constexpr uint32_t sharedAccSlotBytes = (qkAccBytes > pvAccBytes) ? qkAccBytes : pvAccBytes;
    constexpr uint32_t qkAccSlotBytes = skip_rescale ? qkAccBytes : sharedAccSlotBytes;
    constexpr uint32_t l0cCapacityBytes = 0x40000U;
    static_assert(2U * sharedAccSlotBytes <= l0cCapacityBytes, "QK/PV shared L0C ping-pong slots exceed L0C capacity");

    if constexpr (skip_rescale) {
        constexpr uint32_t pvPackedBase = 2U * qkAccBytes;
        TASSIGN(pvAccPendTile, pvPackedBase);
        TASSIGN(pvAccCurrTile, pvPackedBase + pvAccBytes);
        TASSIGN(pvAccTailCurrTile, pvPackedBase + 2U * pvAccBytes);
    } else {
        assign_running_acc_tile(qkAccTile, sharedAccSlotBytes, 0);
        assign_running_acc_tile(pvAccCurrTile, sharedAccSlotBytes, 1);
    }

    // Define tile types for FA softmax P computation. UB offsets for softmax tiles
    // Define per-tile vector tiles sized to Cube_S1
    // DN layout version
    using TileDataF_T = Tile<TileType::Vec, float, Tile_S1, Vec_S0, BLayout::RowMajor, Tile_S1, Vec_S0>;
    using TileDataH_T = Tile<TileType::Vec, half, Tile_S1, Vec_S0, BLayout::RowMajor, Tile_S1, Vec_S0>;
    constexpr uint32_t SubblockRows = Cube_S0 / VEC_CORES;
    // Reduce tiles cover one vector core's rows (Cube_S0 / VEC_CORES); slices are extracted per row_slice
    using ReduceTileF_T = Tile<TileType::Vec, float, 1, SubblockRows, BLayout::RowMajor, 1, SubblockRows>;

    constexpr uint32_t NzBufRows = Cube_S1 + 1;
    using TileDataH_NZ_T = Tile<TileType::Vec, half, NzBufRows, Vec_S0, BLayout::ColMajor, Cube_S1, Vec_S0,
                                SLayout::RowMajor, 512, PadValue::Null, CompactMode::RowPlusOne>;

    TileDataF_T qkVecTile[srcVecTNBuffers];
    ReduceTileF_T m1_local_max;
    TileDataF_T input_reduce_tmp;
    ReduceTileF_T l1_local_sum;
    ReduceTileF_T m2_global_max;
    ReduceTileF_T l2_global_sum;
    ReduceTileF_T l1_exp_max_ififo[qkp_tile_fifo_size];
    TileDataH_T x_expT[xexpVecTNBuffers];
    TileDataH_NZ_T nzConvBuffer[xexpVecTNBuffers];

    using TileOutGuT = Tile<TileType::Vec, float, VecGuRows, HEAD_SIZE, BLayout::RowMajor, VecGuRows, HEAD_SIZE>;
    TileOutGuT pvVecTile[outOTileNBuffers];
    TileOutGuT runningOTile;
    allocate_vec_tile_buffers<TileDataF_T, ReduceTileF_T, TileDataH_T, TileOutGuT, TileDataH_NZ_T, srcVecTNBuffers,
                              xexpVecTNBuffers, outOTileNBuffers>(
        qkVecTile, m1_local_max, input_reduce_tmp, l1_local_sum, m2_global_max, l2_global_sum, l1_exp_max_ififo, x_expT,
        pvVecTile, runningOTile, nzConvBuffer);

    constexpr uint32_t nzBufSize = NzBufRows * Vec_S0 * sizeof(half);
    // softmax skip-rescale uses fixed UB scratch at 254 KiB (deltaMaxTile) and 255 KiB (ctrlTile).
    // Keep the NZ conversion buffers below that scratch area because TINSERT consumes them after
    // softmax decides the skip status.
    constexpr uint32_t softmaxScratchOffset = 254U * 1024U;
    static_assert(softmaxScratchOffset >= 2U * nzBufSize, "NZ conversion buffers overlap softmax scratch");
    // if constexpr (DAV_VEC) {
    //     TASSIGN(nzConvBuffer[0], softmaxScratchOffset - 2U * nzBufSize);
    //     TASSIGN(nzConvBuffer[1], softmaxScratchOffset - nzBufSize);
    // }

    constexpr size_t p_fifo_block_stride =
        static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Tile_S1);
    constexpr size_t p_max_fifo_block_stride = static_cast<size_t>(qkp_tile_fifo_size) * static_cast<size_t>(Cube_S0);
    constexpr size_t qk_fifo_block_stride = p_fifo_block_stride;
    constexpr size_t pv_fifo_block_stride =
        static_cast<size_t>(pv_tile_fifo_size) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(HEAD_SIZE);

    // QK uses L0C->UB (TMOV); Vec must wait PIPE_V, not PIPE_MTE2 (GM path).
    constexpr TSync_Custom<SyncOpType::TMOV_C2UB, SyncOpType::TLOAD> qk2smSync = {BUF0_QK_READY};
    constexpr TSync_Custom<SyncOpType::TINSERT_V2L1, SyncOpType::TLOAD> sm2pvSync = {BUF1_SM_READY};
    constexpr TSync_Custom<SyncOpType::TMOV_C2UB, SyncOpType::TLOAD> pv2guSync = {UPDATE_READY};
    constexpr bool use_cv_comm =
        (!INTERMEDIATE_CHECK) && (launch_block_count >= static_cast<uint32_t>(kFaLaunchCoreCount));
    int pvAccTileEvtID = EVENT_ID2;
    constexpr int pvAccTailTileEvtID = EVENT_ID1;
    const int physical_block_idx = block_idx;
    if constexpr (DAV_CUBE) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
    }
    if constexpr (DAV_VEC) {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        set_intra_block(PIPE_MTE3, RUNNING_O_AVALIABLE);
    }
    const int physical_comm_slot = use_cv_comm ? pto::TSYNC_CVID(physical_block_idx, cv_comm_buf) : physical_block_idx;

    int deferredQkUbDrains = 0;
    int deferredPvUbDrains = 0;
    for (int logical_block_idx = physical_block_idx; logical_block_idx < static_cast<int>(logical_block_count);
         logical_block_idx += static_cast<int>(launch_block_count)) {
        const uint64_t tStart = get_sys_cnt();

        assign_running_acc_tile(qkAccTile, qkAccSlotBytes, 0);

        if constexpr (DAV_VEC && skip_rescale) {
            const bool is_vec1 = static_cast<size_t>(get_subblockid());
            for (int i = 0; i < SKIP_STATUS_FIFO_SIZE; ++i) {
                write_skip_status_slot(i, is_vec1, 0);
            }
        }

        const int block_offset_rows = logical_block_idx * static_cast<int>(Cube_S0);
        const int comm_slot = use_cv_comm ? physical_comm_slot : logical_block_idx;

        __gm__ uint64_t *profile_entry = nullptr;
        if (profile_buf != nullptr) {
            std::size_t profile_block_base = static_cast<std::size_t>(logical_block_idx) * kFaProfileBytesPerBlock;
            std::size_t profile_offset = profile_block_base;
            if constexpr (DAV_VEC) {
                profile_offset +=
                    (static_cast<std::size_t>(get_subblockid()) + 1U) * 1024U; // vec subblock 0/1 use 2nd/3rd KB
            }
            profile_entry = reinterpret_cast<__gm__ uint64_t *>(profile_buf + profile_offset);
            profile_entry[0] = tStart;
        }

        __gm__ half *q_block = q + block_offset_rows * HEAD_SIZE;
        __gm__ half *p_tile_fifo_block = p_tile_fifo + static_cast<size_t>(comm_slot) * p_fifo_block_stride;
        __gm__ float *exp_max_ififo_block = exp_max_ififo + static_cast<size_t>(comm_slot) * p_max_fifo_block_stride;
        __gm__ float *o_out_block = o_out + static_cast<size_t>(block_offset_rows) * static_cast<size_t>(HEAD_SIZE);
        __gm__ float *o_parts_block =
            o_parts_out + static_cast<size_t>(block_offset_rows) * static_cast<size_t>(HEAD_SIZE);
        __gm__ float *qk_tile_fifo_block = qk_tile_fifo + static_cast<size_t>(comm_slot) * qk_fifo_block_stride;
        __gm__ float *pv_tile_fifo_block = pv_tile_fifo + static_cast<size_t>(comm_slot) * pv_fifo_block_stride;
        __gm__ float *pv_pend_tile_fifo_block =
            pv_pend_tile_fifo + static_cast<size_t>(comm_slot) * pv_fifo_block_stride;

        int num_tiles_s1 = S1 / Tile_S1;
        if constexpr (CAUSAL_MASK)
            num_tiles_s1 = (1 + ((logical_block_idx * CUBE_S0) / Tile_S1));

        int p_gu_src_pingpong_id = 0; // shared ping-pong for softmax vec tiles, pv output tiles, and GU input tiles
        int k_src_pingpong_id = 0;    // separate ping-pong for K tiles
        int pv_src_pingpong_id = 0;   // separate ping-pong for P V tiles

        int qkAccTileEvtID = 0;
        const bool has_next_logical_block =
            logical_block_idx + static_cast<int>(launch_block_count) < static_cast<int>(logical_block_count);
        const int qkDrainCarryCount = deferredQkUbDrains;
        const int pvDrainCarryCount = deferredPvUbDrains;
        deferredQkUbDrains = 0;
        deferredPvUbDrains = 0;

        // QK and P pre-computation (tile_id based)
        for (int preload_tile = 0; preload_tile < static_cast<int>(qkPreloadNum) && preload_tile < num_tiles_s1;
             ++preload_tile) {
            if constexpr (DAV_CUBE) {
                for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                    qkAccTileEvtID = assign_running_acc_tile(qkAccTile, qkAccSlotBytes);
                    const int tile_buf_idx = preload_tile % srcVecTNBuffers;
                    compute_qk<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size,
                               CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, srcVecTNBuffers>(
                        preload_tile, sub_tile, tile_buf_idx, q_block, k, qk_tile_fifo_block, qMatTile[0],
                        kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile, qkVecTile[tile_buf_idx],
                        k_src_pingpong_id % kMatTNBuffers, qkAccTileEvtID, qk2smSync, logical_block_idx,
                        qkDrainCarryCount);
                    k_src_pingpong_id++;
                }
            }
            if constexpr (DAV_VEC) {
                for (int row_slice = 0; row_slice < static_cast<int>(kTileFactor); ++row_slice) {
                    const int tile_buf_idx = preload_tile % srcVecTNBuffers;
                    compute_p<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size,
                              CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, pMatTNBuffers>(
                        preload_tile, row_slice, qk_tile_fifo_block, p_tile_fifo_block, exp_max_ififo_block,
                        qkVecTile[tile_buf_idx], x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers], input_reduce_tmp,
                        m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                        l1_exp_max_ififo[preload_tile % qkp_tile_fifo_size], pMatTile[preload_tile % pMatTNBuffers],
                        nzConvBuffer[p_gu_src_pingpong_id % xexpVecTNBuffers], p_gu_src_pingpong_id % xexpVecTNBuffers,
                        qk2smSync, sm2pvSync, logical_block_idx, num_tiles_s1);
                    p_gu_src_pingpong_id++;
                }
            }
        }

        const int steady_tile_end =
            (num_tiles_s1 > static_cast<int>(qkPreloadNum)) ? (num_tiles_s1 - static_cast<int>(qkPreloadNum)) : 0;
        for (int tile_id = 0; tile_id < steady_tile_end; ++tile_id) {
            const int next_qk_tile = tile_id + static_cast<int>(qkPreloadNum);
            qkAccTileEvtID = assign_running_acc_tile(qkAccTile, qkAccSlotBytes);
            if constexpr (!skip_rescale) {
                pvAccTileEvtID = assign_running_acc_tile(pvAccCurrTile, sharedAccSlotBytes);
            }
            for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                if constexpr (DAV_CUBE) {
                    const int tile_buf_idx = next_qk_tile % srcVecTNBuffers;
                    compute_qk<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size,
                               CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, srcVecTNBuffers>(
                        next_qk_tile, sub_tile, tile_buf_idx, q_block, k, qk_tile_fifo_block, qMatTile[0],
                        kMatTile[k_src_pingpong_id % kMatTNBuffers], qkAccTile, qkVecTile[tile_buf_idx],
                        k_src_pingpong_id % kMatTNBuffers, qkAccTileEvtID, qk2smSync, logical_block_idx,
                        qkDrainCarryCount);
                    k_src_pingpong_id++;
                }

                if constexpr (DAV_VEC) {
                    const int tile_buf_idx = next_qk_tile % srcVecTNBuffers;
                    compute_p<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size,
                              CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, pMatTNBuffers>(
                        next_qk_tile, sub_tile, qk_tile_fifo_block, p_tile_fifo_block, exp_max_ififo_block,
                        qkVecTile[tile_buf_idx], x_expT[p_gu_src_pingpong_id % xexpVecTNBuffers], input_reduce_tmp,
                        m1_local_max, l1_local_sum, m2_global_max, l2_global_sum,
                        l1_exp_max_ififo[next_qk_tile % qkp_tile_fifo_size], pMatTile[next_qk_tile % pMatTNBuffers],
                        nzConvBuffer[p_gu_src_pingpong_id % xexpVecTNBuffers], p_gu_src_pingpong_id % xexpVecTNBuffers,
                        qk2smSync, sm2pvSync, logical_block_idx, num_tiles_s1);
                    p_gu_src_pingpong_id++;
                }

                if constexpr (DAV_CUBE) {
                    TileOutGuT &pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers];
                    compute_pv<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, pv_tile_fifo_size,
                               CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, outOTileNBuffers>(
                        tile_id, sub_tile, tile_id % outOTileNBuffers, p_tile_fifo_block, v, pv_tile_fifo_block,
                        pv_pend_tile_fifo_block, pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                        vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccCurrTile, pvAccPendTile, runningOTile,
                        pvPendTile, pvVecTile, pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0, pvAccTileEvtID,
                        sm2pvSync, pv2guSync, logical_block_idx, pvDrainCarryCount);
                    pv_src_pingpong_id++;
                }
            }

            if constexpr (DAV_VEC) {
                TileOutGuT &pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers];
                compute_gu<S0, HEAD_SIZE, S1, CUBE_S0, Tile_S1, pv_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                           INTERMEDIATE_CHECK, CAUSAL_MASK, srcVecTNBuffers, outOTileNBuffers>(
                    tile_id, num_tiles_s1, pv_tile_fifo_block, pv_pend_tile_fifo_block, o_out_block, o_parts_block,
                    runningOTile, pvVecTile[tile_id % outOTileNBuffers], pvPendTile,
                    l1_exp_max_ififo[tile_id % qkp_tile_fifo_size], l2_global_sum, tile_id % outOTileNBuffers,
                    pv2guSync);
                p_gu_src_pingpong_id++;
            }
        }

        for (int tile_id = steady_tile_end; tile_id < num_tiles_s1; ++tile_id) {
            for (int sub_tile = 0; sub_tile < static_cast<int>(kTileFactor); ++sub_tile) {
                if constexpr (DAV_CUBE) {
                    TileOutGuT &pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers];
                    if constexpr (skip_rescale) {
                        const int tail_phase = tile_id - steady_tile_end;
                        const bool has_prior_curr_pv = (steady_tile_end > 0) || (tail_phase > 0);
                        TilePVData &pvAccActiveTile =
                            (has_prior_curr_pv && ((tail_phase & 1) == 0)) ? pvAccTailCurrTile : pvAccCurrTile;
                        const int pvAccActiveTileEvtID =
                            (has_prior_curr_pv && ((tail_phase & 1) == 0)) ? pvAccTailTileEvtID : pvAccTileEvtID;
                        compute_pv<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, pv_tile_fifo_size,
                                   CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, outOTileNBuffers>(
                            tile_id, sub_tile, tile_id % outOTileNBuffers, p_tile_fifo_block, v, pv_tile_fifo_block,
                            pv_pend_tile_fifo_block, pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                            vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccActiveTile, pvAccPendTile, runningOTile,
                            pvPendTile, pvVecTile, pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0,
                            pvAccActiveTileEvtID, sm2pvSync, pv2guSync, logical_block_idx, pvDrainCarryCount);
                    } else {
                        pvAccTileEvtID = assign_running_acc_tile(pvAccCurrTile, sharedAccSlotBytes);
                        compute_pv<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, Tile_S1, qkp_tile_fifo_size, pv_tile_fifo_size,
                                   CV_FIFO_CONS_SYNC_PERIOD, INTERMEDIATE_CHECK, CAUSAL_MASK, outOTileNBuffers>(
                            tile_id, sub_tile, tile_id % outOTileNBuffers, p_tile_fifo_block, v, pv_tile_fifo_block,
                            pv_pend_tile_fifo_block, pMatTile[pv_src_pingpong_id % pMatTNBuffers],
                            vMatTile[pv_src_pingpong_id % vMatTNBuffers], pvAccCurrTile, pvAccPendTile, runningOTile,
                            pvPendTile, pvVecTile, pv_src_pingpong_id % vMatTNBuffers + PV_EVENT_ID0, pvAccTileEvtID,
                            sm2pvSync, pv2guSync, logical_block_idx, pvDrainCarryCount);
                    }
                    pv_src_pingpong_id++;
                }
            }

            if constexpr (DAV_VEC) {
                TileOutGuT &pvPendTile = pvVecTile[(tile_id + 1) % outOTileNBuffers];
                compute_gu<S0, HEAD_SIZE, S1, CUBE_S0, Tile_S1, pv_tile_fifo_size, CV_FIFO_CONS_SYNC_PERIOD,
                           INTERMEDIATE_CHECK, CAUSAL_MASK, srcVecTNBuffers, outOTileNBuffers>(
                    tile_id, num_tiles_s1, pv_tile_fifo_block, pv_pend_tile_fifo_block, o_out_block, o_parts_block,
                    runningOTile, pvVecTile[tile_id % outOTileNBuffers], pvPendTile,
                    l1_exp_max_ififo[tile_id % qkp_tile_fifo_size], l2_global_sum, tile_id % outOTileNBuffers,
                    pv2guSync);
                p_gu_src_pingpong_id++;
            }
        }

        const int paid_qk_carry = (qkDrainCarryCount < num_tiles_s1) ? qkDrainCarryCount : num_tiles_s1;
        const int remaining_qk_carry = qkDrainCarryCount - paid_qk_carry;
        const int pending_qk_ub_consumed =
            remaining_qk_carry + pending_ring_events(num_tiles_s1, static_cast<int>(srcVecTNBuffers));
        const int paid_pv_carry = (pvDrainCarryCount < num_tiles_s1) ? pvDrainCarryCount : num_tiles_s1;
        const int remaining_pv_carry = pvDrainCarryCount - paid_pv_carry;
#if skip_rescale
        const int pending_pv_ub_consumed = remaining_pv_carry + pending_ring_events(num_tiles_s1, static_cast<int>(1));
#else
        const int pending_pv_ub_consumed =
            remaining_pv_carry + pending_ring_events(num_tiles_s1, static_cast<int>(outOTileNBuffers));
#endif
        const int pending_sv_consumed = pending_ring_events(num_tiles_s1, static_cast<int>(pMatTNBuffers));

        if constexpr (DAV_CUBE) {
            if (has_next_logical_block) {
                deferredQkUbDrains = pending_qk_ub_consumed;
            } else {
                for (int i = 0; i < pending_qk_ub_consumed; ++i)
                    qk2smSync.allocate();
            }
            if (has_next_logical_block) {
                deferredPvUbDrains = pending_pv_ub_consumed;
            } else {
                for (int i = 0; i < pending_pv_ub_consumed; ++i)
                    pv2guSync.allocate();
            }
        }

        if constexpr (DAV_VEC) {
            for (int i = 0; i < pending_sv_consumed; ++i) {
                sm2pvSync.allocate();
            }
#if skip_rescale
            const int pending_ss_consumed =
                pending_consumption_events(num_tiles_s1, SKIP_STATUS_FIFO_SIZE, kFaCvFifoConsSyncPeriod);
            for (int i = 0; i < pending_ss_consumed; ++i) {
                wait_intra_block(PIPE_S, FftsBufferFlag::SS_BUF_READY + 1);
            }
#endif
        }

        const uint64_t tEnd = get_sys_cnt();
        if (profile_entry != nullptr) {
            profile_entry[1] = tEnd;
        }
#ifdef _DEBUG
        if constexpr (DAV_CUBE) {
            cce::printf("Core %d Cube Block %d, Start @%d End @%d (%d us)\n", get_coreid(), logical_block_idx,
                        int(tStart), int(tEnd), int(tEnd - tStart) * 20 / 1000);
        } else {
            cce::printf("Core %d Vec Block %d, SubBlock %d, Start @%d End @%d (%d us)\n", get_coreid(),
                        logical_block_idx, int(get_subblockid()), int(tStart), int(tEnd),
                        int(tEnd - tStart) * 20 / 1000);
        }
#endif
    }

    if constexpr (DAV_CUBE) {
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID2);
        wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE);
        wait_intra_block(PIPE_FIX, RUNNING_O_AVALIABLE + 16);
    }
    if constexpr (DAV_VEC) {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
    }
    // pipe_barrier(PIPE_ALL);
}

// Empty kernel to warm up cores
__global__ AICORE __attribute__((aic)) void warmup_kernel()
{}

// Host wrapper
template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
               float *exp_max_ififo, float *o_out, float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo,
               float *pv_pend_tile_fifo, uint8_t *profile_data, aclrtStream stream, uint8_t *cv_comm_buf)
{
    static_assert(S0 % CUBE_S0 == 0, "S0 must be divisible by CUBE_S0");
    constexpr uint32_t logical_block_count = S0 / CUBE_S0;
    constexpr uint32_t launch_block_count = (logical_block_count < static_cast<uint32_t>(kFaLaunchCoreCount)) ?
                                                logical_block_count :
                                                static_cast<uint32_t>(kFaLaunchCoreCount);

    // Warm up all cores first, then prefetch q/k/v into L2
    warmup_kernel<<<32, nullptr, stream>>>();

    const uint64_t tensor_elems = static_cast<uint64_t>(S0) * static_cast<uint64_t>(HEAD_SIZE);
    const uint64_t tensor_bytes = tensor_elems * sizeof(half);
    constexpr bool kPrefetchUseSdma = true; // simulation cannot use sdma
    constexpr int kPrefetchAivCores = 64;   // only used when kPrefetchUseSdma is false

    if constexpr (kPrefetchUseSdma) {
        PTO_PREFETCH((__gm__ void *)q, tensor_bytes, stream);
        PTO_PREFETCH((__gm__ void *)k, tensor_bytes, stream);
        PTO_PREFETCH((__gm__ void *)v, tensor_bytes, stream);
    } else {
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void *)q, tensor_bytes, stream);
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void *)k, tensor_bytes, stream);
        PTO_PREFETCH<false, kPrefetchAivCores>((__gm__ void *)v, tensor_bytes, stream);
    }

    runTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CV_FIFO_SIZE, INTERMEDIATE_CHECK, CAUSAL_MASK,
           CV_FIFO_CONS_SYNC_PERIOD><<<launch_block_count, nullptr, stream>>>(
        (__gm__ uint64_t *)ffts, (half *)q, (half *)k, (half *)v, (half *)p_tile_fifo, exp_max_ififo, o_out,
        o_parts_out, qk_tile_fifo, pv_tile_fifo, pv_pend_tile_fifo, cv_comm_buf, profile_data);
}

// Backward-compatible overload without profiling buffer
template <int S0, int HEAD_SIZE, int S1, int CUBE_S0, int CUBE_S1, int TILE_S1, int QK_PRELOAD, int CV_FIFO_SIZE,
          bool INTERMEDIATE_CHECK, bool CAUSAL_MASK, int CV_FIFO_CONS_SYNC_PERIOD>
void LaunchTFA(uint16_t *ffts, aclFloat16 *q, aclFloat16 *k, aclFloat16 *v, aclFloat16 *p_tile_fifo,
               float *exp_max_ififo, float *o_out, float *o_parts_out, float *qk_tile_fifo, float *pv_tile_fifo,
               float *pv_pend_tile_fifo, aclrtStream stream, uint8_t *cv_comm_buf)
{
    LaunchTFA<S0, HEAD_SIZE, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CV_FIFO_SIZE, INTERMEDIATE_CHECK, CAUSAL_MASK,
              CV_FIFO_CONS_SYNC_PERIOD>(ffts, q, k, v, p_tile_fifo, exp_max_ififo, o_out, o_parts_out, qk_tile_fifo,
                                        pv_tile_fifo, pv_pend_tile_fifo, nullptr, stream, cv_comm_buf);
}

#include "generated_cases.h"

#define INSTANTIATE_TFA(S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, CAUSAL_MASK)                             \
    template void LaunchTFA<S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, CAUSAL_MASK,   \
                            kFaCvFifoConsSyncPeriod>(                                                                 \
        uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, aclFloat16 * p_out, float *p_out_fp32,       \
        float *o_out, float *o_parts_out, float *qk_out, float *pv_out, float *pv_pend_out, uint8_t *profile_data,    \
        aclrtStream stream, uint8_t *cv_comm_buf);                                                                    \
    template void LaunchTFA<S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, false, CAUSAL_MASK,   \
                            kFaCvFifoConsSyncPeriod>(uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, \
                                                     aclFloat16 * p_out, float *p_out_fp32, float *o_out,             \
                                                     float *o_parts_out, float *qk_out, float *pv_out,                \
                                                     float *pv_pend_out, aclrtStream stream, uint8_t *cv_comm_buf);   \
    template void LaunchTFA<S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, true, CAUSAL_MASK,    \
                            kFaCvFifoConsSyncPeriod>(                                                                 \
        uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, aclFloat16 * p_out, float *p_out_fp32,       \
        float *o_out, float *o_parts_out, float *qk_out, float *pv_out, float *pv_pend_out, uint8_t *profile_data,    \
        aclrtStream stream, uint8_t *cv_comm_buf);                                                                    \
    template void LaunchTFA<S0, HEAD, S1, CUBE_S0, CUBE_S1, TILE_S1, QK_PRELOAD, kFaCvFifoSize, true, CAUSAL_MASK,    \
                            kFaCvFifoConsSyncPeriod>(uint16_t * ffts, aclFloat16 * q, aclFloat16 * k, aclFloat16 * v, \
                                                     aclFloat16 * p_out, float *p_out_fp32, float *o_out,             \
                                                     float *o_parts_out, float *qk_out, float *pv_out,                \
                                                     float *pv_pend_out, aclrtStream stream, uint8_t *cv_comm_buf);

TFA_FOR_EACH_CASE(INSTANTIATE_TFA)

#undef INSTANTIATE_TFA
