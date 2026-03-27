/**
 * Ring AllGather + Streaming GEMM shared metadata.
 * Only TileFlagMatrix is used: tile-based flags for streaming pipeline
 * (no FlagMatrix / WorkQueue / Dispatcher).
 */

#pragma once

#include <cstdint>

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
#include "pto/comm/pto_comm_inst.hpp"
#endif

constexpr int MAX_RING_RANKS = 16;

// ============================================================================
// Streaming Pipeline Configuration
// ============================================================================
#ifndef STREAMING_TILE_SIZE
#define STREAMING_TILE_SIZE 4  // Number of blocks per tile for streaming
#endif
constexpr int TILE_SIZE = STREAMING_TILE_SIZE;

// ============================================================================
// TileFlagMatrix: Tile-based flags for streaming pipeline
//
// Design for fine-grained communication-computation overlap:
//   - Each tile contains TILE_SIZE blocks
//   - Communication sets tile flag immediately after tile transfer completes
//   - Computation waits for tile flag before processing tile
//   - Non-competing design: each (src_rank, tile_idx) has unique writer/reader
//
// Layout: [num_ranks][num_tiles_per_src] with 64-byte alignment per row
// ============================================================================
struct alignas(64) TileFlagMatrix {
    int32_t num_ranks;
    int32_t num_tiles_per_src;
    int32_t num_blocks_per_src;
    int32_t tile_size;
    int32_t stride;              // Aligned stride for each rank's tile flags
    int32_t my_rank;             // Local rank id, set at init for compute kernel optimization
    int32_t padding[10];         // Pad header to 64 bytes
    // Followed by int32_t tile_flags[num_ranks * stride]
};

inline size_t TileFlagMatrixSize(int num_ranks, int num_blocks_per_src, int tile_size)
{
    int num_tiles = (num_blocks_per_src + tile_size - 1) / tile_size;
    // Align stride to cache line (16 int32_t = 64 bytes)
    int stride = ((num_tiles + 15) / 16) * 16;
    return sizeof(TileFlagMatrix) + static_cast<size_t>(num_ranks) * stride * sizeof(int32_t);
}

// 每个 src 一个「门铃」ptr，紧接在 TileFlagMatrix 之后；AIC 先轮询 summary，>=1 再查该 src 的 flag 区
inline size_t TileFlagMatrixSummaryOffset(int num_ranks, int num_blocks_per_src, int tile_size)
{
    return TileFlagMatrixSize(num_ranks, num_blocks_per_src, tile_size);
}
inline size_t TileFlagMatrixWithSummarySize(int num_ranks, int num_blocks_per_src, int tile_size)
{
    return TileFlagMatrixSummaryOffset(num_ranks, num_blocks_per_src, tile_size)
           + static_cast<size_t>(num_ranks) * sizeof(int32_t);
}

inline void TileFlagMatrixInit(TileFlagMatrix* flags, int num_ranks, int num_blocks_per_src, int tile_size)
{
    int num_tiles = (num_blocks_per_src + tile_size - 1) / tile_size;
    int stride = ((num_tiles + 15) / 16) * 16;
    
    flags->num_ranks = num_ranks;
    flags->num_tiles_per_src = num_tiles;
    flags->num_blocks_per_src = num_blocks_per_src;
    flags->tile_size = tile_size;
    flags->stride = stride;
    flags->my_rank = -1;  // Will be set by host before kernel launch
    for (int i = 0; i < 10; i++) flags->padding[i] = 0;
    
    int32_t* base = reinterpret_cast<int32_t*>(
        reinterpret_cast<uint8_t*>(flags) + sizeof(TileFlagMatrix));
    for (int i = 0; i < num_ranks * stride; ++i) {
        base[i] = 0;
    }
}

// 初始化/清零 summary 区（caller 传入的 buffer 需至少 TileFlagMatrixWithSummarySize）
inline void TileFlagMatrixSummaryInit(int32_t* summary_base, int num_ranks)
{
    for (int i = 0; i < num_ranks; ++i) summary_base[i] = 0;
}

inline void TileFlagMatrixReset(TileFlagMatrix* flags)
{
    int32_t* base = reinterpret_cast<int32_t*>(
        reinterpret_cast<uint8_t*>(flags) + sizeof(TileFlagMatrix));
    for (int i = 0; i < flags->num_ranks * flags->stride; ++i) {
        base[i] = 0;
    }
}

inline void TileFlagMatrixSetLocalReady(TileFlagMatrix* flags, int my_rank)
{
    int32_t* base = reinterpret_cast<int32_t*>(
        reinterpret_cast<uint8_t*>(flags) + sizeof(TileFlagMatrix));
    int offset = my_rank * flags->stride;
    for (int c = 0; c < flags->num_tiles_per_src; ++c) {
        base[offset + c] = 1;
    }
}

#if !defined(__CCE_KT_TEST__) && defined(__CCE_AICORE__)
// ============================================================================
// dcci compatibility layer
//
// dcci (data cache cache invalidate) is a dav-c220 L2 cache inline operation.
// A5 (dav-c310) with REGISTER_BASE mode has hardware-managed cache coherence
// for shared memory, making explicit dcci unnecessary. Under REGISTER_BASE,
// TNOTIFY AtomicAdd already provides hardware coherence for remote writes.
// If dcci is not available, compile calls as no-ops.
// ============================================================================

#ifndef SINGLE_CACHE_LINE
#define SINGLE_CACHE_LINE 0
#endif

// A5 (REGISTER_BASE): dcci may not be available; use memory barrier only.
// A2/A3 (MEMORY_BASE): dcci is required for L2 cache invalidation of shared memory.
#if defined(REGISTER_BASE) && !defined(DCCI_AVAILABLE)
#define DCCI_NOOP 1
#else
#define DCCI_NOOP 0
#endif

AICORE inline void dcci_compat(__gm__ void* addr, int lines)
{
#if DCCI_NOOP
    // REGISTER_BASE: hardware guarantees coherence; barrier is sufficient.
    __asm__ __volatile__("" ::: "memory");
#else
    // MEMORY_BASE: explicit L2 cache invalidation required.
    dcci(addr, lines);
    __asm__ __volatile__("" ::: "memory");
#endif
}

// TileFlagMatrix device-side functions
// ============================================================================

// Device: 从 TileFlagMatrix 头计算 flag 区字节数，用于得到 summary 基址
AICORE inline size_t TileFlagMatrixBytes(volatile __gm__ TileFlagMatrix* flags)
{
    return sizeof(TileFlagMatrix)
           + static_cast<size_t>(flags->num_ranks) * static_cast<size_t>(flags->stride) * sizeof(int32_t);
}

AICORE inline volatile __gm__ int32_t* GetTileFlagPtr(
    volatile __gm__ TileFlagMatrix* flags,
    int32_t src_rank,
    int32_t tile_idx)
{
    int32_t stride = flags->stride;
    int32_t idx = src_rank * stride + tile_idx;
    volatile __gm__ int32_t* base = reinterpret_cast<volatile __gm__ int32_t*>(
        reinterpret_cast<volatile __gm__ uint8_t*>(flags) + sizeof(TileFlagMatrix));
    return base + idx;
}

// Summary 基址：紧接在 flag 矩阵之后，num_ranks 个 int32，每个 src 一个「门铃」
AICORE inline volatile __gm__ int32_t* GetSummaryBase(volatile __gm__ TileFlagMatrix* flags)
{
    return reinterpret_cast<volatile __gm__ int32_t*>(
        reinterpret_cast<volatile __gm__ uint8_t*>(flags) + TileFlagMatrixBytes(flags));
}

// Set local tile flag ready (for local rank's data)
// Uses TNOTIFY AtomicAdd for hardware atomic semantics.
AICORE inline void SetTileFlagReady(
    volatile __gm__ TileFlagMatrix* flags,
    int32_t src_rank,
    int32_t tile_idx)
{
    volatile __gm__ int32_t* ptr = GetTileFlagPtr(flags, src_rank, tile_idx);
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t*>(const_cast<__gm__ int32_t*>(ptr)));
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
}

// Set remote tile flag ready (notify remote rank that tile data is available)
// 若传入 remote_summary_src_ptr（目标 rank 的 summary[src_rank]），则同时对该 ptr 做 TNOTIFY +1，供 AIC 先轮询 summary。
AICORE inline void SetRemoteTileFlagReady(
    __gm__ TileFlagMatrix* remote_flags,
    int32_t src_rank,
    int32_t tile_idx,
    __gm__ int32_t* remote_summary_src_ptr = nullptr)
{
    volatile __gm__ TileFlagMatrix* r = reinterpret_cast<volatile __gm__ TileFlagMatrix*>(remote_flags);
    if (src_rank < 0 || src_rank >= r->num_ranks ||
        tile_idx < 0 || tile_idx >= r->num_tiles_per_src) {
        return;  // 越界不写，防止并行时错位
    }
    volatile __gm__ int32_t* ptr = GetTileFlagPtr(r, src_rank, tile_idx);
    pto::comm::Signal sig(reinterpret_cast<__gm__ int32_t*>(const_cast<__gm__ int32_t*>(ptr)));
    pto::comm::TNOTIFY(sig, 1, pto::comm::NotifyOp::AtomicAdd);
    if (remote_summary_src_ptr != nullptr) {
        pto::comm::Signal sumSig(remote_summary_src_ptr);
        pto::comm::TNOTIFY(sumSig, 1, pto::comm::NotifyOp::AtomicAdd);
    }
}

// 本地 summary[src_rank] = value（Block 0 置完 local tile 后调用，供 AIC 轮询到「本 rank 有就绪」）
AICORE inline void SetLocalSummaryReady(volatile __gm__ int32_t* summary_base, int32_t src_rank, int32_t value)
{
    if (summary_base == nullptr || src_rank < 0) return;
    volatile __gm__ int32_t* ptr = summary_base + src_rank;
    dcci_compat((__gm__ void*)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    *ptr = value;
    dcci_compat((__gm__ void*)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
}

// Check if tile is ready (non-blocking)
AICORE inline bool IsTileReady(
    volatile __gm__ TileFlagMatrix* flags,
    int32_t src_rank,
    int32_t tile_idx)
{
    volatile __gm__ int32_t* ptr = GetTileFlagPtr(flags, src_rank, tile_idx);
    dcci_compat((__gm__ void*)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return (*ptr >= 1);
}

// 第一级轮询：该 src 是否有至少 1 个 tile 就绪（读 summary[src_rank]）
AICORE inline bool IsAnyReadyFromSrc(volatile __gm__ int32_t* summary_base, int32_t src_rank)
{
    if (summary_base == nullptr || src_rank < 0) return false;
    volatile __gm__ int32_t* ptr = summary_base + src_rank;
    dcci_compat((__gm__ void*)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return (*ptr >= 1);
}

// 读取 summary[src_rank] 的当前值（带 cache 刷新）
AICORE inline int32_t GetReadyCountFromSrc(volatile __gm__ int32_t* summary_base, int32_t src_rank)
{
    if (summary_base == nullptr || src_rank < 0) return 0;
    volatile __gm__ int32_t* ptr = summary_base + src_rank;
    dcci_compat((__gm__ void*)ptr, SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    return *ptr;
}

// 阻塞等待：直到 summary[src_rank] >= expected
AICORE inline void WaitReadyCountFromSrc(volatile __gm__ int32_t* summary_base, int32_t src_rank, int32_t expected)
{
    if (summary_base == nullptr || src_rank < 0 || expected <= 0) return;
    __gm__ int32_t* ptr = const_cast<__gm__ int32_t*>(summary_base + src_rank);
    pto::comm::Signal sig(ptr);
    pto::comm::TWAIT(sig, expected, pto::comm::WaitCmp::GE);
}

#endif
