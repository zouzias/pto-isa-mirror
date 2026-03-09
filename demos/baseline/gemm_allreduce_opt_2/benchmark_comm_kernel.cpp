/**
 * Benchmark Communication Kernel - Device-side code
 * 
 * This file contains device-side kernel code for benchmarking single comm core performance.
 */

#include "pto/comm/pto_comm_inst.hpp"
#include "pto/common/pto_tile.hpp"
#include "common.hpp"

using namespace pto;

// Use same tile size as main program
constexpr uint32_t G_BASE_M = 128;
constexpr uint32_t G_BASE_N = 256;

// ============================================================================
// Single Comm Core Benchmark Kernel
// Tests TPUT performance by transferring tiles to all ranks
// ============================================================================
AICORE inline void SingleCommCoreBenchmark(
    __gm__ float *src_buffer,
    __gm__ float *dst_buffer,
    int nranks,
    int num_tiles)
{
    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    
    using TileData = pto::Tile<pto::TileType::Vec, float, G_BASE_M, G_BASE_N,
                               pto::BLayout::RowMajor, -1, -1>;
    
    constexpr size_t tileUBBytes = ((G_BASE_M * G_BASE_N * sizeof(float) + 1023) / 1024) * 1024;
    
    // Ping-pong tiles for TPUT
    TileData pingTile(G_BASE_M, G_BASE_N);
    TileData pongTile(G_BASE_M, G_BASE_N);
    TASSIGN(pingTile, 0x0);
    TASSIGN(pongTile, tileUBBytes);
    
    ShapeDyn tileShape(1, 1, 1, G_BASE_M, G_BASE_N);
    StrideDyn tileStride(G_BASE_M * G_BASE_N, G_BASE_M * G_BASE_N, G_BASE_M * G_BASE_N, G_BASE_N, 1);
    
    // Transfer num_tiles tiles to all ranks
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint64_t tile_offset = (uint64_t)tile_idx * G_BASE_M * G_BASE_N;
        
        Global srcG(src_buffer + tile_offset, tileShape, tileStride);
        
        // TPUT to all ranks (including self)
        for (int r = 0; r < nranks; ++r) {
            __gm__ float *dst_ptr = ShmemPtr(dst_buffer, r) + tile_offset;
            Global dstG(dst_ptr, tileShape, tileStride);
            
            // Use AtomicAdd to accumulate (simulating AllReduce)
            pto::comm::TPUT<pto::AtomicType::AtomicAdd>(dstG, srcG, pingTile, pongTile);
        }
    }
    
    // Ensure all TPUT writes are complete
    ShmemDeviceQuiet();
    ShmemDeviceBarrierAll();
    pipe_barrier(PIPE_ALL);
}

__global__ AICORE void SingleCommCoreBenchmarkKernel(
    __gm__ uint8_t *src_buffer,
    __gm__ uint8_t *dst_buffer,
    int nranks,
    int num_tiles)
{
    SingleCommCoreBenchmark(
        reinterpret_cast<__gm__ float *>(src_buffer),
        reinterpret_cast<__gm__ float *>(dst_buffer),
        nranks,
        num_tiles);
}

// ============================================================================
// Host-side launch function
// This function is only compiled for host-side code (not device-side)
// ============================================================================
#ifndef __CCE_AICORE__
#include "pto/pto-inst.hpp"

void launchSingleCommCoreBenchmark(
    uint8_t *src_buffer,
    uint8_t *dst_buffer,
    int nranks,
    int num_tiles,
    void *stream)
{
    // Use <<<>>> syntax provided by pto-inst.hpp
    SingleCommCoreBenchmarkKernel<<<1, nullptr, stream>>>(
        src_buffer, dst_buffer, nranks, num_tiles);
}
#endif

