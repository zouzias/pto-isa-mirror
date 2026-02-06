#ifndef TPOP_HPP
#define TPOP_HPP

#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/fifo.hpp>

namespace pto {
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
template <typename PipeCons, typename TileData, typename DataFifo, typename DType>
PTO_INTERNAL void TPOP_IMPL(PipeCons &cons, TileData &tile, DataFifo &fifo) {
    // 1. Cross-Core: Wait for Data
    cons.wait();

    // 2. Address Calculation
    int buffer_idx = cons.iter % DataFifo::fifoDepth;
    using T = typename TileData::DType;
    __gm__ T *addr = (__gm__ T *)fifo.fifoBase + buffer_idx * TileData::Numel;

    using GShape = TileShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GStride = BaseShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    GT global_tensor(addr);

    // 3. Intra-Core Sync & Load
    if constexpr (TileData::Loc == TileType::Vec) {
        // === Case: VECTOR Input (GM -> UB) ===
        // Data Transfer
        TLOAD(tile, global_tensor);
    } else {
        // === Case: CUBE Input (GM -> L1) ===
        // Data Transfer
        TLOAD(tile, global_tensor);
    }

    // 4. Cross-Core: Free Space
    cons.free();
}
} // namespace pto

#endif