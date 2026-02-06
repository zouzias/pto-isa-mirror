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
template <typename PipeCons, typename TileData, typename DataFIFO, typename DType>
PTO_INTERNAL void TPOP_IMPL(PipeCons &cons, TileData &tile, DataFIFO &fifo)
{
    // 1. Cross-Core: Wait for Data
    cons.wait();

    // 2. Address Calculation
    int buffer_idx = cons.iter % DataFIFO::fifoDepth;
    using T = typename TileData::DType;
    __gm__ T *addr = (__gm__ T *)fifo.fifoBase + buffer_idx * TileData::Numel;

    using GShape = TileShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GStride = BaseShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
    GT global_tensor(addr);

    // 3. Intra-Core Sync & Load
    if constexpr (TileData::Loc == TileType::Vec || DataFIFO::fifoType == FIFOType::GM_FIFO) {
        // pop from GM FIFO
        TLOAD(tile, global_tensor);
    } else if constexpr (TileData::Loc == TIleType::Vec && DataFIFO::fifoType == FIFOType::GM_FIFO) {
        // pop from GM FIFO
        TLOAD(tile, global_tensor);
    } else if constexpr (TileData::Loc == TileType::Vec && DataFIFO::fifoType == FIFOType::VEC_FIFO) {
        // pop from VecTile FIFO
    } else if constexpr (TileData::Loc == TileType::Acc && DataFIFO::fifoTYpe == FIFOType::Mat_FIFO) {
        // pop from MatTile FIFO
    } else {
        // not supported
    }

    // 4. Cross-Core: Free Space
    cons.free();
}
} // namespace pto

#endif