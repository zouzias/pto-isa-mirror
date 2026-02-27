#ifndef TPOP_HPP
#define TPOP_HPP

#include <pto/common/fifo.hpp>
#include <pto/npu/a5/TStore.hpp>
#include <pto/npu/a5/TPush.hpp>

namespace pto {
/**
 * TPOP: Pop Tile from FIFO
 * * Flow:
 * 1. [Wait]    Wait for data ready (Cross-Core)
 * 2. [Load]    Load data from GM
 * 3. [Free]    Release GM space (Cross-Core)
 */
template <typename PipeCons, typename TileDataSrc, typename DataFiFo>
PTO_INTERNAL void TPOP_IMPL(PipeCons &cons, TileDataSrc &tile, DataFiFo &fifo)
{
    // // 1. Cross-Core: Wait for Data
    bool isWait = cons.getWaitStatus();
    if (isWait) {
        cons.wait();
    }

    // 2. Address Calculation & Pop
    cons.pop(fifo, tile);

    // 3. Cross-Core: Free Space
    bool isFree = cons.getFreeStatus();
    if (isFree) {
        cons.free();
    }
}

template <typename PipeCons>
PTO_INTERNAL void TPOPRELEASE_IMPL(PipeCons &cons)
{
    bool isFree = cons.getFreeStatus();
    if (isFree) {
        cons.free();
    }
}

} // namespace pto

#endif