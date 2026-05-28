#ifndef PTO_CPU_NZ_UTILS_HPP
#define PTO_CPU_NZ_UTILS_HPP

#include <cstddef>

namespace pto {

constexpr size_t NZ_INNER_ROWS = 16;
constexpr size_t NZ_INNER_COLS = 8;

PTO_INLINE size_t GetNZGlobalOffset(size_t r, size_t c, size_t gShape4)
{
    size_t blockRow = r / NZ_INNER_ROWS;
    size_t innerRow = r % NZ_INNER_ROWS;
    
    size_t blockCol = c / NZ_INNER_COLS;
    size_t innerCol = c % NZ_INNER_COLS;
    
    size_t numBlocksCol = (gShape4 + NZ_INNER_COLS - 1) / NZ_INNER_COLS;
    size_t blockOffset = (blockRow * numBlocksCol + blockCol) * NZ_INNER_ROWS * NZ_INNER_COLS;
    size_t innerOffset = innerRow * NZ_INNER_COLS + innerCol;
    
    return blockOffset + innerOffset;
}

template <typename TileData, typename Func>
PTO_INLINE void ForEachNZElement(
    int totalRows,
    int totalCols,
    Func &&func)
{
    for (size_t r = 0; r < static_cast<std::size_t>(totalRows); r++) {
        size_t subTileR = r / TileData::InnerRows;
        size_t innerR = r % TileData::InnerRows;
        
        for (size_t c = 0; c < static_cast<std::size_t>(totalCols); c++) {
            size_t subTileC = c / TileData::InnerCols;
            size_t innerC = c % TileData::InnerCols;
            
            size_t tile_idx = GetTileElementOffsetSubfractals<TileData>(
                subTileR, innerR, subTileC, innerC);
            
            size_t gd_idx = GetNZGlobalOffset(r, c, totalCols);
            
            func(r, c, tile_idx, gd_idx);
        }
    }
}

} // namespace pto
#endif