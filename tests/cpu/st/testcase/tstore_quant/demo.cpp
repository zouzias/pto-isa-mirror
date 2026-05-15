/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
... (License text omitted for brevity) ...
*/

#include <pto/pto-inst.hpp>
#include "test_common.h"
#include <pto/common/constants.hpp>
#include <iostream>

using namespace std;
using namespace pto;

// Helper to determine the quantization mode bitmask for the hardware register
template<bool is_v_quant>
constexpr QuantMode_t GetQuantMode() {
    return is_v_quant ? QuantMode_t::VectorQuant : QuantMode_t::ScalarQuant;
}

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant, bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreRowMajor(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src, __gm__ uint64_t __in__ *fbQuant)
{
    using ST = std::conditional_t<std::is_same_v<SrcT, aclFloat16>, half, SrcT>;
    using DT = std::conditional_t<std::is_same_v<Dst, aclFloat16>, half, Dst>;

    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4, 1};
    
    constexpr int blockSize = 32 / sizeof(ST);
    constexpr int validRow = gShape0 * gShape1 * gShape2 * gShape3;
    constexpr int validCol = gShape4;
    constexpr int Rows = validRow;
    constexpr int Cols = (validCol + blockSize - 1) / blockSize * blockSize;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<DT, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, ST, Rows, Cols, BLayout::RowMajor, -1, -1>;

    TileData srcTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);

    GlobalTensor<ST, DynShapeDim5, DynStridDim5> srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    constexpr QuantMode_t qMode = GetQuantMode<is_v_quant>();
    constexpr ReluPreMode rMode = apply_relu ? ReluPreMode::Relu : ReluPreMode::NoRelu;

    if constexpr (is_v_quant) {
        // For Vector Quant, the quant tile width matches validCol
        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Cols, BLayout::RowMajor>;
        QuantTile qTile(1, validCol);
        
        // Load the quant vector from GM to a local tile
        __gm__ uint64_t* qPtr = fbQuant;
        TLOAD(qTile, qPtr); 
        
        TStoreAccFp<GlobalData, TileData, QuantTile, qMode, rMode>(
            dstGlobal.data(), srcTile, qTile, 
            gShape0, gShape1, gShape2, gShape3, gShape4, 
            gStride[0], gStride[1], gStride[2], gStride[3], gStride[4], 
            validRow, validCol
        );
    } else {
        // Scalar Quant: pass the single parameter value
        uint64_t scalarQuant = fbQuant[0];
        TSTORE<GlobalData, TileData, qMode, rMode>(
            dstGlobal, srcTile, scalarQuant, gShape3, gShape4, gStride[2], gStride[3], validRow, validCol
        );
    }
}

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant, bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreColMajor(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src, __gm__ uint64_t __in__ *fbQuant)
{
    using ST = std::conditional_t<std::is_same_v<SrcT, aclFloat16>, half, SrcT>;
    using DT = std::conditional_t<std::is_same_v<Dst, aclFloat16>, half, Dst>;

    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, 1, gWholeShape3};

    constexpr int blockSize = 32 / sizeof(ST);
    constexpr int Rows = (gShape3 + blockSize - 1) / blockSize * blockSize;
    constexpr int Cols = gShape0 * gShape1 * gShape2 * gShape4;
    constexpr int validRow = gShape3;
    constexpr int validCol = Cols;

    using DynShapeDim5 = Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<DT, DynShapeDim5, DynStridDim5, Layout::DN>;
    using TileData = Tile<TileType::Vec, ST, Rows, Cols, BLayout::ColMajor, -1, -1>;

    TileData srcTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);

    GlobalTensor<ST, DynShapeDim5, DynStridDim5, Layout::DN> srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    constexpr QuantMode_t qMode = GetQuantMode<is_v_quant>();
    constexpr ReluPreMode rMode = apply_relu ? ReluPreMode::Relu : ReluPreMode::NoRelu;

    if constexpr (is_v_quant) {
        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Cols, BLayout::RowMajor>;
        QuantTile qTile(1, validCol);
        TLOAD(qTile, fbQuant);

        TStoreAccFp<GlobalData, TileData, QuantTile, qMode, rMode>(
            dstGlobal.data(), srcTile, qTile, 
            gShape0, gShape1, gShape2, gShape3, gShape4, 
            gStride[0], gStride[1], gStride[2], gStride[3], gStride[4], 
            validRow, validCol
        );
    } else {
        uint64_t scalarQuant = fbQuant[0];
        TSTORE<GlobalData, TileData, qMode, rMode>(
            dstGlobal, srcTile, scalarQuant, gShape0, gShape1, gShape2, gShape3, gShape4, 
            gStride[0], gStride[3], validRow, validCol
        );
    }
}

template <typename Dst, typename SrcT, int gShape0, int gShape1, int gShape2, int gShape3, int gShape4, int gWholeShape0,
          int gWholeShape1, int gWholeShape2, int gWholeShape3, int gWholeShape4, bool is_v_quant, bool saturate_inf, bool apply_relu>
AICORE inline void RunTStoreNZ(__gm__ Dst __out__ *out, __gm__ SrcT __in__ *src, __gm__ uint64_t __in__ *fbQuant)
{
    using ST = std::conditional_t<std::is_same_v<SrcT, aclFloat16>, half, SrcT>;
    using DT = std::conditional_t<std::is_same_v<Dst, aclFloat16>, half, Dst>;

    constexpr int gStride[5] = {gWholeShape1 * gWholeShape2 * gWholeShape3 * gWholeShape4,
                                gWholeShape2 * gWholeShape3 * gWholeShape4, gWholeShape3 * gWholeShape4, gWholeShape4, 1};

    constexpr int Rows = gShape2 * gShape3;
    constexpr int Cols = gShape0 * gShape1 * gShape4;

    using DynShapeDim5 = pto::Shape<gShape0, gShape1, gShape2, gShape3, gShape4>;
    using DynStridDim5 = pto::Stride<gStride[0], gStride[1], gStride[2], gStride[3], gStride[4]>;
    using GlobalData = GlobalTensor<DT, DynShapeDim5, DynStridDim5, Layout::NZ>;
    using TileData = Tile<TileType::Vec, ST, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512>;

    int validRow = Rows;
    int validCol = Cols;
    TileData srcTile(validRow, validCol);
    TASSIGN(srcTile, 0x0);

    GlobalTensor<ST, DynShapeDim5, DynStridDim5, Layout::NZ> srcGlobal(src);
    GlobalData dstGlobal(out);
    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    constexpr QuantMode_t qMode = GetQuantMode<is_v_quant>();
    constexpr ReluPreMode rMode = apply_relu ? ReluPreMode::Relu : ReluPreMode::NoRelu;

    if constexpr (is_v_quant) {
        using QuantTile = Tile<TileType::Vec, uint64_t, 1, Cols, BLayout::RowMajor>;
        QuantTile qTile(1, validCol);
        TLOAD(qTile, fbQuant);

        TStoreAccFp<GlobalData, TileData, QuantTile, qMode, rMode>(
            dstGlobal.data(), srcTile, qTile, 
            gShape0, gShape1, gShape2, gShape3, gShape4, 
            gStride[0], 0, 0, 0, 0, // Simplified strides for NZ
            validRow, validCol
        );
    } else {
        uint64_t scalarQuant = fbQuant[0];
        TSTORE<GlobalData, TileData, qMode, rMode>(
            dstGlobal, srcTile, scalarQuant, 
            gShape0, gShape1, gShape2, gShape3, gShape4, 
            gStride[0], validRow, validCol
        );
    }
}