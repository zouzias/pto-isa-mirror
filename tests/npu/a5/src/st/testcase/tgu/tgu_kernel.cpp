#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"
#include "TGU.h"

using namespace pto;

template <typename T, int Rows, int Cols>
__global__ AICORE void runTgu(__gm__ T *__out, __gm__ T *__prev, __gm__ T *__est, __gm__ T *__exp_max,
                              __gm__ T *__pv_pend)
{
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
    using ReduceGlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;
    // using ReduceGlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, 1>, pto::Stride<1, 1, 1, 1, 1>>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using ReduceTile = Tile<TileType::Vec, T, Rows, 1, BLayout::ColMajor, -1, -1>;
    // using ReduceTile = Tile<TileType::Vec, T, Rows, 1, BLayout::RowMajor, -1, -1>;

    TileData prevTile(Rows, Cols);
    TileData estTile(Rows, Cols);
    TileData pvTile(Rows, Cols);
    ReduceTile expMaxTile(Rows, 1);

    GlobalData prevGlobal(__prev);
    GlobalData estGlobal(__est);
    GlobalData pvGlobal(__pv_pend);
    ReduceGlobalData expMaxGlobal(__exp_max);
    GlobalData outGlobal(__out);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(Rows) * sizeof(float);
    constexpr std::size_t off0b = 0;
    constexpr std::size_t off1b = off0b + float_tile_bytes;
    constexpr std::size_t off2b = off1b + float_tile_bytes;
    constexpr std::size_t off3b = off2b + float_tile_bytes;
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    constexpr std::size_t last_tile_end = off3b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);

    TASSIGN(prevTile, off0);
    TASSIGN(estTile, off1);
    TASSIGN(pvTile, off2);
    TASSIGN(expMaxTile, off3);

    TLOAD(prevTile, prevGlobal);
    TLOAD(estTile, estGlobal);
    TLOAD(pvTile, pvGlobal);
    TLOAD(expMaxTile, expMaxGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto_macro_fa_gu<ReduceTile, TileData>(prevTile, estTile, expMaxTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, prevTile);
    __out = outGlobal.data();
}

template <typename T, int Rows, int Cols>
__global__ AICORE void runTguLast(__gm__ T *__out, __gm__ T *__prev, __gm__ T *__est, __gm__ T *__exp_max,
                                  __gm__ T *__global_sum, __gm__ T *__pv_pend)
{
    using GlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, Cols>, pto::Stride<1, 1, 1, Cols, 1>>;
    using ReduceGlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, 1>, pto::Stride<1, 1, 1, 1, 1>, Layout::DN>;
    // using ReduceGlobalData = GlobalTensor<T, Shape<1, 1, 1, Rows, 1>, pto::Stride<1, 1, 1, 1, 1>>;
    using TileData = Tile<TileType::Vec, T, Rows, Cols, BLayout::RowMajor, -1, -1>;
    using ReduceTile = Tile<TileType::Vec, T, Rows, 1, BLayout::ColMajor, -1, -1>;
    // using ReduceTile = Tile<TileType::Vec, T, Rows, 1, BLayout::RowMajor, -1, -1>;

    TileData prevTile(Rows, Cols);
    TileData estTile(Rows, Cols);
    TileData pvTile(Rows, Cols);
    ReduceTile expMaxTile(Rows, 1);
    ReduceTile globalSumTile(Rows, 1);

    GlobalData prevGlobal(__prev);
    GlobalData estGlobal(__est);
    GlobalData pvGlobal(__pv_pend);
    GlobalData outGlobal(__out);
    ReduceGlobalData expMaxGlobal(__exp_max);
    ReduceGlobalData globalSumGlobal(__global_sum);

    constexpr std::size_t float_tile_bytes = static_cast<std::size_t>(Rows) * static_cast<std::size_t>(Cols) * sizeof(float);
    constexpr std::size_t reduce_tile_bytes = static_cast<std::size_t>(Rows) * sizeof(float);
    constexpr std::size_t off0b = 0;
    constexpr std::size_t off1b = off0b + float_tile_bytes;
    constexpr std::size_t off2b = off1b + float_tile_bytes;
    constexpr std::size_t off3b = off2b + float_tile_bytes;
    constexpr std::size_t off4b = off3b + reduce_tile_bytes;
    constexpr std::size_t UB_OFFSET_LIMIT = 0x3FFFFULL;
    constexpr std::size_t last_tile_end = off4b - 1ULL;
    static_assert(last_tile_end <= UB_OFFSET_LIMIT, "UB offset overflow: outTile end exceeds 0x3FFFF");

    constexpr uint32_t off0 = static_cast<uint32_t>(off0b);
    constexpr uint32_t off1 = static_cast<uint32_t>(off1b);
    constexpr uint32_t off2 = static_cast<uint32_t>(off2b);
    constexpr uint32_t off3 = static_cast<uint32_t>(off3b);
    constexpr uint32_t off4 = static_cast<uint32_t>(off4b);

    TASSIGN(prevTile, off0);
    TASSIGN(estTile, off1);
    TASSIGN(pvTile, off2);
    TASSIGN(expMaxTile, off3);
    TASSIGN(globalSumTile, off4);

    TLOAD(prevTile, prevGlobal);
    TLOAD(estTile, estGlobal);
    TLOAD(pvTile, pvGlobal);
    TLOAD(expMaxTile, expMaxGlobal);
    TLOAD(globalSumTile, globalSumGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    pto_macro_fa_gu_last<ReduceTile, TileData>(prevTile, estTile, expMaxTile, globalSumTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, prevTile);
    __out = outGlobal.data();
}

template <typename T, int Rows, int Cols>
void LaunchTgu(T *out, T *prev, T *est, T *exp_max, T *pv_pend, void *stream)
{
    runTgu<T, Rows, Cols><<<1, nullptr, stream>>>(out, prev, est, exp_max, pv_pend);
}

template <typename T, int Rows, int Cols>
void LaunchTguLast(T *out, T *prev, T *est, T *exp_max, T *global_sum, T *pv_pend, void *stream)
{
    runTguLast<T, Rows, Cols><<<1, nullptr, stream>>>(out, prev, est, exp_max, global_sum, pv_pend);
}

template void LaunchTgu<float, 32, 128>(float *out, float *prev, float *est, float *exp_max, float *pv_pend, void *stream);

template void LaunchTguLast<float, 32, 128>(float *out, float *prev, float *est, float *exp_max,
                                          float *global_sum, float *pv_pend, void *stream);