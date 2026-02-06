
#ifndef PTO_MACRO_FA_FIFO_HPP
#define PTO_MACRO_FA_FIFO_HPP

#include "fa_performance_kernel.h"
namespace pto {

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

template <int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD, typename TileQKData,
          typename TSyncQK2SM>
AICORE inline void push_qk_to_gm_fifo(int tile_id, int sub_tile_id, __gm__ float *qk_tile_fifo, TileQKData &qkAccTile,
                                      TSyncQK2SM &qk2smSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    const int sync_iter = tile_id;
    const bool should_wait_consume = should_wait_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
    if (sub_tile_id == 0 && should_wait_consume)
        qk2smSync.allocate(); // wait for SM consume data

    using GlobalDataQK = GlobalTensor<float, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
    const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t base_elems =
        static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) * static_cast<size_t>(Cube_S0) *
            static_cast<size_t>(Cube_S1) +
        static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
    GlobalDataQK qkGlobalTile(qk_tile_fifo + base_elems);

#if UF_ENABLE
    TSTORE<STPhase::Final>(qkGlobalTile, qkAccTile);
#else
    TSTORE(qkGlobalTile, qkAccTile);
#endif

    if (sub_tile_id == static_cast<int>(kTileFactor) - 1)
        qk2smSync.record(); // notify for QK produce data
}

template <int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD, typename TileDataF_T,
          typename TSyncQK2SM>
AICORE inline void pop_qk_from_gm_fifo(int tile_id, int row_slice, __gm__ float *qk_tile_fifo, TileDataF_T &qkVecTile,
                                       TSyncQK2SM &qk2smSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    if (row_slice == 0)
        qk2smSync.wait(); // wait for QK produce data

    const int sync_iter = tile_id;
    const bool should_notify_consume = should_notify_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
    const size_t subblock_base_rows = static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
    const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
    const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t base_elems = static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
                              static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
    __gm__ float *qk_ptr = qk_tile_fifo + base_elems + row_offset * static_cast<size_t>(Cube_S1);

    using GlobalDataQK_Sub =
        GlobalTensor<float, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
    using TileDataF_Sub = Tile<TileType::Vec, float, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
    for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
        __gm__ float *qk_ptr_sub =
            qk_ptr + static_cast<size_t>(sub_col) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
        GlobalDataQK_Sub qkGlobalSub(qk_ptr_sub);

        TileDataF_Sub qkVecSub;
        const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * Cube_S1 * sizeof(float));
        TASSIGN(qkVecSub, (uint64_t)qkVecTile.data() + col_byte_offset);
        TLOAD(qkVecSub, qkGlobalSub);
    }

    if (row_slice == static_cast<int>(kTileFactor) - 1 && should_notify_consume)
        qk2smSync.free(); // notify for SM consume data
}

template <int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD, typename TileDataH_T,
          typename TSyncSM2PV>
AICORE inline void push_p_to_gm_fifo(int tile_id, int row_slice, __gm__ half *p_tile_fifo, TileDataH_T &x_expT,
                                     TSyncSM2PV &sm2pvSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    constexpr uint32_t Vec_S0 = Cube_S0 / VEC_CORES / kTileFactor;
    int sync_iter = tile_id;
    const bool should_wait_sv_consumed = should_wait_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
    if (row_slice == 0 && should_wait_sv_consumed)
        sm2pvSync.allocate(); // wait for SV consume data

    const size_t subblock_base_rows = static_cast<size_t>(Cube_S0 / VEC_CORES) * static_cast<size_t>(get_subblockid());
    const size_t row_offset = subblock_base_rows + static_cast<size_t>(row_slice * Vec_S0);
    const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t base_elems = static_cast<size_t>(buf_idx) * static_cast<size_t>(kTileFactor) *
                              static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
    using GlobalPTileHalfSub =
        GlobalTensor<half, pto::Shape<1, 1, 1, Vec_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
    using TileDataH_Sub = Tile<TileType::Vec, half, Vec_S0, Tile_S1, BLayout::RowMajor, Vec_S0, Cube_S1>;
    __gm__ half *p_ptr = p_tile_fifo + base_elems + row_offset * static_cast<size_t>(Cube_S1);
    for (int sub_col = 0; sub_col < static_cast<int>(kTileFactor); ++sub_col) {
        __gm__ half *p_ptr_sub =
            p_ptr + static_cast<size_t>(sub_col) * static_cast<size_t>(Cube_S1) * static_cast<size_t>(Cube_S0);
        GlobalPTileHalfSub pTileHalfSub((__gm__ half *)(p_ptr_sub));

        TileDataH_Sub xExpSub;
        const uint64_t col_byte_offset = static_cast<uint64_t>(sub_col * Cube_S1 * sizeof(half));
        TASSIGN(xExpSub, (uint64_t)x_expT.data() + col_byte_offset);
        TSTORE(pTileHalfSub, xExpSub);
    }
    if (row_slice == static_cast<int>(kTileFactor) - 1)
        sm2pvSync.record(); // notify softmax produce data
}

template <int CUBE_S0, int CUBE_S1, int TILE_S1, int QKP_CV_FIFO, int CV_FIFO_CONS_SYNC_PERIOD,
          typename TileMatPData, typename TSyncSM2PV>
AICORE inline void pop_p_tile_from_gm_fifo(int tile_id, int sub_tile_id, __gm__ half *p_tile_fifo,
                                           TileMatPData &pMatTile, TSyncSM2PV &sm2pvSync)
{
    constexpr uint32_t Cube_S0 = CUBE_S0;
    constexpr uint32_t Cube_S1 = CUBE_S1;
    constexpr uint32_t Tile_S1 = TILE_S1;
    constexpr uint32_t kTileFactor = Tile_S1 / Cube_S1;
    if (sub_tile_id == 0)
        sm2pvSync.wait(); // wait for softmax produce data

    int sync_iter = tile_id;
    const bool should_notify_consume = should_notify_consumption<QKP_CV_FIFO, CV_FIFO_CONS_SYNC_PERIOD>(sync_iter);
// For TILE_S1 > CUBE_S1, need to stride by Tile_S1 for each Cube_S1 chunk
#ifndef P_FIFO_USE_NZ
    using GlobalXexpTileT = GlobalTensor<half, pto::Shape<1, 1, 1, Cube_S0, Cube_S1>, pto::Stride<1, 1, 1, Cube_S1, 1>>;
#else
    using GlobalXexpTileT = GlobalTensor<half, pto::Shape<1, Cube_S1 / 16, Cube_S0 / 16, 16, 16>,
                                         pto::Stride<Cube_S0 * Cube_S1, Cube_S0 * 16, 16 * 16, 16, 1>, Layout::NZ>;
#endif

    const uint32_t buf_idx = static_cast<uint32_t>(tile_id % QKP_CV_FIFO);
    const size_t base_elems =
        static_cast<size_t>(buf_idx) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Tile_S1) +
        static_cast<size_t>(sub_tile_id) * static_cast<size_t>(Cube_S0) * static_cast<size_t>(Cube_S1);
    GlobalXexpTileT xexpLoad(p_tile_fifo + base_elems);
    TLOAD(pMatTile, xexpLoad);
    if (sub_tile_id == static_cast<int>(kTileFactor) - 1 && should_notify_consume)
        sm2pvSync.free(); // notify SV consume data
}

} // namespace pto

#endif
