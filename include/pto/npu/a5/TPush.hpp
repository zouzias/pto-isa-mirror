#ifndef TPUSH_HPP
#define TPUSH_HPP

#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>
#include <pto/common/fifo.hpp>

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class TSyncOpType : uint8_t {
    TSTORE_C2GM,  // Store (Cube core operation via PIPE_FIX) - GM path
    TSTORE_V2GM,  // Store (Vector core operation via PIPE_MTE3) - GM path
    TMOV_C2UB,    // TMOV from L0C to UB (Cube core operation via PIPE_FIX) - UB path
    TINSERT_V2L1, // TINSERT from UB to L1 (Vector core operation via PIPE_MTE3) - UB path
                  // TINSERT uses copy_ubuf_to_cbuf which goes through MTE3 pipe
                  // Cube consumer waits on PIPE_MTE1 (L1 side receives via MTE1)
    TLOAD         // Load operation (consumer operation)
};

// -----------------------------------------------------------------------------
// Compile-time direction inference based on producer/consumer ops
// GM path:
//   TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
//   TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
// UB path:
//   TMOV_C2UB (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
//   TINSERT_V2L1 (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
//   TINSERT (UB->L1) uses MTE3 on Vec side, Cube waits on MTE1
// -----------------------------------------------------------------------------
template <TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSyncTraits {
    // GM path: Cube produces via TSTORE_C2GM (PIPE_FIX) - consumer waits on PIPE_MTE2
    static constexpr bool is_cube_to_vec_gm = (ProducerOp == TSyncOpType::TSTORE_C2GM);
    // UB path: Cube produces via TMOV_C2UB (PIPE_FIX) - consumer waits on PIPE_V
    static constexpr bool is_cube_to_vec_ub = (ProducerOp == TSyncOpType::TMOV_C2UB);
    // Unified Cube-to-Vec detection
    static constexpr bool is_cube_to_vec = is_cube_to_vec_gm || is_cube_to_vec_ub;

    // GM path: Vector produces via TSTORE_V2GM (PIPE_MTE3)
    static constexpr bool is_vec_to_cube_gm = (ProducerOp == TSyncOpType::TSTORE_V2GM);
    // UB path: Vector produces via TINSERT_V2L1 (PIPE_MTE3) - Cube waits on PIPE_MTE1
    static constexpr bool is_vec_to_cube_ub = (ProducerOp == TSyncOpType::TINSERT_V2L1);
    // Unified Vec-to-Cube detection
    static constexpr bool is_vec_to_cube = is_vec_to_cube_gm || is_vec_to_cube_ub;

    static_assert(ConsumerOp == TSyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(is_cube_to_vec || is_vec_to_cube,
        "Producer must be TSTORE_C2GM, TMOV_C2UB (Cube) or TSTORE_V2GM, TINSERT_V2L1 (Vector)");
};

enum TSyncCVMode : uint8_t { CUBE_ALL_CORE_SYNC = 0, VEC_ALL_CORE_SYNC = 0, VEC_SUBCORES_SYNC = 1, CV_CORES_SYNC = 2 };

/**
 * Pipe: Manages Cross-Core FIFO Synchronization
 * @tparam ReadyFlag    Signal from Producer to Consumer (Data Ready)
 * @tparam ConsumedFlag Signal from Consumer to Producer (Space Released)
 * @tparam Depth        FIFO Depth (e.g., 2 for Double Buffering)
 * @tparam Period       Sync Period (Sync once every N tiles)
 * @tparam ProdRole     Logic role of Producer (CUBE/VECTOR) -> Deduce signal pipe
 * @tparam ConsRole     Logic role of Consumer (CUBE/VECTOR) -> Deduce signal pipe
 */
template <uint16_t FlagID, int Depth, int Period, TSyncOpType ProducerOp, TSyncOpType ConsumerOp>
struct TSync {
    using Traits = SyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_c2v_gm = Traits::is_cube_to_vec_gm;
    static constexpr bool is_c2v_ub = Traits::is_cube_to_vec_ub;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;
    static constexpr bool is_v2c_gm = Traits::is_vec_to_cube_gm;
    static constexpr bool is_v2c_ub = Traits::is_vec_to_cube_ub;

    uint16_t flag_id = FlagID;

    // -------------------------------------------------------------------------
    // Producer Interface
    // -------------------------------------------------------------------------
    struct Producer {
        int iter = 0; // Private iterator for Producer

        /**
         * alloc: Request space in FIFO
         * Logic:
         * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is empty.
         * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
         */
        PTO_INTERNAL void allocate() {
            if constexpr (is_c2v) {
                // Cube producer waits for Vec consumer to free buffer
                // Vec signals on flag_id+1 only, but Cube must wait on BOTH
                // (because Vec0 signals flag_id+1, Vec1 signals flag_id+1+16 from Cube's view)
                wait_intra_block(PIPE_FIX, flag_id + 1);
                wait_intra_block(PIPE_FIX, flag_id + 1 + 16);
            } else { // is_v2c (both gm and ub)
                // Vec producer waits for Cube consumer to free buffer
                // Cube signals on BOTH, Vec waits on flag_id+1 only
                wait_intra_block(PIPE_MTE3, flag_id + 1);
            }
        }

        // Forward dependency: record (producer) and wait (consumer)
        /**
         * record - Producer signals that data is ready
         * Called by the producer after completing the operation (TSTORE_C2GM or TSTORE_V2GM)
         */
        PTO_INTERNAL void record() {
            if constexpr (is_c2v) {
                // Cube -> Vec: Cube sets BOTH flags on PIPE_FIX
                set_intra_block(PIPE_FIX, flag_id);
                set_intra_block(PIPE_FIX, flag_id + 16);
            } else { // is_v2c (both gm and ub)
                // Vec -> Cube: Vec sets flag_id only on PIPE_MTE3
                // Each Vec subblock executes this; hardware maps subblock 1's flag to flag_id+16
                set_intra_block(PIPE_MTE3, flag_id);
            }
        }
        iter++;
    };

    // -------------------------------------------------------------------------
    // Consumer Interface
    // -------------------------------------------------------------------------
    struct Consumer {
        int iter = 0; // Private iterator for Consumer

        /**
         * wait: Block until data is ready
         * Consumers strictly wait for data (no sparse optimization for safety).
         */
        PTO_INTERNAL void wait() {
            if constexpr (is_c2v_gm) {
                // Cube -> Vec (GM path): Vec waits on PIPE_MTE2 (data loaded from GM)
                wait_intra_block(PIPE_MTE2, flag_id);
            } else if constexpr (is_c2v_ub) {
                // Cube -> Vec (UB path): Vec waits on PIPE_V before vector ops on UB data
                // Cube sets PIPE_FIX, Vec waits PIPE_V (Vec does vector ops, not TLOAD)
                wait_intra_block(PIPE_V, flag_id);
            } else if constexpr (is_v2c_gm) {
                // Vec -> Cube (GM path): Cube waits on PIPE_MTE2, BOTH flags
                wait_intra_block(PIPE_MTE2, flag_id);
                wait_intra_block(PIPE_MTE2, flag_id + 16);
            } else { // is_v2c_ub
                // Vec -> Cube (UB path - TINSERT): Cube waits on PIPE_MTE1, BOTH flags
                wait_intra_block(PIPE_MTE1, flag_id);
                wait_intra_block(PIPE_MTE1, flag_id + 16);
            }
        }

        /**
         * free: Release space in FIFO
         * Logic:
         * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
         * is still enjoying the initial free buffer space.
         * 2. (is_sync_step): Accumulate free slots and signal in batches.
         */
        PTO_INTERNAL void free() {
            bool is_sync_step = ((iter + 1) % Period) == 0;
            if (iter >= Depth - Period && is_sync_step) {
                if constexpr (is_c2v_gm) {
                    // Vec consumer frees buffer for Cube - signals on PIPE_MTE2, flag_id+1 only
                    set_intra_block(PIPE_MTE2, flag_id + 1);
                } else if constexpr (is_c2v_ub) {
                    // Vec consumer frees buffer for Cube - signals on PIPE_V, flag_id+1 only
                    // Vec signals after vector ops complete (PIPE_V)
                    set_intra_block(PIPE_V, flag_id + 1);
                } else { // is_v2c (both gm and ub)
                    // Cube consumer frees buffer for Vec - signals BOTH flags on PIPE_MTE1
                    set_intra_block(PIPE_MTE1, flag_id + 1);
                    set_intra_block(PIPE_MTE1, flag_id + 1 + 16);
                }
            }
        };
    };

    /**
     * TPUSH: Push Tile to Global Memory FIFO
     * * Flow:
     * 1. [Alloc]   Check GM space (Cross-Core)
     * 2. [Calc]    Calculate GM address based on iter
     * 3. [Barrier] Wait for calculation to finish (Intra-Core RAW protection)
     * 4. [Store]   Write data to GM
     * 5. [Commit]  Signal Consumer (Cross-Core)
     */
    template <typename PipeProd, typename TileData, typename DataFifo>
    PTO_INTERNAL void TPUSH_IMPL(PipeProd &prod, TileData &tile, DataFifo &fifo) {
        // 1. Cross-Core: Wait for space
        prod.allocate();

        /*
                // 2. Address Calculation
                int buffer_idx = prod.iter % DataFifo::fifoDepth;
                using T = typename TileData::DType;
                __ubuf__ T *addr = (__gm__ T *)fifo.fifoBase + buffer_idx * TileData::Numel;

                // 3. Intra-Core Sync & Store
                // Identify Tile location to enforce correct barriers
                if constexpr (TileData::Loc == TileType::Acc && DataFifo::memType == MemType::GM) {
                    // === Case: CUBE Output (L0C -> GM) ===
                    // Construct GlobalTensor helper
                    using GShape = TileShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
                    using GStride = BaseShape2D<T, TileData::Rows, TileData::Cols, Layout::ND>;
                    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;
                    GT global_tensor(addr);
                    TSTORE(global_tensor, tile);
                } else if constexpr (TileData::Loc == TileType::Acc && DataFifo::memType == MemType::UB) {
                    // === Case: CUBE Output (L0C -> UB) ===
                    // construct UB tile helper
                    using TileData = Tile<TileType::Vec, T, TileData::Rows, TileData::Cols, BLayout::RowMajor, -1, -1>;
                    TileData ub_tile(TileData::Rows, TileData::Cols);
                    TMOV(ub_tile, tile);
                } else if constexpr (TileData::Loc == TileType::Vec && DataFifo::memType == MemType::L1) {
                    // === Case: VECTOR Output (UB -> L1) ===
                    // cnstruct AccTile helper
                    using TileMatPData =
                        Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512>;
                    const uint32_t col_idx = static_cast<uint32_t>(sub_col * Cube_S1);
                    TINSERT_CUSTOM<TInsertMode::NZ_PLUS_1>(tile, nzConvBuffer, static_cast<uint32_t>(row_offset),
           col_idx); } else {
                    // === Case: VECTOR Output (UB -> GM) ===
                    TSTORE(global_tensor, tile);
                }
        */

        // 4. Cross-Core: Commit & Signal
        prod.record();
    }

} // namespace pto

#endif