/**
 * scatter_kernel.cpp - auto-mode A3 prototype.
 *
 * Pack tokens by expert assignment. Generic over kTopK in {1, 2, 4, 8, 16}.
 *
 * Inputs   (GM): X         [kT, kH]            fp16
 *                expert_id [kT, kTopK]         int32
 * Outputs  (GM): A         [kT*kTopK + 16, kH] fp16  (kT*kTopK rows valid, +16 overspill pad)
 *                A_id      [kT*kTopK + 16]     int32 (kT*kTopK rows valid, +16 ignored)
 *                rank_id   [kT*kTopK + 16]     int32 (kT*kTopK rows valid, +16 ignored)
 *                expert_count [kE]             int32
 *                expert_start [kE]             int32
 *
 * rank_id[r] is the k slot (0..kTopK-1) of the (t, k) pair that the packed
 * row r came from. Used downstream by the gather kernel to look up
 * softmax_weight[t, rank_id[r]] when kTopK > 1.
 *
 * Semantics (three passes — same shape as moe_top1_permute, extended to
 * (t, k) pairs for kTopK > 1):
 *
 *   1. histogram         : count[e] = #{(t, k) : expert_id[t, k] == e}
 *   2. prefix sum        : start[0] = 0;
 *                          start[e] = start[e-1] + count[e-1]
 *   3. pack              : for each (t, k) in row-major order,
 *                          slot       = counter[expert_id[t, k]]++;
 *                          packed_pos = start[expert_id[t, k]] + slot;
 *                          A_id[packed_pos]    = t;
 *                          rank_id[packed_pos] = k;
 *                          A[packed_pos, :]    = X[t, :];
 *
 * The trailing 16 rows of A and A_id are NOT written here. They are the
 * overspill landing pad for the downstream expert_ffn kernel and contain
 * undefined contents — the FFN's per-expert outer loop overwrites them
 * (or they remain as scratch beyond the gather's valid range).
 *
 * Auto-mode constraints honored (mirror moe_top1_permute / add_tile_array):
 *   - Single AICORE (<<<1, nullptr, stream>>>).
 *   - Static row tile (Tile<Vec, half, 1, kH, RowMajor, 1, kH>) declared
 *     once outside both loops.
 *   - Per-expert histogram / start / counter in small local int32 arrays.
 *   - GlobalTensor reconstructed per iteration with runtime offset.
 *   - No TASSIGN aliasing; no Tile::data() in kernel; no *_IMPL calls;
 *     no raw CCE intrinsics; no Event<>; no manual sync; no TPipe / TPUSH /
 *     TPOP; no double buffering.
 *
 * Pattern source:
 *   - kernels/automode/a2a3/moe_top1_permute/moe_top1_permute_kernel.cpp
 *     (confirmed-built §11.4; we extend the (t) loop to (t, k) pairs and
 *     emit A_id as a back-map instead of token_to_packed as a forward-map)
 *   - kernels/automode/a2a3/add_tile_array/ (per-iter GM offsets baseline)
 *
 * Host boundary: uint8_t* for the typed fp16 buffers (host TU is -xc++ and
 * cannot name `half`; see compile_error_logbook.md §E13).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace scatter_cfg {

// v1 shape — must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT     = 256;
constexpr unsigned kH     = 64;
constexpr unsigned kE     = 32;
constexpr unsigned kTopK  = 1;

constexpr unsigned kPackedRows    = kT * kTopK;
constexpr unsigned kOverspillPad  = 16;
constexpr unsigned kAlloc         = kPackedRows + kOverspillPad;

}  // namespace scatter_cfg

template <typename TIn>
__global__ AICORE void runScatter(
    __gm__ uint8_t *A_raw,
    __gm__ int32_t *A_id,
    __gm__ int32_t *rank_id,
    __gm__ int32_t *expert_count,
    __gm__ int32_t *expert_start,
    __gm__ uint8_t *X_raw,
    __gm__ int32_t *expert_id)
{
    using namespace scatter_cfg;

    __gm__ TIn *A = reinterpret_cast<__gm__ TIn *>(A_raw);
    __gm__ TIn *X = reinterpret_cast<__gm__ TIn *>(X_raw);

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<TIn, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, TIn,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    RowTile rowTile;

    // ============================================================
    // Pass 1: histogram. count[e] = #{(t, k) : expert_id[t, k] == e}.
    // ============================================================
    int32_t count[kE];
    for (unsigned e = 0; e < kE; ++e) {
        count[e] = 0;
    }
    for (unsigned t = 0; t < kT; ++t) {
        for (unsigned k = 0; k < kTopK; ++k) {
            int32_t e = expert_id[t * kTopK + k];   // GM scalar read
            count[e]++;
        }
    }
    for (unsigned e = 0; e < kE; ++e) {
        expert_count[e] = count[e];                 // GM scalar write
    }

    // ============================================================
    // Pass 2: prefix sum. start[e] = sum_{i<e} count[i].
    // ============================================================
    int32_t start[kE];
    start[0] = 0;
    for (unsigned e = 1; e < kE; ++e) {
        start[e] = start[e - 1] + count[e - 1];
    }
    for (unsigned e = 0; e < kE; ++e) {
        expert_start[e] = start[e];                 // GM scalar write
    }

    // ============================================================
    // Pass 3: pack tokens into expert-grouped order, preserving original
    // (t, k) order within each expert. counter[e] is the running slot
    // index within expert e's segment.
    // ============================================================
    int32_t counter[kE];
    for (unsigned e = 0; e < kE; ++e) {
        counter[e] = 0;
    }

    for (unsigned t = 0; t < kT; ++t) {
        for (unsigned k = 0; k < kTopK; ++k) {
            pipe_barrier(PIPE_ALL);

            int32_t e          = expert_id[t * kTopK + k];   // GM scalar read
            int32_t slot       = counter[e]++;
            int32_t packed_pos = start[e] + slot;

            A_id[packed_pos]   = static_cast<int32_t>(t);    // GM scalar write
            rank_id[packed_pos] = static_cast<int32_t>(k);   // GM scalar write

            size_t src_off = static_cast<size_t>(t)          * kH;
            size_t dst_off = static_cast<size_t>(packed_pos) * kH;

            RowGlobal srcGlobal(X + src_off);
            RowGlobal dstGlobal(A + dst_off);

            TLOAD (rowTile, srcGlobal);
            TSTORE(dstGlobal, rowTile);
        }
    }
}

template <typename TIn>
void launchScatter(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                   int32_t *expert_count, int32_t *expert_start,
                   uint8_t *X, int32_t *expert_id,
                   void *stream)
{
    runScatter<TIn><<<1, nullptr, stream>>>(
        A, A_id, rank_id, expert_count, expert_start, X, expert_id);
}

template void launchScatter<half>(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                                  int32_t *expert_count, int32_t *expert_start,
                                  uint8_t *X, int32_t *expert_id,
                                  void *stream);

// Non-template wrapper: host TU is -xc++ and cannot name `half`.
// (compile_error_logbook.md §E13)
extern "C" void launchScatterFp16(uint8_t *A, int32_t *A_id, int32_t *rank_id,
                                  int32_t *expert_count, int32_t *expert_start,
                                  uint8_t *X, int32_t *expert_id,
                                  void *stream)
{
    launchScatter<half>(A, A_id, rank_id, expert_count, expert_start, X, expert_id, stream);
}
