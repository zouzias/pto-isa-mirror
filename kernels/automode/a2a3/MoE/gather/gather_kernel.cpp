/**
 * gather_kernel.cpp - auto-mode A3 prototype.
 *
 * Unpack and accumulate per-expert outputs back into per-token rows. Generic
 * over kTopK in {1, 2, 4, 8, 16} — for kTopK=1 each token row is written
 * exactly once (pure permute); for kTopK>1 multiple packed rows per token
 * are summed (scatter-add).
 *
 * Inputs   (GM): B    [kT*kTopK + 16, kH] fp32  (only first kT*kTopK rows consulted)
 *                A_id [kT*kTopK + 16]     int32 (only first kT*kTopK consulted)
 * Outputs  (GM): C    [kT, kH]            fp32
 *
 * Semantics:
 *   C := 0
 *   for r in [0, kT*kTopK):
 *       C[A_id[r]] += B[r]
 *
 * Implementation per packed row:
 *   1. Read A_id[r] as a GM scalar -> token index t.
 *   2. TLOAD C[t] (current accumulated value)
 *   3. TLOAD B[r] (this row's contribution)
 *   4. TADD  sum = C[t] + B[r]
 *   5. TSTORE C[t]
 *
 * Initialization:
 *   The kernel does NOT pre-zero C. The host driver memsets C to zero before
 *   launch (aclrtMemset 0x00 on a float32 buffer = 0.0f), so the first TLOAD
 *   of any C[t] reads zeros. For kTopK=1 that means each C[t] is written
 *   exactly once with B[A_id^{-1}(t)] + 0; for kTopK>1 successive writes
 *   accumulate.
 *
 * Auto-mode constraints honored (mirror moe_top1_unpermute / moe_top1_permute):
 *   - Single AICORE (<<<1, nullptr, stream>>>).
 *   - Static row tiles declared once outside the loop.
 *   - pipe_barrier(PIPE_ALL) at the start of each iteration — same
 *     hardware-confirmed cross-iter auto-sync guard used in topk_kernel.cpp
 *     and the expert_ffn placeholder's row loops.
 *   - GlobalTensor reconstructed per iter with (base + runtime offset).
 *   - No TASSIGN; no Tile::data() in kernel; no *_IMPL calls; no raw CCE
 *     intrinsics; no Event<>; no manual sync; no TPipe/TPUSH/TPOP; no
 *     double buffering.
 *
 * Pattern source:
 *   - kernels/automode/a2a3/moe_top1_unpermute/moe_top1_unpermute_kernel.cpp
 *     (§11.5; same row-tile-in-loop pattern; we extend the simple TLOAD->TSTORE
 *     to TLOAD->TLOAD->TADD->TSTORE for accumulation)
 *   - kernels/automode/a2a3/add_tile_array/ (the TADD primitive on row tiles).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace gather_cfg {

// v1 shape — must match scripts/gen_data.py and main.cpp.
constexpr unsigned kT    = 256;
constexpr unsigned kH    = 64;
constexpr unsigned kTopK = 1;

constexpr unsigned kPackedRows   = kT * kTopK;
constexpr unsigned kOverspillPad = 16;
constexpr unsigned kAlloc        = kPackedRows + kOverspillPad;

}  // namespace gather_cfg

template <typename T>
__global__ AICORE void runGather(
    __gm__ T       __out__ *C,
    __gm__ T       __in__  *B,
    __gm__ int32_t __in__  *A_id)
{
    using namespace gather_cfg;

    using RowShape  = Shape <1, 1, 1, 1, kH>;
    using RowStride = Stride<1, 1, 1, kH, 1>;
    using RowGlobal = GlobalTensor<T, RowShape, RowStride>;

    using RowTile = Tile<TileType::Vec, T,
                         1, kH,
                         BLayout::RowMajor,
                         1, kH>;

    RowTile bTile;
    RowTile cTile;
    RowTile sumTile;

    for (unsigned r = 0; r < kPackedRows; ++r) {
        pipe_barrier(PIPE_ALL);

        int32_t t = A_id[r];                              // GM scalar read

        size_t src_off = static_cast<size_t>(r) * kH;
        size_t dst_off = static_cast<size_t>(t) * kH;

        RowGlobal bGlobal(B + src_off);
        RowGlobal cGlobal(C + dst_off);

        TLOAD(bTile, bGlobal);
        TLOAD(cTile, cGlobal);
        TADD (sumTile, cTile, bTile);
        TSTORE(cGlobal, sumTile);
    }
}

template <typename T>
void launchGather(T *C, T *B, int32_t *A_id, void *stream)
{
    runGather<T><<<1, nullptr, stream>>>(C, B, A_id);
}

template void launchGather<float>(float *C, float *B, int32_t *A_id, void *stream);
