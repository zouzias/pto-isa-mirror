/**
 * outval_pad_kernel.cpp - auto-mode A3 prototype.
 *
 * Bridge kernel for the full_moe_combined pipeline. moe_topk_padded outputs
 * outVal in (kT, kTopK) compact layout; gather expects (kT, kPadded) where
 * `kPadded = max(8, kTopK)` and cols kTopK..kPadded-1 are sentinel `-1e30`
 * (so they neutralize the per-row softmax: `exp(-1e30 - real_max) = 0`).
 *
 * The host pre-fills the destination (kT, kPadded) GM buffer with `-1e30`
 * before the combined launcher fires. This kernel then overwrites only the
 * first kTopK columns of each row with the actual sorted top-K values:
 *
 *   for t in 0..kT:
 *       TLOAD  rowTile     <- outVal_compact[t * kTopK : t * kTopK + kTopK]
 *       TSTORE outVal_padded[t * kPadded : t * kPadded + kTopK] <- rowTile
 *                                       (cols kTopK..kPadded-1 stay -1e30)
 *
 * For `kTopK < 8` the row tile's static Cols must be padded up to 8 floats
 * (32-byte UB alignment); valid region stays kTopK, so TLOAD/TSTORE only
 * touch the first kTopK elements.
 *
 * Auto-mode constraints honored:
 *   - Single AICORE.
 *   - Static row tile declared once outside the loop.
 *   - pipe_barrier(PIPE_ALL) at the start of each iteration.
 *   - GlobalTensor reconstructed per iteration with runtime offset.
 *
 * Pattern source: kernels/automode/a2a3/moe_top1_permute / moe_top1_unpermute
 * (row TLOAD-TSTORE inside-loop pattern).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

#include "generated_cases.h"

using namespace pto;

namespace outval_pad_cfg {

// Input-shape constants pulled from generated_cases.h (single-case per binary).
constexpr unsigned kT     = kMoeT;
constexpr unsigned kTopK  = kMoeTopK;

// Softmax tile column padding for 32-byte UB alignment (fp32: Cols % 8 == 0).
constexpr unsigned kPadded = (kTopK < 8) ? 8 : kTopK;

}  // namespace outval_pad_cfg

template <typename T>
__global__ AICORE void runOutValPad(__gm__ T __out__ *outVal_padded,
                                    __gm__ T __in__  *outVal_compact)
{
    using namespace outval_pad_cfg;

    using CompactShape  = Shape <1, 1, 1, 1, kTopK>;
    using CompactStride = Stride<1, 1, 1, kTopK, 1>;
    using CompactGlobal = GlobalTensor<T, CompactShape, CompactStride>;

    using PaddedShape   = Shape <1, 1, 1, 1, kPadded>;
    using PaddedStride  = Stride<1, 1, 1, kPadded, 1>;
    using PaddedGlobal  = GlobalTensor<T, PaddedShape, PaddedStride>;

    // Static Cols = kPadded (always >= 8 -> 32-byte aligned). Valid Cols = kTopK
    // so TLOAD/TSTORE only touch the first kTopK elements; cols kTopK..kPadded-1
    // of the GM destination retain the host's -1e30 sentinel.
    using RowTile = Tile<TileType::Vec, T, 1, kPadded, BLayout::RowMajor, -1, -1>;

    RowTile rowTile(1, kTopK);

    for (unsigned t = 0; t < kT; ++t) {
        pipe_barrier(PIPE_ALL);

        CompactGlobal compactRow(outVal_compact + t * kTopK);
        PaddedGlobal  paddedRow (outVal_padded  + t * kPadded);

        TLOAD (rowTile, compactRow);
        TSTORE(paddedRow, rowTile);
    }
}

template <typename T>
void launchOutValPad(T *outVal_padded, T *outVal_compact, void *stream)
{
    runOutValPad<T><<<1, nullptr, stream>>>(outVal_padded, outVal_compact);
}

template void launchOutValPad<float>(float *outVal_padded, float *outVal_compact, void *stream);
