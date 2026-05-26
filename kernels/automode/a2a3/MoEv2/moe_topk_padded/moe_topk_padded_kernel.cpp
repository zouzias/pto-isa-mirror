/**
 * moe_topk_padded_kernel.cpp - auto-mode A3 prototype.
 *
 * Per-row top-K selection over MoE router logits, generic over
 * kTopK in {1, 2, 4, 8, 16}.
 *
 * Input:  logits (kT, kE)     float32, one row per token.
 *         idx    (1, kE)      uint32   identity row [0..kE-1].
 * Output: outVal (kT, kTopK)  float32  top-K values, descending.
 *         outIdx (kT, kTopK)  uint32   matching expert indices.
 *
 * Pipeline per row (kE = 32 fits in a single TSORT32 block):
 *   1. TLOAD  logits row + idx (identity)
 *   2. TSORT32 -> sort32DstTile (one 32-element block -> 64 packed (val,idx))
 *   3. TSUBVIEW first kPackedTopK packed elements of sort32DstTile
 *   4. TGATHER P0101 -> outValTile (top-K float values)
 *   5. TRESHAPE + TGATHER P1010 -> outIdxTile (top-K uint32 indices)
 *   6. TSTORE outVal + outIdx
 *
 * Padding trick (mirrors MoE/moe_topk's kGatherWidth = 8 workaround):
 *   pto_tile.hpp:1510 asserts the output tile's Cols * sizeof(DType) is a
 *   multiple of 32 bytes. For kTopK in {1,2,4} the natural tile width fails
 *   that check. We declare the output tile with Cols = kGatherWidth, where
 *   kGatherWidth = max(8, kTopK), and set its dynamic valid region to
 *   (1, kTopK). TGATHER and TSTORE only emit valid-region elements, so only
 *   kTopK values per row land in GM. For kTopK in {8, 16} kGatherWidth ==
 *   kTopK; no padding needed.
 *
 * Pattern source: MoE/moe_topk/moe_topk_kernel.cpp (confirmed-built for
 *   kTopK = 2) and kernels/automode/a2a3/topk/topk_kernel.cpp (full topk,
 *   confirmed-built). This kernel generalizes moe_topk to arbitrary kTopK
 *   in {1, 2, 4, 8, 16}.
 *
 * Limitations (v1):
 *   - Single AICORE; no block_idx work split.
 *   - Float32 only (TYPE_COEF = 1).
 *   - kE must be exactly 32 (one TSORT32 block; no merge / tail).
 *   - kTopK must be in {1, 2, 4, 8, 16}.
 *   - No double-buffering.
 */

#include <pto/pto-inst.hpp>
#include "generated_cases.h"
#include <pto/common/pto_tile.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

template <typename T, int kT_, int kE_, int kTopK_>
__global__ AICORE void RunMoeTopkPadded(__gm__ uint8_t *outVal_raw,
                                        __gm__ uint8_t *outIdx_raw,
                                        __gm__ uint8_t *src_raw,
                                        __gm__ uint8_t *idx_raw)
{
    using indexT = uint32_t;
    constexpr int TYPE_COEF = sizeof(float) / sizeof(T);
    constexpr int kPackedCols = kE_   * 2 * TYPE_COEF;  // 32*2 = 64
    constexpr int kPackedTopK = kTopK_ * 2 * TYPE_COEF;  // kTopK*2

    // Pad output tile width up to the smallest multiple of 8 floats so the
    // 32-byte-alignment assertion passes. valid region (set in constructor
    // below) is (1, kTopK_), so TGATHER + TSTORE only emit kTopK_ values.
    constexpr int kGatherWidth = (kTopK_ < 8) ? 8 : kTopK_;

    __gm__ T      *outVal = reinterpret_cast<__gm__ T *>(outVal_raw);
    __gm__ indexT *outIdx = reinterpret_cast<__gm__ indexT *>(outIdx_raw);
    __gm__ T      *src    = reinterpret_cast<__gm__ T *>(src_raw);
    __gm__ indexT *idx    = reinterpret_cast<__gm__ indexT *>(idx_raw);

    using SrcGlobal    = GlobalTensor<T,      Shape<1, 1, 1, 1, kE_>,    Stride<1, 1, 1, kE_,    1>>;
    using IdxGlobal    = GlobalTensor<indexT, Shape<1, 1, 1, 1, kE_>,    Stride<1, 1, 1, kE_,    1>>;
    using OutValGlobal = GlobalTensor<T,      Shape<1, 1, 1, 1, kTopK_>, Stride<1, 1, 1, kTopK_, 1>>;
    using OutIdxGlobal = GlobalTensor<indexT, Shape<1, 1, 1, 1, kTopK_>, Stride<1, 1, 1, kTopK_, 1>>;

    using SrcTile       = Tile<TileType::Vec, T,      1, kE_,         BLayout::RowMajor, -1, -1>;
    using IdxTile       = Tile<TileType::Vec, indexT, 1, kE_,         BLayout::RowMajor, -1, -1>;
    using PackedTile    = Tile<TileType::Vec, T,      1, kPackedCols, BLayout::RowMajor, -1, -1>;
    using PackedIdxTile = Tile<TileType::Vec, indexT, 1, kPackedCols, BLayout::RowMajor, -1, -1>;
    using OutValTile    = Tile<TileType::Vec, T,      1, kGatherWidth, BLayout::RowMajor, -1, -1>;
    using OutIdxTile    = Tile<TileType::Vec, indexT, 1, kGatherWidth, BLayout::RowMajor, -1, -1>;

    if constexpr (kE_ <= 32) {
        // ----------------------------------------------------------------
        // Tile path: TSORT32 handles exactly one 32-element block.
        // ----------------------------------------------------------------
        for (int row = 0; row < kT_; ++row) {
            pipe_barrier(PIPE_ALL);

            SrcGlobal     srcGlobal(src + row * kE_);
            IdxGlobal     idxGlobal(idx);
            OutValGlobal  outValGlobal(outVal + row * kTopK_);
            OutIdxGlobal  outIdxGlobal(outIdx + row * kTopK_);

            SrcTile     srcTile(1, kE_);
            IdxTile     idxTile(1, kE_);
            SrcTile     sort32TmpTile(1, kE_);
            PackedTile  sort32DstTile(1, kPackedCols);
            OutValTile  outValTile(1, kTopK_);
            OutIdxTile  outIdxTile(1, kTopK_);

            TLOAD(srcTile, srcGlobal);
            TLOAD(idxTile, idxGlobal);
            TSORT32(sort32DstTile, srcTile, idxTile, sort32TmpTile);

            PackedTile sortedTopKView(1, kPackedTopK);
            TSUBVIEW(sortedTopKView, sort32DstTile, 0, 0);
            if constexpr (std::is_same_v<T, half>) {
                TGATHER<OutValTile, PackedTile, MaskPattern::P0001>(outValTile, sortedTopKView);
            } else {
                TGATHER<OutValTile, PackedTile, MaskPattern::P0101>(outValTile, sortedTopKView);
            }

            PackedIdxTile sortedTopKIdxView(1, kPackedTopK);
            TRESHAPE(sortedTopKIdxView, sort32DstTile);
            TGATHER<OutIdxTile, PackedIdxTile, MaskPattern::P1010>(outIdxTile, sortedTopKIdxView);

            TSTORE(outValGlobal, outValTile);
            TSTORE(outIdxGlobal, outIdxTile);
        }
    } else {
        // ----------------------------------------------------------------
        // Scalar fallback for kE > 32 (e.g., kE=64).
        // TSORT32 only handles 32 elements; a tile-based 2-pass merge is
        // complex in auto-mode.  We use a scalar selection sort on the SPU
        // instead.  kE <= 64 → 256 bytes per value/index array on the
        // scalar stack (well within the 32 KB limit).
        // ----------------------------------------------------------------
        static_assert(kE_ <= 64,
            "Scalar fallback supports kE up to 64; extend or implement tile merge for larger kE");

        for (int row = 0; row < kT_; ++row) {
            pipe_barrier(PIPE_ALL);

            // Load all kE logit values and build identity index array.
            T        logit_vals[kE_];
            indexT   logit_idx[kE_];
            for (int i = 0; i < kE_; ++i) {
                logit_vals[i] = src[row * kE_ + i];
                logit_idx[i]  = static_cast<indexT>(i);
            }

            // Selection sort: pull the top-kTopK_ elements to the front.
            for (int k = 0; k < kTopK_; ++k) {
                int best = k;
                for (int i = k + 1; i < kE_; ++i) {
                    if (logit_vals[i] > logit_vals[best]) best = i;
                }
                if (best != k) {
                    T      tv = logit_vals[k]; logit_vals[k] = logit_vals[best]; logit_vals[best] = tv;
                    indexT ti = logit_idx[k];  logit_idx[k]  = logit_idx[best];  logit_idx[best]  = ti;
                }
                outVal[row * kTopK_ + k] = logit_vals[k];
                outIdx[row * kTopK_ + k] = logit_idx[k];
            }
        }
    }
}

template <typename T>
void launchMoeTopkPadded(uint8_t *outVal, uint8_t *outIdx,
                         uint8_t *src,    uint8_t *idx,
                         void *stream)
{
    // Compile-time shape constants — must match main.cpp and gen_data.py.
    constexpr int kT    = kMoeT;
    constexpr int kE    = kMoeE;
    constexpr int kTopK = kMoeTopK;   // v1: kTopK = 1; change to 2/4/8/16 for sweep.
    RunMoeTopkPadded<T, kT, kE, kTopK><<<1, nullptr, stream>>>(outVal, outIdx, src, idx);
}

template void launchMoeTopkPadded<float>(uint8_t *outVal, uint8_t *outIdx,
                                         uint8_t *src,    uint8_t *idx,
                                         void *stream);
