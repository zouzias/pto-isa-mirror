/**
 * mla_basic_vec_kernel.cpp - auto-mode A3 prototype, vec-arch stages.
 *
 * Two `__global__ AICORE` entries for the MLA pipeline that need the vec
 * pipe (`--cce-aicore-arch=dav-c220-vec`):
 *
 *   Stage 3 -- runKVCacheStore
 *       C_cache[B,S,L] = C_kv[B,S,L]   single tile copy (128 x 64 FP16)
 *
 *   Stage 5b -- runAttnSoftmax
 *       probs[h]  =  softmax(scale * scores[h])    per head h in [0, Nh)
 *       scale = 1/sqrt(kHeadDim) baked in by the host
 *
 * Both kernels follow the §A11 add_tile_array recipe: static-valid Vec tile,
 * declared once outside the loop, GlobalTensor reconstructed per iteration
 * with a runtime offset baked into the GM pointer.
 *
 * Softmax decomposition matches docs/coding/tutorials/row-softmax.md:
 *   scaled  =  scores * scale
 *   rowMax  =  TROWMAX(scaled)             (per-row reduction)
 *   bcast   =  TROWEXPAND(rowMax)          (broadcast back to [S, S])
 *   shifted =  scaled - bcast              (numerical stability: subtract max)
 *   ex      =  exp(shifted)
 *   rowSum  =  TROWSUM(ex)
 *   bcast'  =  TROWEXPAND(rowSum)
 *   probs   =  ex / bcast'
 *
 * UB budget (per head iteration, FP16 throughout):
 *   4 x [128, 128] FP16 tiles  =  4 * 32 KB  = 128 KB
 *   2 x [128,  16] FP16 tiles  =  ~1 KB
 *   total                       ~= 129 KB  (UB = 256 KB)
 *
 * No in-place ops (dst always distinct from sources); §A15 documented the
 * TADDS in-place failure mode and we are conservative here.
 *
 * No bidirectional / no causal mask in v1 (user choice).
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>
#include "generated_cases.h"

#include "generated_cases.h"   // emitted by scripts/generate_cases.py

using namespace pto;

namespace mla_basic_cfg_vec {

// Mirror of mla_basic_cube_kernel.cpp's namespace. The two TUs compile
// independently so we cannot share a header without adding plumbing; both
// pull the active case from build/generated_cases.h.
constexpr unsigned kBatch    = 1;
constexpr unsigned kSeqLen   = kMlaSeqLen;
constexpr unsigned kHidden   = kMlaHidden;
constexpr unsigned kNumHeads = kMlaNumHeads;
constexpr unsigned kHeadDim  = kMlaHeadDim;
constexpr unsigned kLatent   = kMlaLatent;
constexpr unsigned kRopeDim  = kMlaRopeDim;
constexpr unsigned kRopeHalf = kRopeDim / 2;

// Vec-side M-chunk sizes. Per-row softmax / RoPE / cache copy are independent
// along M (sequence axis), so we process the S rows in fixed-size sub-tiles.
// This keeps UB usage independent of kSeqLen — increasing kSeqLen just adds
// more inner-loop iterations, not bigger tiles.
//
//   UB budget at kSeqLen=128 / 512 / 1024 (FP16):
//   softmax: 5 tiles [TileM, S] + 2 row-reduce [TileM, 16]
//     TileM=16:  20 KB / 82 KB / 164 KB
//   rope: 7 half-tiles [TileM, 32]
//     TileM=64:  28 KB independent of S
//   cache:  1 tile [TileM, 64]
//     TileM=64:   8 KB independent of S
constexpr unsigned kSoftmaxTileM = 16;
constexpr unsigned kRopeTileM    = 64;
constexpr unsigned kCacheTileM   = 64;
static_assert(kSeqLen % kSoftmaxTileM == 0, "kSeqLen must be a multiple of kSoftmaxTileM");
static_assert(kSeqLen % kRopeTileM    == 0, "kSeqLen must be a multiple of kRopeTileM");
static_assert(kSeqLen % kCacheTileM   == 0, "kSeqLen must be a multiple of kCacheTileM");

// Multi-core launch dimension; mirrors the cube TU. Each kernel uses
// `get_block_idx()` to claim a strided slice of its output work items.
constexpr unsigned kBlockDim = 24;

}  // namespace mla_basic_cfg_vec

// =============================================================================
// Stage 3 -- KV cache store: C_cache = C_kv  (prefill-only v1, no append).
//
// Chunked along M (sequence) to keep UB usage independent of kSeqLen. Each
// inner iter copies a [kCacheTileM, kLatent] sub-tile.
// =============================================================================
template <typename T>
__global__ AICORE void runKVCacheStore(__gm__ T *c_cache, __gm__ T *c_kv)
{
    using namespace mla_basic_cfg_vec;

    using TileShape  = Shape <1, 1, 1, kCacheTileM, kLatent>;
    using TileStride = Stride<1, 1, 1, kLatent, 1>;
    using GlobalData = GlobalTensor<T, TileShape, TileStride>;

    using TileData = Tile<TileType::Vec, T,
                          kCacheTileM, kLatent,
                          BLayout::RowMajor,
                          kCacheTileM, kLatent>;

    TileData buf;

    // Multi-core: each core takes a strided slice of m-chunks.
    const unsigned core_id   = get_block_idx();
    const unsigned chunkStep = kCacheTileM * kBlockDim;

    for (unsigned m0 = core_id * kCacheTileM; m0 < kSeqLen; m0 += chunkStep) {
        const size_t rowOffset = static_cast<size_t>(m0) * kLatent;

        GlobalData srcGlobal(c_kv    + rowOffset);
        GlobalData dstGlobal(c_cache + rowOffset);

        TLOAD (buf, srcGlobal);
        TSTORE(dstGlobal, buf);
    }
}

// =============================================================================
// Stage 5g (NEW) -- RoPE half-rotation.
//
//   For each block b in [0, numBlocks):
//     for m-chunk (kRopeTileM rows at a time):
//       x1 = x[b, m..m+TileM, :half],  x2 = x[b, m..m+TileM, half:]
//       y[b, m..m+TileM, :half] = x1 * cos[m..] - x2 * sin[m..]
//       y[b, m..m+TileM, half:] = x1 * sin[m..] + x2 * cos[m..]
//
//   x, y     : [numBlocks, S, kRopeDim]   half
//   cos, sin : [S, kRopeHalf]             half  (shared across blocks)
//
// Chunked along M so UB usage doesn't grow with kSeqLen. cos/sin are
// re-loaded per m-chunk; the inner-loop ordering (b outer, m inner) keeps
// the kernel simple at the cost of redundant cos/sin DMA — negligible for
// a correctness kernel.
//
// numBlocks is a runtime argument: for Q_rope it is kNumHeads (32 head-major
// blocks); for K_rope it is 1 (shared across heads).
// =============================================================================
template <typename T>
__global__ AICORE void runRoPE(__gm__ T *y, __gm__ T *x,
                               __gm__ T *cos, __gm__ T *sin,
                               unsigned numBlocks)
{
    using namespace mla_basic_cfg_vec;

    using HalfShape       = Shape <1, 1, 1, kRopeTileM, kRopeHalf>;
    using HalfStrideX     = Stride<1, 1, 1, kRopeDim,  1>;   // x has row stride kRopeDim
    using HalfStrideTable = Stride<1, 1, 1, kRopeHalf, 1>;   // cos/sin tightly packed

    using GlobalDataHalfX     = GlobalTensor<T, HalfShape, HalfStrideX>;
    using GlobalDataHalfTable = GlobalTensor<T, HalfShape, HalfStrideTable>;

    using HalfTile = Tile<TileType::Vec, T,
                          kRopeTileM, kRopeHalf,
                          BLayout::RowMajor,
                          kRopeTileM, kRopeHalf>;

    HalfTile x1Tile;      // x[b, m..m+TileM, :half]
    HalfTile x2Tile;      // x[b, m..m+TileM, half:]
    HalfTile cosTile;     // cos[m..m+TileM, :]
    HalfTile sinTile;     // sin[m..m+TileM, :]
    HalfTile productTile; // scratch for one TMUL result
    HalfTile y1Tile;      // y[b, m..m+TileM, :half] before TSTORE
    HalfTile y2Tile;      // y[b, m..m+TileM, half:] before TSTORE

    // Multi-core: flatten (b, m-chunk) work-item space and stride across cores.
    constexpr unsigned mIters = kSeqLen / kRopeTileM;
    const unsigned core_id    = get_block_idx();
    const unsigned totalWork  = numBlocks * mIters;

    for (unsigned w = core_id; w < totalWork; w += kBlockDim) {
        const unsigned b           = w / mIters;
        const unsigned m_chunk     = w % mIters;
        const size_t blockOffset   = static_cast<size_t>(b) * kSeqLen * kRopeDim;
        const size_t m0            = static_cast<size_t>(m_chunk) * kRopeTileM;
        const size_t xRowOffset    = m0 * kRopeDim;
        const size_t tableOffset   = m0 * kRopeHalf;

        GlobalDataHalfX     x1Global (x + blockOffset + xRowOffset);
        GlobalDataHalfX     x2Global (x + blockOffset + xRowOffset + kRopeHalf);
        GlobalDataHalfX     y1Global (y + blockOffset + xRowOffset);
        GlobalDataHalfX     y2Global (y + blockOffset + xRowOffset + kRopeHalf);
        GlobalDataHalfTable cosGlobal(cos + tableOffset);
        GlobalDataHalfTable sinGlobal(sin + tableOffset);

        TLOAD(x1Tile,  x1Global);
        TLOAD(x2Tile,  x2Global);
        TLOAD(cosTile, cosGlobal);
        TLOAD(sinTile, sinGlobal);

        // y1 = x1 * cos - x2 * sin
        TMUL(y1Tile,      x1Tile, cosTile);
        TMUL(productTile, x2Tile, sinTile);
        TSUB(y1Tile,      y1Tile, productTile);

        // y2 = x1 * sin + x2 * cos
        TMUL(y2Tile,      x1Tile, sinTile);
        TMUL(productTile, x2Tile, cosTile);
        TADD(y2Tile,      y2Tile, productTile);

        TSTORE(y1Global, y1Tile);
        TSTORE(y2Global, y2Tile);
    }
}

// =============================================================================
// Stage 5b -- Attention softmax:
//   For each head h, for each m-chunk of kSoftmaxTileM rows:
//     scoresSum = scores_nope[h, m..m+TileM, :] + scores_rope[h, m..m+TileM, :]
//     probs[h, m..m+TileM, :] = softmax(scale * scoresSum)
//
// Softmax is row-independent, so chunking M is exact (no online-softmax
// needed). UB usage at runtime:
//   5 tiles [kSoftmaxTileM, kSeqLen] + 2 row-reduce [kSoftmaxTileM, 16]
// which scales linearly with kSeqLen but is divided by kSoftmaxTileM relative
// to a full-row tile, so a kSoftmaxTileM=16 chunk keeps the kernel in UB for
// kSeqLen up to ~1024.
//
//   scores_nope, scores_rope : [kNumHeads, kSeqLen, kSeqLen]  half
//   probs                    : [kNumHeads, kSeqLen, kSeqLen]  half
//   scale                    : 1/sqrt(kHeadDim), supplied by host.
//
// The nope+rope add happens INSIDE this kernel via TADD on the loaded
// scoresTile (avoids needing a separate combine kernel / atomic-add TSTORE).
// =============================================================================
template <typename T>
__global__ AICORE void runAttnSoftmax(__gm__ T *probs,
                                      __gm__ T *scores_nope,
                                      __gm__ T *scores_rope, T scale)
{
    using namespace mla_basic_cfg_vec;

    using TileShape  = Shape <1, 1, 1, kSoftmaxTileM, kSeqLen>;
    using TileStride = Stride<1, 1, 1, kSeqLen, 1>;
    using GlobalData = GlobalTensor<T, TileShape, TileStride>;

    using ScoresTile = Tile<TileType::Vec, T,
                            kSoftmaxTileM, kSeqLen,
                            BLayout::RowMajor,
                            kSoftmaxTileM, kSeqLen>;

    // Row-reduction output tile: storage width 16 cols (32-byte aligned for
    // FP16 per Tile constexpr asserts), valid width 1 col. Pattern from
    // tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp:30.
    using RowReduceTile = Tile<TileType::Vec, T,
                               kSoftmaxTileM, 16,
                               BLayout::RowMajor,
                               -1, -1>;

    ScoresTile     scoresTile;     // role 1: nope+rope combined; role 2: shifted; role 3: probs
    ScoresTile     scoresRopeTile; // staging for the rope term before TADD
    ScoresTile     scaledTile;     // role 1: scaled scores;  role 2: exp
    ScoresTile     broadcastTile;  // role 1: broadcast(max); role 2: broadcast(sum)
    ScoresTile     tmpTile;        // TROWMAX / TROWSUM scratch (shape == src)
    RowReduceTile  rowMaxTile(kSoftmaxTileM, 1);
    RowReduceTile  rowSumTile(kSoftmaxTileM, 1);

    // Multi-core: flatten (h, m-chunk) work-item space. Each work item is one
    // [kSoftmaxTileM, kSeqLen] probs tile; softmax is row-independent so
    // chunking M is exact (no online softmax needed).
    constexpr unsigned mIters    = kSeqLen / kSoftmaxTileM;
    constexpr unsigned totalWork = kNumHeads * mIters;
    const unsigned core_id       = get_block_idx();

    for (unsigned w = core_id; w < totalWork; w += kBlockDim) {
        const unsigned h           = w / mIters;
        const unsigned m_chunk     = w % mIters;
        const size_t headOffset    = static_cast<size_t>(h) * kSeqLen * kSeqLen;
        const size_t m0            = static_cast<size_t>(m_chunk) * kSoftmaxTileM;
        const size_t rowOffset     = m0 * kSeqLen;

        GlobalData srcNopeGlobal(scores_nope + headOffset + rowOffset);
        GlobalData srcRopeGlobal(scores_rope + headOffset + rowOffset);
        GlobalData dstGlobal    (probs       + headOffset + rowOffset);

        TLOAD(scoresTile,     srcNopeGlobal);
        TLOAD(scoresRopeTile, srcRopeGlobal);
        TADD (scoresTile, scoresTile, scoresRopeTile);  // combined scores

        // 1. Scale by 1/sqrt(headDim).
        TMULS(scaledTile, scoresTile, scale);

        // 2. Row max (numerical stability).
        TROWMAX(rowMaxTile, scaledTile, tmpTile);

        // 3. Broadcast row max to [TileM, kSeqLen].
        TROWEXPAND(broadcastTile, rowMaxTile);

        // 4. shifted = scaled - broadcast(max).      (reuse scoresTile)
        TSUB(scoresTile, scaledTile, broadcastTile);

        // 5. exp(shifted).                            (reuse scaledTile)
        TEXP(scaledTile, scoresTile);

        // 6. Row sum of exp.
        TROWSUM(rowSumTile, scaledTile, tmpTile);

        // 7. Broadcast row sum.                       (reuse broadcastTile)
        TROWEXPAND(broadcastTile, rowSumTile);

        // 8. probs = exp / broadcast(sum).            (reuse scoresTile)
        TDIV(scoresTile, scaledTile, broadcastTile);

        TSTORE(dstGlobal, scoresTile);
    }
}

// ----------------------------------------------------------------------------
// Templated launchers + explicit instantiations + non-template `…Fp16`
// wrappers. Same shape as the cube TU.
// ----------------------------------------------------------------------------

template <typename T>
void launchKVCacheStore(T *c_cache, T *c_kv, void *stream)
{
    runKVCacheStore<T><<<mla_basic_cfg_vec::kBlockDim, nullptr, stream>>>(c_cache, c_kv);
}

template <typename T>
void launchAttnSoftmax(T *probs, T *scores_nope, T *scores_rope, T scale, void *stream)
{
    runAttnSoftmax<T><<<mla_basic_cfg_vec::kBlockDim, nullptr, stream>>>(probs, scores_nope, scores_rope, scale);
}

template <typename T>
void launchRoPE(T *y, T *x, T *cos, T *sin, unsigned numBlocks, void *stream)
{
    runRoPE<T><<<mla_basic_cfg_vec::kBlockDim, nullptr, stream>>>(y, x, cos, sin, numBlocks);
}

template void launchKVCacheStore<half>(half *, half *, void *);
template void launchAttnSoftmax <half>(half *, half *, half *, half, void *);
template void launchRoPE        <half>(half *, half *, half *, half *, unsigned, void *);

// Host wrappers (host TU compiled `-xc++` cannot see `half`).
// We take the scalar `scale` as a `float` from the host and cast on entry.
extern "C" void launchMlaKVCacheStoreFp16(uint8_t *c_cache, uint8_t *c_kv, void *stream)
{
    launchKVCacheStore<half>(reinterpret_cast<half *>(c_cache),
                             reinterpret_cast<half *>(c_kv),
                             stream);
}

extern "C" void launchMlaAttnSoftmaxFp16(uint8_t *probs,
                                         uint8_t *scores_nope,
                                         uint8_t *scores_rope,
                                         float    scale,
                                         void    *stream)
{
    launchAttnSoftmax<half>(reinterpret_cast<half *>(probs),
                            reinterpret_cast<half *>(scores_nope),
                            reinterpret_cast<half *>(scores_rope),
                            static_cast<half>(scale),
                            stream);
}

extern "C" void launchMlaRoPEFp16(uint8_t *y, uint8_t *x,
                                  uint8_t *cos, uint8_t *sin,
                                  unsigned numBlocks, void *stream)
{
    launchRoPE<half>(reinterpret_cast<half *>(y),
                     reinterpret_cast<half *>(x),
                     reinterpret_cast<half *>(cos),
                     reinterpret_cast<half *>(sin),
                     numBlocks,
                     stream);
}
