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

using namespace pto;

namespace mla_basic_cfg_vec {

// Mirror of mla_basic_cube_kernel.cpp's namespace. The two TUs compile
// independently so we cannot share a header without adding plumbing.
constexpr unsigned kBatch    = 1;
constexpr unsigned kSeqLen   = 128;
constexpr unsigned kHidden   = 4096;
constexpr unsigned kNumHeads = 32;
constexpr unsigned kHeadDim  = 128;
constexpr unsigned kLatent   = 64;
constexpr unsigned kRopeDim  = 64;
constexpr unsigned kRopeHalf = kRopeDim / 2;  // 32

}  // namespace mla_basic_cfg_vec

// =============================================================================
// Stage 3 -- KV cache store: C_cache = C_kv  (prefill-only v1, no append).
//
// Pure tile copy: TLOAD a Vec tile from C_kv, TSTORE it to C_cache. One
// iteration since [kSeqLen, kLatent] = [128, 64] FP16 = 16 KB fits in a
// single tile. Same shape as §A11 add_tile_array minus the TADD.
// =============================================================================
template <typename T>
__global__ AICORE void runKVCacheStore(__gm__ T *c_cache, __gm__ T *c_kv)
{
    using namespace mla_basic_cfg_vec;

    using TileShape  = Shape <1, 1, 1, kSeqLen, kLatent>;
    using TileStride = Stride<1, 1, 1, kLatent, 1>;
    using GlobalData = GlobalTensor<T, TileShape, TileStride>;

    using TileData = Tile<TileType::Vec, T,
                          kSeqLen, kLatent,
                          BLayout::RowMajor,
                          kSeqLen, kLatent>;

    TileData buf;

    GlobalData srcGlobal(c_kv);
    GlobalData dstGlobal(c_cache);

    TLOAD (buf, srcGlobal);
    TSTORE(dstGlobal, buf);
}

// =============================================================================
// Stage 5g (NEW) -- RoPE half-rotation.
//
//   For each block b in [0, numBlocks):
//     for s in [0, S):
//       x1 = x[b, s, :half],  x2 = x[b, s, half:]
//       y[b, s, :half] = x1 * cos[s] - x2 * sin[s]
//       y[b, s, half:] = x1 * sin[s] + x2 * cos[s]
//
//   x, y     : [numBlocks, S, kRopeDim]   half
//   cos, sin : [S, kRopeHalf]             half  (shared, host-precomputed)
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

    // Half-tile shape: [S, kRopeHalf] = [128, 32]. For FP16: 32 * 2 = 64
    // bytes per row -> 32-byte aligned. OK.
    using HalfShape       = Shape <1, 1, 1, kSeqLen, kRopeHalf>;
    using HalfStrideX     = Stride<1, 1, 1, kRopeDim, 1>;   // x has row stride kRopeDim
    using HalfStrideTable = Stride<1, 1, 1, kRopeHalf, 1>;  // cos/sin tightly packed

    using GlobalDataHalfX     = GlobalTensor<T, HalfShape, HalfStrideX>;
    using GlobalDataHalfTable = GlobalTensor<T, HalfShape, HalfStrideTable>;

    using HalfTile = Tile<TileType::Vec, T,
                          kSeqLen, kRopeHalf,
                          BLayout::RowMajor,
                          kSeqLen, kRopeHalf>;

    HalfTile x1Tile;      // x[b, :, :half]
    HalfTile x2Tile;      // x[b, :, half:]
    HalfTile cosTile;     // cos[:, :]
    HalfTile sinTile;     // sin[:, :]
    HalfTile productTile; // scratch for one TMUL result
    HalfTile y1Tile;      // y[b, :, :half] before TSTORE
    HalfTile y2Tile;      // y[b, :, half:] before TSTORE

    // cos/sin are block-invariant. Load once before the loop.
    GlobalDataHalfTable cosGlobal(cos);
    GlobalDataHalfTable sinGlobal(sin);
    TLOAD(cosTile, cosGlobal);
    TLOAD(sinTile, sinGlobal);

    for (unsigned b = 0; b < numBlocks; ++b) {
        const size_t blockOffset = static_cast<size_t>(b) * kSeqLen * kRopeDim;

        GlobalDataHalfX x1Global(x + blockOffset);                 // first half
        GlobalDataHalfX x2Global(x + blockOffset + kRopeHalf);     // second half
        GlobalDataHalfX y1Global(y + blockOffset);
        GlobalDataHalfX y2Global(y + blockOffset + kRopeHalf);

        TLOAD(x1Tile, x1Global);
        TLOAD(x2Tile, x2Global);

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
//   For each head h in [0, kNumHeads):
//     scoresSum[h] = scores_nope[h] + scores_rope[h]
//     probs[h]     = softmax(scale * scoresSum[h])
//
//   scores_nope, scores_rope : [kNumHeads, kSeqLen, kSeqLen]  half
//   probs                    : [kNumHeads, kSeqLen, kSeqLen]  half
//   scale                    : 1/sqrt(kHeadDim + kRopeDim), supplied by host.
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

    using TileShape  = Shape <1, 1, 1, kSeqLen, kSeqLen>;
    using TileStride = Stride<1, 1, 1, kSeqLen, 1>;
    using GlobalData = GlobalTensor<T, TileShape, TileStride>;

    using ScoresTile = Tile<TileType::Vec, T,
                            kSeqLen, kSeqLen,
                            BLayout::RowMajor,
                            kSeqLen, kSeqLen>;

    // Row-reduction output tile: storage width 16 cols (32-byte aligned for
    // FP16 per Tile constexpr asserts), valid width 1 col. Pattern from
    // tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp:30.
    using RowReduceTile = Tile<TileType::Vec, T,
                               kSeqLen, 16,
                               BLayout::RowMajor,
                               -1, -1>;

    ScoresTile     scoresTile;     // role 1: nope+rope combined; role 2: shifted; role 3: probs
    ScoresTile     scoresRopeTile; // staging for the rope term before TADD
    ScoresTile     scaledTile;     // role 1: scaled scores;  role 2: exp
    ScoresTile     broadcastTile;  // role 1: broadcast(max); role 2: broadcast(sum)
    ScoresTile     tmpTile;        // TROWMAX / TROWSUM scratch (shape == src)
    RowReduceTile  rowMaxTile(kSeqLen, 1);
    RowReduceTile  rowSumTile(kSeqLen, 1);

    for (unsigned h = 0; h < kNumHeads; ++h) {
        const size_t headOffset = static_cast<size_t>(h) * kSeqLen * kSeqLen;

        GlobalData srcNopeGlobal(scores_nope + headOffset);
        GlobalData srcRopeGlobal(scores_rope + headOffset);
        GlobalData dstGlobal    (probs       + headOffset);

        TLOAD(scoresTile,     srcNopeGlobal);
        TLOAD(scoresRopeTile, srcRopeGlobal);
        TADD (scoresTile, scoresTile, scoresRopeTile);  // combined scores

        // 1. Scale by 1/sqrt(headDim).
        TMULS(scaledTile, scoresTile, scale);

        // 2. Row max (numerical stability).
        TROWMAX(rowMaxTile, scaledTile, tmpTile);

        // 3. Broadcast row max to [S, S].
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
    runKVCacheStore<T><<<1, nullptr, stream>>>(c_cache, c_kv);
}

template <typename T>
void launchAttnSoftmax(T *probs, T *scores_nope, T *scores_rope, T scale, void *stream)
{
    runAttnSoftmax<T><<<1, nullptr, stream>>>(probs, scores_nope, scores_rope, scale);
}

template <typename T>
void launchRoPE(T *y, T *x, T *cos, T *sin, unsigned numBlocks, void *stream)
{
    runRoPE<T><<<1, nullptr, stream>>>(y, x, cos, sin, numBlocks);
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
