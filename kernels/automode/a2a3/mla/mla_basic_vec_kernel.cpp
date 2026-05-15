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
// Stage 5b -- Attention softmax:
//   For each head h in [0, kNumHeads):
//     probs[h] = softmax(scale * scores[h])
//
//   scores : [kNumHeads, kSeqLen, kSeqLen]  half
//   probs  : [kNumHeads, kSeqLen, kSeqLen]  half
//   scale  : 1/sqrt(kHeadDim), supplied by the host as a half.
//
// One `[kSeqLen, kSeqLen] = [128, 128]` Vec tile per buffer; per-head loop
// reuses all tiles (auto-allocator pins the addresses once, per the §A11 /
// §A15 patterns).
// =============================================================================
template <typename T>
__global__ AICORE void runAttnSoftmax(__gm__ T *probs, __gm__ T *scores, T scale)
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

    ScoresTile     scoresTile;     // role 1: loaded scores;  role 2: shifted; role 3: probs
    ScoresTile     scaledTile;     // role 1: scaled scores;  role 2: exp
    ScoresTile     broadcastTile;  // role 1: broadcast(max); role 2: broadcast(sum)
    ScoresTile     tmpTile;        // TROWMAX / TROWSUM scratch (shape == src)
    RowReduceTile  rowMaxTile(kSeqLen, 1);
    RowReduceTile  rowSumTile(kSeqLen, 1);

    for (unsigned h = 0; h < kNumHeads; ++h) {
        const size_t headOffset = static_cast<size_t>(h) * kSeqLen * kSeqLen;

        GlobalData srcGlobal(scores + headOffset);
        GlobalData dstGlobal(probs  + headOffset);

        TLOAD(scoresTile, srcGlobal);

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
void launchAttnSoftmax(T *probs, T *scores, T scale, void *stream)
{
    runAttnSoftmax<T><<<1, nullptr, stream>>>(probs, scores, scale);
}

template void launchKVCacheStore<half>(half *, half *, void *);
template void launchAttnSoftmax <half>(half *, half *, half, void *);

// Host wrappers (host TU compiled `-xc++` cannot see `half`).
// We take the scalar `scale` as a `float` from the host and cast on entry.
extern "C" void launchMlaKVCacheStoreFp16(uint8_t *c_cache, uint8_t *c_kv, void *stream)
{
    launchKVCacheStore<half>(reinterpret_cast<half *>(c_cache),
                             reinterpret_cast<half *>(c_kv),
                             stream);
}

extern "C" void launchMlaAttnSoftmaxFp16(uint8_t *probs, uint8_t *scores,
                                         float scale, void *stream)
{
    launchAttnSoftmax<half>(reinterpret_cast<half *>(probs),
                            reinterpret_cast<half *>(scores),
                            static_cast<half>(scale),
                            stream);
}
