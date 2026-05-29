// --------------------------------------------------------------------------------
// gate_softmax — DeepSeek-V4 router score activation (auto-mode A3, vector).
//
// One of three variants is selected at compile time:
//   default                  : scores = sqrt(softplus(scores))    [ModelArgs default]
//   -DSCORE_FUNC_SOFTMAX     : scores = softmax(scores, dim=-1)
//   -DSCORE_FUNC_SIGMOID     : scores = sigmoid(scores)
//
// Reference: deepseek/model.py:567-572.
//
// DESIGN / SKELETON ONLY — pseudocode body below. Real implementation is
// deferred. Do not claim compile or runtime success until run.sh produces
// `test data success` on the compiler server.
//
// Pattern sources:
//   - MoE/gather/gather_kernel.cpp                (FA softmax composition)
//   - tests/.../tfa/pto_macro_fa_softmax.hpp      (TROWMAX → ... → TROWEXPANDDIV)
//   - add_tile_array/add_tile_array_kernel.cpp    (pure elementwise tile baseline)
// --------------------------------------------------------------------------------

#include <cstdint>
#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include "generated_cases.h"

using namespace pto;

namespace gate_softmax_cfg {
constexpr unsigned kT        = kDsmoeT;
constexpr unsigned kNRouted  = kDsmoeNRouted;
}  // namespace gate_softmax_cfg

extern "C" __global__ __aicore__ void gate_softmax_kernel(
    __gm__ uint8_t *scores_out,   // FP32, (T, N_ROUTED) row-major — out
    __gm__ uint8_t *scores_in)    // FP32, (T, N_ROUTED) row-major — in
{
    using namespace gate_softmax_cfg;
    // ----------------------------------------------------------------------
    // PSEUDOCODE — auto-mode A3 vector activation.
    //
    //   GlobalTensor<float> gIn (scores_in,  {kT, kNRouted});
    //   GlobalTensor<float> gOut(scores_out, {kT, kNRouted});
    //
    //   Tile<float, MemUB, Shape{kT, kNRouted}> inTile;
    //   Tile<float, MemUB, Shape{kT, kNRouted}> tmpTile;
    //   Tile<float, MemUB, Shape{kT, kNRouted}> outTile;
    //
    //   TLOAD(inTile, gIn);
    //
    // #if defined(SCORE_FUNC_SOFTMAX)
    //   // Row-softmax over N_ROUTED — see MoE/gather/gather_kernel.cpp.
    //   Tile<float, MemUB, Shape{kT, 1}, BLayout::ColMajor> maxTile, sumTile;
    //   TROWMAX       (maxTile, inTile, tmpTile);
    //   TROWEXPANDSUB (tmpTile, inTile, maxTile);
    //   TEXP          (tmpTile, tmpTile);
    //   TROWSUM       (sumTile, tmpTile, /*scratch=*/ outTile);
    //   TROWEXPANDDIV (outTile, tmpTile, sumTile);
    //
    // #elif defined(SCORE_FUNC_SIGMOID)
    //   // Elementwise sigmoid: sigmoid(x) = 1 / (1 + exp(-x)).
    //   TNEG (tmpTile, inTile);             // -x
    //   TEXP (tmpTile, tmpTile);            // exp(-x)
    //   TADDS(tmpTile, tmpTile, 1.0f);      // 1 + exp(-x)
    //   TRCP (outTile, tmpTile);            // 1 / (1 + exp(-x))
    //
    // #else
    //   // Default: sqrt(softplus(x)) = sqrt(log(1 + exp(x))).
    //   // Stable form for large |x|:
    //   //   softplus(x) = max(0, x) + log(1 + exp(-|x|))
    //   // For the prototype, use the direct form; revisit if logits exceed
    //   // ~80 (FP32 exp overflow boundary).
    //   TEXP (tmpTile, inTile);             // exp(x)
    //   TADDS(tmpTile, tmpTile, 1.0f);      // 1 + exp(x)
    //   TLN  (tmpTile, tmpTile);            // softplus(x) = log(1 + exp(x))
    //   TSQRT(outTile, tmpTile);            // sqrt(softplus(x))
    // #endif
    //
    //   TSTORE(gOut, outTile);
    // ----------------------------------------------------------------------
    // TODO(deepseekv4): replace pseudocode above with real auto-mode body.
    // Anchor on add_tile_array_kernel.cpp for the basic vector-tile shape
    // and gather_kernel.cpp for the softmax composition.
    (void)scores_out; (void)scores_in;
}

// Host launcher (kept minimal).
void launch_gate_softmax(uint8_t *scores_out, uint8_t *scores_in,
                         uint64_t T, uint64_t N, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    // TODO(deepseekv4): wire ICache, AICORE config, then launch gate_softmax_kernel.
    (void)scores_out; (void)scores_in; (void)T; (void)N; (void)stream;
}
