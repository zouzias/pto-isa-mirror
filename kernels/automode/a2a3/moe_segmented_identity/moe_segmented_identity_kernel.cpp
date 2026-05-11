/**
 * moe_segmented_identity_kernel.cpp - auto-mode A3 prototype.
 *
 * First kernel that walks dynamic expert segments described by
 * (expert_start[e], expert_count[e]) and runs a fixed TILE_M = 128 microtile
 * inside each segment. The microtile op is the simplest auto-mode-safe
 * elementwise instruction (TADDS, dst = src + 1.0f) — this prototype proves
 * the per-expert outer / per-microtile inner loop shape that the future
 * expert FFN (GEMM1 → activation → GEMM2) will reuse.
 *
 * v1 tail policy:
 *   Host pads every expert segment length up to a multiple of TILE_M
 *   (scripts/gen_data.py). Padded rows are initialised to 0.0 and the
 *   golden adds 1.0 to them too — so the kernel processes padded rows
 *   identically to real rows. This sidesteps SetValidRow / partial-tile
 *   stores entirely (still an open Assumption — see
 *   docs_for_ai/tile_type_reference.md §6 and §11 item 12).
 *
 * Semantics (host-visible):
 *   for (e = 0; e < kNumExperts; ++e) {
 *       start = expert_start[e];      // padded values
 *       count = expert_count[e];      // padded; multiple of kTileM
 *       for (m0 = 0; m0 < count; m0 += kTileM) {
 *           row = start + m0;
 *           packed_output[row .. row+kTileM, :] =
 *               packed_tokens[row .. row+kTileM, :] + 1.0f;
 *       }
 *   }
 *
 * Inputs   (GM): packed_tokens [T_PADDED, H] float32,
 *                expert_count  [kNumExperts] int32 (PADDED counts),
 *                expert_start  [kNumExperts] int32 (PADDED starts).
 * Outputs  (GM): packed_output [T_PADDED, H] float32.
 *
 * Auto-mode constraints honored:
 *   - Single AICORE (<<<1, nullptr, stream>>>); no block_idx work split.
 *   - Static segment tile (Tile<Vec, float, kTileM, kH, RowMajor, kTileM, kH>)
 *     declared once, reused across all inner iterations — auto allocator
 *     pins its UB address.
 *   - GlobalTensor reconstructed per inner iter with (base + runtime offset).
 *   - TADDS is the user-facing public wrapper at
 *     include/pto/common/pto_instr.hpp:1517-1524 (signature
 *     `TADDS(dst, src, scalar)`); the test kernel
 *     tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp uses the same
 *     three-argument form and is in `ALL_TESTCASES` (auto-mode build list).
 *   - No TASSIGN aliasing; no Tile::data() in kernel; no *_IMPL calls;
 *     no raw CCE intrinsics; no Event<>; no manual sync; no TPipe / TPUSH /
 *     TPOP; no double buffering; no SetValidRow / SetValidShape; no
 *     partial-tile stores; no A5-only instructions.
 *
 * Patterns reused (all already user-confirmed):
 *   - scalar GM read of int32_t metadata        — §11.4 (moe_top1_permute)
 *   - runtime scalar GM row offset              — §11.4 / §11.5 (permute/unpermute)
 *   - single reused static Vec tile             — §A11 (add_tile_array)
 *   - TLOAD → elementwise op → TSTORE, no sync  — §A1 / §A2 / §A11
 *
 * NEW (not yet confirmed by experiment; this kernel's verification target):
 *   - Nested loop: outer over experts (kNumExperts), inner over microtiles
 *     within an expert's padded segment, with both bounds coming from
 *     scalar GM reads of int32 metadata.
 *   - Two 128 x 64 float static Vec tiles (srcTile, dstTile) — 64 KB total
 *     UB allocation, well below A3's UB budget. Previously this kernel used
 *     a single in-place tile (`TADDS(t, t, 1.0f)`); that form produced a
 *     zero-filled region near flat indices ~0x1088..0x1132 on the first run
 *     and has been replaced with the separate-tile form below. Whether the
 *     in-place form is fundamentally unsafe in A3 auto mode or whether it
 *     was a tooling/codegen quirk is **Unknown** — do not reintroduce it
 *     without a separate confirmed experiment.
 */

#include <pto/common/constants.hpp>
#include <pto/pto-inst.hpp>

using namespace pto;

namespace moe_segmented_identity_cfg {

// Static configuration. Must match scripts/gen_data.py and main.cpp.
constexpr unsigned kH          = 64;        // hidden dim per token row
constexpr unsigned kTileM      = 128;       // microtile rows per inner iter
constexpr unsigned kNumExperts = 4;         // number of experts (compile-time)

}  // namespace moe_segmented_identity_cfg

template <typename T>
__global__ AICORE void runMoeSegmentedIdentity(__gm__ T       __out__ *packed_output,
                                                __gm__ T       __in__  *packed_tokens,
                                                __gm__ int32_t __in__  *expert_count,
                                                __gm__ int32_t __in__  *expert_start)
{
    using namespace moe_segmented_identity_cfg;

    using SegShape  = Shape <1, 1, 1, kTileM, kH>;
    using SegStride = Stride<1, 1, 1, kH,     1>;
    using SegGlobal = GlobalTensor<T, SegShape, SegStride>;

    // 128 x 64 float = 32 KB per tile; two tiles = 64 KB UB total. The
    // tadds ST kernel (tests/npu/a2a3/src/st/testcase/tadds/tadds_kernel.cpp)
    // also uses separate src/dst tiles for TADDS — we mirror that confirmed
    // shape here. Static valid region.
    using SegTile = Tile<TileType::Vec, T,
                         kTileM, kH,
                         BLayout::RowMajor,
                         kTileM, kH>;

    SegTile srcTile;
    SegTile dstTile;

    for (unsigned e = 0; e < kNumExperts; ++e) {
        int32_t start = expert_start[e];      // GM scalar read (resolved §11.4)
        int32_t count = expert_count[e];      // GM scalar read (resolved §11.4)

        for (int32_t m0 = 0; m0 < count; m0 += static_cast<int32_t>(kTileM)) {
            size_t row = static_cast<size_t>(start) + static_cast<size_t>(m0);
            size_t off = row * kH;

            SegGlobal srcGlobal(packed_tokens + off);
            SegGlobal dstGlobal(packed_output + off);

            TLOAD (srcTile, srcGlobal);
            // TADDS<TileDataDst, TileDataSrc>(dst, src, scalar) — public PTO
            // wrapper at include/pto/common/pto_instr.hpp:1517-1524. dst and
            // src are distinct tiles (matches tadds_kernel.cpp's shape).
            TADDS(dstTile, srcTile, static_cast<T>(1.0f));
            TSTORE(dstGlobal, dstTile);
        }
    }
}

template <typename T>
void launchMoeSegmentedIdentity(T *packed_output, T *packed_tokens,
                                int32_t *expert_count, int32_t *expert_start,
                                void *stream)
{
    runMoeSegmentedIdentity<T><<<1, nullptr, stream>>>(
        packed_output, packed_tokens, expert_count, expert_start);
}

template void launchMoeSegmentedIdentity<float>(float *packed_output,
                                                float *packed_tokens,
                                                int32_t *expert_count,
                                                int32_t *expert_start,
                                                void *stream);
