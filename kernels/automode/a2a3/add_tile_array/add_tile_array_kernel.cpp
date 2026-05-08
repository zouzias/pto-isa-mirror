/**
 * add_tile_array_kernel.cpp - auto-mode A3 prototype kernel.
 *
 * Element-wise C = A + B on a 2-D float32 array (NUM_TILES * TILE_ROWS, TILE_COLS),
 * processed as a serial loop of TILE_ROWS x TILE_COLS tiles by a single AICORE.
 *
 * Pattern source:
 *   - demos/auto_mode/baseline/add/csrc/kernel/add_custom.cpp  (auto-mode add demo)
 *   - tests/npu/a2a3/src/st/testcase/tadd/tadd_kernel.cpp      (closest dual-mode tadd ST)
 * See docs_for_ai/known_good_kernel_examples.md §A1 and §A2.
 *
 * Build: this file is compiled with --cce-aicore-arch=dav-c220-vec
 *        and --cce-enable-pto-passes -O2 (auto mode), per CMakeLists.txt.
 *
 * Auto-mode constraints honored (correctness-first v1):
 *   - Single AICORE invocation (<<<1, nullptr, stream>>>); no block_idx work
 *     distribution yet. Multi-core split is a follow-up.
 *   - No double-buffering, no TPipe / TPUSH / TPOP, no manual ping-pong.
 *   - No set_flag / wait_flag / pipe_barrier; no Event<>.
 *   - No TASSIGN aliasing tricks; tile addresses are pinned by the auto allocator.
 *   - No Tile::data() in kernel code; no *_IMPL calls; no raw CCE intrinsics.
 *   - Tail handling: total length must be a multiple of TILE_ROWS*TILE_COLS;
 *     enforced by the constexpr layout. Non-aligned shapes are a follow-up.
 */

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>

using namespace pto;

namespace add_tile_array_cfg {

// Static configuration. Total array shape: (NUM_TILES * TILE_ROWS, TILE_COLS).
// Total elements: 4 * 64 * 64 = 16384. Must match scripts/gen_data.py / main.cpp.
constexpr unsigned TILE_ROWS  = 64;
constexpr unsigned TILE_COLS  = 64;
constexpr unsigned NUM_TILES  = 4;
constexpr unsigned TOTAL_ROWS = NUM_TILES * TILE_ROWS;  // 256

}  // namespace add_tile_array_cfg

template <typename T>
__global__ AICORE void runAddTileArray(__gm__ T __out__ *c,
                                       __gm__ T __in__  *a,
                                       __gm__ T __in__  *b)
{
    using namespace add_tile_array_cfg;

    // GM tensor view: each tile is (TILE_ROWS, TILE_COLS) with row stride
    // TILE_COLS (the parent array is laid out row-major, contiguous).
    using TileShape  = Shape <1, 1, 1, TILE_ROWS, TILE_COLS>;
    using TileStride = Stride<1, 1, 1, TILE_COLS, 1>;
    using GlobalData = GlobalTensor<T, TileShape, TileStride>;

    // Static valid region (no DYNAMIC) - the simplest correct shape.
    using TileData = Tile<TileType::Vec, T,
                          TILE_ROWS, TILE_COLS,
                          BLayout::RowMajor,
                          TILE_ROWS, TILE_COLS>;

    TileData aTile;
    TileData bTile;
    TileData cTile;

    // Auto mode handles MTE/V pipeline sync; no manual flags here. Tile
    // addresses are pinned by the auto allocator across iterations
    // (per docs/auto_mode/Kernel_Developer_Rules_And_Limitations.md §2.3).
    for (unsigned i = 0; i < NUM_TILES; ++i) {
        const unsigned offset = i * TILE_ROWS * TILE_COLS;
        GlobalData aGlobal(a + offset);
        GlobalData bGlobal(b + offset);
        GlobalData cGlobal(c + offset);

        TLOAD(aTile, aGlobal);
        TLOAD(bTile, bGlobal);
        TADD (cTile, aTile, bTile);
        TSTORE(cGlobal, cTile);
    }
}

template <typename T>
void launchAddTileArray(T *c, T *a, T *b, void *stream)
{
    runAddTileArray<T><<<1, nullptr, stream>>>(c, a, b);
}

template void launchAddTileArray<float>(float *c, float *a, float *b, void *stream);
