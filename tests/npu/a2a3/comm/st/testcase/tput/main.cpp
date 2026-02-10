// 验证通过 PTO TPut（Shmem 后端）进行环形互传：从前一 rank 拉取数据
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

// 声明在 tput_kernel.cpp 中实现的设备侧逻辑（返回执行是否成功）
// 1D Vector Tile tests
template <typename T, size_t count>
bool RunPutRing(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Matrix Tile tests
template <typename T, size_t rows, size_t cols>
bool RunPutRing2D(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// AtomicAdd tests
template <typename T, size_t count>
bool RunPutAtomicAdd(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Large Shape Chunked tests (GlobalTensor > UB tile, auto-chunked by TPUT_IMPL)
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunPutRingLargeShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Multi-Dimensional Chunked tests (outer dims > 1, auto-chunked by TPUT_IMPL)
template <typename T, size_t d0, size_t d1, size_t d2, size_t d3, size_t cols, size_t tile_rows>
bool RunPutRingMultiDim(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Irregular Shape Chunked tests (total_rows % tile_rows != 0, partial last chunk)
template <typename T, size_t total_rows, size_t cols, size_t tile_rows>
bool RunPutRingIrregularShape(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// 2D Sliding tests (both rows and cols exceed tile, auto 2D chunking)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunPutRing2DSliding(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// Ping-Pong tests (double buffering: TPUT with two staging tiles)
template <typename T, size_t total_rows, size_t total_cols, size_t tile_rows, size_t tile_cols>
bool RunPutRingPingPong(int n_ranks, int n_devices, int first_rank_id, int first_device_id);

// ============================================================================
// 1D Vector Tile Tests
// ============================================================================
TEST(TPut, Vec_FloatSmall)
{
    ASSERT_TRUE((RunPutRing<float, 256>(4, 4, 0, 0)));
}
TEST(TPut, Vec_Int32Large)
{
    ASSERT_TRUE((RunPutRing<int32_t, 4096>(2, 2, 0, 0)));
}
TEST(TPut, Vec_Uint8Small)
{
    ASSERT_TRUE((RunPutRing<uint8_t, 512>(8, 8, 0, 0)));
}

// ============================================================================
// 2D Shape Tests (GlobalTensor with 2D shape, using Vec Tile for transfer)
// ============================================================================
TEST(TPut, Shape2D_Float16x16)
{
    ASSERT_TRUE((RunPutRing2D<float, 16, 16>(2, 2, 0, 0)));
}
TEST(TPut, Shape2D_Float8x32)
{
    ASSERT_TRUE((RunPutRing2D<float, 8, 32>(2, 2, 0, 0)));
}
TEST(TPut, Shape2D_Int32_4x64)
{
    ASSERT_TRUE((RunPutRing2D<int32_t, 4, 64>(2, 2, 0, 0)));
}

// ============================================================================
// AtomicAdd Tests
// ============================================================================
TEST(TPut, AtomicAdd_Int32)
{
    ASSERT_TRUE((RunPutAtomicAdd<int32_t, 256>(4, 4, 0, 0)));
}

// ============================================================================
// Large Shape Chunked Tests
// GlobalTensor shape exceeds UB tile capacity, TPUT_IMPL auto-chunks the transfer
// ============================================================================
// float: 128 rows x 64 cols (8192 elems), tile 16 rows → 8 chunks
TEST(TPut, LargeShape_Float_128x64_tile16)
{
    ASSERT_TRUE((RunPutRingLargeShape<float, 128, 64, 16>(2, 2, 0, 0)));
}
// int32: 256 rows x 32 cols (8192 elems), tile 32 rows → 8 chunks
TEST(TPut, LargeShape_Int32_256x32_tile32)
{
    ASSERT_TRUE((RunPutRingLargeShape<int32_t, 256, 32, 32>(2, 2, 0, 0)));
}
// float: 512 rows x 32 cols (16384 elems), tile 64 rows → 8 chunks
TEST(TPut, LargeShape_Float_512x32_tile64)
{
    ASSERT_TRUE((RunPutRingLargeShape<float, 512, 32, 64>(2, 2, 0, 0)));
}
// float: 2048 rows x 32 cols (65536 elems = 256KB), tile 64 rows → 32 chunks
TEST(TPut, LargeShape_Float_2048x32_tile64)
{
    ASSERT_TRUE((RunPutRingLargeShape<float, 2048, 32, 64>(2, 2, 0, 0)));
}
// float: 4096 rows x 32 cols (131072 elems = 512KB), tile 64 rows → 64 chunks
TEST(TPut, LargeShape_Float_4096x32_tile64)
{
    ASSERT_TRUE((RunPutRingLargeShape<float, 4096, 32, 64>(2, 2, 0, 0)));
}
// float: 2048 rows x 64 cols (131072 elems = 512KB), tile 128 rows → 16 chunks
TEST(TPut, LargeShape_Float_2048x64_tile128)
{
    ASSERT_TRUE((RunPutRingLargeShape<float, 2048, 64, 128>(2, 2, 0, 0)));
}
// int32: 4096 rows x 64 cols (262144 elems = 1MB), tile 128 rows → 32 chunks
TEST(TPut, LargeShape_Int32_4096x64_tile128)
{
    ASSERT_TRUE((RunPutRingLargeShape<int32_t, 4096, 64, 128>(2, 2, 0, 0)));
}

// ============================================================================
// Multi-Dimensional Chunked Tests
// GlobalTensor has outer dims > 1, TPUT_IMPL iterates outer dims + chunks dim3
// ============================================================================
// float: (2,2,1,32,32)=4096 elems, tile 16 rows → 4 outer iters × 2 inner chunks
TEST(TPut, MultiDim_Float_2x2x1x32x32_tile16)
{
    ASSERT_TRUE((RunPutRingMultiDim<float, 2, 2, 1, 32, 32, 16>(2, 2, 0, 0)));
}
// int32: (4,1,1,32,64)=8192 elems, tile 16 rows → 4 outer iters × 2 inner chunks
TEST(TPut, MultiDim_Int32_4x1x1x32x64_tile16)
{
    ASSERT_TRUE((RunPutRingMultiDim<int32_t, 4, 1, 1, 32, 64, 16>(2, 2, 0, 0)));
}

// ============================================================================
// Irregular Shape Chunked Tests (Non-power-of-2, non-divisible by tile rows)
// Tests partial last chunk path in TPUT_IMPL via DYNAMIC ValidRow tiles
// ============================================================================
// float: 2047 rows x 32 cols, tile 64 → 31 full chunks + 1 partial (63 rows)
TEST(TPut, Irregular_Float_2047x32_tile64)
{
    ASSERT_TRUE((RunPutRingIrregularShape<float, 2047, 32, 64>(2, 2, 0, 0)));
}
// int32: 1025 rows x 32 cols, tile 64 → 16 full chunks + 1 partial (1 row)
TEST(TPut, Irregular_Int32_1025x32_tile64)
{
    ASSERT_TRUE((RunPutRingIrregularShape<int32_t, 1025, 32, 64>(2, 2, 0, 0)));
}
// float: 4095 rows x 32 cols, tile 128 → 31 full chunks + 1 partial (127 rows)
TEST(TPut, Irregular_Float_4095x32_tile128)
{
    ASSERT_TRUE((RunPutRingIrregularShape<float, 4095, 32, 128>(2, 2, 0, 0)));
}

// ============================================================================
// 2D Sliding Tests (both rows and cols exceed tile, TPUT_IMPL auto 2D chunking)
// ============================================================================
// ---- Regular 2D sliding ----
// float: 64x128, tile 16x32 → 4×4 = 16 chunks
TEST(TPut, Sliding2D_Float_64x128_tile16x32)
{
    ASSERT_TRUE((RunPutRing2DSliding<float, 64, 128, 16, 32>(2, 2, 0, 0)));
}
// int32: 128x256, tile 32x64 → 4×4 = 16 chunks
TEST(TPut, Sliding2D_Int32_128x256_tile32x64)
{
    ASSERT_TRUE((RunPutRing2DSliding<int32_t, 128, 256, 32, 64>(2, 2, 0, 0)));
}
// float: 256x512, tile 64x128 → 4×4 = 16 chunks (large data)
TEST(TPut, Sliding2D_Float_256x512_tile64x128)
{
    ASSERT_TRUE((RunPutRing2DSliding<float, 256, 512, 64, 128>(2, 2, 0, 0)));
}
// ---- Irregular 2D sliding (partial chunks) ----
// float: 65x64, tile 16x32 → row 4+1, col 2 (irregular rows only)
TEST(TPut, Sliding2D_IrregRow_Float_65x64_tile16x32)
{
    ASSERT_TRUE((RunPutRing2DSliding<float, 65, 64, 16, 32>(2, 2, 0, 0)));
}
// float: 64x104, tile 16x32 → row 4, col 3+1(8) (irregular cols only)
TEST(TPut, Sliding2D_IrregCol_Float_64x104_tile16x32)
{
    ASSERT_TRUE((RunPutRing2DSliding<float, 64, 104, 16, 32>(2, 2, 0, 0)));
}
// float: 65x104, tile 16x32 → row 4+1(1), col 3+1(8) (both irregular)
TEST(TPut, Sliding2D_IrregBoth_Float_65x104_tile16x32)
{
    ASSERT_TRUE((RunPutRing2DSliding<float, 65, 104, 16, 32>(2, 2, 0, 0)));
}

// ============================================================================
// Ping-Pong Double Buffering Tests
// TPUT with two staging tiles: overlap TLOAD(chunk N+1) with TSTORE(chunk N)
// ============================================================================
// Regular: float 128x128, tile 16x32 → 32 chunks with MTE2/MTE3 overlap
TEST(TPut, PingPong_Float_128x128_tile16x32)
{
    ASSERT_TRUE((RunPutRingPingPong<float, 128, 128, 16, 32>(2, 2, 0, 0)));
}
// Regular: int32 256x256, tile 32x64 → 32 chunks
TEST(TPut, PingPong_Int32_256x256_tile32x64)
{
    ASSERT_TRUE((RunPutRingPingPong<int32_t, 256, 256, 32, 64>(2, 2, 0, 0)));
}
// Irregular: float 65x104, tile 16x32 → 20 chunks with partial rows+cols
TEST(TPut, PingPong_Irregular_Float_65x104_tile16x32)
{
    ASSERT_TRUE((RunPutRingPingPong<float, 65, 104, 16, 32>(2, 2, 0, 0)));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
