#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void cube_matmul_4buf_preload(__gm__ half* v1, __gm__ half* v2, __gm__ float* v3) {
  unsigned v4 = 3;
  unsigned v5 = 2;
  unsigned v6 = 1;
  unsigned v7 = 0;
  const int32_t v8 = 0;
  const int32_t v9 = 1;
  const int32_t v10 = 2;
  const int32_t v11 = 4;
  const int32_t v12 = 16;
  const int32_t v13 = 60;
  const int32_t v14 = 64;
  const int32_t v15 = 256;
  const int32_t v16 = 512;
  const int32_t v17 = 4096;
  const int64_t v18 = 32768;
  const int64_t v19 = 0;
  const int64_t v20 = 8192;
  const int64_t v21 = 16384;
  const int64_t v22 = 24576;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24;
  TASSIGN(v24, v19);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v20);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v21);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v22);
  Tile<TileType::Left, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v19);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v19);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v20);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v31;
  TASSIGN(v31, v21);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v32;
  TASSIGN(v32, v22);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v33;
  TASSIGN(v33, v19);
  pto::Shape<1, 1, 1, 32, 256> v34 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v35 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v15 + v7 * (unsigned) v9), v34, v35);
  pto::Shape<1, 1, 1, 16, 256> v37 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v38 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v39 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v7 * (unsigned) v17) + v7 * (unsigned) v15 + v7 * (unsigned) v9), v37, v38);
  pto::Shape<1, 1, 1, 16, 256> v40 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v41 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v42 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v6 * (unsigned) v17) + v7 * (unsigned) v15 + v7 * (unsigned) v9), v40, v41);
  pto::Shape<1, 1, 1, 16, 256> v43 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v44 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v45 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v5 * (unsigned) v17) + v7 * (unsigned) v15 + v7 * (unsigned) v9), v43, v44);
  pto::Shape<1, 1, 1, 16, 256> v46 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v47 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v48 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v4 * (unsigned) v17) + v7 * (unsigned) v15 + v7 * (unsigned) v9), v46, v47);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  TLOAD(v24, v39);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TLOAD(v25, v42);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TLOAD(v26, v45);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  TLOAD(v27, v48);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TMOV(v29, v24);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
  TMOV(v30, v25);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
  TMOV(v31, v26);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
  TMOV(v32, v27);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  for (size_t v49 = (size_t) v8; v49 < ((size_t) v14); v49 += (size_t) v9) {
    int32_t v50 = (int32_t) v49;
    pto::Shape<1, 1, 1, 32, 16> v51 = pto::Shape<1, 1, 1, 32, 16>();
    pto::Stride<512, 512, 512, 16, 1> v52 = pto::Stride<512, 512, 512, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND> v53 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND>(v1 + ((v7 + (unsigned) v50 * (unsigned) v16) + v7 * (unsigned) v12 + v7 * (unsigned) v9), v51, v52);
    int32_t v54 = (int32_t) ((uint32_t) v50 % (uint32_t) v11);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    TLOAD(v23, v53);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    pipe_barrier(PIPE_MTE1);
    TMOV(v28, v23);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    int32_t v55 = (int32_t) ((uint32_t) v50 + (uint32_t) v11);
    if (v50 < v13) {
      int32_t v56 = (int32_t) ((uint32_t) v55 % (uint32_t) v11);
      pto::Shape<1, 1, 1, 16, 256> v57 = pto::Shape<1, 1, 1, 16, 256>();
      pto::Stride<4096, 4096, 4096, 256, 1> v58 = pto::Stride<4096, 4096, 4096, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v59 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + (unsigned) v55 * (unsigned) v17) + v7 * (unsigned) v15 + v7 * (unsigned) v9), v57, v58);
      if (v56 == v8) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
        TLOAD(v24, v59);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID5);
        TMOV(v29, v24);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
      } else {
        if (v56 == v9) {
          wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
          TLOAD(v25, v59);
          set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
          wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
          TMOV(v30, v25);
          set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        } else {
          if (v56 == v10) {
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            TLOAD(v26, v59);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
            TMOV(v31, v26);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
          } else {
            TLOAD(v27, v59);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            TMOV(v32, v27);
          };
        };
      };
    };
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    if (v50 == v8) {
      TMATMUL(v33, v28, v29);
    } else {
      if (v54 == v8) {
        TMATMUL_ACC(v33, v33, v28, v29);
      } else {
        if (v54 == v9) {
          TMATMUL_ACC(v33, v33, v28, v30);
        } else {
          if (v54 == v10) {
            TMATMUL_ACC(v33, v33, v28, v31);
          } else {
            TMATMUL_ACC(v33, v33, v28, v32);
          };
        };
      };
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v36, v33);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}

// Launcher template for test harness
template <typename T_A, typename T_B, typename T_C>
void LaunchCubeMatmul4BufPreload(T_A *a, T_B *b, T_C *c, void *stream)
{
    cube_matmul_4buf_preload(a, b, c);
}

// Explicit instantiation
template void LaunchCubeMatmul4BufPreload<half, half, float>(half *, half *, float *, void *);

