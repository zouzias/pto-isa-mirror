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

__global__ AICORE void cube_matmul_4buf(__gm__ half* v1, __gm__ half* v2, __gm__ float* v3) {
  unsigned v4 = 0;
  const int32_t v5 = 0;
  const int32_t v6 = 1;
  const int32_t v7 = 4;
  const int32_t v8 = 16;
  const int32_t v9 = 64;
  const int32_t v10 = 256;
  const int32_t v11 = 512;
  const int32_t v12 = 4096;
  const int64_t v13 = 0;
  const int64_t v14 = 1024;
  const int64_t v15 = 9216;
  const int64_t v16 = 17408;
  const int64_t v17 = 8192;
  const int64_t v18 = 16384;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v19;
  TASSIGN(v19, v13);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v20;
  TASSIGN(v20, v14);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v21;
  TASSIGN(v21, v15);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v22;
  TASSIGN(v22, v16);
  Tile<TileType::Left, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v23;
  TASSIGN(v23, v13);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v24;
  TASSIGN(v24, v13);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v17);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v18);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v13);
  pto::Shape<1, 1, 1, 32, 256> v28 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v29 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v30 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v4 + v4 * (unsigned) v10 + v4 * (unsigned) v6), v28, v29);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  for (size_t v31 = (size_t) v5; v31 < ((size_t) v9); v31 += (size_t) v6) {
    int32_t v32 = (int32_t) v31;
    pto::Shape<1, 1, 1, 32, 16> v33 = pto::Shape<1, 1, 1, 32, 16>();
    pto::Stride<512, 512, 512, 16, 1> v34 = pto::Stride<512, 512, 512, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND> v35 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND>(v1 + ((v4 + (unsigned) v32 * (unsigned) v11) + v4 * (unsigned) v8 + v4 * (unsigned) v6), v33, v34);
    pto::Shape<1, 1, 1, 16, 256> v36 = pto::Shape<1, 1, 1, 16, 256>();
    pto::Stride<4096, 4096, 4096, 256, 1> v37 = pto::Stride<4096, 4096, 4096, 256, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v4 + (unsigned) v32 * (unsigned) v12) + v4 * (unsigned) v10 + v4 * (unsigned) v6), v36, v37);
    int32_t v39 = (int32_t) ((uint32_t) v32 % (uint32_t) v7);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v19, v35);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    if (v39 == v5) {
      TLOAD(v20, v38);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      pipe_barrier(PIPE_MTE1);
      TMOV(v24, v20);
    } else {
      if (v39 == v6) {
        TLOAD(v21, v38);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_MTE1);
        TMOV(v25, v21);
      } else {
        TLOAD(v22, v38);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        pipe_barrier(PIPE_MTE1);
        TMOV(v26, v22);
      };
    };
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    TMOV(v23, v19);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    if (v32 == v5) {
      TMATMUL(v27, v23, v24);
    } else {
      TMATMUL_ACC(v27, v27, v23, v24);
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  }
  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
  TSTORE(v30, v27);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
