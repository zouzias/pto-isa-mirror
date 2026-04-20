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
  const int32_t v13 = 64;
  const int32_t v14 = 256;
  const int32_t v15 = 512;
  const int32_t v16 = 4096;
  const int64_t v17 = 32768;
  const int64_t v18 = 0;
  const int64_t v19 = 8192;
  const int64_t v20 = 16384;
  const int64_t v21 = 24576;
  using T = float;

  #if defined(__DAV_CUBE__)
  Tile<TileType::Mat, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v22;
  TASSIGN(v22, v17);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v23;
  TASSIGN(v23, v18);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v24;
  TASSIGN(v24, v19);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v25;
  TASSIGN(v25, v20);
  Tile<TileType::Mat, half, 16, 256, BLayout::ColMajor, 16, 256, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v26;
  TASSIGN(v26, v21);
  Tile<TileType::Left, half, 32, 16, BLayout::ColMajor, 32, 16, SLayout::RowMajor, 512, PadValue::Null, CompactMode::Null> v27;
  TASSIGN(v27, v18);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v28;
  TASSIGN(v28, v18);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v29;
  TASSIGN(v29, v19);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v30;
  TASSIGN(v30, v21);
  Tile<TileType::Right, half, 16, 256, BLayout::RowMajor, 16, 256, SLayout::ColMajor, 512, PadValue::Null, CompactMode::Null> v31;
  TASSIGN(v31, v20);
  Tile<TileType::Acc, float, 32, 256, BLayout::ColMajor, 32, 256, SLayout::RowMajor, 1024, PadValue::Null, CompactMode::Null> v32;
  TASSIGN(v32, v18);
  pto::Shape<1, 1, 1, 32, 256> v33 = pto::Shape<1, 1, 1, 32, 256>();
  pto::Stride<8192, 8192, 8192, 256, 1> v34 = pto::Stride<8192, 8192, 8192, 256, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 256>, pto::Stride<8192, 8192, 8192, 256, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v14 + v7 * (unsigned) v9), v33, v34);
  pto::Shape<1, 1, 1, 16, 256> v36 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v37 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v38 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v7 * (unsigned) v16) + v7 * (unsigned) v14 + v7 * (unsigned) v9), v36, v37);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  TLOAD(v23, v38);
  pto::Shape<1, 1, 1, 16, 256> v39 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v40 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v41 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v6 * (unsigned) v16) + v7 * (unsigned) v14 + v7 * (unsigned) v9), v39, v40);
  TLOAD(v24, v41);
  pto::Shape<1, 1, 1, 16, 256> v42 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v43 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v44 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v5 * (unsigned) v16) + v7 * (unsigned) v14 + v7 * (unsigned) v9), v42, v43);
  TLOAD(v25, v44);
  pto::Shape<1, 1, 1, 16, 256> v45 = pto::Shape<1, 1, 1, 16, 256>();
  pto::Stride<4096, 4096, 4096, 256, 1> v46 = pto::Stride<4096, 4096, 4096, 256, 1>();
  GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v47 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + v4 * (unsigned) v16) + v7 * (unsigned) v14 + v7 * (unsigned) v9), v45, v46);
  TLOAD(v26, v47);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID5);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  for (size_t v48 = (size_t) v8; v48 < ((size_t) v13); v48 += (size_t) v9) {
    int32_t v49 = (int32_t) v48;
    int32_t v50 = (int32_t) ((uint32_t) v49 % (uint32_t) v11);
    int32_t v51 = (int32_t) ((uint32_t) v49 + (uint32_t) v11);
    pto::Shape<1, 1, 1, 32, 16> v52 = pto::Shape<1, 1, 1, 32, 16>();
    pto::Stride<512, 512, 512, 16, 1> v53 = pto::Stride<512, 512, 512, 16, 1>();
    GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND> v54 = GlobalTensor<half, pto::Shape<1, 1, 1, 32, 16>, pto::Stride<512, 512, 512, 16, 1>, pto::Layout::ND>(v1 + ((v7 + (unsigned) v49 * (unsigned) v15) + v7 * (unsigned) v12 + v7 * (unsigned) v9), v52, v53);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    TLOAD(v22, v54);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
    pipe_barrier(PIPE_MTE1);
    TMOV(v27, v22);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    if (v51 < v13) {
      pto::Shape<1, 1, 1, 16, 256> v55 = pto::Shape<1, 1, 1, 16, 256>();
      pto::Stride<4096, 4096, 4096, 256, 1> v56 = pto::Stride<4096, 4096, 4096, 256, 1>();
      GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND> v57 = GlobalTensor<half, pto::Shape<1, 1, 1, 16, 256>, pto::Stride<4096, 4096, 4096, 256, 1>, pto::Layout::ND>(v2 + ((v7 + (unsigned) v51 * (unsigned) v16) + v7 * (unsigned) v14 + v7 * (unsigned) v9), v55, v56);
      int32_t v58 = (int32_t) ((uint32_t) v51 % (uint32_t) v11);
      if (v58 == v8) {
        pipe_barrier(PIPE_MTE2);
        TLOAD(v23, v57);
      } else {
        if (v58 == v9) {
          pipe_barrier(PIPE_MTE2);
          TLOAD(v24, v57);
        } else {
          if (v58 == v10) {
            pipe_barrier(PIPE_MTE2);
            TLOAD(v25, v57);
          } else {
            pipe_barrier(PIPE_MTE2);
            TLOAD(v26, v57);
          };
        };
      };
    };
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    bool v59 = v50 == v8;
    bool v60 = v50 == v9;
    bool v61 = v50 == v10;
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    if (v59) {
      TMOV(v28, v23);
    } else {
      if (v60) {
        TMOV(v29, v24);
      } else {
        if (v61) {
          TMOV(v30, v25);
        } else {
          TMOV(v31, v26);
        };
      };
    };
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID4);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID6);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID2);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID3);
    if (v49 == v8) {
      if (v59) {
        TMATMUL(v32, v27, v28);
      } else {
        TMATMUL(v32, v27, v28);
      };
    } else {
      if (v59) {
        TMATMUL_ACC(v32, v32, v27, v28);
      } else {
        if (v60) {
          TMATMUL_ACC(v32, v32, v27, v29);
        } else {
          if (v61) {
            TMATMUL_ACC(v32, v32, v27, v30);
          } else {
            TMATMUL_ACC(v32, v32, v27, v31);
          };
        };
      };
    };
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
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
  TSTORE(v35, v32);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID4);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);
  #endif // __DAV_CUBE__

  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
