#include "common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include <cmath>
using namespace pto;

template <int H, int D, int BLOCK = 64>
AICORE void sparse_attn_kernel__kernel(__gm__ bfloat16_t *q_handle,
                                       __gm__ bfloat16_t *kv_handle,
                                       __gm__ float *output_handle,
                                       __gm__ float *attn_sink_handle,
                                       __gm__ int *topk_idxs_handle,
                                       __gm__ bfloat16_t *workspace_1_handle,
                                       __gm__ float *workspace_2_handle,
                                       __gm__ bfloat16_t *workspace_3_handle,
                                       __gm__ float *workspace_4_handle,
                                       int64_t b,
                                       int64_t m,
                                       int64_t n,
                                       int64_t topk,
                                       uint64_t ffts_Addr) {
//   static_assert(BLOCK == 64, "Current kernel is specialized for BLOCK == 64.");

  constexpr int VBLOCK = H / 2;
  constexpr int Q_STRIDE = H * D;
  constexpr int W1_STRIDE = BLOCK * D;
  constexpr int W2_STRIDE = H * BLOCK;
  constexpr int W3_STRIDE = H * BLOCK;
  constexpr int W4_STRIDE = H * D;
  constexpr int V_W2_OFFSET = VBLOCK * BLOCK;
  constexpr int V_W4_OFFSET = VBLOCK * D;
  constexpr int K_TAIL_D = (D % 128 == 0) ? 128 : D % 128;
  constexpr float SOFTMAX_SCALE = (D == 512) ? 4.419417e-02f :
                                  (D == 256) ? 6.250000e-02f :
                                  (D == 128) ? 8.838835e-02f :
                                  (D == 64)  ? 1.250000e-01f :
                                  1.0f;

  auto cid = get_block_idx();
  auto vid = get_subblockid();
  set_ffts_base_addr(ffts_Addr);

#if defined(__DAV_C220_CUBE__)
  tl::ascend_pto::TileMatL1<bfloat16_t, H, D, H, D> q_l1;
  tl::ascend_pto::TileMatL1<bfloat16_t, BLOCK, D, BLOCK, D> kv_l1;
  tl::ascend_pto::TileMatL1<bfloat16_t, H, BLOCK, H, BLOCK> acc_s_l1;

  TileAcc<float, H, BLOCK, H, BLOCK> acc_s_l0c;
  TileAcc<float, H, D, H, D> acc_o_l0c;

  TASSIGN(q_l1, 0);
  TASSIGN(kv_l1, 65536);
  TASSIGN(acc_s_l1, 131072);

  TASSIGN(acc_s_l0c, 0);
  TASSIGN(acc_o_l0c, 0);
#endif

#if defined(__DAV_C220_VEC__)
  tl::ascend_pto::TileUbDataND<float, VBLOCK, D, VBLOCK, D> acc_o_ub;
  tl::ascend_pto::TileUbDataND<float, 1, VBLOCK, 1, VBLOCK> sum_exp;
  tl::ascend_pto::TileUbDataND<bfloat16_t, 1, D, 1, D> kv_ub_tmp;
  tl::ascend_pto::TileUbDataND<float, 1, VBLOCK, 1, VBLOCK> scores_max;
  tl::ascend_pto::TileUbDataND<float, 1, BLOCK, 1, BLOCK> acc_s_ub_singledim;
  tl::ascend_pto::TileUbDataND<int, 1, BLOCK, 1, BLOCK> idxs_ub;
  tl::ascend_pto::TileUbDataND<bfloat16_t, BLOCK, D, BLOCK, D> kv_ub;
  tl::ascend_pto::TileUbDataND<float, VBLOCK, BLOCK, VBLOCK, BLOCK> acc_s_ub;
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 4096, 1, 4096> tmp_ub;
  tl::ascend_pto::TileUbDataND<float, VBLOCK, BLOCK, VBLOCK, BLOCK> acc_s_ub_;
  tl::ascend_pto::TileUbDataND<float, 1, VBLOCK, 1, VBLOCK> scores_max_prev;
  tl::ascend_pto::TileUbDataND<float, VBLOCK, BLOCK, VBLOCK, BLOCK> scores_max_brd;
  tl::ascend_pto::TileUbDataND<float, 1, VBLOCK, 1, VBLOCK> scores_sum;
  tl::ascend_pto::TileUbDataND<bfloat16_t, VBLOCK, BLOCK, VBLOCK, BLOCK> acc_s_half;
  tl::ascend_pto::TileUbDataND<float, 1, VBLOCK, 1, VBLOCK> attn_sink_ub;
  tl::ascend_pto::TileUbDataND<bfloat16_t, 1, BLOCK, 1, BLOCK> attn_sink_ub1;

  tl::ascend_pto::TileUbDataND<float, VBLOCK, D, VBLOCK, D> sum_exp_brd;
  tl::ascend_pto::TileUbDataND<float, VBLOCK, D, VBLOCK, D> scores_max_prev_brd;
  tl::ascend_pto::TileUbDataND<float, VBLOCK, D, VBLOCK, D> acc_o_ub_;

#ifdef __PTO_AUTO__
//   TRESHAPE(sum_exp_brd, kv_ub);
//   TRESHAPE(kv_ub, acc_o_ub_);
//   TRESHAPE(acc_o_ub_, scores_max_prev_brd);
#else
  TASSIGN(acc_o_ub, 0);
  TASSIGN(sum_exp, 65536);
  TASSIGN(kv_ub_tmp, 65664);
  TASSIGN(scores_max, 66688);
  TASSIGN(acc_s_ub_singledim, 66816);
  TASSIGN(idxs_ub, 67072);
  TASSIGN(kv_ub, 67328);
  TASSIGN(acc_s_ub, 132864);
  TASSIGN(tmp_ub, 141056);
  TASSIGN(acc_s_ub_, 145152);
  TASSIGN(scores_max_prev, 153344);
  TASSIGN(scores_max_brd, 153472);
  TASSIGN(scores_sum, 161664);
  TASSIGN(acc_s_half, 161792);
  TASSIGN(attn_sink_ub, 165888);

  TASSIGN(sum_exp_brd, 67328);
  TASSIGN(scores_max_prev_brd, 67328);
  TASSIGN(acc_o_ub_, 67328);
#endif
#endif

#if defined(__DAV_C220_CUBE__)
  {
    pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> q_shape;
    q_shape.shape[3] = H;
    q_shape.shape[4] = D;
    pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, Q_STRIDE, D, 1>>
        q_global(q_handle + (cid * Q_STRIDE), q_shape,
                 pto::Stride<-1, -1, Q_STRIDE, D, 1>(b, m));
    TLOAD(q_l1, q_global);
  }

  for (int32_t t = 0; t < ((topk + BLOCK - 1) / BLOCK); ++t) {
    tl::ascend_pto::wait_cross_flag(0);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w1_shape;
      w1_shape.shape[3] = BLOCK;
      w1_shape.shape[4] = D;
      pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W1_STRIDE, D, 1>>
          w1_global(workspace_1_handle + (cid * W1_STRIDE), w1_shape,
                    pto::Stride<1, -1, W1_STRIDE, D, 1>(b * m));
      TLOAD(kv_l1, w1_global);
    }

    tl::ascend_pto::gemm_v0<bfloat16_t, float, H, BLOCK, D, H, BLOCK, D, K_TAIL_D, false, true>(q_l1, kv_l1, acc_s_l0c, (bool)1);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w2_shape;
      w2_shape.shape[3] = H;
      w2_shape.shape[4] = BLOCK;
      pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W2_STRIDE, BLOCK, 1>>
          w2_global(workspace_2_handle + (cid * W2_STRIDE), w2_shape,
                    pto::Stride<1, -1, W2_STRIDE, BLOCK, 1>(b * m));
      TSTORE(w2_global, acc_s_l0c);
    }

    ffts_cross_core_sync(PIPE_FIX, 289);
    tl::ascend_pto::wait_cross_flag(2);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w3_shape;
      w3_shape.shape[3] = H;
      w3_shape.shape[4] = BLOCK;
      pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W3_STRIDE, BLOCK, 1>>
          w3_global(workspace_3_handle + (cid * W3_STRIDE), w3_shape,
                    pto::Stride<1, -1, W3_STRIDE, BLOCK, 1>(b * m));
      TLOAD(acc_s_l1, w3_global);
    }

    tl::ascend_pto::gemm_v1<bfloat16_t, float, H, D, BLOCK, H, D, BLOCK, BLOCK, false, false>(acc_s_l1, kv_l1, acc_o_l0c, (bool)1);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w4_shape;
      w4_shape.shape[3] = H;
      w4_shape.shape[4] = D;
      pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W4_STRIDE, D, 1>>
          w4_global(workspace_4_handle + (cid * W4_STRIDE), w4_shape,
                    pto::Stride<1, -1, W4_STRIDE, D, 1>(b * m));
      TSTORE(w4_global, acc_o_l0c);
    }

    ffts_cross_core_sync(PIPE_FIX, 801);
  }
#endif

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  TEXPANDS(acc_o_ub, 3.000000e+00f);
  TEXPANDS(sum_exp, 0.000000e+00f);
  TEXPANDS(kv_ub_tmp, 0.000000e+00f);
  TEXPANDS(scores_max, -CUDART_INF_F);

  for (int32_t t_1 = 0; t_1 < ((topk + BLOCK - 1) / BLOCK); ++t_1) {
    TEXPANDS(acc_s_ub_singledim, 0.000000e+00f);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> idx_shape;
      idx_shape.shape[3] = ((1 <= (m - (cid % m))) ? 1 : 0);
      idx_shape.shape[4] = ((1 <= ((topk / BLOCK) - t_1)) ? BLOCK : ((0 < (topk - (t_1 * BLOCK))) ? (topk - (t_1 * BLOCK)) : 0));
      pto::GlobalTensor<int, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, -1, -1, 1>>
          idx_global(topk_idxs_handle + ((t_1 * BLOCK) + (cid * topk)), idx_shape,
                     pto::Stride<1, -1, -1, -1, 1>(b, m, topk));
      TLOAD(idxs_ub, idx_global);
    }

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    for (int32_t i = 0; i < BLOCK; ++i) {
      if (topk <= ((t_1 * BLOCK) + i)) {
        idxs_ub.SetValue(i, -1);
      }
    }

    for (int32_t b_i = 0; b_i < BLOCK; ++b_i) {
      int32_t idx_num = idxs_ub.GetValue(b_i);

#ifdef __PTO_AUTO__
      tl::ascend_pto::TileUbDataND<bfloat16_t, BLOCK, D, -1, -1> kv_ub_temp_0(1, D);
      TSUBVIEW(kv_ub_temp_0, kv_ub, b_i, 0);
      tl::ascend_pto::TileUbDataND<bfloat16_t, 1, D> kv_ub_row;
      TRESHAPE(kv_ub_row, kv_ub_temp_0);
#else
      tl::ascend_pto::TileUbDataND<bfloat16_t, 1, D, 1, D> kv_ub_temp_0;
      TASSIGN(kv_ub_temp_0, 67328 + (b_i * D) * 2);
#endif

      if (idx_num != -1) {
        pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> kv_shape;
        kv_shape.shape[3] = ((1 <= (n - idx_num)) ? 1 : 0);
        kv_shape.shape[4] = D;
        pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, -1, D, 1>>
            kv_global(kv_handle + ((((cid / m) * n) * D) + (idx_num * D)),
                      kv_shape, pto::Stride<1, -1, -1, D, 1>(b, n));
        TLOAD(kv_ub_temp_0, kv_global);
      } else {
#ifdef __PTO_AUTO__
        TMOV(kv_ub_row, kv_ub_tmp);
#else
        TMOV(kv_ub_temp_0, kv_ub_tmp);
#endif
      }
    }

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w1_shape;
      w1_shape.shape[3] = BLOCK;
      w1_shape.shape[4] = D;
      pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W1_STRIDE, D, 1>>
          w1_global(workspace_1_handle + (cid * W1_STRIDE), w1_shape,
                    pto::Stride<1, -1, W1_STRIDE, D, 1>(b * m));
      TSTORE(w1_global, kv_ub);
    }
    ffts_cross_core_sync(PIPE_MTE3, 33);

    for (int32_t i_1 = 0; i_1 < BLOCK; ++i_1) {
      if (idxs_ub.GetValue(i_1) == -1) {
        acc_s_ub_singledim.SetValue(i_1, -CUDART_INF_F);
      }
    }

    TCOLEXPAND(acc_s_ub, acc_s_ub_singledim);
    tl::ascend_pto::wait_cross_flag(1);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w2_shape;
      w2_shape.shape[3] = VBLOCK;
      w2_shape.shape[4] = BLOCK;
      pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W2_STRIDE, BLOCK, 1>>
          w2_global(workspace_2_handle + ((cid * W2_STRIDE) + (vid * V_W2_OFFSET)), w2_shape,
                    pto::Stride<1, -1, W2_STRIDE, BLOCK, 1>(b * m));
      TLOAD(acc_s_ub_, w2_global);
    }

    TADD(acc_s_ub, acc_s_ub, acc_s_ub_);
    TMULS(acc_s_ub, acc_s_ub, SOFTMAX_SCALE);
    TMOV(scores_max_prev, scores_max);

    tl::ascend_pto::TileUbDataDN<float, VBLOCK, 1, VBLOCK, 1> scores_max_temp_0;
#ifdef __PTO_AUTO__
    TRESHAPE(scores_max_temp_0, scores_max);
#else
    TASSIGN(scores_max_temp_0, 66688 + 0 * 4);
#endif
    TROWMAX(scores_max_temp_0, acc_s_ub, tmp_ub);

    TMAX(scores_max, scores_max, scores_max_prev);
    TSUB(scores_max_prev, scores_max_prev, scores_max);
    TEXP(scores_max_prev, scores_max_prev);

    TROWEXPAND(scores_max_brd, scores_max_temp_0);

    TSUB(acc_s_ub, acc_s_ub, scores_max_brd);
    TEXP(acc_s_ub, acc_s_ub);

    tl::ascend_pto::TileUbDataDN<float, VBLOCK, 1, VBLOCK, 1> scores_sum_temp_0;
#ifdef __PTO_AUTO__
    TRESHAPE(scores_sum_temp_0, scores_sum);
#else
    TASSIGN(scores_sum_temp_0, 161664 + 0 * 4);
#endif
    TROWSUM(scores_sum_temp_0, acc_s_ub, tmp_ub);

    TMUL(sum_exp, sum_exp, scores_max_prev);
    TADD(sum_exp, sum_exp, scores_sum);

    TCVT(acc_s_half, acc_s_ub, pto::RoundMode::CAST_NONE);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w3_shape;
      w3_shape.shape[3] = VBLOCK;
      w3_shape.shape[4] = BLOCK;
      pto::GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W3_STRIDE, BLOCK, 1>>
          w3_global(workspace_3_handle + ((cid * W3_STRIDE) + (vid * V_W2_OFFSET)), w3_shape,
                    pto::Stride<1, -1, W3_STRIDE, BLOCK, 1>(b * m));
      TSTORE(w3_global, acc_s_half);
    }

    ffts_cross_core_sync(PIPE_MTE3, 545);

    tl::ascend_pto::TileUbDataDN<float, VBLOCK, 1, VBLOCK, 1> scores_max_prev_temp_0;
#ifdef __PTO_AUTO__
    TRESHAPE(scores_max_prev_temp_0, scores_max_prev);
#else
    TASSIGN(scores_max_prev_temp_0, 153344 + 0 * 4);
#endif
    TROWEXPAND(scores_max_prev_brd, scores_max_prev_temp_0);

    TMUL(acc_o_ub, acc_o_ub, scores_max_prev_brd);

    tl::ascend_pto::wait_cross_flag(3);

    {
      pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> w4_shape;
      w4_shape.shape[3] = VBLOCK;
      w4_shape.shape[4] = D;
      pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, -1, W4_STRIDE, D, 1>>
          w4_global(workspace_4_handle + ((cid * W4_STRIDE) + (vid * V_W4_OFFSET)), w4_shape,
                    pto::Stride<1, -1, W4_STRIDE, D, 1>(b * m));
      TLOAD(acc_o_ub_, w4_global);
    }

    TADD(acc_o_ub, acc_o_ub, acc_o_ub_);
  }

  {
    pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> sink_shape;
    sink_shape.shape[3] = 1;
    sink_shape.shape[4] = VBLOCK;
    pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<1, 1, 1, H, 1>>
        sink_global(attn_sink_handle + (vid * VBLOCK), sink_shape);
    TLOAD(attn_sink_ub, sink_global);
  }

  TSUB(attn_sink_ub, attn_sink_ub, scores_max);
  TEXP(attn_sink_ub, attn_sink_ub);
  TADD(sum_exp, sum_exp, attn_sink_ub);

  tl::ascend_pto::TileUbDataDN<float, VBLOCK, 1, VBLOCK, 1> sum_exp_temp_0;
#ifdef __PTO_AUTO__
  TRESHAPE(sum_exp_temp_0, sum_exp);
#else
  TASSIGN(sum_exp_temp_0, 65536 + 0 * 4);
#endif
  TROWEXPAND(sum_exp_brd, sum_exp_temp_0);

  TDIV(acc_o_ub, acc_o_ub, sum_exp_brd);

  {
    pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC> out_shape;
    out_shape.shape[3] = VBLOCK;
    out_shape.shape[4] = D;
    pto::GlobalTensor<float, pto::Shape<1, 1, 1, -1, -1>, pto::Stride<-1, -1, Q_STRIDE, D, 1>>
        out_global(output_handle + ((cid * Q_STRIDE) + (vid * V_W4_OFFSET)), out_shape,
                   pto::Stride<-1, -1, Q_STRIDE, D, 1>(b, m));
    TSTORE(out_global, acc_o_ub);
  }
#endif
}

template <int H, int D, int BLOCK>
__global__ AICORE void launch_kernel(__gm__ uint8_t *q_handle,
                                     __gm__ uint8_t *kv_handle,
                                     __gm__ uint8_t *output_handle,
                                     __gm__ uint8_t *attn_sink_handle,
                                     __gm__ uint8_t *topk_idxs_handle,
                                     __gm__ uint8_t *workspace_1_handle,
                                     __gm__ uint8_t *workspace_2_handle,
                                     __gm__ uint8_t *workspace_3_handle,
                                     __gm__ uint8_t *workspace_4_handle,
                                     int64_t b,
                                     int64_t m,
                                     int64_t n,
                                     int64_t topk,
                                     uint64_t fftsAddr) {
  sparse_attn_kernel__kernel<H, D, BLOCK>(
      reinterpret_cast<__gm__ bfloat16_t *>(q_handle),
      reinterpret_cast<__gm__ bfloat16_t *>(kv_handle),
      reinterpret_cast<__gm__ float *>(output_handle),
      reinterpret_cast<__gm__ float *>(attn_sink_handle),
      reinterpret_cast<__gm__ int *>(topk_idxs_handle),
      reinterpret_cast<__gm__ bfloat16_t *>(workspace_1_handle),
      reinterpret_cast<__gm__ float *>(workspace_2_handle),
      reinterpret_cast<__gm__ bfloat16_t *>(workspace_3_handle),
      reinterpret_cast<__gm__ float *>(workspace_4_handle),
      b, m, n, topk,
      reinterpret_cast<uint64_t>(fftsAddr));
}

template <int H, int D, int BLOCK>
void call(uint8_t *q_handle,
                     uint8_t *kv_handle,
                     uint8_t *output_handle,
                     uint8_t *attn_sink_handle,
                     uint8_t *topk_idxs_handle,
                     uint8_t *workspace_1_handle,
                     uint8_t *workspace_2_handle,
                     uint8_t *workspace_3_handle,
                     uint8_t *workspace_4_handle,
                     int64_t b,
                     int64_t m,
                     int64_t n,
                     int64_t topk,
                     void *stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_kernel<H, D, BLOCK><<<(b * m), nullptr, stream>>>(q_handle, kv_handle, output_handle, attn_sink_handle,
                                              topk_idxs_handle, workspace_1_handle, workspace_2_handle,
                                              workspace_3_handle, workspace_4_handle, b, m, n, topk, fftsAddr);
}

// template void call<64, 512, 64>(
//     uint8_t *q_handle,
//     uint8_t *kv_handle,
//     uint8_t *output_handle,
//     uint8_t *attn_sink_handle,
//     uint8_t *topk_idxs_handle,
//     uint8_t *workspace_1_handle,
//     uint8_t *workspace_2_handle,
//     uint8_t *workspace_3_handle,
//     uint8_t *workspace_4_handle,
//     int64_t b,
//     int64_t m,
//     int64_t n,
//     int64_t topk,
//     void *stream
// );

template void call<16, 256, 64>(
    uint8_t *q_handle,
    uint8_t *kv_handle,
    uint8_t *output_handle,
    uint8_t *attn_sink_handle,
    uint8_t *topk_idxs_handle,
    uint8_t *workspace_1_handle,
    uint8_t *workspace_2_handle,
    uint8_t *workspace_3_handle,
    uint8_t *workspace_4_handle,
    int64_t b,
    int64_t m,
    int64_t n,
    int64_t topk,
    void *stream
);