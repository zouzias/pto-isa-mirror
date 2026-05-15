#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# mla_basic - gen_data.py
#
# Builds inputs and golden output for the very-basic Multi-Head Latent Attention
# kernel (DeepSeek V2/V3 style, no RoPE, no causal mask, prefill-only,
# single sequence). Mirrors the kernel pipeline byte-for-byte:
#
#   Q       = X @ W_q                      (FP32 acc, FP16 GM)
#   C_kv    = X @ W_dkv                    (FP32 acc, FP16 GM)
#   C_cache = C_kv                         (prefill-only)
#   K       = C_cache @ W_uk               (FP32 acc, FP16 GM)
#   V       = C_cache @ W_uv               (FP32 acc, FP16 GM)
#   per head h:
#     scores[h] = (Q_h @ K_h^T) * (1/sqrt(Hd))   (FP32 acc, FP16 GM, FP16 scale)
#     probs[h]  = softmax(scores[h], axis=-1)    (FP32 math, FP16 GM)
#     out_h     = probs[h] @ V_h                 (FP32 acc, FP16 GM)
#
# The "FP32 acc, FP16 GM" pattern matches the §A18 fused TSTORE; we emulate
# it with `np.float32` accumulators that we cast to `np.float16` at each
# stage's GM hand-off. Softmax math is done in FP32 with FP16 inputs (the
# kernel does FP16 vec math; this is a known precision divergence and is why
# `main.cpp`'s ResultCmp uses a relatively generous tolerance for v1).
#
# Output files (all raw little-endian, contiguous, no header):
#   ./input/input_x.bin       (S * H              FP16)
#   ./input/input_w_q.bin     (H * Nh*Hd          FP16)
#   ./input/input_w_dkv.bin   (H * L              FP16)
#   ./input/input_w_uk.bin    (L * Nh*Hd          FP16)
#   ./input/input_w_uv.bin    (L * Nh*Hd          FP16)
#   ./output/golden_q.bin     (S * Nh*Hd          FP16)   debug
#   ./output/golden_c_kv.bin  (S * L              FP16)   debug
#   ./output/golden_k.bin     (S * Nh*Hd          FP16)   debug
#   ./output/golden_v.bin     (S * Nh*Hd          FP16)   debug
#   ./output/golden_scores.bin(Nh * S * S         FP16)   debug
#   ./output/golden_probs.bin (Nh * S * S         FP16)   debug
#   ./output/golden_out.bin   (S * Nh*Hd          FP16)   THE main golden
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(42)


def small_uniform(shape, lo=-1.0, hi=1.0):
    """Small uniform values to keep FP16 GEMM outputs inside the FP16 range
    even when accumulated across kHidden=4096 K dimension."""
    return np.random.uniform(low=lo, high=hi, size=shape).astype(np.float16)


def gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent):
    kQKVHidden = kNumHeads * kHeadDim
    assert kQKVHidden == kHidden, "v1 assumes num_heads * head_dim == hidden"

    # ---- Inputs / weights ------------------------------------------------
    # Values are small enough that:
    #   GEMM accumulators stay well inside the FP16 exact range.
    #   - X @ W_q : K=4096 sums of (|x| <= 1)*(|w| <= 0.1) -> max ~ 410. OK.
    # Pick weight magnitudes proportional to 1/sqrt(K) to keep the activations
    # in a roughly unit-variance regime (Xavier-style init), which is what
    # transformer activations look like in practice.
    w_q_scale   = 1.0 / math.sqrt(kHidden)
    w_dkv_scale = 1.0 / math.sqrt(kHidden)
    w_uk_scale  = 1.0 / math.sqrt(kLatent)
    w_uv_scale  = 1.0 / math.sqrt(kLatent)

    x      = small_uniform((kSeqLen, kHidden),       -1.0, 1.0)
    w_q    = small_uniform((kHidden, kQKVHidden),   -w_q_scale,   w_q_scale)
    w_dkv  = small_uniform((kHidden, kLatent),      -w_dkv_scale, w_dkv_scale)
    w_uk   = small_uniform((kLatent, kQKVHidden),   -w_uk_scale,  w_uk_scale)
    w_uv   = small_uniform((kLatent, kQKVHidden),   -w_uv_scale,  w_uv_scale)

    # ---- Stage 1 : Q projection ------------------------------------------
    # FP32 acc -> FP16 GM (matches the kernel's fused TSTORE).
    q_fp32 = x.astype(np.float32) @ w_q.astype(np.float32)
    q      = q_fp32.astype(np.float16)
    # Reshape view-only: the kernel writes Q laid out as [S, Nh*Hd] contiguous,
    # which is bit-identical to [S, Nh, Hd] (row-major). No actual reorder.

    # ---- Stage 2 : KV compression ----------------------------------------
    c_kv_fp32 = x.astype(np.float32) @ w_dkv.astype(np.float32)
    c_kv      = c_kv_fp32.astype(np.float16)

    # ---- Stage 3 : KV cache store (prefill-only: identity) ---------------
    c_cache = c_kv.copy()

    # ---- Stage 4 : KV reconstruction -------------------------------------
    k_fp32 = c_cache.astype(np.float32) @ w_uk.astype(np.float32)
    v_fp32 = c_cache.astype(np.float32) @ w_uv.astype(np.float32)
    k      = k_fp32.astype(np.float16)
    v      = v_fp32.astype(np.float16)

    # ---- Stage 5 : Attention per head ------------------------------------
    scale = np.float16(1.0 / math.sqrt(float(kHeadDim)))  # match kernel scalar

    scores = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    probs  = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    out    = np.zeros((kSeqLen, kQKVHidden),        dtype=np.float16)

    # Reshape Q, K, V from [S, Nh*Hd] to [S, Nh, Hd] for per-head slicing.
    q_h = q.reshape(kSeqLen, kNumHeads, kHeadDim)
    k_h = k.reshape(kSeqLen, kNumHeads, kHeadDim)
    v_h = v.reshape(kSeqLen, kNumHeads, kHeadDim)

    for h in range(kNumHeads):
        Q = q_h[:, h, :]                          # [S, Hd]
        K = k_h[:, h, :]                          # [S, Hd]
        V = v_h[:, h, :]                          # [S, Hd]

        # 5a -- QK^T : FP32 acc, FP16 GM, then scale by 1/sqrt(Hd) (FP16 scalar)
        s_fp32 = Q.astype(np.float32) @ K.astype(np.float32).T   # [S, S]
        s_fp16 = s_fp32.astype(np.float16)
        # The kernel computes (FP32 acc) and writes FP16; the FP16 *= scale
        # happens INSIDE the vec softmax kernel via TMULS. So the "scores GM"
        # contents are the un-scaled FP16 scores.
        scores[h, :, :] = s_fp16

        # 5b -- softmax with FP16 scale: replicate the kernel's vec FP16 math.
        # NOTE: the kernel does TMULS -> TROWMAX -> TROWEXPAND -> TSUB ->
        # TEXP -> TROWSUM -> TROWEXPAND -> TDIV, all in FP16. We mirror that
        # closely (FP16 each step) so the golden matches.
        scaled    = (s_fp16 * scale).astype(np.float16)
        row_max   = scaled.max(axis=1, keepdims=True).astype(np.float16)
        shifted   = (scaled - row_max).astype(np.float16)
        ex        = np.exp(shifted.astype(np.float32)).astype(np.float16)
        row_sum   = ex.sum(axis=1, keepdims=True).astype(np.float16)
        p_fp16    = (ex / row_sum).astype(np.float16)
        probs[h, :, :] = p_fp16

        # 5c -- PV : FP32 acc, FP16 GM
        o_fp32 = p_fp16.astype(np.float32) @ V.astype(np.float32)
        out[:, h * kHeadDim:(h + 1) * kHeadDim] = o_fp32.astype(np.float16)

    # ---- Save ------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x.tofile     ("./input/input_x.bin")
    w_q.tofile   ("./input/input_w_q.bin")
    w_dkv.tofile ("./input/input_w_dkv.bin")
    w_uk.tofile  ("./input/input_w_uk.bin")
    w_uv.tofile  ("./input/input_w_uv.bin")

    q.tofile     ("./output/golden_q.bin")
    c_kv.tofile  ("./output/golden_c_kv.bin")
    k.tofile     ("./output/golden_k.bin")
    v.tofile     ("./output/golden_v.bin")
    scores.tofile("./output/golden_scores.bin")
    probs.tofile ("./output/golden_probs.bin")
    out.tofile   ("./output/golden_out.bin")

    # ---- Debug summary ---------------------------------------------------
    print(f"[gen_data] B=1  S={kSeqLen}  H={kHidden}  Nh={kNumHeads}  "
          f"Hd={kHeadDim}  L={kLatent}")
    print(f"[gen_data] dtypes: x={x.dtype} w_q={w_q.dtype} c_kv={c_kv.dtype} "
          f"q={q.dtype} scores={scores.dtype} probs={probs.dtype} out={out.dtype}")

    def stats(name, arr):
        a = arr.astype(np.float32)
        print(f"[gen_data] {name:>10s}: shape={list(arr.shape)} "
              f"min={a.min():+.4f} max={a.max():+.4f} "
              f"mean={a.mean():+.4f} std={a.std():.4f}")

    stats("x",        x)
    stats("w_q",      w_q)
    stats("w_dkv",    w_dkv)
    stats("w_uk",     w_uk)
    stats("w_uv",     w_uv)
    stats("q",        q)
    stats("c_kv",     c_kv)
    stats("k",        k)
    stats("v",        v)
    stats("scores",   scores)
    stats("probs",    probs)
    stats("out",      out)

    print(f"[gen_data] softmax scale (FP16) = {float(scale):.6f}")
    print(f"[gen_data] probs row sums first row head 0: "
          f"{probs[0, 0, :].astype(np.float32).sum():.4f} (should be ~1.0)")


if __name__ == "__main__":
    kSeqLen   = 128
    kHidden   = 4096
    kNumHeads = 32
    kHeadDim  = 128
    kLatent   = 64
    gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent)
