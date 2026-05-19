#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# mla_basic - gen_data.py
#
# Multi-Head Latent Attention with DeepSeek-V2 style **decoupled RoPE**:
#
#   Q_nope     = X @ W_q                  (FP32 acc, FP16 GM) (existing)
#   C_kv       = X @ W_dkv                (FP32 acc, FP16 GM) (existing)
#   C_cache    = C_kv                                         (existing)
#   K_nope     = C_cache @ W_uk           (FP32 acc, FP16 GM) (existing)
#   V          = C_cache @ W_uv           (FP32 acc, FP16 GM) (existing)
#   Q_rope     = X @ W_q_rope             (FP32 acc, FP16 GM) NEW   [Nh,S,Rd]
#   K_rope     = X @ W_k_rope             (FP32 acc, FP16 GM) NEW   [S,Rd]
#   Q_rope_rot = RoPE(Q_rope, cos, sin)                       NEW
#   K_rope_rot = RoPE(K_rope, cos, sin)                       NEW
#   per head h:
#     scores_nope[h] = Q_nope_h @ K_nope_h^T                  (existing)
#     scores_rope[h] = Q_rope_rot_h @ K_rope_rot^T  (K shared)NEW
#     scores[h]      = scores_nope[h] + scores_rope[h]
#     probs[h]       = softmax(scores[h] * scale)
#     out_h          = probs[h] @ V_h
#
#   scale = 1/sqrt(kHeadDim + kRopeDim) = 1/sqrt(192)
#
# Decoupled RoPE notes (matches DeepSeek-V2 reference):
#   - kRopeDim = 64 (separate from kHeadDim = 128).
#   - K_rope has NO heads dim: shape [S, Rd]. Broadcast across all heads.
#   - Half-rotation variant: for x in [S, Rd],
#       x1 = x[..., :Rd/2], x2 = x[..., Rd/2:]
#       y[..., :Rd/2] = x1 * cos - x2 * sin
#       y[..., Rd/2:] = x1 * sin + x2 * cos
#   - cos/sin tables are host-precomputed [S, Rd/2] FP16 and uploaded as
#     kernel inputs (no transcendentals on device).
#
# Q_rope storage layout: [Nh, S, Rd] (head-major). Chosen so the RoPE vec
# kernel and the runAttnQKRope cube kernel can read each head's [S, Rd]
# block contiguously, without GlobalTensor stride tricks.
# --------------------------------------------------------------------------------

import math
import os
import numpy as np

np.random.seed(42)


def small_uniform(shape, lo=-1.0, hi=1.0):
    return np.random.uniform(low=lo, high=hi, size=shape).astype(np.float16)


def build_rope_tables(kSeqLen, kRopeDim, base=10000.0):
    """Standard RoPE: theta_i = base^(-2i/Rd), angle = pos * theta_i.
    cos/sin shape [S, Rd/2], FP16."""
    half = kRopeDim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) * 2.0 / kRopeDim))  # [half]
    positions = np.arange(kSeqLen, dtype=np.float32)                                    # [S]
    angles = np.outer(positions, inv_freq)                                              # [S, half]
    cos = np.cos(angles).astype(np.float16)
    sin = np.sin(angles).astype(np.float16)
    return cos, sin


def apply_rope_half(x_fp16, cos_fp16, sin_fp16):
    """Half-rotation RoPE.
    x:   [..., Rd]  FP16
    cos: [S,   Rd/2] FP16
    sin: [S,   Rd/2] FP16
    The leading axes of x must broadcast such that x's S-axis aligns
    with cos/sin's S-axis. For Q_rope shape [Nh, S, Rd] -> cos/sin
    broadcast across heads via [None, S, half]. For K_rope shape
    [S, Rd] -> cos/sin used directly.
    Computation done in FP16 to mirror the kernel's FP16 vec math.
    """
    half = x_fp16.shape[-1] // 2
    x1 = x_fp16[..., :half]
    x2 = x_fp16[..., half:]
    if x_fp16.ndim == 3:  # [Nh, S, Rd]
        c = cos_fp16[None, :, :]  # broadcast across heads
        s = sin_fp16[None, :, :]
    else:                  # [S, Rd]
        c = cos_fp16
        s = sin_fp16
    out = np.empty_like(x_fp16)
    out[..., :half] = (x1 * c - x2 * s).astype(np.float16)
    out[..., half:] = (x1 * s + x2 * c).astype(np.float16)
    return out


def gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent, kRopeDim):
    kQKVHidden = kNumHeads * kHeadDim
    assert kQKVHidden == kHidden, "v1 assumes num_heads * head_dim == hidden"
    assert kRopeDim % 2 == 0, "kRopeDim must be even (half-rotation)"
    kQRopeWidth = kNumHeads * kRopeDim   # 32 * 64 = 2048

    # ---- Inputs / weights ------------------------------------------------
    w_q_scale     = 1.0 / math.sqrt(kHidden)
    w_dkv_scale   = 1.0 / math.sqrt(kHidden)
    w_uk_scale    = 1.0 / math.sqrt(kLatent)
    w_uv_scale    = 1.0 / math.sqrt(kLatent)
    w_q_rope_scl  = 1.0 / math.sqrt(kHidden)
    w_k_rope_scl  = 1.0 / math.sqrt(kHidden)

    x        = small_uniform((kSeqLen, kHidden),               -1.0, 1.0)
    w_q      = small_uniform((kHidden, kQKVHidden),    -w_q_scale,    w_q_scale)
    w_dkv    = small_uniform((kHidden, kLatent),       -w_dkv_scale,  w_dkv_scale)
    w_uk     = small_uniform((kLatent, kQKVHidden),    -w_uk_scale,   w_uk_scale)
    w_uv     = small_uniform((kLatent, kQKVHidden),    -w_uv_scale,   w_uv_scale)
    w_q_rope = small_uniform((kHidden, kQRopeWidth),   -w_q_rope_scl, w_q_rope_scl)
    w_k_rope = small_uniform((kHidden, kRopeDim),      -w_k_rope_scl, w_k_rope_scl)

    cos_table, sin_table = build_rope_tables(kSeqLen, kRopeDim)

    # ---- Existing nope/V/C path -----------------------------------------
    q_nope_fp32 = x.astype(np.float32) @ w_q.astype(np.float32)
    q_nope      = q_nope_fp32.astype(np.float16)

    c_kv_fp32 = x.astype(np.float32) @ w_dkv.astype(np.float32)
    c_kv      = c_kv_fp32.astype(np.float16)
    c_cache   = c_kv.copy()

    k_nope_fp32 = c_cache.astype(np.float32) @ w_uk.astype(np.float32)
    v_fp32      = c_cache.astype(np.float32) @ w_uv.astype(np.float32)
    k_nope = k_nope_fp32.astype(np.float16)
    v      = v_fp32.astype(np.float16)

    # ---- Q_rope (head-major output layout) -------------------------------
    # GEMM produces [S, Nh * Rd] in normal layout; the kernel writes it
    # rearranged to [Nh, S, Rd] so RoPE can read each head contiguously.
    q_rope_flat_fp32 = x.astype(np.float32) @ w_q_rope.astype(np.float32)        # [S, Nh*Rd]
    q_rope_flat      = q_rope_flat_fp32.astype(np.float16)
    q_rope = (q_rope_flat.reshape(kSeqLen, kNumHeads, kRopeDim)                  # [S, Nh, Rd]
                          .transpose(1, 0, 2).copy())                            # [Nh, S, Rd]

    # ---- K_rope (single shared head) -------------------------------------
    k_rope_fp32 = x.astype(np.float32) @ w_k_rope.astype(np.float32)             # [S, Rd]
    k_rope      = k_rope_fp32.astype(np.float16)

    # ---- RoPE rotation (FP16, mirrors the vec kernel) --------------------
    q_rope_rot = apply_rope_half(q_rope, cos_table, sin_table)                   # [Nh, S, Rd]
    k_rope_rot = apply_rope_half(k_rope, cos_table, sin_table)                   # [S, Rd]

    # ---- Attention -------------------------------------------------------
    head_dim_total = kHeadDim + kRopeDim
    scale = np.float16(1.0 / math.sqrt(float(head_dim_total)))

    scores_nope = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    scores_rope = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    scores      = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    probs       = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    out         = np.zeros((kSeqLen, kQKVHidden),        dtype=np.float16)

    q_nope_h = q_nope.reshape(kSeqLen, kNumHeads, kHeadDim)
    k_nope_h = k_nope.reshape(kSeqLen, kNumHeads, kHeadDim)
    v_h      = v.reshape(kSeqLen, kNumHeads, kHeadDim)

    for h in range(kNumHeads):
        Q_n = q_nope_h[:, h, :]                                                  # [S, Hd]
        K_n = k_nope_h[:, h, :]                                                  # [S, Hd]
        V_h = v_h[:,    h, :]                                                    # [S, Hd]
        Q_r = q_rope_rot[h]                                                      # [S, Rd]
        K_r = k_rope_rot                                                         # [S, Rd]  shared

        s_n_fp32 = Q_n.astype(np.float32) @ K_n.astype(np.float32).T
        s_r_fp32 = Q_r.astype(np.float32) @ K_r.astype(np.float32).T
        s_n_fp16 = s_n_fp32.astype(np.float16)
        s_r_fp16 = s_r_fp32.astype(np.float16)
        scores_nope[h] = s_n_fp16
        scores_rope[h] = s_r_fp16

        s_sum = (s_n_fp16 + s_r_fp16).astype(np.float16)
        scores[h] = s_sum

        # Softmax mirrors FP16 vec sequence (TMULS -> TROWMAX -> TSUB -> TEXP
        # -> TROWSUM -> TDIV).
        scaled  = (s_sum * scale).astype(np.float16)
        row_max = scaled.max(axis=1, keepdims=True).astype(np.float16)
        shifted = (scaled - row_max).astype(np.float16)
        ex      = np.exp(shifted.astype(np.float32)).astype(np.float16)
        row_sum = ex.sum(axis=1, keepdims=True).astype(np.float16)
        p_fp16  = (ex / row_sum).astype(np.float16)
        probs[h] = p_fp16

        o_fp32 = p_fp16.astype(np.float32) @ V_h.astype(np.float32)
        out[:, h * kHeadDim:(h + 1) * kHeadDim] = o_fp32.astype(np.float16)

    # ---- Save ------------------------------------------------------------
    os.makedirs("input",  exist_ok=True)
    os.makedirs("output", exist_ok=True)

    x.tofile        ("./input/input_x.bin")
    w_q.tofile      ("./input/input_w_q.bin")
    w_dkv.tofile    ("./input/input_w_dkv.bin")
    w_uk.tofile     ("./input/input_w_uk.bin")
    w_uv.tofile     ("./input/input_w_uv.bin")
    w_q_rope.tofile ("./input/input_w_q_rope.bin")
    w_k_rope.tofile ("./input/input_w_k_rope.bin")
    cos_table.tofile("./input/input_cos.bin")
    sin_table.tofile("./input/input_sin.bin")

    q_nope.tofile      ("./output/golden_q.bin")
    c_kv.tofile        ("./output/golden_c_kv.bin")
    k_nope.tofile      ("./output/golden_k.bin")
    v.tofile           ("./output/golden_v.bin")
    q_rope.tofile      ("./output/golden_q_rope.bin")        # [Nh, S, Rd]
    k_rope.tofile      ("./output/golden_k_rope.bin")        # [S, Rd]
    q_rope_rot.tofile  ("./output/golden_q_rope_rot.bin")
    k_rope_rot.tofile  ("./output/golden_k_rope_rot.bin")
    scores_nope.tofile ("./output/golden_scores_nope.bin")   # [Nh, S, S]
    scores_rope.tofile ("./output/golden_scores_rope.bin")
    scores.tofile      ("./output/golden_scores.bin")        # combined
    probs.tofile       ("./output/golden_probs.bin")
    out.tofile         ("./output/golden_out.bin")

    # ---- Debug summary ---------------------------------------------------
    print(f"[gen_data] B=1  S={kSeqLen}  H={kHidden}  Nh={kNumHeads}  "
          f"Hd={kHeadDim}  L={kLatent}  Rd={kRopeDim}")
    print(f"[gen_data] head_dim_total = {head_dim_total}  scale = {float(scale):.6f}")

    def stats(name, arr):
        a = arr.astype(np.float32)
        print(f"[gen_data] {name:>12s}: shape={list(arr.shape)} "
              f"min={a.min():+.4f} max={a.max():+.4f} "
              f"mean={a.mean():+.4f} std={a.std():.4f}")

    stats("x",            x)
    stats("w_q_rope",     w_q_rope)
    stats("w_k_rope",     w_k_rope)
    stats("cos",          cos_table)
    stats("sin",          sin_table)
    stats("q_nope",       q_nope)
    stats("c_kv",         c_kv)
    stats("k_nope",       k_nope)
    stats("v",            v)
    stats("q_rope",       q_rope)
    stats("k_rope",       k_rope)
    stats("q_rope_rot",   q_rope_rot)
    stats("k_rope_rot",   k_rope_rot)
    stats("scores_nope",  scores_nope)
    stats("scores_rope",  scores_rope)
    stats("scores",       scores)
    stats("probs",        probs)
    stats("out",          out)

    print(f"[gen_data] probs row sums first row head 0: "
          f"{probs[0, 0, :].astype(np.float32).sum():.4f} (should be ~1.0)")


if __name__ == "__main__":
    kSeqLen   = 128
    kHidden   = 4096
    kNumHeads = 32
    kHeadDim  = 128
    kLatent   = 64
    kRopeDim  = 64
    gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent, kRopeDim)
