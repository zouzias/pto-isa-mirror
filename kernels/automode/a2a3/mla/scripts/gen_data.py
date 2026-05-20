#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# mla_basic - gen_data.py
#
# Multi-Head Latent Attention, DeepSeek-V2 architecture:
#   * Query path is compressed too (W_dq -> W_uq), not a single W_q.
#   * head_dim is partitioned: head_dim = nope_dim + rope_dim, both per-head.
#     - Q_nope/K_nope per-head dim = nope_dim (64)
#     - Q_rope/K_rope per-head dim = rope_dim (64)
#     - V/Out per-head dim       = head_dim (128 = nope_dim + rope_dim)
#   * Attention score adds nope and rope branches:
#       score[h,i,j] = Q_nope_h[i,:]    · K_nope_h[j,:]
#                   + Q_rope_rot_h[i,:] · K_rope_rot[j,:]
#     scale = 1/sqrt(head_dim)
#
# Pipeline mirrored by the kernel:
#   C_q       = X @ W_dq                                 (compressed Q)
#   Q_nope    = C_q @ W_uq            -> [S, Nh, nope_d]
#   C_kv      = X @ W_dkv                                (compressed KV, unchanged)
#   C_cache   = C_kv                                     (prefill identity)
#   K_nope    = C_cache @ W_uk        -> [S, Nh, nope_d]
#   V         = C_cache @ W_uv        -> [S, Nh, head_d]
#   Q_rope    = X @ W_q_rope          -> [Nh, S, rope_d]
#   K_rope    = X @ W_k_rope          -> [S, rope_d]   (shared across heads)
#   Q_rope_rot= RoPE(Q_rope, cos, sin)
#   K_rope_rot= RoPE(K_rope, cos, sin)
#   per head h:
#     scores_nope[h] = Q_nope_h @ K_nope_h^T
#     scores_rope[h] = Q_rope_rot_h @ K_rope_rot^T
#     scores[h]      = scores_nope[h] + scores_rope[h]
#     probs[h]       = softmax(scores[h] * (1/sqrt(head_dim)))
#     out_h          = probs[h] @ V_h
#
# Buffer layouts (chosen to avoid GlobalTensor stride tricks in the kernel):
#   Q_nope, K_nope, V    : [S, Nh, *]            row stride = Nh*per_head_dim
#   Q_rope, Q_rope_rot   : [Nh, S, rope_dim]     head-major; row stride = rope_dim
#   K_rope, K_rope_rot   : [S, rope_dim]         shared single block
# --------------------------------------------------------------------------------

import math
import os
import json
from pathlib import Path
import numpy as np

np.random.seed(42)


def load_generated_case():
    case_path = Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"
    if not case_path.exists():
        return {"seq_len": 128, "hidden": 4096, "num_heads": 32, "head_dim": 128, "latent": 64, "rope_dim": 64}
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases[0]


def small_uniform(shape, lo=-1.0, hi=1.0):
    return np.random.uniform(low=lo, high=hi, size=shape).astype(np.float16)


def build_rope_tables(kSeqLen, kRopeDim, base=10000.0):
    """Standard RoPE: theta_i = base^(-2i/Rd), angle = pos * theta_i.
    cos/sin shape [S, Rd/2], FP16."""
    half = kRopeDim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) * 2.0 / kRopeDim))
    positions = np.arange(kSeqLen, dtype=np.float32)
    angles = np.outer(positions, inv_freq)
    cos = np.cos(angles).astype(np.float16)
    sin = np.sin(angles).astype(np.float16)
    return cos, sin


def apply_rope_half(x_fp16, cos_fp16, sin_fp16):
    half = x_fp16.shape[-1] // 2
    x1 = x_fp16[..., :half]
    x2 = x_fp16[..., half:]
    if x_fp16.ndim == 3:
        c = cos_fp16[None, :, :]
        s = sin_fp16[None, :, :]
    else:
        c = cos_fp16
        s = sin_fp16
    out = np.empty_like(x_fp16)
    out[..., :half] = (x1 * c - x2 * s).astype(np.float16)
    out[..., half:] = (x1 * s + x2 * c).astype(np.float16)
    return out


def gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent, kRopeDim,
                    kNopeDim, kQLatent):
    assert kNopeDim + kRopeDim == kHeadDim, "Must have nope_dim + rope_dim == head_dim"
    assert kRopeDim % 2 == 0, "rope_dim must be even"

    kQNopeWidth = kNumHeads * kNopeDim    # 32 * 64 = 2048
    kVWidth     = kNumHeads * kHeadDim    # 32 * 128 = 4096
    kQRopeWidth = kNumHeads * kRopeDim    # 32 * 64 = 2048

    # ---- Inputs / weights ------------------------------------------------
    # Xavier-style scales (1/sqrt(in_dim)).
    w_dq_scale    = 1.0 / math.sqrt(kHidden)
    w_uq_scale    = 1.0 / math.sqrt(kQLatent)
    w_dkv_scale   = 1.0 / math.sqrt(kHidden)
    w_uk_scale    = 1.0 / math.sqrt(kLatent)
    w_uv_scale    = 1.0 / math.sqrt(kLatent)
    w_q_rope_scl  = 1.0 / math.sqrt(kHidden)
    w_k_rope_scl  = 1.0 / math.sqrt(kHidden)

    x        = small_uniform((kSeqLen, kHidden),            -1.0, 1.0)
    w_dq     = small_uniform((kHidden,  kQLatent),    -w_dq_scale,    w_dq_scale)
    w_uq     = small_uniform((kQLatent, kQNopeWidth), -w_uq_scale,    w_uq_scale)
    w_dkv    = small_uniform((kHidden,  kLatent),     -w_dkv_scale,   w_dkv_scale)
    w_uk     = small_uniform((kLatent,  kQNopeWidth), -w_uk_scale,    w_uk_scale)  # NOTE: width kQNopeWidth (was kVWidth)
    w_uv     = small_uniform((kLatent,  kVWidth),     -w_uv_scale,    w_uv_scale)
    w_q_rope = small_uniform((kHidden,  kQRopeWidth), -w_q_rope_scl,  w_q_rope_scl)
    w_k_rope = small_uniform((kHidden,  kRopeDim),    -w_k_rope_scl,  w_k_rope_scl)

    cos_table, sin_table = build_rope_tables(kSeqLen, kRopeDim)

    # ---- Compressed Q path -----------------------------------------------
    c_q_fp32 = x.astype(np.float32) @ w_dq.astype(np.float32)
    c_q      = c_q_fp32.astype(np.float16)

    q_nope_fp32 = c_q.astype(np.float32) @ w_uq.astype(np.float32)
    q_nope      = q_nope_fp32.astype(np.float16)                    # [S, Nh*nope_d]

    # ---- KV compression / reconstruction --------------------------------
    c_kv_fp32 = x.astype(np.float32) @ w_dkv.astype(np.float32)
    c_kv      = c_kv_fp32.astype(np.float16)
    c_cache   = c_kv.copy()

    k_nope_fp32 = c_cache.astype(np.float32) @ w_uk.astype(np.float32)
    v_fp32      = c_cache.astype(np.float32) @ w_uv.astype(np.float32)
    k_nope      = k_nope_fp32.astype(np.float16)                    # [S, Nh*nope_d]
    v           = v_fp32.astype(np.float16)                          # [S, Nh*head_d]

    # ---- Q_rope (head-major), K_rope (shared) ---------------------------
    q_rope_flat_fp32 = x.astype(np.float32) @ w_q_rope.astype(np.float32)
    q_rope_flat      = q_rope_flat_fp32.astype(np.float16)           # [S, Nh*rope_d]
    q_rope = (q_rope_flat.reshape(kSeqLen, kNumHeads, kRopeDim)
                          .transpose(1, 0, 2).copy())                # [Nh, S, rope_d]

    k_rope_fp32 = x.astype(np.float32) @ w_k_rope.astype(np.float32)
    k_rope      = k_rope_fp32.astype(np.float16)                     # [S, rope_d]

    q_rope_rot = apply_rope_half(q_rope, cos_table, sin_table)
    k_rope_rot = apply_rope_half(k_rope, cos_table, sin_table)

    # ---- Attention -------------------------------------------------------
    scale = np.float16(1.0 / math.sqrt(float(kHeadDim)))             # 1/sqrt(128)

    scores_nope = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    scores_rope = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    scores      = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    probs       = np.zeros((kNumHeads, kSeqLen, kSeqLen), dtype=np.float16)
    out         = np.zeros((kSeqLen, kVWidth),            dtype=np.float16)

    q_nope_h = q_nope.reshape(kSeqLen, kNumHeads, kNopeDim)
    k_nope_h = k_nope.reshape(kSeqLen, kNumHeads, kNopeDim)
    v_h      = v.reshape    (kSeqLen, kNumHeads, kHeadDim)

    for h in range(kNumHeads):
        Q_n = q_nope_h[:, h, :]              # [S, nope_d]
        K_n = k_nope_h[:, h, :]              # [S, nope_d]
        V_h = v_h[:,    h, :]                # [S, head_d]
        Q_r = q_rope_rot[h]                  # [S, rope_d]
        K_r = k_rope_rot                     # [S, rope_d]  shared

        s_n_fp32 = Q_n.astype(np.float32) @ K_n.astype(np.float32).T
        s_r_fp32 = Q_r.astype(np.float32) @ K_r.astype(np.float32).T
        s_n_fp16 = s_n_fp32.astype(np.float16)
        s_r_fp16 = s_r_fp32.astype(np.float16)
        scores_nope[h] = s_n_fp16
        scores_rope[h] = s_r_fp16

        s_sum = (s_n_fp16 + s_r_fp16).astype(np.float16)
        scores[h] = s_sum

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
    w_dq.tofile     ("./input/input_w_dq.bin")
    w_uq.tofile     ("./input/input_w_uq.bin")
    w_dkv.tofile    ("./input/input_w_dkv.bin")
    w_uk.tofile     ("./input/input_w_uk.bin")
    w_uv.tofile     ("./input/input_w_uv.bin")
    w_q_rope.tofile ("./input/input_w_q_rope.bin")
    w_k_rope.tofile ("./input/input_w_k_rope.bin")
    cos_table.tofile("./input/input_cos.bin")
    sin_table.tofile("./input/input_sin.bin")

    c_q.tofile         ("./output/golden_c_q.bin")           # [S, kQLatent]
    q_nope.tofile      ("./output/golden_q.bin")             # [S, Nh*nope_d]  (Q_nope)
    c_kv.tofile        ("./output/golden_c_kv.bin")
    k_nope.tofile      ("./output/golden_k.bin")             # [S, Nh*nope_d]  (K_nope)
    v.tofile           ("./output/golden_v.bin")             # [S, Nh*head_d]
    q_rope.tofile      ("./output/golden_q_rope.bin")
    k_rope.tofile      ("./output/golden_k_rope.bin")
    q_rope_rot.tofile  ("./output/golden_q_rope_rot.bin")
    k_rope_rot.tofile  ("./output/golden_k_rope_rot.bin")
    scores_nope.tofile ("./output/golden_scores_nope.bin")
    scores_rope.tofile ("./output/golden_scores_rope.bin")
    scores.tofile      ("./output/golden_scores.bin")
    probs.tofile       ("./output/golden_probs.bin")
    out.tofile         ("./output/golden_out.bin")

    # ---- Debug summary ---------------------------------------------------
    print(f"[gen_data] DeepSeek-V2 MLA  B=1  S={kSeqLen}  H={kHidden}  Nh={kNumHeads}")
    print(f"           head_dim={kHeadDim}  nope_dim={kNopeDim}  rope_dim={kRopeDim}  "
          f"q_latent={kQLatent}  kv_latent={kLatent}")
    print(f"           scale=1/sqrt({kHeadDim})={float(scale):.6f}")

    def stats(name, arr):
        a = arr.astype(np.float32)
        print(f"[gen_data] {name:>12s}: shape={list(arr.shape)} "
              f"min={a.min():+.4f} max={a.max():+.4f} "
              f"mean={a.mean():+.4f} std={a.std():.4f}")

    stats("x",            x)
    stats("w_dq",         w_dq)
    stats("w_uq",         w_uq)
    stats("w_uk",         w_uk)
    stats("w_uv",         w_uv)
    stats("c_q",          c_q)
    stats("q_nope",       q_nope)
    stats("c_kv",         c_kv)
    stats("k_nope",       k_nope)
    stats("v",            v)
    stats("q_rope",       q_rope)
    stats("k_rope",       k_rope)
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
    kNopeDim  = 64
    kQLatent  = 64
    gen_golden_data(kSeqLen, kHidden, kNumHeads, kHeadDim, kLatent, kRopeDim,
                    kNopeDim, kQLatent)
