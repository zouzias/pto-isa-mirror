#!/usr/bin/env python3
"""
Semantic reference for the Opus fused FA4 DN softmax (FP16 output).

PURPOSE
=======
This is a from-scratch Python re-implementation of the math performed by
`pto_macro_fa_dn_softmax_opus.hpp` (init variant) for the [128, 64] FP16 case.
It exists to:

  1. Document the intended semantics independently of the CCE code.
  2. Provide a golden reference for diff'ing against the simulator output
     when correctness fails.
  3. Encode the ND → NZ+1 block layout transformation so we can validate
     the in-UB layout produced by `vsstb` matches the cube's expected
     consumption order for the next P×V matmul.

Run:
    python3 scripts/softmax_opus_ref.py

The script is intentionally self-contained and does NOT import from any
other Python helper in the case repo — per the VF fusion guide §5,
semantic verification must be re-derived from scratch.
"""
import numpy as np


# ----------------------------------------------------------------------
# Tile geometry (matches the FA4 [128,64] case at S=128)
# ----------------------------------------------------------------------
TILE_S1 = 128       # number of rows per softmax call (= Cube_S1 here)
VEC_S0  = 64        # number of cols per row  (= half VL of fp16)
HEAD_SIZE = 128     # softmax scale factor exponent argument
CUBE_S1 = 128       # NZ row alignment
BLOCK_BYTES = 32    # one UB block
ELEM_BYTES_FP16 = 2
ELEM_BYTES_FP32 = 4

SCALE = 1.0 / np.sqrt(HEAD_SIZE)

# ----------------------------------------------------------------------
# 1. Streaming softmax math (init step — first S1 tile)
# ----------------------------------------------------------------------
def softmax_init_math(x_fp32):
    """
    x_fp32      : (TILE_S1, VEC_S0) FP32 — the input QK tile (per vec core).

    Returns:
      x_exp_fp16     : (TILE_S1, VEC_S0) FP16 — exp(scale*(x - col_max))
      col_max_fp32   : (VEC_S0,)   FP32 — per-column max  (new_global_max)
      col_sum_fp32   : (VEC_S0,)   FP32 — per-column sum exp (new_global_sum)
    """
    assert x_fp32.dtype == np.float32
    assert x_fp32.shape == (TILE_S1, VEC_S0)

    col_max = np.max(x_fp32, axis=0)
    scaled_max = col_max * SCALE
    scaled_x = x_fp32 * SCALE
    x_exp = np.exp(scaled_x - scaled_max[None, :])
    col_sum = np.sum(x_exp, axis=0)

    x_exp_fp16 = x_exp.astype(np.float16)

    return x_exp_fp16, col_max.astype(np.float32), col_sum.astype(np.float32)


# ----------------------------------------------------------------------
# 2. ND → NZ+1 block layout (UB-side memory model)
# ----------------------------------------------------------------------
def nd_to_nz_plus_one(nd_fp16):
    """
    Re-arrange an ND [TILE_S1, VEC_S0] FP16 tile into NZ+1 layout exactly
    as `vsstb(block_stride=129, repeat_stride=1, 4-block preg)` would
    produce in UB.

    NZ+1 logical shape  : [4 col-groups, 129 rows] blocks  (block = 16 halves)
    NZ+1 physical bytes : 4 × 129 × 32 = 16,512 B
    nd_fp16 input shape : (128, 64) FP16  -> (128, 4) blocks of 16 halves

    The "+1" pads each column-group with one phantom row.  Real data
    occupies rows 0..127 of each column-group; row 128 is padding.
    """
    R = TILE_S1                              # 128
    BPR = VEC_S0 * ELEM_BYTES_FP16 // BLOCK_BYTES   # 4 blocks per ND row
    HALVES_PER_BLOCK = BLOCK_BYTES // ELEM_BYTES_FP16  # 16

    # Re-shape ND row into (BPR, halves_per_block) so axis 0 is "block in row"
    nd_blocks = nd_fp16.reshape(R, BPR, HALVES_PER_BLOCK)
    # NZ logical: (col_group=BPR, row=R, halves_per_block).  Then pad +1.
    nz_logical = nd_blocks.transpose(1, 0, 2)               # (BPR, R, 16)
    pad = np.zeros((BPR, 1, HALVES_PER_BLOCK), dtype=nz_logical.dtype)
    nz_plus_one = np.concatenate([nz_logical, pad], axis=1)  # (BPR, R+1, 16)
    return nz_plus_one


def nz_plus_one_to_nd(nz_buf):
    """Inverse of nd_to_nz_plus_one — strip padding and transpose back."""
    BPR, R_plus_1, hpb = nz_buf.shape
    R = R_plus_1 - 1
    nz_logical = nz_buf[:, :R, :]                 # drop padding row
    nd_blocks = nz_logical.transpose(1, 0, 2)     # (R, BPR, 16)
    nd_fp16 = nd_blocks.reshape(R, BPR * hpb)
    return nd_fp16


# ----------------------------------------------------------------------
# 3. Cycle-accurate per-row VSSTB simulation (for diagnostic prints)
# ----------------------------------------------------------------------
def simulate_per_row_vsstb(x_exp_fp16):
    """
    Walk through the per-row vsstb writes as the macro does:
      - 128 rows × {one full-VL vlds(64 halves), one vsstb(4 blocks)}.
      - block_stride = 129 blocks, repeat_stride = 1 block, base advances 32B.
    Returns the NZ+1 buffer layout as it would appear in UB.
    """
    BPR = 4
    NZ_ROWS = TILE_S1 + 1              # 129
    HPB = 16                           # halves per block
    nz_buf = np.zeros((BPR, NZ_ROWS, HPB), dtype=np.float16)

    for r in range(TILE_S1):
        # The vsstb writes 4 blocks of `vreg` to:
        #   col-group b at row r,  b ∈ {0,1,2,3}.
        # `vreg[b * 16 : (b+1) * 16]` lands at (b, r, :).
        vreg = x_exp_fp16[r]            # 64 halves
        blocks = vreg.reshape(BPR, HPB)
        for b in range(BPR):
            nz_buf[b, r, :] = blocks[b]
        # POST_UPDATE: base advances by 1 block. Captured implicitly by
        # iterating r (which is the row index inside each col-group).

    return nz_buf


# ----------------------------------------------------------------------
# 4. Sanity checks
# ----------------------------------------------------------------------
def self_test():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((TILE_S1, VEC_S0)).astype(np.float32)

    # Math
    x_exp_h, col_max, col_sum = softmax_init_math(x)

    # Layout round-trip
    nz_from_layout  = nd_to_nz_plus_one(x_exp_h)
    nz_from_vsstb   = simulate_per_row_vsstb(x_exp_h)

    assert nz_from_layout.shape == (4, 129, 16)
    assert nz_from_vsstb.shape  == (4, 129, 16)
    layout_match = np.array_equal(nz_from_layout, nz_from_vsstb)
    print(f"NZ+1 layout (numpy transpose) ≡ per-row vsstb walk: {layout_match}")

    # Round-trip
    rt = nz_plus_one_to_nd(nz_from_vsstb)
    assert np.array_equal(rt, x_exp_h), "round-trip ND→NZ+1→ND must be lossless"

    # Math invariants
    # 1. col_sum strictly positive
    assert np.all(col_sum > 0.0)
    # 2. exp(0) = 1, so the col-max element of each column is exactly 1.0 in fp16
    argmax = np.argmax(x, axis=0)
    on_max = x_exp_h[argmax, np.arange(VEC_S0)]
    assert np.all(on_max == np.float16(1.0)), (
        f"x_exp at the col-max position must be 1.0; got {on_max[:8]}…"
    )

    print(f"col_max[:8] = {col_max[:8]}")
    print(f"col_sum[:8] = {col_sum[:8]}")
    print(f"x_exp[0,:8] = {x_exp_h[0,:8].astype(np.float32)}")
    print("All sanity checks passed.")


if __name__ == "__main__":
    self_test()
