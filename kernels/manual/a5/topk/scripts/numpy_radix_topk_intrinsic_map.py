#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# --------------------------------------------------------------------------------
"""
NumPy reference of the Radix-Select Top-K algorithm in `draft.cpp`.

Every NumPy expression below is annotated with the PTO/CCE intrinsic it stands
for. Tile shapes/dtypes match the device tiles. The five `phase{1..5}` functions
correspond 1-to-1 with `Phase1_..Phase5_` in draft.cpp.

Two levels of abstraction live side-by-side:
  * **Tile-level (`T*`) intrinsics**: TLOAD, TASSIGN, TEXPANDS, TADD, TSUB, TOR,
    TSHLS, TCVT, TCMPS, TCI, TSELS, TSEL, TROWMIN, TGATHER, TCONCAT_IMPL, TSTORE,
    TMOV, TSTORE. These match the PTO tile ops used directly in `draft.cpp`.
  * **SIMD-register (`v*`) intrinsics** that implement THISTOGRAM internally
    (see include/pto/npu/a5/THistogram.hpp): vbr, vlds (BRC_B8 / DINTLV_B8),
    vcmp_eq, chistv2 (Bin_N0/Bin_N1), vcvt (PART_EVEN/PART_ODD), vadd, vsts
    (INTLV_B32).

SIMD API conventions (matching the actual hardware):
  * **Every vector argument/return is a VL-byte register (or n×VL "long vector").**
    VL = 256 bytes ⇒ one register holds 256 u8 / 128 u16 / 64 u32 elements.
  * **Every predicate argument/return is a 256-bool register** (`PRED_LANES = 256`).
    A b{8,16,32} op only consults predicate positions {0, W, 2W, ...} where
    W = sizeof(elem) (1, 2, or 4 bytes). E.g. an fp32 op reads pred at indices
    0, 4, 8, ...; the in-between bits are don't-cares (kept True for clarity).
  * `create_predicate(n, elem_bytes=W)` returns a 256-bool with the first `n`
    elements active under the b{8,16,32} stride.

Layout conventions
------------------
* ``HistTile``         : uint32, 1 row x 256 cols           (THISTOGRAM out / chist)
* ``WinnerLaneTile``   : uint32, 1 row x 256 cols           (TSELS lanes)
* ``WinnerBinTile``    : uint32, 1 row x  32 cols           (TGATHER broadcast)
* ``RowMinDstTile``    : uint32, 1 row x  16 cols (col0 used by TROWMIN)
* ``RemainKTile``      : uint32, 1 row x  32 cols           (broadcast scalar)
* ``PackedU16Tile``    : uint16, 1 row x  32 cols           (packed threshold)
* ``IdxFilterTile``    : uint8                              (THISTOGRAM filter)
* ``MaskCmpTile``      : uint8                              (TCMPS predicate)

The PTO `TSELS(dst, mask, idx_tile, tmp, false_value)` is the "indexed select":
``dst[i] = mask[i] ? idx_tile[i] : false_value``.
`TROWMIN(dst, src, tmp)` writes ``dst[0] = min(src)`` to lane 0.
`TGATHER(dst, table, idx, tmp)` is ``dst[i] = table[idx[i]]``; with ``idx[:]==0``
this is a broadcast of ``table[0]`` to all 32 lanes.
`THISTOGRAM<BYTE_k>(dst, src, idx_filter)` returns an **ascending cumulative**
histogram of byte-k of ``src`` (optionally filtered by ``idx_filter``):
``dst[b] = |{i : byte_k(src[i]) <= b and (filter==None or msb(src[i])==filter)}|``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants — must match draft.cpp's `topk_radix_detail` namespace.            #
# --------------------------------------------------------------------------- #
N = 8192
TOPK = 512
TILE_COLS = 2048
HIST_CHUNK_COLS = 2048
CHUNK_COLS = 256
BIN_NUM = 256
SELS_FALSE = 0xFFFFFFFF  # default value written by TSELS when mask=false

# --------------------------------------------------------------------------- #
# SIMD register widths — from <pto/common/constants.hpp>: REPEAT_BYTE = 256.    #
#                                                                            #
# Every vector register is **VL = 256 bytes wide**; an n×VL "long vector" is  #
# n consecutive registers operated on as a unit (e.g. THistogram's per-row    #
# work). The element count per register depends on the dtype:                 #
#   VL_B8  = 256 u8  lanes  (W = 1)                                          #
#   VL_B16 = 128 u16 lanes  (W = 2)                                          #
#   VL_B32 =  64 u32 lanes  (W = 4)                                          #
#                                                                            #
# **Predicate registers are always 256 booleans** — one bool per byte lane of #
# a VL register. An op on dtype of width W only consults predicate positions  #
# {0, W, 2W, ...}; the in-between bits are don't-cares (set True for clarity).#
# --------------------------------------------------------------------------- #
VL = 256
VL_B8 = VL
VL_B16 = VL // 2
VL_B32 = VL // 4
PRED_LANES = VL


# --------------------------------------------------------------------------- #
# Primitive PTO ops — pure NumPy stand-ins.                                    #
# --------------------------------------------------------------------------- #
def _u32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.uint32)


def t_expands(shape: Tuple[int, ...], value, dtype=np.uint32) -> np.ndarray:
    """TEXPANDS(dst, scalar): broadcast `scalar` to every element of dst."""
    return np.full(shape, value, dtype=dtype)


def t_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TADD(dst, a, b): elementwise add (wrap-around in the chosen dtype)."""
    return (a + b).astype(a.dtype, copy=False)


def t_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TSUB(dst, a, b): elementwise sub (wrap-around in the chosen dtype)."""
    return (a - b).astype(a.dtype, copy=False)


def t_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TOR(dst, a, b): elementwise bitwise OR."""
    return (a | b).astype(a.dtype, copy=False)


def t_shls(src: np.ndarray, shift: int) -> np.ndarray:
    """TSHLS(dst, src, imm): elementwise logical shift-left by immediate."""
    return (src.astype(np.uint32) << shift).astype(src.dtype, copy=False)


def t_cvt(src: np.ndarray, dst_dtype) -> np.ndarray:
    """TCVT(dst, src, CAST_TRUNC): bit-truncating numeric cast."""
    return src.astype(dst_dtype, copy=False)


def t_cmps_ge(src: np.ndarray, thr) -> np.ndarray:
    """TCMPS(mask, src, thr, GE): mask[i] = (src[i] >= thr) as 0/1 uint8."""
    return (src >= thr).astype(np.uint8)


def t_cmps_gt(src: np.ndarray, thr) -> np.ndarray:
    """TCMPS(mask, src, thr, GT): mask[i] = (src[i] >  thr) as 0/1 uint8."""
    return (src > thr).astype(np.uint8)


def t_cmps_eq(src: np.ndarray, thr) -> np.ndarray:
    """TCMPS(mask, src, thr, EQ): mask[i] = (src[i] == thr) as 0/1 uint8."""
    return (src == thr).astype(np.uint8)


def t_ci(start: int, length: int) -> np.ndarray:
    """TCI(dst, start): dst[i] = start + i (uint32 counter init)."""
    return (np.arange(length, dtype=np.uint32) + np.uint32(start))


def t_sels(mask: np.ndarray, idx_tile: np.ndarray, false_val: int) -> np.ndarray:
    """TSELS(dst, mask, idx, tmp, false_val): dst[i] = mask[i] ? idx[i] : false_val."""
    out = np.full(idx_tile.shape, false_val, dtype=idx_tile.dtype)
    out[mask.astype(bool)] = idx_tile[mask.astype(bool)]
    return out


def t_sel(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """TSEL(dst, mask, a, b, tmp): dst[i] = mask[i] ? a[i] : b[i]."""
    return np.where(mask.astype(bool), a, b).astype(a.dtype, copy=False)


def t_rowmin(src: np.ndarray) -> np.ndarray:
    """TROWMIN(dst, src, tmp): write the row-min into dst[0]; remaining lanes are scratch."""
    out = np.zeros(16, dtype=np.uint32)
    out[0] = np.uint32(src.min())
    return out


def t_gather_broadcast(table: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """TGATHER(dst, table, idx, tmp): dst[i] = table[idx[i]] (used here for broadcast)."""
    return table[idx.astype(np.int64)].astype(table.dtype, copy=False)


def t_load(gm: np.ndarray, base: int, valid: int, tile_capacity: int) -> np.ndarray:
    """TLOAD(ubTile, gmTile): MTE2 DMA copy `valid` elements from GM[base..base+valid) into a UB tile.

    This is a bulk **MTE2 pipe** transfer, *not* a vector load — the per-lane `vlds` happens later
    inside the vector pipe (see `t_histogram_simd` for the SIMD register-level inner loop).
    """
    tile = np.zeros(tile_capacity, dtype=gm.dtype)
    tile[:valid] = gm[base:base + valid]
    return tile


# --------------------------------------------------------------------------- #
# SIMD register primitives — transliterations of the CCE intrinsics used in   #
# pto::THistogram / pto::histogram_b8i_b32o (see include/pto/npu/a5/THistogram.hpp). #
#                                                                            #
# Convention: every vector argument is a VL-byte register (or n×VL long       #
# vector) represented as a numpy array of `VL // sizeof(elem)` elements with  #
# the appropriate dtype. Every predicate argument is a numpy bool array of    #
# length PRED_LANES (= 256); ops with element width W consult positions       #
# {0, W, 2W, ...} via `_pred_take_b{8,16,32}`.                                #
# --------------------------------------------------------------------------- #
def _pred_take_b8(pred256: np.ndarray) -> np.ndarray:
    """Extract the 256-element bool mask consulted by b8 ops."""
    return pred256.astype(bool)


def _pred_take_b16(pred256: np.ndarray) -> np.ndarray:
    """Extract the 128-element bool mask consulted by b16 ops (stride 2 over PRED_LANES)."""
    return pred256[0::2].astype(bool)


def _pred_take_b32(pred256: np.ndarray) -> np.ndarray:
    """Extract the 64-element bool mask consulted by b32 ops (stride 4 over PRED_LANES)."""
    return pred256[0::4].astype(bool)


def vbr(elem_bytes: int = 4, value=0, dtype=np.uint32) -> np.ndarray:
    """vbr(reg, value): broadcast scalar `value` into every active lane of a VL-byte register.

    Returns `VL // elem_bytes` elements (the consulted lanes of one VL register).
    """
    return np.full(VL // elem_bytes, value, dtype=dtype)


def create_predicate(num_active_elements: int, elem_bytes: int = 1) -> np.ndarray:
    """CreatePredicate<T>(sreg): 256-bool predicate; True at byte positions {0, W, ..., (n-1)*W}
    for `n = num_active_elements` elements, where W = sizeof(T).

    Used by the THistogram inner loop to mask off the tail of the last repeat.
    """
    pred = np.zeros(PRED_LANES, dtype=bool)
    max_elements = PRED_LANES // elem_bytes
    n = max(0, min(num_active_elements, max_elements))
    pred[0 : n * elem_bytes : elem_bytes] = True
    return pred


def create_predicate_b8(active_lanes: int) -> np.ndarray:
    """CreatePredicate<uint8_t>(sreg): convenience wrapper — first `active_lanes` byte lanes True."""
    return create_predicate(active_lanes, elem_bytes=1)


def pset_all() -> np.ndarray:
    """pset_b{8,16,32}(PAT_ALL): full-True 256-bool predicate.

    All stride positions become True; b16/b32 ops simply consult every 2nd/4th bit.
    """
    return np.ones(PRED_LANES, dtype=bool)


def pset_b8_all() -> np.ndarray:
    """pset_b8(PAT_ALL): 256-bool, all True (b8 stride consults every bit)."""
    return pset_all()


def pset_b16_all() -> np.ndarray:
    """pset_b16(PAT_ALL): 256-bool, all True (b16 stride consults bits 0, 2, 4, ...)."""
    return pset_all()


def pset_b32_all() -> np.ndarray:
    """pset_b32(PAT_ALL): 256-bool, all True (b32 stride consults bits 0, 4, 8, ...)."""
    return pset_all()


def vlds_brc_b8(ub_u8: np.ndarray, byte_offset: int) -> np.ndarray:
    """vlds(reg, idxPtr, 1, BRC_B8, POST_UPDATE): broadcast one byte from UB to all VL_B8 lanes.

    The `POST_UPDATE` form also advances `idxPtr` by 1 byte; the caller models that by
    passing `byte_offset = row_index` since each row provides one filter byte.
    """
    return np.full(VL_B8, ub_u8[byte_offset], dtype=np.uint8)


def vlds_dintlv_b8_u16(ub_u16: np.ndarray, elem_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """vlds(vb8_LSB, vb8_MSB, srcPtr, byteOff, DINTLV_B8) for a uint16 source.

    Consumes a **2×VL byte** source window (= 256 uint16 elements = 512 bytes) starting at
    `elem_offset` (in u16 elements) and **deinterleaves** it into two VL byte registers:
      * `lsb_vec[i] = src[i] & 0xff`   (one 256-lane uint8 vector register)
      * `msb_vec[i] = src[i] >> 8`     (one 256-lane uint8 vector register)
    Tail lanes beyond `validCols` are zeroed.
    """
    block = np.zeros(VL_B8, dtype=np.uint16)
    end = min(elem_offset + VL_B8, ub_u16.size)
    block[: end - elem_offset] = ub_u16[elem_offset:end].astype(np.uint16)
    lsb_vec = (block & 0xFF).astype(np.uint8)
    msb_vec = (block >> 8).astype(np.uint8)
    return lsb_vec, msb_vec


def vcmp_eq_b8(a_u8: np.ndarray, b_u8: np.ndarray, src_pred256: np.ndarray) -> np.ndarray:
    """vcmp_eq(dst_pred, a, b, src_pred): dst_pred[i] = src_pred[i] AND (a[i] == b[i]).

    Both `src_pred` and the returned `dst_pred` are 256-bool predicate registers (b8 stride).
    """
    out = np.zeros(PRED_LANES, dtype=bool)
    active = _pred_take_b8(src_pred256)
    out[:] = active & (a_u8 == b_u8)
    return out


def chistv2(src_u8: np.ndarray, pred256: np.ndarray, bin_part: int) -> np.ndarray:
    """chistv2(dst_u16, src_u8, pred, Bin_N{0|1}): per-repeat *cumulative* histogram.

    `src_u8` is one VL byte register (256 lanes). `pred256` is a 256-bool predicate (b8 stride).
    Returns one VL register holding VL_B16 (=128) u16 elements:
      * bin_part == 0  (Bin_N0)  -> dst[b]  = |{i : src[i] <= b      AND pred[i]}|, b in [0..127]
      * bin_part == 1  (Bin_N1)  -> dst[b'] = |{i : src[i] <= 128+b' AND pred[i]}|, b' in [0..127]

    Concatenating N0 (low 128 lanes) and N1 (high 128 lanes) yields the full ascending cumulative
    histogram over byte values 0..255 for the **current 256-byte repeat**. Per-repeat counts are then
    summed across repeats with `vadd` to obtain the global cumulative histogram.
    """
    active = _pred_take_b8(pred256)
    selected = src_u8[active]
    counts = np.bincount(selected.astype(np.int64), minlength=256).astype(np.uint32)
    cum256 = np.cumsum(counts).astype(np.uint32)
    base = 0 if bin_part == 0 else 128
    return cum256[base:base + VL_B16].astype(np.uint16)


def vcvt_b16_to_b32(src_u16: np.ndarray, pred256: np.ndarray, part: str) -> np.ndarray:
    """vcvt(dst_u32, src_u16, pred, PART_{EVEN,ODD}): widen half of the u16 lanes to u32.

    Source is one VL register (128 u16 elements); output is one VL register (64 u32 elements).
    Predicate is the b16 source predicate (256 bools; positions {0, 2, 4, ...} consulted).
    """
    pred_b16 = _pred_take_b16(pred256)                # 128 bools matching the 128 u16 source lanes
    if part == "EVEN":
        widened_active = pred_b16[0::2]               # 64 lanes corresponding to even u16 positions
        widened_vals = src_u16[0::2].astype(np.uint32)
    elif part == "ODD":
        widened_active = pred_b16[1::2]               # 64 lanes corresponding to odd u16 positions
        widened_vals = src_u16[1::2].astype(np.uint32)
    else:
        raise ValueError(part)
    out = np.zeros(VL_B32, dtype=np.uint32)
    out[widened_active] = widened_vals[widened_active]
    return out


def vcvt_b16_to_b32_part_even(src_u16: np.ndarray, pred256: np.ndarray | None = None) -> np.ndarray:
    """Backwards-compatible shim for `vcvt PART_EVEN`; defaults to a full predicate."""
    return vcvt_b16_to_b32(src_u16, pred256 if pred256 is not None else pset_b16_all(), "EVEN")


def vcvt_b16_to_b32_part_odd(src_u16: np.ndarray, pred256: np.ndarray | None = None) -> np.ndarray:
    """Backwards-compatible shim for `vcvt PART_ODD`; defaults to a full predicate."""
    return vcvt_b16_to_b32(src_u16, pred256 if pred256 is not None else pset_b16_all(), "ODD")


def vadd_u32(acc: np.ndarray, addend: np.ndarray,
             pred256: np.ndarray | None = None, mode: str = "zeroing") -> np.ndarray:
    """vadd(dst, acc, addend, pred, MODE_{ZEROING,MERGING}): elementwise u32 add under predicate.

    `pred256` defaults to full-true. With MODE_ZEROING, inactive lanes are written 0; with
    MODE_MERGING they retain the prior `acc` value. Both inputs are VL u32 registers (64 lanes).
    """
    if pred256 is None:
        return (acc + addend).astype(np.uint32)
    active = _pred_take_b32(pred256)
    out = np.zeros_like(acc) if mode == "zeroing" else acc.copy()
    out[active] = (acc[active] + addend[active]).astype(np.uint32)
    return out.astype(np.uint32)


def vsts_intlv_b32(ub_dst_u32: np.ndarray, dst_elem_offset: int,
                   even_u32: np.ndarray, odd_u32: np.ndarray,
                   pred256: np.ndarray | None = None) -> None:
    """vsts(even, odd, dstPtr, off, INTLV_B32, pred): interleave 64+64 u32 lanes → 128 u32 in UB.

        ub_dst_u32[off + 2*i]     = even[i]
        ub_dst_u32[off + 2*i + 1] = odd [i]    for i in [0..VL_B32) where pred is active

    `pred256` defaults to full-true; only positions {0, 4, ...} are consulted (b32 stride).
    """
    active = _pred_take_b32(pred256) if pred256 is not None else np.ones(VL_B32, dtype=bool)
    even_slot = slice(dst_elem_offset + 0, dst_elem_offset + 2 * VL_B32, 2)
    odd_slot  = slice(dst_elem_offset + 1, dst_elem_offset + 2 * VL_B32, 2)
    e_view = ub_dst_u32[even_slot]
    o_view = ub_dst_u32[odd_slot]
    e_view[active] = even_u32[active]
    o_view[active] = odd_u32[active]
    ub_dst_u32[even_slot] = e_view
    ub_dst_u32[odd_slot]  = o_view


# --------------------------------------------------------------------------- #
# SIMD u32 primitives — each call models ONE VL register operation.            #
#                                                                            #
# A multi-VL tile (e.g. the 256-bin chist = 4 × VL_B32) is consumed by these  #
# primitives in an explicit Python loop:                                      #
#                                                                            #
#     for c in range(n_chunks):                                              #
#         vb32_x = vlds_u32(ub_tile, c * VL_B32)         # 1 instruction     #
#         vb32_y = vop_u32(vb32_x, ..., preg_full_b32)   # 1 instruction     #
#         vsts_u32(ub_dst,  c * VL_B32, vb32_y)          # 1 instruction     #
#                                                                            #
# Sub-VL tiles (e.g. 32-lane broadcast tiles) are handled by a single VL op   #
# with `create_predicate(32, elem_bytes=4)` restricting writes to the first   #
# 32 u32 lanes; the remaining 32 lanes are don't-cares.                       #
# --------------------------------------------------------------------------- #
def vlds_u32(ub_tile_u32: np.ndarray, elem_offset: int,
             pred256: np.ndarray | None = None) -> np.ndarray:
    """vlds(vb32, srcPtr, off, B32, pred): load one VL_B32 u32 chunk from a UB tile.

    Returns one VL register (`VL_B32 = 64` u32 lanes); lanes past the tile end are zero.
    """
    out = np.zeros(VL_B32, dtype=np.uint32)
    avail = max(0, min(VL_B32, ub_tile_u32.size - elem_offset))
    if avail > 0:
        out[:avail] = ub_tile_u32[elem_offset:elem_offset + avail].astype(np.uint32)
    return out


def vsts_u32(ub_tile_u32: np.ndarray, elem_offset: int, vb32: np.ndarray,
             pred256: np.ndarray | None = None) -> None:
    """vsts(vb32, dstPtr, off, B32, pred): store one VL_B32 u32 chunk into a UB tile."""
    avail = max(0, min(VL_B32, ub_tile_u32.size - elem_offset))
    if avail == 0:
        return
    active = _pred_take_b32(pred256) if pred256 is not None else np.ones(VL_B32, dtype=bool)
    write_lanes = active[:avail]
    view = ub_tile_u32[elem_offset:elem_offset + avail]
    view[write_lanes] = vb32[:avail][write_lanes]
    ub_tile_u32[elem_offset:elem_offset + avail] = view


def vci_u32(start_value: int, pred256: np.ndarray | None = None) -> np.ndarray:
    """vci_u32(vb32, start, pred): vb32[i] = start + i for i in [0..VL_B32), masked by pred."""
    seq = (np.arange(VL_B32, dtype=np.uint32) + np.uint32(start_value))
    if pred256 is None:
        return seq
    active = _pred_take_b32(pred256)
    out = np.zeros(VL_B32, dtype=np.uint32)
    out[active] = seq[active]
    return out


def vbr_u32(value, pred256: np.ndarray | None = None) -> np.ndarray:
    """vbr_u32(vb32, scalar, pred): broadcast scalar to active VL_B32 lanes; inactive lanes are 0."""
    full = np.full(VL_B32, np.uint32(value), dtype=np.uint32)
    if pred256 is None:
        return full
    active = _pred_take_b32(pred256)
    full[~active] = 0
    return full


def vcmps_u32(src_vb32: np.ndarray, thr_scalar, src_pred256: np.ndarray, op: str) -> np.ndarray:
    """vcmps_u32(dst_pred, vb32, thr, src_pred, {GE,GT,EQ,LT,LE,NE}): scalar-threshold compare.

    Returns a 256-bool predicate (b32 stride; positions {0,4,...} carry the per-element result).
    """
    active = _pred_take_b32(src_pred256)
    thr = np.uint32(thr_scalar)
    if   op == "GE": elem = active & (src_vb32 >= thr)
    elif op == "GT": elem = active & (src_vb32 >  thr)
    elif op == "EQ": elem = active & (src_vb32 == thr)
    elif op == "LT": elem = active & (src_vb32 <  thr)
    elif op == "LE": elem = active & (src_vb32 <= thr)
    elif op == "NE": elem = active & (src_vb32 != thr)
    else: raise ValueError(op)
    out = np.zeros(PRED_LANES, dtype=bool)
    out[0:PRED_LANES:4] = elem
    return out


def vsels_u32(mask_pred256: np.ndarray, idx_vb32: np.ndarray, false_val,
              src_pred256: np.ndarray | None = None) -> np.ndarray:
    """vsels_u32(vb32_dst, mask, idx, false): vb32_dst[i] = mask[i] ? idx[i] : false."""
    out = np.full(VL_B32, np.uint32(false_val), dtype=np.uint32)
    sel = _pred_take_b32(mask_pred256)
    out[sel] = idx_vb32[sel]
    return out


def vsel_u32(mask_pred256: np.ndarray, a_vb32: np.ndarray, b_vb32: np.ndarray,
             src_pred256: np.ndarray | None = None) -> np.ndarray:
    """vsel_u32(vb32_dst, mask, a, b): vb32_dst[i] = mask[i] ? a[i] : b[i]."""
    sel = _pred_take_b32(mask_pred256)
    out = b_vb32.copy()
    out[sel] = a_vb32[sel]
    return out


def vsub_u32(a_vb32: np.ndarray, b_vb32: np.ndarray,
             pred256: np.ndarray | None = None, mode: str = "zeroing") -> np.ndarray:
    """vsub_u32(vb32_dst, a, b, pred, MODE_{ZEROING,MERGING}): elementwise u32 subtract."""
    if pred256 is None:
        return (a_vb32 - b_vb32).astype(np.uint32)
    active = _pred_take_b32(pred256)
    out = np.zeros_like(a_vb32) if mode == "zeroing" else a_vb32.copy()
    out[active] = (a_vb32[active] - b_vb32[active]).astype(np.uint32)
    return out.astype(np.uint32)


def vmin_u32(a_vb32: np.ndarray, b_vb32: np.ndarray,
             pred256: np.ndarray | None = None, mode: str = "merging") -> np.ndarray:
    """vmin_u32(vb32_dst, a, b, pred, MODE): elementwise u32 min within one VL register."""
    if pred256 is None:
        return np.minimum(a_vb32, b_vb32).astype(np.uint32)
    active = _pred_take_b32(pred256)
    out = np.zeros_like(a_vb32) if mode == "zeroing" else a_vb32.copy()
    out[active] = np.minimum(a_vb32[active], b_vb32[active]).astype(np.uint32)
    return out.astype(np.uint32)


def vrmin_u32(vb32: np.ndarray, pred256: np.ndarray | None = None) -> int:
    """vrmin_u32(sreg, vb32, pred): cross-lane u32 min reduction of one VL register → scalar."""
    if pred256 is None:
        pred256 = pset_b32_all()
    active = _pred_take_b32(pred256)
    if not active.any():
        return np.uint32(0xFFFFFFFF)
    return np.uint32(vb32[active].min())


def vgather_u32(table_u32: np.ndarray, idx_vb32: np.ndarray,
                pred256: np.ndarray | None = None) -> np.ndarray:
    """vgather_u32(vb32_dst, table, idx, pred): vb32_dst[i] = table[idx[i]] on active lanes."""
    if pred256 is None:
        pred256 = pset_b32_all()
    active = _pred_take_b32(pred256)
    out = np.zeros(VL_B32, dtype=np.uint32)
    if active.any():
        idx_clip = np.clip(idx_vb32[active].astype(np.int64), 0, table_u32.size - 1)
        out[active] = table_u32[idx_clip].astype(np.uint32)
    return out


def vshls_u16(src_vb16: np.ndarray, shift_imm: int,
              pred256: np.ndarray | None = None, mode: str = "merging") -> np.ndarray:
    """vshls_u16(vb16_dst, src, imm, pred, MODE): elementwise u16 logical-shift-left by immediate."""
    shifted = ((src_vb16.astype(np.uint32) << shift_imm) & 0xFFFF).astype(np.uint16)
    if pred256 is None:
        return shifted
    active = _pred_take_b16(pred256)
    out = np.zeros_like(src_vb16) if mode == "zeroing" else src_vb16.copy()
    out[active] = shifted[active]
    return out


def vor_u16(a_vb16: np.ndarray, b_vb16: np.ndarray,
            pred256: np.ndarray | None = None, mode: str = "merging") -> np.ndarray:
    """vor_u16(vb16_dst, a, b, pred, MODE): elementwise u16 bitwise OR."""
    full = (a_vb16 | b_vb16).astype(np.uint16)
    if pred256 is None:
        return full
    active = _pred_take_b16(pred256)
    out = np.zeros_like(a_vb16) if mode == "zeroing" else a_vb16.copy()
    out[active] = full[active]
    return out


def vcvt_u32_to_u16(src_vb32: np.ndarray, pred256: np.ndarray | None = None) -> np.ndarray:
    """vcvt(vb16_dst, vb32_src, pred, CAST_TRUNC): narrow VL_B32 u32 lanes to u16 (truncating).

    Output is one VL register holding VL_B16 (=128) u16 elements; only the first
    VL_B32 (=64) lanes carry meaningful data (the narrowed u32 values). The upper
    half is zero — a consumer using the b16 predicate at positions {0, 2, ...,
    2*(VL_B32-1)} will only ever read the lower half.
    """
    out = np.zeros(VL_B16, dtype=np.uint16)
    if pred256 is None:
        out[:VL_B32] = (src_vb32 & 0xFFFF).astype(np.uint16)
        return out
    active = _pred_take_b32(pred256)
    lower = np.zeros(VL_B32, dtype=np.uint16)
    lower[active] = (src_vb32[active] & 0xFFFF).astype(np.uint16)
    out[:VL_B32] = lower
    return out


# --------------------------------------------------------------------------- #
# SIMD-register-level THISTOGRAM — mirrors pto::THistogram<isMSB> 1:1.         #
# --------------------------------------------------------------------------- #
def t_histogram_simd(in_tile_u16: np.ndarray, valid_rows: int, valid_cols: int,
                     idx_filter_u8: np.ndarray, *, is_msb: bool) -> np.ndarray:
    """Transliteration of `pto::THistogram<TileDst, TileSrc, TileIdx, isMSB>`.

    Inputs are UB tiles already populated by TLOAD (MTE2). The vector pipe walks the tile in
    VL_B8-wide repeats, building four u32 accumulator registers per row (N0/N1 x even/odd) and
    finally interleaving them back into a 256-lane uint32 histogram in UB.

    Returns a (valid_rows, 256) uint32 array; when valid_rows==1, the row vector is returned.
    """
    bin_count = np.zeros((valid_rows, 2 * VL_B16), dtype=np.uint32)
    repeat_times_per_row = (valid_cols + VL_B8 - 1) // VL_B8
    row_stride_u16 = in_tile_u16.shape[-1]
    flat_u16 = in_tile_u16.reshape(-1)

    # `preg_all_b16` / `preg_all_b32` are full-true 256-bool predicates used by vcvt / vadd / vsts.
    preg_all_b16 = pset_b16_all()
    preg_all_b32 = pset_b32_all()

    # `vbr(vb16_BIN_N0, 0)` / `vbr(vb16_BIN_N1, 0)` outside the row loop — the per-chistv2 reset
    # at the end of histogram_b8i_b32o keeps them zeroed for every repeat.
    for r in range(valid_rows):
        # vlds(vb8_idx, idxPtr, 1, BRC_B8, POST_UPDATE) — broadcast one filter byte (row r).
        vb8_idx = vlds_brc_b8(idx_filter_u8, r)

        # vbr(<accumulators>, 0): zero the four u32 partial-sum VL registers for this row.
        vb32_n0_even_inc = vbr(elem_bytes=4, value=0, dtype=np.uint32)
        vb32_n0_odd_inc  = vbr(elem_bytes=4, value=0, dtype=np.uint32)
        vb32_n1_even_inc = vbr(elem_bytes=4, value=0, dtype=np.uint32)
        vb32_n1_odd_inc  = vbr(elem_bytes=4, value=0, dtype=np.uint32)

        sreg_even = valid_cols   # CreatePredicate source register for N0 (BYTE_LSB half)
        sreg_odd  = valid_cols   # CreatePredicate source register for N1 (BYTE_MSB half)

        for c in range(repeat_times_per_row):
            consumed = c * VL_B8
            preg_b8_0 = create_predicate_b8(sreg_even - consumed)  # CreatePredicate<uint8_t>(sreg_even) -> 256 bools
            preg_b8_1 = create_predicate_b8(sreg_odd  - consumed)  # CreatePredicate<uint8_t>(sreg_odd)  -> 256 bools

            # vlds DINTLV_B8 on uint16 source → (LSB byte vec, MSB byte vec), each VL_B8 lanes.
            elem_off = r * row_stride_u16 + consumed
            vb8_src_LSB, vb8_src_MSB = vlds_dintlv_b8_u16(flat_u16, elem_off)

            if is_msb:
                # MSB pass (BYTE_1): histogram the MSB byte under the lane-active predicate.
                src_for_hist = vb8_src_MSB
                pred0 = preg_b8_0
                pred1 = preg_b8_1
            else:
                # LSB pass (BYTE_0): vcmp_eq filter — keep lanes where MSB == filter byte.
                pred_filter = vcmp_eq_b8(vb8_src_MSB, vb8_idx, preg_b8_0)
                src_for_hist = vb8_src_LSB
                pred0 = pred_filter
                pred1 = pred_filter

            # ---- histogram_b8i_b32o: cumulative-128 × 2 → vcvt widen → vadd accumulate ----
            vb16_BIN_N0 = chistv2(src_for_hist, pred0, bin_part=0)   # u16[128], bins   0..127
            vb16_BIN_N1 = chistv2(src_for_hist, pred1, bin_part=1)   # u16[128], bins 128..255
            # vcvt u16 → u32 splits each 128-lane u16 into two 64-lane u32 registers.
            vb32_n0_even = vcvt_b16_to_b32(vb16_BIN_N0, preg_all_b16, "EVEN")
            vb32_n0_odd  = vcvt_b16_to_b32(vb16_BIN_N0, preg_all_b16, "ODD")
            vb32_n1_even = vcvt_b16_to_b32(vb16_BIN_N1, preg_all_b16, "EVEN")
            vb32_n1_odd  = vcvt_b16_to_b32(vb16_BIN_N1, preg_all_b16, "ODD")
            # vadd MODE_ZEROING under the full b32 predicate → elementwise u32 accumulate.
            vb32_n0_even_inc = vadd_u32(vb32_n0_even_inc, vb32_n0_even, preg_all_b32, mode="zeroing")
            vb32_n0_odd_inc  = vadd_u32(vb32_n0_odd_inc,  vb32_n0_odd,  preg_all_b32, mode="zeroing")
            vb32_n1_even_inc = vadd_u32(vb32_n1_even_inc, vb32_n1_even, preg_all_b32, mode="zeroing")
            vb32_n1_odd_inc  = vadd_u32(vb32_n1_odd_inc,  vb32_n1_odd,  preg_all_b32, mode="zeroing")
            # vbr(vb16_BIN_N0, 0) / vbr(vb16_BIN_N1, 0): reset the chistv2 scratch u16 regs
            # so the next repeat's chistv2 produces *just* that repeat's cumulative counts.

        # vsts INTLV_B32: store this row's 256 bins.
        #   dstPtr            <-- (n0_even, n0_odd) at lane offset 0,   stride 256*r u32 (handled via row idx)
        #   dstPtr + 128 elem <-- (n1_even, n1_odd) at lane offset 128, stride 256*r u32
        vsts_intlv_b32(bin_count[r], 0,            vb32_n0_even_inc, vb32_n0_odd_inc, preg_all_b32)
        vsts_intlv_b32(bin_count[r], 2 * VL_B32,   vb32_n1_even_inc, vb32_n1_odd_inc, preg_all_b32)

    return bin_count[0] if valid_rows == 1 else bin_count


# --------------------------------------------------------------------------- #
# Phase 1 — MSB cumulative histogram (per-tile TLOAD + THISTOGRAM + TADD).     #
# --------------------------------------------------------------------------- #
def phase1_load_and_histogram_msb(src_gm: np.ndarray) -> np.ndarray:
    """Returns ``chistMSB`` (uint32[256], ascending cumulative)."""
    # TASSIGN tiles, TEXPANDS(chistMSB, 0u): start the accumulator at zero.
    chist_msb = t_expands((BIN_NUM,), 0, np.uint32)
    # The idx filter UB tile is unused on the MSB pass (isMSB=true skips vcmp_eq), but THistogram
    # still issues `vlds(vb8_idx, idxPtr, 1, BRC_B8, POST_UPDATE)` once per row, so we provide a byte.
    idx_filter_ub = np.zeros(1, dtype=np.uint8)

    n_chist_chunks = BIN_NUM // VL_B32  # 4: chistMSB is a 4 × VL_B32 long-vector.
    preg_full_b32 = pset_b32_all()

    n_loop = (N + TILE_COLS - 1) // TILE_COLS
    for i in range(n_loop):
        base = i * TILE_COLS
        valid = min(TILE_COLS, N - base)
        for sub in range(0, valid, HIST_CHUNK_COLS):
            sub_valid = min(HIST_CHUNK_COLS, valid - sub)
            # TLOAD (MTE2 pipe): GM -> UB tile. The vector pipe consumes this via `vlds` inside THISTOGRAM.
            in_tile = t_load(src_gm, base + sub, sub_valid, HIST_CHUNK_COLS)

            # THISTOGRAM<BYTE_1>(tileHist, inTile, idxFilter): SIMD-register-level expansion below.
            tile_hist = t_histogram_simd(in_tile.reshape(1, -1), valid_rows=1,
                                          valid_cols=sub_valid, idx_filter_u8=idx_filter_ub,
                                          is_msb=True)

            # TADD(chistMSB, chistMSB, tileHist): explicit 4 × VL_B32 SIMD loop.
            for c in range(n_chist_chunks):
                off = c * VL_B32
                vb32_acc  = vlds_u32(chist_msb, off)                                 # vlds B32
                vb32_inc  = vlds_u32(tile_hist, off)                                 # vlds B32
                vb32_sum  = vadd_u32(vb32_acc, vb32_inc, preg_full_b32, "merging")  # vadd u32
                vsts_u32(chist_msb, off, vb32_sum)                                   # vsts B32

    return chist_msb


# --------------------------------------------------------------------------- #
# Phase 2 — MSB winner bin + remainK.                                          #
# --------------------------------------------------------------------------- #
@dataclass
class Phase2Out:
    msb_winner_saved: np.ndarray   # WinnerBinTile uint32[32], = raw winner b
    msb_winner_bin: np.ndarray     # WinnerBinTile uint32[32], = max(0, raw-1) (TGATHER idx)
    remain_k_tile: np.ndarray      # RemainKTile  uint32[32], = (N-TopK) - C[winner-1]


def phase2_winner_msb_and_remain_k(chist_msb: np.ndarray, topk: int) -> Phase2Out:
    """SIMD-register-level transliteration with an explicit VL-chunk loop.

    The 256-bin `chist_msb` tile is 4 × VL_B32 (64 u32 lanes per VL register), so every
    "tile op" expands into a Python loop of `n_chunks = BIN_NUM // VL_B32 = 4` SIMD ops.
    The 32-lane "broadcast" tiles fit in one VL register with `create_predicate(32, 4)`
    restricting writes to the first 32 u32 lanes; remaining lanes are don't-cares.
    """
    thr_msb = np.uint32(N - topk)
    n_chunks = BIN_NUM // VL_B32                  # 256 / 64 = 4

    preg_full_b32  = pset_b32_all()               # all 64 u32 lanes active
    preg_first_32  = create_predicate(32, 4)      # first 32 u32 lanes active (b32 stride)

    # =========================================================================
    # Block A: msb_winner_lanes[b] = (C[b] >= thr_msb) ? b : SELS_FALSE
    #   Loop the (TCMPS, TCI, TSELS) trio over the 4 VL_B32 chunks of chist_msb.
    # =========================================================================
    msb_winner_lanes = np.full(BIN_NUM, SELS_FALSE, dtype=np.uint32)
    for c in range(n_chunks):
        base = c * VL_B32
        vb32_chist = vlds_u32(chist_msb, base)                            # vlds B32
        vb32_idx   = vci_u32(start_value=base)                            # vci u32 (start = c*VL_B32)
        preg_ge    = vcmps_u32(vb32_chist, thr_msb, preg_full_b32, "GE") # vcmps GE
        vb32_lanes = vsels_u32(preg_ge, vb32_idx, SELS_FALSE)             # vsels
        vsts_u32(msb_winner_lanes, base, vb32_lanes)                      # vsts B32

    # =========================================================================
    # Block B: scalar_winner = min(msb_winner_lanes); broadcast to a 32-lane tile.
    #   Per-chunk vmin into an accumulator VL register, then one vrmin reduction.
    # =========================================================================
    vb32_acc = vbr_u32(SELS_FALSE)                                       # vbr u32 (init = +inf)
    for c in range(n_chunks):
        vb32_chunk = vlds_u32(msb_winner_lanes, c * VL_B32)              # vlds B32
        vb32_acc   = vmin_u32(vb32_acc, vb32_chunk, preg_full_b32, "merging")  # vmin u32
    scalar_winner = int(vrmin_u32(vb32_acc, preg_full_b32))              # vrmin u32 → scalar

    # vbr broadcast scalar to a 32-lane "WinnerBinTile" view (first 32 u32 lanes).
    vb32_winner_saved      = vbr_u32(scalar_winner, preg_first_32)       # vbr u32 (pred=first_32)
    msb_winner_saved_tile  = vb32_winner_saved[:32].copy()

    # =========================================================================
    # Block C: msb_winner_bin = (winner==0) ? 0 : winner-1 ; broadcast to 32 lanes.
    #   Stays within ONE VL register: vbr(min) → vsub(-1) → vcmps(GT 256) → vsel(0).
    # =========================================================================
    vb32_min   = vbr_u32(scalar_winner)                                   # vbr u32 broadcast min
    vb32_one   = vbr_u32(1)                                               # vbr u32 ones
    vb32_sub   = vsub_u32(vb32_min, vb32_one, preg_full_b32, "merging")  # vsub u32 (wraps on 0)
    preg_wrap  = vcmps_u32(vb32_sub, 256, preg_full_b32, "GT")           # vcmps GT 256
    vb32_zero  = vbr_u32(0)                                               # vbr u32 zero
    vb32_bin0  = vsel_u32(preg_wrap, vb32_zero, vb32_sub, preg_full_b32) # vsel u32 (wrap→0)
    scalar_winner_bin = int(vb32_bin0[0])
    vb32_winner_bin   = vbr_u32(scalar_winner_bin, preg_first_32)        # vbr u32 (pred=first_32)
    msb_winner_bin_tile = vb32_winner_bin[:32].copy()

    # =========================================================================
    # Block D: remain_k = (N-TopK) - C_fix[winner-1] on the 32-lane RemainKTile.
    #   One VL register; predicate restricts writes to the first 32 u32 lanes.
    # =========================================================================
    vb32_thr        = vbr_u32(int(thr_msb), preg_first_32)               # vbr u32
    vb32_cw         = vgather_u32(chist_msb, vb32_winner_bin, preg_first_32)  # vgather u32 C[winner-1]
    preg_w_eq_0     = vcmps_u32(vb32_winner_saved, 0, preg_first_32, "EQ")    # vcmps EQ (winner==0?)
    vb32_zero32     = vbr_u32(0, preg_first_32)                          # vbr u32
    vb32_cw_fix     = vsel_u32(preg_w_eq_0, vb32_zero32, vb32_cw, preg_first_32)  # vsel
    vb32_remain_k   = vsub_u32(vb32_thr, vb32_cw_fix, preg_first_32, "merging")   # vsub u32
    remain_k_tile   = vb32_remain_k[:32].copy()

    return Phase2Out(msb_winner_saved=msb_winner_saved_tile,
                     msb_winner_bin=msb_winner_bin_tile,
                     remain_k_tile=remain_k_tile)


# --------------------------------------------------------------------------- #
# Phase 3 — LSB cumulative histogram, filtered by MSB == winner.               #
# --------------------------------------------------------------------------- #
def phase3_histogram_lsb(src_gm: np.ndarray, msb_winner_saved: np.ndarray) -> np.ndarray:
    """Returns ``chistLSB`` (uint32[256])."""
    # TCVT(idxFilter, msbWinnerSaved): u32 winner -> u8 filter byte (one byte written to UB).
    idx_filter_ub = np.zeros(1, dtype=np.uint8)
    idx_filter_ub[0] = np.uint8(t_cvt(msb_winner_saved[:1], np.uint8)[0])

    chist_lsb = t_expands((BIN_NUM,), 0, np.uint32)     # TEXPANDS(chistLSB, 0u)

    n_chist_chunks = BIN_NUM // VL_B32  # 4
    preg_full_b32 = pset_b32_all()

    n_loop = (N + TILE_COLS - 1) // TILE_COLS
    for i in range(n_loop):
        base = i * TILE_COLS
        valid = min(TILE_COLS, N - base)
        for sub in range(0, valid, HIST_CHUNK_COLS):
            sub_valid = min(HIST_CHUNK_COLS, valid - sub)
            in_tile = t_load(src_gm, base + sub, sub_valid, HIST_CHUNK_COLS)
            # THISTOGRAM<BYTE_0>(tileHist, inTile, idxFilter): SIMD-register-level expansion.
            # The inner loop:
            #   * vlds DINTLV_B8 → (LSB, MSB) byte vectors
            #   * vcmp_eq(preg_idx, MSB, broadcast(filter_byte), preg_b8_0)  — keep MSB==filter lanes
            #   * chistv2(LSB, preg_idx, Bin_N{0,1}) → cumulative LSB hist (filtered)
            tile_hist = t_histogram_simd(in_tile.reshape(1, -1), valid_rows=1,
                                          valid_cols=sub_valid, idx_filter_u8=idx_filter_ub,
                                          is_msb=False)
            # TADD across tiles: explicit 4 × VL_B32 SIMD loop.
            for c in range(n_chist_chunks):
                off = c * VL_B32
                vb32_acc = vlds_u32(chist_lsb, off)                                 # vlds B32
                vb32_inc = vlds_u32(tile_hist, off)                                 # vlds B32
                vb32_sum = vadd_u32(vb32_acc, vb32_inc, preg_full_b32, "merging")  # vadd u32
                vsts_u32(chist_lsb, off, vb32_sum)                                  # vsts B32

    return chist_lsb


# --------------------------------------------------------------------------- #
# Phase 4 — LSB winner + packed uint16 threshold (TOR).                        #
# --------------------------------------------------------------------------- #
def phase4_winner_lsb_and_packed(
    chist_lsb: np.ndarray,
    remain_k_tile: np.ndarray,
    msb_winner_saved: np.ndarray,
) -> np.ndarray:
    """SIMD-register-level transliteration with an explicit VL-chunk loop.

    `chist_lsb` is 256 u32 = 4 × VL_B32 (one VL register per chunk). The final pack
    (TSHLS + TOR) operates on a single u16 lane, expressed as a 1-lane vector op
    with `create_predicate(1, 2)` restricting writes to lane 0.

    Returns ``packedThrU`` (uint16[32]); only lane 0 carries the threshold.
    """
    n_chunks = BIN_NUM // VL_B32                  # 4
    remain_k_u32 = np.uint32(remain_k_tile[0])

    preg_full_b32 = pset_b32_all()
    preg_first_32 = create_predicate(32, 4)       # first 32 u32 lanes (broadcast tile)
    preg_first_1_b16 = create_predicate(1, 2)     # first 1 u16 lane (pack lane 0)

    # =========================================================================
    # Block A: lsb_winner_lanes[b] = (chist_lsb[b] > remainK) ? b : SELS_FALSE
    # =========================================================================
    lsb_winner_lanes = np.full(BIN_NUM, SELS_FALSE, dtype=np.uint32)
    for c in range(n_chunks):
        base = c * VL_B32
        vb32_clsb = vlds_u32(chist_lsb, base)                                  # vlds B32
        vb32_idx  = vci_u32(start_value=base)                                  # vci u32
        preg_gt   = vcmps_u32(vb32_clsb, remain_k_u32, preg_full_b32, "GT")   # vcmps GT
        vb32_lns  = vsels_u32(preg_gt, vb32_idx, SELS_FALSE)                   # vsels
        vsts_u32(lsb_winner_lanes, base, vb32_lns)                             # vsts B32

    # =========================================================================
    # Block B: scalar_lsb_winner = min(lanes); broadcast to a 32-lane tile.
    # =========================================================================
    vb32_acc = vbr_u32(SELS_FALSE)
    for c in range(n_chunks):
        vb32_chunk = vlds_u32(lsb_winner_lanes, c * VL_B32)                    # vlds B32
        vb32_acc   = vmin_u32(vb32_acc, vb32_chunk, preg_full_b32, "merging")  # vmin u32
    scalar_lsb_winner = int(vrmin_u32(vb32_acc, preg_full_b32))                # vrmin u32

    vb32_lsb_winner_bin = vbr_u32(scalar_lsb_winner, preg_first_32)            # vbr u32 (first 32)

    # =========================================================================
    # Block C: pack = (msbWinner << 8) | lsbWinner  (single u16 lane).
    #   One VL register; predicate restricts writes to lane 0 of the u16 view.
    #   vcvt u32 → u16 narrows the broadcast tiles; vshls + vor produces the pack.
    # =========================================================================
    vb32_msb_winner = vbr_u32(int(msb_winner_saved[0]), preg_first_32)         # vbr u32
    vb16_msb_u      = vcvt_u32_to_u16(vb32_msb_winner, preg_first_32)          # vcvt CAST_TRUNC
    vb16_lsb_u      = vcvt_u32_to_u16(vb32_lsb_winner_bin, preg_first_32)      # vcvt CAST_TRUNC
    vb16_hi_u       = vshls_u16(vb16_msb_u, 8, preg_first_1_b16, "merging")    # vshls << 8
    vb16_pack_u     = vor_u16(vb16_hi_u, vb16_lsb_u, preg_first_1_b16, "merging")  # vor u16

    # TASSIGN packedThrU; broadcast lane 0 (a single u16 scalar) to the 32-lane tile.
    packed_thr_u = np.zeros(32, dtype=np.uint16)
    packed_thr_u[:] = vb16_pack_u[0]
    return packed_thr_u


# --------------------------------------------------------------------------- #
# Phase 5 — Per-tile GT/EQ TGATHER + TCONCAT, final TSTORE.                    #
# --------------------------------------------------------------------------- #
def _t_gather_compare(
    src_tile: np.ndarray, valid: int, packed_thr_u: np.ndarray, base: int, mode: str,
) -> Tuple[np.ndarray, int]:
    """TGATHER<..., CmpMode::{GT,EQ}>(dst, src, packedThr, concat, tmp, offset=base).

    Returns (indices, count) where indices are *global* (base-offset added by the
    intrinsic) and count is what `concat` reports.
    """
    thr = np.uint16(packed_thr_u[0])
    keys = src_tile[:valid].astype(np.uint16)
    if mode == "GT":
        sel = keys > thr
    elif mode == "EQ":
        sel = keys == thr
    else:
        raise ValueError(mode)
    local = np.flatnonzero(sel).astype(np.uint32)
    indices = local + np.uint32(base)
    return indices, int(indices.size)


def phase5_gather_concat_store(src_gm: np.ndarray, packed_thr_u: np.ndarray, topk: int) -> np.ndarray:
    """Returns the Top-K index vector written by TSTORE.

    The outer `(i, sub)` loops are the SIMD-register-level loops here: each
    `_t_gather_compare` consumes one CHUNK_COLS-sized u16 UB tile (256 u16 =
    2 × VL_B16 long-vector). The `TGATHER<CmpMode>` intrinsic itself is the
    hardware primitive — it performs a per-lane `vcmps_u16` against the broadcast
    threshold, then a compress/pack step that emits a contiguous list of
    qualifying *global* indices (one VL_B32 worth at a time) and updates the
    write-cursor (`idxGtAcc` / `idxEqAcc`) in bytes. `TCONCAT_IMPL` is the
    follow-up vector store that appends the produced indices to the running
    segment in UB.
    """
    # TASSIGN + TEXPANDS(gtSeg/eqSeg, 0u): zeroed segment accumulators.
    gt_seg = np.zeros(topk, dtype=np.uint32)
    eq_seg = np.zeros(topk, dtype=np.uint32)
    # `idx*Acc` holds the running write cursor in **bytes** (sizeof(uint32)=4).
    idx_gt_acc_bytes = 0
    idx_eq_acc_bytes = 0

    n_loop = (N + TILE_COLS - 1) // TILE_COLS

    # ---- GT pass: collect strictly-greater indices up to TopK total. -------
    for i in range(n_loop):
        base = i * TILE_COLS
        valid = min(TILE_COLS, N - base)
        for sub in range(0, valid, CHUNK_COLS):
            sub_valid = min(CHUNK_COLS, valid - sub)
            src_g = t_load(src_gm, base + sub, sub_valid, CHUNK_COLS)
            dst_g, cnt_g = _t_gather_compare(src_g, sub_valid, packed_thr_u, base + sub, "GT")

            # TCONCAT_IMPL(segTmp, gtSeg, dstG, idxGtOut, idxGtAcc, concatG).
            cursor = idx_gt_acc_bytes // 4
            take = min(cnt_g, topk - cursor)
            seg_tmp = gt_seg.copy()
            seg_tmp[cursor:cursor + take] = dst_g[:take]
            # TMOV(gtSeg, segTmp); TMOV(idxGtAcc, idxGtOut)
            gt_seg = seg_tmp
            idx_gt_acc_bytes = (cursor + take) * 4

    gt_count = idx_gt_acc_bytes // 4
    eq_cap = max(topk - gt_count, 0)

    # ---- EQ pass: pad up to TopK with indices equal to the threshold. ------
    for i in range(n_loop):
        base = i * TILE_COLS
        valid = min(TILE_COLS, N - base)
        for sub in range(0, valid, CHUNK_COLS):
            sub_valid = min(CHUNK_COLS, valid - sub)
            src_e = t_load(src_gm, base + sub, sub_valid, CHUNK_COLS)
            dst_e, cnt_e = _t_gather_compare(src_e, sub_valid, packed_thr_u, base + sub, "EQ")

            cursor = idx_eq_acc_bytes // 4
            take = min(cnt_e, eq_cap - cursor)
            if take <= 0:
                continue
            seg_tmp = eq_seg.copy()
            seg_tmp[cursor:cursor + take] = dst_e[:take]
            eq_seg = seg_tmp
            idx_eq_acc_bytes = (cursor + take) * 4

    eq_count = idx_eq_acc_bytes // 4

    # Final TCONCAT_IMPL(mergedIdx, gtSeg, eqSeg, idxGtAcc, idxEqAcc).
    merged_idx = np.zeros(2 * topk, dtype=np.uint32)
    merged_idx[:gt_count] = gt_seg[:gt_count]
    merged_idx[gt_count:gt_count + eq_count] = eq_seg[:eq_count]

    # TSTORE: UB -> GM, first TopK lanes.
    return merged_idx[:topk].copy()


# --------------------------------------------------------------------------- #
# Top-level driver — same call order as `RunRadixTopKDraft`.                   #
# --------------------------------------------------------------------------- #
def radix_topk(keys: np.ndarray, topk: int = TOPK) -> np.ndarray:
    assert keys.dtype == np.uint16 and keys.size == N, "expect uint16[N]"
    chist_msb = phase1_load_and_histogram_msb(keys)
    p2 = phase2_winner_msb_and_remain_k(chist_msb, topk)
    chist_lsb = phase3_histogram_lsb(keys, p2.msb_winner_saved)
    packed = phase4_winner_lsb_and_packed(chist_lsb, p2.remain_k_tile, p2.msb_winner_saved)
    return phase5_gather_concat_store(keys, packed, topk)


# --------------------------------------------------------------------------- #
# CLI: run against `input/keys.bin` and verify multiset == golden.             #
# --------------------------------------------------------------------------- #
def _multiset_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return np.array_equal(np.sort(a), np.sort(b))


def main() -> int:
    ap = argparse.ArgumentParser(description="NumPy reference for draft.cpp radix-select Top-K.")
    ap.add_argument("--keys", default=None, help="Path to uint16 keys.bin (default: ../input/keys.bin)")
    ap.add_argument("--golden", default=None,
                    help="Path to golden_topk_multiset.bin (uint16 keys, default: ../input/golden_topk_multiset.bin)")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--show", type=int, default=8, help="Print first N picked indices for sanity")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    topk_dir = os.path.dirname(script_dir)
    keys_path = args.keys or os.path.join(topk_dir, "input", "keys.bin")
    if not os.path.isfile(keys_path):
        print(f"Missing {keys_path}; run scripts/gen_data.py first.", file=sys.stderr)
        return 1
    keys = np.fromfile(keys_path, dtype=np.uint16)
    if keys.size != N:
        print(f"keys.bin has {keys.size} elems, expected {N}.", file=sys.stderr)
        return 1

    out_idx = radix_topk(keys, args.topk)

    # Phase-by-phase trace.
    chist_msb = phase1_load_and_histogram_msb(keys)
    p2 = phase2_winner_msb_and_remain_k(chist_msb, args.topk)
    chist_lsb = phase3_histogram_lsb(keys, p2.msb_winner_saved)
    packed = phase4_winner_lsb_and_packed(chist_lsb, p2.remain_k_tile, p2.msb_winner_saved)

    print(f"Phase1  chistMSB[255]            = {int(chist_msb[255])}  (expect {N})")
    print(f"Phase2  msbWinnerSaved[0]        = {int(p2.msb_winner_saved[0])}")
    print(f"Phase2  msbWinnerBin[0] (idx)    = {int(p2.msb_winner_bin[0])}")
    print(f"Phase2  remainKTile[0]           = {int(p2.remain_k_tile[0])}")
    print(f"Phase3  chistLSB[255]            = {int(chist_lsb[255])}")
    print(f"Phase4  packedThrU[0] (uint16)   = 0x{int(packed[0]):04x}")
    print(f"Phase5  out_idx[:{args.show}]            = {out_idx[:args.show].tolist()}")

    # Multiset verification against the host golden file (same one used by main.cpp).
    golden_path = args.golden or os.path.join(topk_dir, "input", "golden_topk_multiset.bin")
    if os.path.isfile(golden_path):
        golden_keys = np.fromfile(golden_path, dtype=np.uint16)
        picked_keys = keys[out_idx.astype(np.int64)]
        ok = _multiset_equal(picked_keys, golden_keys)
        print(f"\nmultiset(keys[out_idx]) == multiset(golden) ? {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2
    else:
        # Fall back to a direct top-K sanity check.
        kth_sorted = np.sort(keys)[::-1][:args.topk]
        picked_keys = np.sort(keys[out_idx.astype(np.int64)])[::-1]
        ok = np.array_equal(picked_keys, kth_sorted)
        print(f"\nmultiset(keys[out_idx]) == sort_desc(keys)[:TopK] ? {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
