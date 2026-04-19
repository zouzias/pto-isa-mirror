#!/usr/bin/env python3
"""
Generate tile_perf_kernel.cpp and main.cpp from input.csv.

Supported op categories:
  binary:           TADD, TSUB, TMUL, TMAX, TMIN           (src0 + src1 → dst)
  unary:            TEXP, TLOG, TSQRT, TABS, TNEG, TRCP    (src → dst)
  scalar:           TADDS, TSUBS, TMULS, TDIVS              (src + scalar → dst)
  reduce_row:       TROWSUM                                 (src(H,W) → dst(H,1), tmp)
  reduce_col:       TCOLSUM                                 (src(H,W) → dst(1,W), tmp)
  broadcast_scalar: TEXPANDS                                (scalar → dst(H,W))
  broadcast_row:    TROWEXPAND                              (src(H,1) → dst(H,W))
  broadcast_col:    TCOLEXPAND                              (src(1,W) → dst(H,W))

CSV format:  op,dtype,tile_h,tile_w,valid_h,valid_w[,scalar]
  H,W always refer to the "big" 2-D tile shape.

Usage:  python3 generate_code.py [--csv input.csv]
"""
import argparse
from collections import Counter
from pathlib import Path

# ── Op classification ────────────────────────────────────────────────────────

UNARY_OPS = {"TEXP", "TLOG", "TSQRT", "TABS", "TNEG", "TRCP", "TRSQRT"}
SCALAR_OPS = {"TADDS", "TSUBS", "TMULS", "TDIVS", "TMAXS", "TMINS"}
REDUCE_ROW_OPS = {"TROWSUM"}
REDUCE_COL_OPS = {"TCOLSUM"}
BROADCAST_SCALAR_OPS = {"TEXPANDS"}
BROADCAST_ROW_OPS = {"TROWEXPAND"}
BROADCAST_COL_OPS = {"TCOLEXPAND"}


def get_op_category(op: str) -> str:
    u = op.upper()
    if u in UNARY_OPS:
        return "unary"
    if u in SCALAR_OPS:
        return "scalar"
    if u in REDUCE_ROW_OPS:
        return "reduce_row"
    if u in REDUCE_COL_OPS:
        return "reduce_col"
    if u in BROADCAST_SCALAR_OPS:
        return "broadcast_scalar"
    if u in BROADCAST_ROW_OPS:
        return "broadcast_row"
    if u in BROADCAST_COL_OPS:
        return "broadcast_col"
    return "binary"


# ── CSV parser ───────────────────────────────────────────────────────────────

def read_input_csv(csv_path):
    cases = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            op, dtype, tile_h, tile_w, valid_h, valid_w = parts[:6]
            scalar = float(parts[6]) if len(parts) > 6 else 1.5
            cases.append(dict(
                op=op, dtype=dtype,
                tile_h=int(tile_h), tile_w=int(tile_w),
                valid_h=int(valid_h), valid_w=int(valid_w),
                scalar=scalar,
                cat=get_op_category(op),
            ))
    return cases


# ── Helpers ──────────────────────────────────────────────────────────────────

DTYPE_CPP = {"float": "float", "half": "half", "float32": "float", "float16": "half"}
DTYPE_ACL = {"float": "float", "half": "aclFloat16", "float32": "float", "float16": "aclFloat16"}
DTYPE_GTEST = {"float": "float", "half": "half", "float32": "float", "float16": "half"}


def cpp(d):
    return DTYPE_CPP.get(d, d)


def acl_t(d):
    return DTYPE_ACL.get(d, d)


def gtest(d):
    return DTYPE_GTEST.get(d, d)


COPYRIGHT = """\
/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
"""


# ── Kernel code generators per op category ───────────────────────────────────

def gen_kernel_binary(op, T, th, tw, vh, vw):
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src0, __gm__ {T} *src1)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<{T}, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    GlobalData src0Global(src0, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    GlobalData src1Global(src1, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    GlobalData dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    TileData src0Tile({vh}, {vw});
    TileData src1Tile({vh}, {vw});
    TileData dstTile({vh}, {vw});

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x100);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(src0Tile, src0Global);
        TLOAD(src1Tile, src1Global);
        {op}(dstTile, src0Tile, src1Tile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_unary(op, T, th, tw, vh, vw):
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<{T}, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    GlobalData dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    TileData srcTile({vh}, {vw});
    TileData dstTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_scalar(op, T, th, tw, vh, vw):
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src, {T} scalar)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<{T}, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    GlobalData srcGlobal(src, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    GlobalData dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    TileData srcTile({vh}, {vw});
    TileData dstTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_reduce_row(op, T, th, tw, vh, vw):
    """TROWSUM: src(H,W) → dst(H,1), needs tmp(H,W)"""
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    // Source: full (H, W)
    using SrcTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;
    // Destination: (H, 1) — one value per row
    using DstTileData = Tile<TileType::Vec, {T}, {th}, 1, BLayout::RowMajor, -1, -1>;
    // Tmp: same shape as src (scratch buffer)
    using TmpTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    using SrcGlobal = GlobalTensor<{T}, DynShape, DynStride>;
    using DstGlobal = GlobalTensor<{T}, DynShape, DynStride>;

    SrcGlobal srcGlobal(src, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    DstGlobal dstGlobal(out, DynShape({vh}, 1), DynStride({th}, 1));
    SrcTileData srcTile({vh}, {vw});
    DstTileData dstTile({vh}, 1);
    TmpTileData tmpTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x100);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile, tmpTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_reduce_col(op, T, th, tw, vh, vw):
    """TCOLSUM: src(H,W) → dst(1,W), needs tmp(H,W)"""
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    // Source: full (H, W)
    using SrcTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;
    // Destination: (1, W) — one value per column
    using DstTileData = Tile<TileType::Vec, {T}, 1, {tw}, BLayout::RowMajor, -1, -1>;
    // Tmp: same shape as src (scratch buffer)
    using TmpTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    using SrcGlobal = GlobalTensor<{T}, DynShape, DynStride>;
    using DstGlobal = GlobalTensor<{T}, DynShape, DynStride>;

    SrcGlobal srcGlobal(src, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    DstGlobal dstGlobal(out, DynShape(1, {vw}), DynStride(1, {tw}));
    SrcTileData srcTile({vh}, {vw});
    DstTileData dstTile(1, {vw});
    TmpTileData tmpTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x100);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile, tmpTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_broadcast_scalar(op, T, th, tw, vh, vw):
    """TEXPANDS: scalar → dst(H,W)"""
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, {T} scalar)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    using GlobalData = GlobalTensor<{T}, DynShape, DynStride>;
    using TileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    GlobalData dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    TileData dstTile({vh}, {vw});

    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        {op}(dstTile, scalar);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_broadcast_row(op, T, th, tw, vh, vw):
    """TROWEXPAND: src(H,1) → dst(H,W)"""
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    // Source: column vector (H, 1)
    using SrcTileData = Tile<TileType::Vec, {T}, {th}, 1, BLayout::RowMajor, -1, -1>;
    // Destination: full tile (H, W)
    using DstTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    using SrcGlobal = GlobalTensor<{T}, DynShape, DynStride>;
    using DstGlobal = GlobalTensor<{T}, DynShape, DynStride>;

    SrcGlobal srcGlobal(src, DynShape({vh}, 1), DynStride({th}, 1));
    DstGlobal dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    SrcTileData srcTile({vh}, 1);
    DstTileData dstTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


def gen_kernel_broadcast_col(op, T, th, tw, vh, vw):
    """TCOLEXPAND: src(1,W) → dst(H,W)"""
    return f"""template <>
__global__ AICORE void run{op}<{T}, {th}, {tw}, {vh}, {vw}>(__gm__ {T} *out, __gm__ {T} *src)
{{
    using DynShape = Shape<1, 1, 1, -1, -1>;
    using DynStride = pto::Stride<1, 1, -1, -1, 1>;
    // Source: row vector (1, W)
    using SrcTileData = Tile<TileType::Vec, {T}, 1, {tw}, BLayout::RowMajor, -1, -1>;
    // Destination: full tile (H, W)
    using DstTileData = Tile<TileType::Vec, {T}, {th}, {tw}, BLayout::RowMajor, -1, -1>;

    using SrcGlobal = GlobalTensor<{T}, DynShape, DynStride>;
    using DstGlobal = GlobalTensor<{T}, DynShape, DynStride>;

    SrcGlobal srcGlobal(src, DynShape(1, {vw}), DynStride(1, {tw}));
    DstGlobal dstGlobal(out, DynShape({vh}, {vw}), DynStride({th}, {tw}));
    SrcTileData srcTile(1, {vw});
    DstTileData dstTile({vh}, {vw});

    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    for (int i = 0; i < 10; i++) {{
        TLOAD(srcTile, srcGlobal);
        {op}(dstTile, srcTile);
        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    }}
}}

"""


KERNEL_GEN = {
    "binary": gen_kernel_binary,
    "unary": gen_kernel_unary,
    "scalar": gen_kernel_scalar,
    "reduce_row": gen_kernel_reduce_row,
    "reduce_col": gen_kernel_reduce_col,
    "broadcast_scalar": gen_kernel_broadcast_scalar,
    "broadcast_row": gen_kernel_broadcast_row,
    "broadcast_col": gen_kernel_broadcast_col,
}

# Launcher signature category → (run_kernel_params, launcher_params, instantiation_params)
CAT_LAUNCHER = {
    "binary": ("__gm__ T *out, __gm__ T *src0, __gm__ T *src1",
               "void *out, void *src0, void *src1, aclrtStream stream",
               "void*, void*, void*, aclrtStream"),
    "unary": ("__gm__ T *out, __gm__ T *src",
              "void *out, void *src, aclrtStream stream",
              "void*, void*, aclrtStream"),
    "scalar": ("__gm__ T *out, __gm__ T *src, T scalar",
               "void *out, void *src, float scalar, aclrtStream stream",
               "void*, void*, float, aclrtStream"),
    "reduce_row": ("__gm__ T *out, __gm__ T *src",
                   "void *out, void *src, aclrtStream stream",
                   "void*, void*, aclrtStream"),
    "reduce_col": ("__gm__ T *out, __gm__ T *src",
                   "void *out, void *src, aclrtStream stream",
                   "void*, void*, aclrtStream"),
    "broadcast_scalar": ("__gm__ T *out, T scalar",
                         "void *out, float scalar, aclrtStream stream",
                         "void*, float, aclrtStream"),
    "broadcast_row": ("__gm__ T *out, __gm__ T *src",
                      "void *out, void *src, aclrtStream stream",
                      "void*, void*, aclrtStream"),
    "broadcast_col": ("__gm__ T *out, __gm__ T *src",
                      "void *out, void *src, aclrtStream stream",
                      "void*, void*, aclrtStream"),
}


def _launcher_call_args(cat):
    """Return the argument forwarding inside launcher body."""
    if cat == "binary":
        return "(T*)out, (T*)src0, (T*)src1"
    elif cat == "scalar":
        return "(T*)out, (T*)src, (T)scalar"
    elif cat == "broadcast_scalar":
        return "(T*)out, (T)scalar"
    else:
        return "(T*)out, (T*)src"


def _launcher_call_args_half(cat):
    if cat == "binary":
        return "(half*)out, (half*)src0, (half*)src1"
    elif cat == "scalar":
        return "(half*)out, (half*)src, (half)scalar"
    elif cat == "broadcast_scalar":
        return "(half*)out, (half)scalar"
    else:
        return "(half*)out, (half*)src"


# ── Generate tile_perf_kernel.cpp ────────────────────────────────────────────

def generate_kernel_cpp(cases, output_path):
    header = COPYRIGHT + """/**
 * Tile Performance Benchmark Kernel — Auto-generated by generate_code.py
 * Re-generate: python3 generate_code.py
 */
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <acl/acl.h>

using namespace pto;

"""
    # ── Forward-declare all kernel templates (one per (op, cat)) ─────────────
    fwd_decls = "// ===== Kernel forward declarations =====\n\n"
    ops_declared = set()
    for c in cases:
        op = c["op"].upper()
        cat = c["cat"]
        if op in ops_declared:
            continue
        ops_declared.add(op)
        run_params = CAT_LAUNCHER[cat][0]
        fwd_decls += f"""template <typename T, int tileH, int tileW, int vRows, int vCols>
__global__ AICORE void run{op}({run_params});

"""

    # ── Kernel specialisations ───────────────────────────────────────────────
    seen_kernels = set()
    kernel_code = "// ===== Kernel specialisations =====\n\n"
    for c in cases:
        op = c["op"].upper()
        T = cpp(c["dtype"])
        key = (op, T, c["tile_h"], c["tile_w"], c["valid_h"], c["valid_w"])
        if key in seen_kernels:
            continue
        seen_kernels.add(key)
        kernel_code += KERNEL_GEN[c["cat"]](op, T, c["tile_h"], c["tile_w"], c["valid_h"], c["valid_w"])

    # ── Launcher wrappers ────────────────────────────────────────────────────
    launchers = "// ===== Launcher wrappers =====\n\n"
    ops_launched = set()
    for c in cases:
        op = c["op"].upper()
        cat = c["cat"]
        if op in ops_launched:
            continue
        ops_launched.add(op)
        _, launcher_params, _ = CAT_LAUNCHER[cat]
        call = _launcher_call_args(cat)
        call_half = _launcher_call_args_half(cat)
        launchers += f"""template <typename T, int tileH, int tileW, int vRows, int vCols>
void launch{op}({launcher_params})
{{
    if constexpr (std::is_same_v<T, aclFloat16>)
        run{op}<half, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>({call_half});
    else
        run{op}<T, tileH, tileW, vRows, vCols><<<1, nullptr, stream>>>({call});
}}

"""

    # ── Explicit template instantiations ─────────────────────────────────────
    insts = "// ===== Explicit template instantiations =====\n\n"
    seen_inst = set()
    for c in cases:
        op = c["op"].upper()
        cat = c["cat"]
        T = acl_t(c["dtype"])
        th, tw, vh, vw = c["tile_h"], c["tile_w"], c["valid_h"], c["valid_w"]
        inst_key = (op, T, th, tw, vh, vw)
        if inst_key in seen_inst:
            continue
        seen_inst.add(inst_key)
        _, _, inst_params = CAT_LAUNCHER[cat]
        insts += f"template void launch{op}<{T}, {th}, {tw}, {vh}, {vw}>({inst_params});\n"

    with open(output_path, "w") as f:
        f.write(header + fwd_decls + kernel_code + launchers + insts)
    print(f"Generated: {output_path}")


# ── Generate main.cpp ────────────────────────────────────────────────────────

def generate_main_cpp(cases, output_path):
    header = COPYRIGHT + """/**
 * Tile Performance Benchmark — Auto-generated by generate_code.py
 * Re-generate: python3 generate_code.py
 */
#include "test_common.h"
#include "acl/acl.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace std;
using namespace PtoTestCommon;

class TilePerfTest : public testing::Test {};

"""
    # ── Forward declarations ─────────────────────────────────────────────────
    decls = "// ===== Launcher forward declarations =====\n\n"
    decl_seen = set()
    for c in cases:
        op = c["op"].upper()
        cat = c["cat"]
        if op in decl_seen:
            continue
        decl_seen.add(op)
        _, launcher_params, _ = CAT_LAUNCHER[cat]
        decls += f"""template <typename T, int tileH, int tileW, int vRows, int vCols>
void launch{op}({launcher_params});

"""

    # ── Test macros ──────────────────────────────────────────────────────────
    macros = """// ========== Test macros ==========

#define BINARY_TEST(OP, DTYPE, NAME, H, W, ELEM, ACL_T) \\
TEST_F(TilePerfTest, OP##_##DTYPE##_##NAME) { \\
    size_t bytes = (ELEM) * sizeof(ACL_T); \\
    aclInit(nullptr); aclrtSetDevice(0); \\
    void *devOut, *devSrc0, *devSrc1; \\
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtMalloc(&devSrc0, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtMalloc(&devSrc1, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \\
    launch##OP<ACL_T, H, W, H, W>(devOut, devSrc0, devSrc1, stream); \\
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \\
    aclrtFree(devOut); aclrtFree(devSrc0); aclrtFree(devSrc1); \\
    aclrtResetDevice(0); aclFinalize(); \\
}

#define UNARY_TEST(OP, DTYPE, NAME, H, W, ELEM, ACL_T) \\
TEST_F(TilePerfTest, OP##_##DTYPE##_##NAME) { \\
    size_t bytes = (ELEM) * sizeof(ACL_T); \\
    aclInit(nullptr); aclrtSetDevice(0); \\
    void *devOut, *devSrc; \\
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \\
    launch##OP<ACL_T, H, W, H, W>(devOut, devSrc, stream); \\
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \\
    aclrtFree(devOut); aclrtFree(devSrc); \\
    aclrtResetDevice(0); aclFinalize(); \\
}

#define SCALAR_TEST(OP, DTYPE, NAME, H, W, ELEM, ACL_T, SCALAR) \\
TEST_F(TilePerfTest, OP##_##DTYPE##_##NAME) { \\
    size_t bytes = (ELEM) * sizeof(ACL_T); \\
    aclInit(nullptr); aclrtSetDevice(0); \\
    void *devOut, *devSrc; \\
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtMalloc(&devSrc, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \\
    launch##OP<ACL_T, H, W, H, W>(devOut, devSrc, SCALAR, stream); \\
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \\
    aclrtFree(devOut); aclrtFree(devSrc); \\
    aclrtResetDevice(0); aclFinalize(); \\
}

#define BROADCAST_SCALAR_TEST(OP, DTYPE, NAME, H, W, ELEM, ACL_T, SCALAR) \\
TEST_F(TilePerfTest, OP##_##DTYPE##_##NAME) { \\
    size_t bytes = (ELEM) * sizeof(ACL_T); \\
    aclInit(nullptr); aclrtSetDevice(0); \\
    void *devOut; \\
    aclrtMalloc(&devOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST); \\
    aclrtStream stream = nullptr; aclrtCreateStream(&stream); \\
    launch##OP<ACL_T, H, W, H, W>(devOut, SCALAR, stream); \\
    aclrtSynchronizeStream(stream); aclrtDestroyStream(stream); \\
    aclrtFree(devOut); \\
    aclrtResetDevice(0); aclFinalize(); \\
}

"""

    # ── Test case instantiations ─────────────────────────────────────────────
    tests = "// ========== Test cases (auto-generated from input.csv) ==========\n\n"
    for c in cases:
        op = c["op"].upper()
        cat = c["cat"]
        d = gtest(c["dtype"])
        h, w = c["valid_h"], c["valid_w"]
        elems = h * w
        T = acl_t(c["dtype"])
        name = f"{h}x{w}"

        if cat == "binary":
            tests += f"BINARY_TEST({op}, {d}, {name}, {h}, {w}, {elems}, {T})\n"
        elif cat in ("unary", "reduce_row", "reduce_col", "broadcast_row", "broadcast_col"):
            tests += f"UNARY_TEST({op}, {d}, {name}, {h}, {w}, {elems}, {T})\n"
        elif cat == "scalar":
            tests += f"SCALAR_TEST({op}, {d}, {name}, {h}, {w}, {elems}, {T}, {c['scalar']}f)\n"
        elif cat == "broadcast_scalar":
            tests += f"BROADCAST_SCALAR_TEST({op}, {d}, {name}, {h}, {w}, {elems}, {T}, {c['scalar']}f)\n"

    with open(output_path, "w") as f:
        f.write(header + decls + macros + tests)
    print(f"Generated: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate tile_perf_kernel.cpp and main.cpp from input.csv")
    parser.add_argument("--csv", type=Path, default=Path(__file__).parent / "input.csv",
                        help="Path to input CSV (default: input.csv in same dir)")
    args = parser.parse_args()

    csv_path = args.csv
    script_dir = csv_path.parent

    cases = read_input_csv(csv_path)
    if not cases:
        print(f"No cases found in {csv_path}")
        return

    cat_counts = Counter(c["cat"] for c in cases)
    op_counts = Counter(c["op"].upper() for c in cases)
    print(f"Loaded {len(cases)} test cases from {csv_path}")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")
    print(f"  Ops: {', '.join(f'{o}({n})' for o, n in sorted(op_counts.items()))}")

    generate_kernel_cpp(cases, script_dir / "tile_perf_kernel.cpp")
    generate_main_cpp(cases, script_dir / "main.cpp")

    print(f"\nDone! Rebuild with:")
    print(f"  cd ~/pto-isa/tests/npu/a5/src/st/build && make tile_perf -j8")


if __name__ == "__main__":
    main()
