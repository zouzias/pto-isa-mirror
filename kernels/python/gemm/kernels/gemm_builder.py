# ruff: noqa: F821
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shape-specialized GEMM kernel built with PTO-DSL.

The kernel computes ``C = A * B`` on Ascend A2/A3 with:
- persistent scheduling over ``base_m x base_n`` output tiles
- N-group swizzling for L2 reuse
- L1 panel staging with double buffering
- Ping-pong L0A/L0B buffers

Tensor contract:
- A is logical and physical ND, shape ``[m, k]``, dtype fp16.
- B is logical ``[k, n]`` and passed in GM as transposed DN storage.
- C is ND, shape ``[m, n]``, dtype fp16. Accumulation is fp32 in ACC.

Default build() arguments:
  m=6144, k=6144, n=6144
  base_m=128, base_k=64, base_n=256
  step_ka=4, step_kb=4
  block_dim=24, swizzle_count_n=1
"""

import importlib
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
BODY_DIR = Path(__file__).with_name("oversize_bodies")


def _exec_body(filename, local_vars):
    env = dict(globals())
    env.update(local_vars)
    body_path = BODY_DIR / filename
    exec(compile(body_path.read_text(encoding="utf-8"), str(body_path), "exec"), env, env)


def _patch_mlir_pto_quant_type():
    """Install the QuantType symbols expected by ptodsl when they are absent.

    Some MLIR/PTO Python environments do not export ``QuantType`` even though
    ptodsl imports it while constructing tensor metadata. The GEMM kernel does
    not use quantized types, so these placeholder symbols are enough for IR
    generation. Remove this shim once the active bindings always provide
    ``mlir.dialects.pto.QuantType``.
    """
    pto_dialect = importlib.import_module("mlir.dialects.pto")
    if hasattr(pto_dialect, "QuantType"):
        return

    class _QuantType:
        INT8_SYM = object()
        INT8_ASYM = object()

    pto_dialect.QuantType = _QuantType


_patch_mlir_pto_quant_type()

_ptodsl = importlib.import_module("ptodsl")
pto = _ptodsl.pto
tile = _ptodsl.tile
to_ir_module = _ptodsl.to_ir_module
s = importlib.import_module("ptodsl.scalar")


def _require_build_param(condition, message):
    if not condition:
        raise ValueError(message)


def _gemm_meta_base():
    dtype_in = pto.float16
    dtype_out = pto.float16
    dtype_acc = pto.float32
    return {
        "dtype_in": dtype_in,
        "dtype_out": dtype_out,
        "dtype_acc": dtype_acc,
        "ptr_type_in": pto.PtrType(dtype_in),
        "ptr_type_out": pto.PtrType(dtype_out),
        "tensor_type_in": pto.TensorType(rank=2, dtype=dtype_in),
        "tensor_type_out": pto.TensorType(rank=2, dtype=dtype_out),
    }


def _gemm_meta_views(cfg, dtype_in, dtype_out):
    base_m = cfg["base_m"]
    base_k = cfg["base_k"]
    base_n = cfg["base_n"]
    step_ka = cfg["step_ka"]
    step_kb = cfg["step_kb"]
    return {
        "tile_view_a_mat": pto.SubTensorType(shape=[base_m, base_k * step_ka], dtype=dtype_in),
        "tile_view_b_mat": pto.SubTensorType(shape=[base_k * step_kb, base_n], dtype=dtype_in),
        "tile_view_out": pto.SubTensorType(shape=[base_m, base_n], dtype=dtype_out),
    }


def _gemm_meta_tiles(cfg, dtype_in, dtype_acc):
    base_m = cfg["base_m"]
    base_k = cfg["base_k"]
    base_n = cfg["base_n"]
    step_ka = cfg["step_ka"]
    step_kb = cfg["step_kb"]
    b_mat_cfg = pto.TileBufConfig(blayout="RowMajor", slayout="ColMajor", s_fractal_size=512)
    return {
        "tile_buf_a_mat": pto.TileBufType(shape=[base_m, base_k * step_ka], dtype=dtype_in, memory_space="MAT"),
        "tile_buf_b_mat": pto.TileBufType(
            shape=[base_k * step_kb, base_n], dtype=dtype_in, memory_space="MAT", config=b_mat_cfg
        ),
        "tile_buf_a_tile": pto.TileBufType(shape=[base_m, base_k], dtype=dtype_in, memory_space="LEFT"),
        "tile_buf_b_tile": pto.TileBufType(shape=[base_k, base_n], dtype=dtype_in, memory_space="RIGHT"),
        "tile_buf_c_tile": pto.TileBufType(shape=[base_m, base_n], dtype=dtype_acc, memory_space="ACC"),
    }


def _gemm_meta_data(cfg):
    meta = _gemm_meta_base()
    meta.update(_gemm_meta_views(cfg, meta["dtype_in"], meta["dtype_out"]))
    meta.update(_gemm_meta_tiles(cfg, meta["dtype_in"], meta["dtype_acc"]))
    del meta["dtype_in"]
    del meta["dtype_out"]
    del meta["dtype_acc"]
    return meta


DEFAULT_BUILD_CONFIG = {
    "m": 6144,
    "k": 6144,
    "n": 6144,
    "base_m": 128,
    "base_k": 64,
    "base_n": 256,
    "step_ka": 4,
    "step_kb": 4,
    "block_dim": 24,
    "swizzle_count_n": 1,
}

BUILD_CONFIG_KEY_ALIASES = {
    "baseM": "base_m",
    "baseK": "base_k",
    "baseN": "base_n",
    "stepKa": "step_ka",
    "stepKb": "step_kb",
    "blockDim": "block_dim",
    "swizzleCountN": "swizzle_count_n",
}


def _normalized_build_config(build_cfg):
    merged = dict(DEFAULT_BUILD_CONFIG)
    if build_cfg is not None:
        for key, value in build_cfg.items():
            merged[BUILD_CONFIG_KEY_ALIASES.get(key, key)] = value
    return merged


def build(build_cfg=None):
    """Build and return the shape-specialized ``Gemm`` IR module.

    The emitted kernel persistently iterates over the global ``base_m x base_n``
    tile grid. ``swizzle_count_n`` groups adjacent N tiles while walking M rows so
    B panels can stay hot in L2 across multiple output rows.
    """
    cfg = _normalized_build_config(build_cfg)
    m = cfg["m"]
    k = cfg["k"]
    n = cfg["n"]
    base_m = cfg["base_m"]
    base_k = cfg["base_k"]
    base_n = cfg["base_n"]
    step_ka = cfg["step_ka"]
    step_kb = cfg["step_kb"]
    block_dim = cfg["block_dim"]
    swizzle_count_n = cfg["swizzle_count_n"]

    _require_build_param(m % base_m == 0, "m must be divisible by base_m")
    _require_build_param(n % base_n == 0, "n must be divisible by base_n")
    _require_build_param(k % base_k == 0, "k must be divisible by base_k")
    _require_build_param(step_ka == step_kb, "step_ka and step_kb must match")
    _require_build_param(step_ka == 4, "step_ka must be 4")
    _require_build_param((k // base_k) % (step_ka * 2) == 0, "k/base_k must be divisible by step_ka*2")
    _require_build_param(1 <= block_dim <= (m // base_m) * (n // base_n), "block_dim is outside the output tile range")
    _require_build_param(1 <= swizzle_count_n <= (n // base_n), "swizzle_count_n is outside the N tile range")

    def meta_data():
        return _gemm_meta_data(cfg)

    const = s.const
    build_env = locals().copy()

    @to_ir_module(meta_data=meta_data)
    def gemm(out_ptr: "ptr_type_out", a_ptr: "ptr_type_in", b_ptr: "ptr_type_in") -> None:
        _exec_body("gemm_body.inl", dict(build_env, **locals()))

    return gemm


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info("%s", build())
