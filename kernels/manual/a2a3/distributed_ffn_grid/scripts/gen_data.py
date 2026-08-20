#!/usr/bin/python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import argparse
from dataclasses import dataclass

import numpy as np

np.random.seed(19)

PRELU_ALPHA = 0.1


@dataclass
class FfnDataConfig:
    t: int           # token count per row (== FFN_TOKEN_TILE)
    h: int           # hidden dim (== FFN_MODEL_TILE)
    fi: int          # per-rank intermediate dim per col (== FFN_FFN_TILE)
    grid_rows: int
    grid_cols: int
    split_mode: str = "reduce"
    output_dir: str = "./out"


def _prelu(x: np.ndarray, alpha: float = PRELU_ALPHA) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def _make_ffn_arrays(cfg: FfnDataConfig):
    t, h, fi = cfg.t, cfg.h, cfg.fi
    grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
    t_total = t * grid_rows
    f_total = fi * grid_cols
    src_type = np.float16
    dst_type = np.float32

    x_full = np.random.randint(0, 2, [t_total, h]).astype(src_type)
    w_gate_full = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_up_full = np.random.randint(0, 2, [h, f_total]).astype(src_type)
    w_down_full = np.random.randint(0, 2, [f_total, h]).astype(src_type)

    x_f32 = x_full.astype(dst_type)
    gate_full = x_f32 @ w_gate_full.astype(dst_type)
    up_full = x_f32 @ w_up_full.astype(dst_type)
    act_full = _prelu(gate_full * up_full, PRELU_ALPHA)
    hidden_full = act_full.astype(src_type).astype(dst_type)
    golden = (hidden_full @ w_down_full.astype(dst_type)).astype(dst_type)
    return x_full, w_gate_full, w_up_full, w_down_full, golden


def _write_rank_data(cfg: FfnDataConfig, arrays) -> None:
    x_full, w_gate_full, w_up_full, w_down_full, _golden = arrays
    t, h, fi = cfg.t, cfg.h, cfg.fi
    grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
    h_per_col = h // grid_cols
    split_mode = cfg.split_mode
    src_type = np.float16

    for row in range(grid_rows):
        t_start = row * t
        t_end = t_start + t
        x_row = x_full[t_start:t_end, :]

        for col in range(grid_cols):
            rank = row * grid_cols + col
            fi_start = col * fi
            fi_end = fi_start + fi
            h_start = col * h_per_col
            h_end = h_start + h_per_col

            x_path = os.path.join(cfg.output_dir, f"pe_{rank}_x.bin")
            w_gate_path = os.path.join(cfg.output_dir, f"pe_{rank}_w_gate.bin")
            w_up_path = os.path.join(cfg.output_dir, f"pe_{rank}_w_up.bin")
            w_down_path = os.path.join(cfg.output_dir, f"pe_{rank}_w_down.bin")

            # X is the same for every col rank within the same row.
            x_row.tofile(x_path)
            w_gate_full[:, fi_start:fi_end].astype(src_type).tofile(w_gate_path)
            w_up_full[:, fi_start:fi_end].astype(src_type).tofile(w_up_path)
            if split_mode == "allgather":
                w_down = w_down_full[:, h_start:h_end]
            else:
                w_down = w_down_full[fi_start:fi_end, :]
            w_down.astype(src_type).tofile(w_down_path)

            label = "block" if grid_cols == 1 else "rank"
            print(f"  - {label}{rank} (row={row},col={col}): x{tuple(x_row.shape)} -> {x_path}")
            print(f"             w_gate[{h},{fi}] -> {w_gate_path}")
            print(f"             w_up[{h},{fi}]   -> {w_up_path}")
            print(f"             w_down{tuple(w_down.shape)} -> {w_down_path}")


def gen_data(cfg: FfnDataConfig) -> None:
    t, h, fi = cfg.t, cfg.h, cfg.fi
    grid_rows, grid_cols = cfg.grid_rows, cfg.grid_cols
    n_ranks = grid_rows * grid_cols
    t_total = t * grid_rows
    f_total = fi * grid_cols
    if cfg.split_mode == "allgather" and h % grid_cols != 0:
        raise ValueError(f"allgather split requires h ({h}) divisible by grid_cols ({grid_cols})")
    os.makedirs(cfg.output_dir, exist_ok=True)

    arrays = _make_ffn_arrays(cfg)
    _write_rank_data(cfg, arrays)
    golden = arrays[-1]
    golden_path = os.path.join(cfg.output_dir, "golden.bin")
    golden.tofile(golden_path)
    print(f"  - golden{tuple(golden.shape)} fp32 -> {golden_path}")
    print(
        f"[INFO] Generated FFN data: T={t} (T_total={t_total}) H={h} Fi={fi} (F_total={f_total}) "
        f"grid={grid_rows}x{grid_cols} n_ranks={n_ranks} split_mode={split_mode}"
    )
    print(f"[INFO] alpha={PRELU_ALPHA} (must match kernel TLRELU constant FFN_PRELU_ALPHA)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data for distributed FFN GridPipe demo")
    parser.add_argument("--grid-rows", type=int, default=None,
                        help="Grid rows (M4 2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--grid-cols", type=int, default=None,
                        help="Grid cols (M4 2D layout).  If omitted, falls back to --n-ranks (1xN).")
    parser.add_argument("--n-ranks", type=int, default=2,
                        help="Back-compat: total ranks for 1xN grid (used when --grid-rows/--grid-cols absent)")
    parser.add_argument("--t", type=int, required=True, help="Token count per row (T)")
    parser.add_argument("--h", type=int, required=True, help="Hidden dim (H)")
    parser.add_argument("--fi", type=int, required=True, help="Per-rank intermediate dim per col (Fi)")
    parser.add_argument("--split-mode", choices=("reduce", "allgather"), default="reduce",
                        help="W_down sharding mode: reduce keeps [Fi,H], allgather keeps [F,Hc]")
    parser.add_argument("--output-dir", type=str, default="./out", help="Output directory")

    args = parser.parse_args()
    if args.grid_rows is None and args.grid_cols is None:
        grid_rows = 1
        grid_cols = args.n_ranks
    else:
        if args.grid_rows is None or args.grid_cols is None:
            parser.error("--grid-rows and --grid-cols must be specified together")
        grid_rows = args.grid_rows
        grid_cols = args.grid_cols

    cfg = FfnDataConfig(
        t=args.t,
        h=args.h,
        fi=args.fi,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        split_mode=args.split_mode,
        output_dir=args.output_dir,
    )
    gen_data(cfg)


if __name__ == "__main__":
    main()
