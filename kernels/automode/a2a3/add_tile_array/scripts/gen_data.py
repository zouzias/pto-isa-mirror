#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# add_tile_array - gen_data.py
# Generates input and golden output for the add_tile_array auto-mode prototype.
# Mirrors kernels/manual/a2a3/topk/scripts/gen_data.py in style; the layout is
# simpler because this kernel is just element-wise C = A + B.
#
# Output files (all raw little-endian float32, contiguous, no header):
#   ./input/input_a.bin    (TOTAL_ROWS * TILE_COLS floats)
#   ./input/input_b.bin    (same)
#   ./output/golden_c.bin  (same; A + B)
#
# Inputs are integer values in [1, 10] cast to float32 so element-wise
# addition is exact in IEEE-754 float32. main.cpp uses ResultCmp tolerance
# 0.001f (comfortably wide).
# --------------------------------------------------------------------------------

import os
import json
from pathlib import Path
import numpy as np
np.random.seed(19)


def load_generated_case():
    case_path = Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"
    if not case_path.exists():
        return {"num_tiles": 4, "tile_rows": 64, "tile_cols": 64}
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return cases[0]


def gen_golden_data(param):
    src_type  = param.src_type
    num_tiles = param.num_tiles
    tile_rows = param.tile_rows
    tile_cols = param.tile_cols

    rows = num_tiles * tile_rows
    cols = tile_cols

    a = np.random.randint(1, 10, size=(rows, cols)).astype(src_type)
    b = np.random.randint(1, 10, size=(rows, cols)).astype(src_type)
    golden = (a + b).astype(src_type)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    a.tofile("./input/input_a.bin")
    b.tofile("./input/input_b.bin")
    golden.tofile("./output/golden_c.bin")


class AddTileArrayParams:
    def __init__(self, src_type, num_tiles, tile_rows, tile_cols):
        self.src_type  = src_type
        self.num_tiles = num_tiles
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols


if __name__ == "__main__":
    case = load_generated_case()
    case_params_list = [
        AddTileArrayParams(np.float32, case["num_tiles"], case["tile_rows"], case["tile_cols"]),
    ]
    gen_golden_data(case_params_list[0])
