import os
import numpy as np

np.random.seed(42)

CASES = [
    ("case_row_float_128x8", 128, 8, "row"),
    ("case_col_float_64x64", 64, 64, "col"),
]

BASE = "TRowColExpandSubSubtileTest"


def save_case(case_name, rows, cols, mode):
    case_dir = f"{BASE}.{case_name}"
    os.makedirs(case_dir, exist_ok=True)

    src0 = (np.random.rand(rows, cols).astype(np.float32) - 0.5) * 10.0
    if mode == "row":
        src1 = (np.random.rand(rows).astype(np.float32) - 0.5) * 10.0
        golden = src0 - src1[:, None]
    else:
        src1 = (np.random.rand(cols).astype(np.float32) - 0.5) * 10.0
        golden = src0 - src1[None, :]

    src0.tofile(os.path.join(case_dir, "input1.bin"))
    src1.tofile(os.path.join(case_dir, "input2.bin"))
    golden.astype(np.float32).tofile(os.path.join(case_dir, "golden.bin"))


if __name__ == "__main__":
    for case_name, rows, cols, mode in CASES:
        save_case(case_name, rows, cols, mode)
