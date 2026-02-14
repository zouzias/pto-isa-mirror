import os
from pathlib import Path
import numpy as np

np.random.seed(42)

CASES = [
    ("case_row_float_128x8", 128, 8, "row"),
    ("case_col_float_64x64", 64, 64, "col"),
]

BASE = "TRowColExpandSubSubtileTest"

# Write data to the location expected by GetGoldenDir():
# ../trowcolexpandsub_subtile relative to tests/npu/a2a3/src/st/testcase
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_BASE = SCRIPT_DIR.parent.parent / "trowcolexpandsub_subtile"


def save_case(case_name, rows, cols, mode):
    case_dir = OUT_BASE / f"{BASE}.{case_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    src0 = (np.random.rand(rows, cols).astype(np.float32) - 0.5) * 10.0
    if mode == "row":
        src1 = (np.random.rand(rows).astype(np.float32) - 0.5) * 10.0
        golden = src0 - src1[:, None]
    else:
        src1 = (np.random.rand(cols).astype(np.float32) - 0.5) * 10.0
        golden = src0 - src1[None, :]

    src0.tofile(case_dir / "input1.bin")
    src1.tofile(case_dir / "input2.bin")
    golden.astype(np.float32).tofile(case_dir / "golden.bin")


if __name__ == "__main__":
    for case_name, rows, cols, mode in CASES:
        save_case(case_name, rows, cols, mode)
