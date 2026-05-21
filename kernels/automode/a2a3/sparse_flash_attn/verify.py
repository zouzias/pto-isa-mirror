#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "shape.json", "r", encoding="utf-8") as f:
        shape = json.load(f)

    expected = shape["b"] * shape["m"] * shape["h"] * shape["d"]
    output = np.fromfile(data_dir / "output.bin", dtype=np.float32)
    golden = np.fromfile(data_dir / "golden.bin", dtype=np.float32)

    if output.size != expected:
        print(f"Output size mismatch: got {output.size}, expected {expected}")
        sys.exit(1)
    if golden.size != expected:
        print(f"Golden size mismatch: got {golden.size}, expected {expected}")
        sys.exit(1)

    output = output.reshape(shape["b"], shape["m"], shape["h"], shape["d"])
    golden = golden.reshape(shape["b"], shape["m"], shape["h"], shape["d"])

    diff = np.abs(output - golden)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(diff / np.maximum(np.abs(golden), 1e-12)))
    ok = np.allclose(output, golden, rtol=args.rtol, atol=args.atol)

    print(f"max_abs_diff = {max_abs:.8e}")
    print(f"max_rel_diff = {max_rel:.8e}")
    
    print(f"Output: {output.flatten()[:8]}")
    print(f"Golden: {golden.flatten()[:8]}")
    if not ok:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Mismatch at {idx}: output={output[idx]}, golden={golden[idx]}, abs_diff={diff[idx]}")
        sys.exit(1)

    print("Verification passed.")


if __name__ == "__main__":
    main()
