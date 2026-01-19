from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "python"))

import pypto  # noqa: E402
from pypto import DType, MemRefType  # noqa: E402


@pypto.kernel
def vec_add(
    a: MemRefType("gm", (64, 64), DType.f32),
    b: MemRefType("gm", (64, 64), DType.f32),
    out: MemRefType("gm", (64, 64), DType.f32),
):
    t0 = tload(a)
    t1 = tload(b)
    t2 = tadd(t0, t1)
    tstore(out, t2)


def tload(x):
    raise RuntimeError("This is a PyPTO DSL marker; use `python -m pypto emit ...`")


def tadd(a, b):
    raise RuntimeError("This is a PyPTO DSL marker; use `python -m pypto emit ...`")


def tstore(out, t):
    raise RuntimeError("This is a PyPTO DSL marker; use `python -m pypto emit ...`")


if __name__ == "__main__":
    out_pto = repo_root / "demo" / "pto" / "vec_add.pto"
    pypto.emit_mlir_file(vec_add, out_pto)
    print(out_pto)
