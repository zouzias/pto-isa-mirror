from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "python"))

import pypto  # noqa: E402
from pypto import DType, MemRefType  # noqa: E402


@pypto.kernel
def vec_adds(
    a: MemRefType("gm", (64, 64), DType.f32),
    out: MemRefType("gm", (64, 64), DType.f32),
):
    t0 = tload(a)
    t1 = tadds(t0, 1.5)
    tstore(out, t1)


def tload(x):
    raise RuntimeError("PyPTO DSL marker; use `python -m pypto emit ...`")


def tadds(x, scalar):
    raise RuntimeError("PyPTO DSL marker; use `python -m pypto emit ...`")


def tstore(out, t):
    raise RuntimeError("PyPTO DSL marker; use `python -m pypto emit ...`")


if __name__ == "__main__":
    out_pto = repo_root / "demo" / "pto" / "vec_adds.pto"
    pypto.emit_mlir_file(vec_adds, out_pto)
    print(out_pto)

