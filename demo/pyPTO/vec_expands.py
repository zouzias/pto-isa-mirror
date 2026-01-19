from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "python"))

import pypto  # noqa: E402
from pypto import DType, MemRefType  # noqa: E402


@pypto.kernel
def vec_expands(
    out: MemRefType("gm", (64, 64), DType.f32),
):
    t0 = texpands(2.0, like=out)
    tstore(out, t0)


def texpands(x, like=None):
    raise RuntimeError("PyPTO DSL marker; use `python -m pypto emit ...`")


def tstore(out, t):
    raise RuntimeError("PyPTO DSL marker; use `python -m pypto emit ...`")


if __name__ == "__main__":
    out_pto = repo_root / "demo" / "pto" / "vec_expands.pto"
    pypto.emit_mlir_file(vec_expands, out_pto)
    print(out_pto)

