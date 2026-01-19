from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from pypto.sim import simulate_mlir_module  # noqa: E402


def _pattern(rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    x = (np.arange(n, dtype=np.int64) * 13 + 7) % 97
    return x.astype(np.float32).reshape(rows, cols)


class PyPTOCpuSimTests(unittest.TestCase):
    def _run(self, pto_rel: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        p = REPO_ROOT / pto_rel
        text = p.read_text(encoding="utf-8")
        simulate_mlir_module(text, inputs)
        return inputs

    def test_vec_add(self):
        a = _pattern(64, 64)
        b = _pattern(64, 64) * 2
        out = np.zeros((64, 64), dtype=np.float32)
        self._run("demo/pto/vec_add.pto", {"a": a, "b": b, "out": out})
        np.testing.assert_allclose(out, a + b, rtol=0, atol=0)

    def test_vec_abs(self):
        a = -_pattern(64, 64)
        out = np.zeros((64, 64), dtype=np.float32)
        self._run("demo/pto/vec_abs.pto", {"a": a, "out": out})
        np.testing.assert_allclose(out, np.abs(a), rtol=0, atol=0)

    def test_vec_mul(self):
        a = _pattern(64, 64)
        b = _pattern(64, 64) + 1
        out = np.zeros((64, 64), dtype=np.float32)
        self._run("demo/pto/vec_mul.pto", {"a": a, "b": b, "out": out})
        np.testing.assert_allclose(out, a * b, rtol=0, atol=0)

    def test_vec_adds(self):
        a = _pattern(64, 64)
        out = np.zeros((64, 64), dtype=np.float32)
        self._run("demo/pto/vec_adds.pto", {"a": a, "out": out})
        np.testing.assert_allclose(out, a + 1.5, rtol=0, atol=0)

    def test_vec_expands(self):
        out = np.zeros((64, 64), dtype=np.float32)
        self._run("demo/pto/vec_expands.pto", {"out": out})
        np.testing.assert_allclose(out, np.full((64, 64), 2.0, dtype=np.float32), rtol=0, atol=0)

    def test_linear_matmul(self):
        x = _pattern(8, 8)
        w = _pattern(8, 8) * 2
        bias = _pattern(8, 8) + 1
        out = np.zeros((8, 8), dtype=np.float32)
        self._run("demo/pto/linear.pto", {"x": x, "w": w, "bias": bias, "out": out})
        np.testing.assert_allclose(out, x @ w + bias, rtol=0, atol=0)

    def test_sdpa_with_scale_pto_as(self):
        q = (_pattern(8, 8) / 97.0).astype(np.float32)
        k = (_pattern(8, 8)[::-1] / 97.0).astype(np.float32)
        v = ((_pattern(8, 8) + 3) / 97.0).astype(np.float32)
        out = np.zeros((8, 8), dtype=np.float32)
        self._run(
            "pto-isa-main/examples/output_pto/flex_attention/sdpa_with_scale.pto",
            {"Q_mem": q, "K_mem": k, "V_mem": v, "output_mem": out},
        )
        scores = q @ k
        scaled = scores * np.float32(0.35355339059327373)
        row_sum = np.sum(scaled, axis=1, keepdims=True)
        row_sum = row_sum / np.float32(8.0)
        shifted = scaled - row_sum
        exp_scores = np.exp(shifted)
        row_sum2 = np.sum(exp_scores, axis=1, keepdims=True)
        attn = exp_scores / row_sum2
        expect = attn @ v
        np.testing.assert_allclose(out, expect, rtol=1e-6, atol=1e-6)

    def test_scf_for(self):
        a = _pattern(4, 1)
        out = np.zeros((4, 1), dtype=np.float32)
        text = """
module {
  func.func @main(%a: !pto.memref<gm,4x1xf32>, %out: !pto.memref<gm,4x1xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %t0 = pto.alloc_tile : !pto.tilebuf<1x1xf32>
    %t1 = pto.alloc_tile : !pto.tilebuf<1x1xf32>
    scf.for %iv = %c0 to %c4 step %c1 {
      pto.tload ins(%a[%iv, %c0] : !pto.memref<gm,4x1xf32>) outs(%t0 : !pto.tilebuf<1x1xf32>)
      pto.tadds ins(%t0, 1.0 : !pto.tilebuf<1x1xf32>, f32) outs(%t1 : !pto.tilebuf<1x1xf32>)
      pto.tstore ins(%t1 : !pto.tilebuf<1x1xf32>) outs(%out[%iv, %c0] : !pto.memref<gm,4x1xf32>)
    }
    return
  }
}
""".strip()
        simulate_mlir_module(text, {"a": a, "out": out})
        np.testing.assert_allclose(out, a + 1.0, rtol=0, atol=0)

    def test_scf_if(self):
        a = _pattern(1, 1)
        out = np.zeros((1, 1), dtype=np.float32)
        text = """
module {
  func.func @main(%a: !pto.memref<gm,1x1xf32>, %out: !pto.memref<gm,1x1xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %t0 = pto.alloc_tile : !pto.tilebuf<1x1xf32>
    %t1 = pto.alloc_tile : !pto.tilebuf<1x1xf32>

    %cond = arith.cmpi slt, %c0, %c1 : index
    scf.if %cond {
      pto.tload ins(%a[%c0, %c0] : !pto.memref<gm,1x1xf32>) outs(%t0 : !pto.tilebuf<1x1xf32>)
      pto.tadds ins(%t0, 2.0 : !pto.tilebuf<1x1xf32>, f32) outs(%t1 : !pto.tilebuf<1x1xf32>)
      pto.tstore ins(%t1 : !pto.tilebuf<1x1xf32>) outs(%out[%c0, %c0] : !pto.memref<gm,1x1xf32>)
    } else {
      pto.tload ins(%a[%c0, %c0] : !pto.memref<gm,1x1xf32>) outs(%t0 : !pto.tilebuf<1x1xf32>)
      pto.tadds ins(%t0, 3.0 : !pto.tilebuf<1x1xf32>, f32) outs(%t1 : !pto.tilebuf<1x1xf32>)
      pto.tstore ins(%t1 : !pto.tilebuf<1x1xf32>) outs(%out[%c0, %c0] : !pto.memref<gm,1x1xf32>)
    }
    return
  }
}
""".strip()
        simulate_mlir_module(text, {"a": a, "out": out})
        np.testing.assert_allclose(out, a + 2.0, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
