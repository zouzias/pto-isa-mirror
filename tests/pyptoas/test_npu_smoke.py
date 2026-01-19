from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _have_npu() -> bool:
    try:
        subprocess.run(["npu-smi", "info"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _pattern(rows: int, cols: int) -> np.ndarray:
    n = rows * cols
    x = (np.arange(n, dtype=np.int64) * 13 + 7) % 97
    return x.astype(np.float32).reshape(rows, cols)


def _read_f32_tile(path: Path, rows: int, cols: int) -> np.ndarray:
    data = path.read_bytes()
    expect = rows * cols * 4
    if len(data) != expect:
        raise AssertionError(f"Unexpected output size {len(data)} for {path}, expected {expect}")
    vals = struct.unpack(f"<{rows*cols}f", data)
    return np.array(vals, dtype=np.float32).reshape(rows, cols)

def _write_f32_tile(path: Path, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float32)
    path.write_bytes(a.tobytes(order="C"))


class PyPTOASNpuSmokeTests(unittest.TestCase):
    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_vec_add_npu(self):
        pto = REPO_ROOT / "demo/pto/vec_add.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "vec_add.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "out.bin", 64, 64)
            a = _pattern(64, 64)
            b = _pattern(64, 64)
            np.testing.assert_allclose(out, a + b, rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_vec_mul_npu(self):
        pto = REPO_ROOT / "demo/pto/vec_mul.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "vec_mul.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "out.bin", 64, 64)
            a = _pattern(64, 64)
            b = _pattern(64, 64)
            np.testing.assert_allclose(out, a * b, rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_vec_adds_npu(self):
        pto = REPO_ROOT / "demo/pto/vec_adds.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "vec_adds.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "out.bin", 64, 64)
            a = _pattern(64, 64)
            np.testing.assert_allclose(out, a + 1.5, rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_vec_abs_npu(self):
        pto = REPO_ROOT / "demo/pto/vec_abs.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "vec_abs.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "out.bin", 64, 64)
            a = _pattern(64, 64)
            np.testing.assert_allclose(out, np.abs(a), rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_vec_expands_npu(self):
        pto = REPO_ROOT / "demo/pto/vec_expands.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "vec_expands.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "out.bin", 64, 64)
            np.testing.assert_allclose(out, np.full((64, 64), 2.0, dtype=np.float32), rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_linear_tmatmul_npu(self):
        pto = REPO_ROOT / "pto-isa-main/examples/output_pto/torch_functional/F_linear.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bin_path = td / "F_linear.bin"
            out_dir = td / "out"
            out_dir.mkdir()
            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run([str(bin_path), "--out-dir", str(out_dir)], check=True, cwd=str(out_dir))
            out = _read_f32_tile(out_dir / "output_mem.bin", 8, 8)
            x = _pattern(8, 8)
            w = _pattern(8, 8)
            b = _pattern(8, 8)
            np.testing.assert_allclose(out, x @ w + b, rtol=0, atol=0)

    @unittest.skipUnless(_have_npu(), "No NPU detected (npu-smi missing or failed)")
    def test_sdpa_with_scale_npu(self):
        pto = REPO_ROOT / "pto-isa-main/examples/output_pto/flex_attention/sdpa_with_scale.pto"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            in_dir = td / "in"
            out_dir = td / "out"
            in_dir.mkdir()
            out_dir.mkdir()
            bin_path = td / "sdpa_with_scale.bin"

            q = (_pattern(8, 8) / 97.0).astype(np.float32)
            k = (_pattern(8, 8)[::-1] / 97.0).astype(np.float32)
            v = ((_pattern(8, 8) + 3) / 97.0).astype(np.float32)
            _write_f32_tile(in_dir / "Q_mem.bin", q)
            _write_f32_tile(in_dir / "K_mem.bin", k)
            _write_f32_tile(in_dir / "V_mem.bin", v)

            subprocess.run(
                ["python3", "-m", "ptoas", str(pto), "-o", str(bin_path), "--soc", "a3", "--run-mode", "npu"],
                check=True,
                cwd=str(REPO_ROOT),
            )
            subprocess.run(
                [str(bin_path), "--out-dir", str(out_dir), "--in-dir", str(in_dir)],
                check=True,
                cwd=str(out_dir),
            )
            out = _read_f32_tile(out_dir / "output_mem.bin", 8, 8)

            scores = q @ k
            scaled = scores * np.float32(0.35355339059327373)
            row_sum = np.sum(scaled, axis=1, keepdims=True) / np.float32(8.0)
            shifted = scaled - row_sum
            exp_scores = np.exp(shifted)
            attn = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            expect = attn @ v
            np.testing.assert_allclose(out, expect, rtol=5e-4, atol=5e-4)


if __name__ == "__main__":
    os.environ.setdefault("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH", ""))
    unittest.main()
