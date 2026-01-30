import argparse
import os
import sys

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BINDING_PY = _REPO_ROOT / "binding" / "python"
if str(_BINDING_PY) not in sys.path:
    sys.path.insert(0, str(_BINDING_PY))

from ptoas.python import binding, pipeline  # noqa: E402
from ptoas.python.host_spec import prepend_host_spec_to_pto  # noqa: E402

def _default_ptoas(repo: Path) -> Path:
    for p in (
        repo / "ptoas/mlir/build-macos/bin/ptoas",
        repo / "ptoas/mlir/build/bin/ptoas",
    ):
        if p.exists():
            return p
    return repo / "ptoas/mlir/build/bin/ptoas"

def main() -> int:
    repo = pipeline.repo_root()
    ap = argparse.ArgumentParser(description="Run GEMM 256x256x256 (tiled 16x16) end-to-end.")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu")
    ap.add_argument("--soc", default="a3", help="Simulator SoC alias when --run-mode=sim (a3|a5|other)")
    ap.add_argument("--ascend-home", type=Path, default=pipeline.default_ascend_home())
    ap.add_argument("--ptoas", type=Path, default=_default_ptoas(repo))
    ap.add_argument("--outdir", type=Path, default=Path("./pto_output/"))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--memory-model", default="MEMORY_BASE")
    ap.add_argument("--no-insert-events", dest="insert_events", action="store_false", default=True)
    ap.add_argument("--verbose-build", action="store_true", help="Print compiler commands/warnings")
    args = ap.parse_args()

    py = Path(__file__).resolve().with_name("gemm256.py")
    spec = binding.compile_file(py, kernel="gemm256")
    pto_text = prepend_host_spec_to_pto(pto=spec.pto, spec=binding.default_host_spec(spec))

    pto_path = args.outdir / f"{spec.name}.pto"
    pto_path.write_text(pto_text, encoding="utf-8")

    cfg = pipeline.CompileConfig(
        ptoas=args.ptoas,
        ascend_home=args.ascend_home,
        arch="dav-c220-cube",
        memory_model=args.memory_model,
        insert_events=args.insert_events,
    )
    cce_cpp, _bin = pipeline.compile_pto_to_cce_and_bin(pto_path=pto_path, outdir=args.outdir, cfg=cfg)
    print(cce_cpp)

if __name__ == "__main__":
    raise SystemExit(main())