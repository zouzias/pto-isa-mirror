from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from .codegen_cpp import generate_program
from .mlir_parser import parse_pto


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ptoas", description="PTO assembler demo: .pto (MLIR-like) -> C++ -> .bin")
    ap.add_argument("pto", help="Input .pto file (MLIR-like text)")
    ap.add_argument("-o", "--out", help="Output binary path (executable). Required unless --emit-dir is set.")
    ap.add_argument("--emit-dir", help="Emit generated C++/CMake sources to this directory and exit (no build).")
    ap.add_argument(
        "--kernel-name",
        help="Kernel base name for generated sources (defaults to output stem, else input stem).",
    )
    ap.add_argument("--soc", choices=["a3", "a5"], default="a3", help="Target SoC family (a3 covers a2/a3)")
    ap.add_argument("--run-mode", choices=["npu", "sim"], default="npu", help="Link against runtime or runtime_camodel")
    ap.add_argument("--build-dir", default="build/ptoas", help="Build directory")
    ap.add_argument("--keep", action="store_true", help="Keep generated sources under build dir")
    ap.add_argument("--run", action="store_true", help="Run the produced binary after build")
    args = ap.parse_args(argv)

    func = parse_pto(args.pto)
    kernel_name = args.kernel_name
    if kernel_name is None:
        kernel_name = Path(args.out).stem if args.out else Path(args.pto).stem

    repo_root = Path(__file__).resolve().parents[1]

    gen = generate_program(func, kernel_name=kernel_name, soc=args.soc, run_mode=args.run_mode, repo_root=str(repo_root))

    if args.emit_dir:
        emit_dir = Path(args.emit_dir)
        emit_dir.mkdir(parents=True, exist_ok=True)
        (emit_dir / f"{kernel_name}_kernel.cpp").write_text(gen.kernel_cpp, encoding="utf-8")
        if gen.kernel_cube_cpp is not None:
            (emit_dir / f"{kernel_name}_kernel_cube.cpp").write_text(gen.kernel_cube_cpp, encoding="utf-8")
        (emit_dir / f"{kernel_name}_main.cpp").write_text(gen.main_cpp, encoding="utf-8")
        (emit_dir / "CMakeLists.txt").write_text(gen.cmake, encoding="utf-8")
        return 0

    if not args.out:
        raise SystemExit("Missing -o/--out (required unless --emit-dir is set).")

    if "ASCEND_HOME_PATH" not in os.environ:
        raise SystemExit("ASCEND_HOME_PATH is not set (source Ascend setenv.bash first).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    build_dir = Path(args.build_dir) / out.stem
    if build_dir.exists() and not args.keep:
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    (build_dir / f"{out.stem}_kernel.cpp").write_text(gen.kernel_cpp, encoding="utf-8")
    if gen.kernel_cube_cpp is not None:
        (build_dir / f"{out.stem}_kernel_cube.cpp").write_text(gen.kernel_cube_cpp, encoding="utf-8")
    (build_dir / f"{out.stem}_main.cpp").write_text(gen.main_cpp, encoding="utf-8")
    (build_dir / "CMakeLists.txt").write_text(gen.cmake, encoding="utf-8")

    cmake_build = build_dir / "build"
    cmake_build.mkdir(exist_ok=True)
    _run(["cmake", "-S", ".", "-B", "build"], cwd=build_dir)
    jobs = os.cpu_count() or 4
    jobs = min(jobs, 32)
    _run(["cmake", "--build", "build", "-j", str(jobs)], cwd=build_dir)

    produced = build_dir / "build" / out.stem
    if not produced.exists():
        raise SystemExit(f"Build did not produce expected binary: {produced}")
    shutil.copy2(produced, out)
    out.chmod(0o755)

    if args.run:
        _run([str(out)], cwd=out.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
