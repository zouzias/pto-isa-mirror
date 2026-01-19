from __future__ import annotations

import argparse
import re
from pathlib import Path

from .ast_frontend import emit_mlir_file, load_kernels_from_file
from .sim import simulate_mlir_module


def _cmd_emit(args: argparse.Namespace) -> int:
    kernels = load_kernels_from_file(args.py)
    if not kernels:
        raise SystemExit(f"No @pypto.kernel functions found in {args.py}")
    if len(kernels) > 1:
        raise SystemExit(f"Multiple kernels found in {args.py}; keep only one for now")
    emit_mlir_file(kernels[0], args.out)
    return 0


def _cmd_pydb(args: argparse.Namespace) -> int:
    from ptoas.mlir_parser import parse_pto  # local repo import

    import numpy as np

    pto_path = Path(args.pto)
    pto_text = pto_path.read_text(encoding="utf-8", errors="replace")
    func = parse_pto(pto_path)

    re_memref = re.compile(r"!pto\.memref<(?P<space>\\w+),(?P<shape>[^>]+)>")
    re_tile = re.compile(r"!pto\\.(?:tile|tilebuf)<(?:(?P<kind>\\w+),)?(?P<r>\\d+)x(?P<c>\\d+)x(?P<dt>\\w+)>")

    # Collect tile SSA types (from PTO-AS decls and op type sigs).
    tile_types: dict[str, tuple[int, int, str]] = {}
    for ssa, ty in func.tile_decls.items():
        m = re_tile.search(ty)
        if m:
            tile_types[ssa] = (int(m["r"]), int(m["c"]), m["dt"])

    for op in func.ops:
        if not op.results or not op.type_sig:
            continue
        m = re_tile.search(op.type_sig)
        if m:
            tile_types[op.results[0]] = (int(m["r"]), int(m["c"]), m["dt"])

    # Collect output memrefs (tstore / mscatter).
    outputs: set[str] = set()
    for op in func.ops:
        if op.name in {"pto.tstore", "pto.mscatter"} and op.operands:
            dst = op.operands[0].split("[", 1)[0]
            outputs.add(dst)

    # Infer memref shapes.
    mem_shapes: dict[str, tuple[int, int, str]] = {}
    for ssa, ty in func.args.items():
        m = re_memref.fullmatch(ty.strip())
        if not m:
            continue
        shape = m["shape"].strip()
        if shape.startswith("..."):
            parts = [p.strip() for p in shape.split(",")]
            mem_shapes[ssa] = (64, 64, parts[-1])
            continue
        parts = shape.split("x")
        if len(parts) == 3:
            mem_shapes[ssa] = (int(parts[0]), int(parts[1]), parts[2].strip())

    # Refine unknown shapes from tload/tstore tile shapes.
    for op in func.ops:
        if op.name == "pto.tload" and op.results and op.operands:
            mem = op.operands[0].split("[", 1)[0]
            if mem in mem_shapes and mem_shapes[mem][0] == 64 and op.results[0] in tile_types:
                r, c, dt = tile_types[op.results[0]]
                mem_shapes[mem] = (r, c, dt)
        if op.name == "pto.tstore" and len(op.operands) == 2:
            mem = op.operands[0].split("[", 1)[0]
            src = op.operands[1]
            if mem in mem_shapes and mem_shapes[mem][0] == 64 and src in tile_types:
                r, c, dt = tile_types[src]
                mem_shapes[mem] = (r, c, dt)

    in_dir = Path(args.in_dir) if args.in_dir else None
    in_files: dict[str, Path] = {}
    for kv in args.in_files or []:
        if "=" not in kv:
            raise SystemExit(f"bad --in (expected name=path): {kv}")
        k, v = kv.split("=", 1)
        in_files[k] = Path(v)

    def load_or_pattern(name: str, rows: int, cols: int, is_output: bool) -> np.ndarray:
        if is_output:
            return np.zeros((rows, cols), dtype=np.float32)
        path = in_files.get(name)
        if path is None and in_dir is not None:
            cand = in_dir / f"{name}.bin"
            if cand.exists():
                path = cand
        if path is not None:
            data = path.read_bytes()
            expect = rows * cols * 4
            if len(data) != expect:
                raise SystemExit(f"{path} has {len(data)} bytes, expected {expect}")
            return np.frombuffer(data, dtype=np.float32).reshape(rows, cols).copy()
        n = rows * cols
        x = (np.arange(n, dtype=np.int64) * 13 + 7) % 97
        return x.astype(np.float32).reshape(rows, cols)

    inputs: dict[str, np.ndarray] = {}
    for ssa in func.arg_order:
        rows, cols, dt = mem_shapes.get(ssa, (64, 64, "f32"))
        if dt != "f32":
            raise SystemExit(f"pydb currently supports only f32 memrefs, got {ssa}: {dt}")
        inputs[ssa[1:]] = load_or_pattern(ssa[1:], rows, cols, ssa in outputs)

    simulate_mlir_module(pto_text, inputs, dump=tuple(args.dump or ()))
    if outputs:
        any_out = sorted(outputs)[0][1:]
        arr = inputs[any_out]
        print(f"[pypto sim] {any_out}[0,0] = {float(arr.reshape(-1)[0])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="pypto", description="PyPTO: emit/debug PTO MLIR-like programs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_emit = sub.add_parser("emit", help="Emit a .pto file from a Python kernel")
    ap_emit.add_argument("py", help="Path to python file containing a single @pypto.kernel")
    ap_emit.add_argument("-o", "--out", required=True, help="Output .pto path")
    ap_emit.set_defaults(func=_cmd_emit)

    ap_pydb = sub.add_parser("pydb", help="CPU-sim debug a .pto file and dump tiles")
    ap_pydb.add_argument("pto", help="Input .pto file")
    ap_pydb.add_argument("--in-dir", help="Directory containing raw f32 inputs named <memref>.bin")
    ap_pydb.add_argument("--in", dest="in_files", action="append", help="Override input: name=path (repeatable)")
    ap_pydb.add_argument("--dump", action="append", help="Tile SSA to dump, e.g. %t2 (repeatable)")
    ap_pydb.set_defaults(func=_cmd_pydb)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
