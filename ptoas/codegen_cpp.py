from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .mlir_parser import Func, Op
from .op_classify import Pipe, classify_pto_op


_RE_TILE = re.compile(r"!pto\.(?:tile|tilebuf)<(?P<body>[^>]+)>")
_RE_MEMREF = re.compile(r"!pto\.memref<(?P<space>\w+),(?P<shape>[^>]+)>")


@dataclass(frozen=True)
class TileInfo:
    kind: str  # "vec" (default)
    rows: int
    cols: int
    dtype: str

    def key(self) -> Tuple[str, int, int, str]:
        return (self.kind, self.rows, self.cols, self.dtype)


@dataclass(frozen=True)
class MemRefInfo:
    space: str
    rows: Optional[int]
    cols: Optional[int]
    dtype: str


@dataclass(frozen=True)
class Generated:
    kernel_cpp: str
    kernel_cube_cpp: Optional[str]
    main_cpp: str
    cmake: str


def _cpp_dtype(dt: str) -> Tuple[str, str]:
    if dt == "f32":
        return ("float", "float")
    if dt == "i32":
        return ("int32_t", "int32_t")
    if dt == "i16":
        return ("int16_t", "int16_t")
    if dt == "f16":
        return ("aclFloat16", "half")
    if dt == "u8":
        return ("uint8_t", "uint8_t")
    if dt == "i8":
        return ("int8_t", "int8_t")
    raise ValueError(f"Unsupported dtype for codegen: {dt}")


def _parse_memref(type_str: str) -> MemRefInfo:
    m = _RE_MEMREF.fullmatch(type_str.strip())
    if not m:
        raise ValueError(f"Unsupported memref type: {type_str}")
    space = m["space"]
    shape = m["shape"].strip()
    # `pto-isa-main` often uses `...` for unknown shapes: `!pto.memref<gm,...,f32>`.
    if shape.startswith("..."):
        parts = [p.strip() for p in shape.split(",")]
        dtype = parts[-1]
        return MemRefInfo(space=space, rows=None, cols=None, dtype=dtype)
    parts = shape.split("x")
    if len(parts) != 3:
        raise ValueError(f"Unsupported memref shape: {type_str}")
    rows, cols = int(parts[0]), int(parts[1])
    dtype = parts[2].strip()
    return MemRefInfo(space=space, rows=rows, cols=cols, dtype=dtype)


def _parse_tile_type(type_str: str) -> TileInfo:
    """
    Supports:
      - !pto.tile<64x64xf32>
      - !pto.tilebuf<64x64xf32>
      - !pto.tile<vec,64x64xf32>  (future-proof)
    """
    m = _RE_TILE.fullmatch(type_str.strip())
    if not m:
        raise ValueError(f"Unsupported tile type: {type_str}")
    body = m["body"].strip()
    if "," in body:
        kind, rest = [x.strip() for x in body.split(",", 1)]
    else:
        kind, rest = "vec", body
    parts = rest.split("x")
    if len(parts) != 3:
        raise ValueError(f"Unsupported tile shape: {type_str}")
    rows, cols = int(parts[0]), int(parts[1])
    dtype = parts[2].strip()
    return TileInfo(kind=kind, rows=rows, cols=cols, dtype=dtype)


def _tile_kind_cpp(kind: str) -> str:
    k = kind.strip().lower()
    mapping = {
        "vec": "Vec",
        "mat": "Mat",
        "left": "Left",
        "right": "Right",
        "acc": "Acc",
        "bias": "Bias",
        "scaling": "Scaling",
        "scaleleft": "ScaleLeft",
        "scaleright": "ScaleRight",
        "scale_left": "ScaleLeft",
        "scale_right": "ScaleRight",
    }
    if k not in mapping:
        raise ValueError(f"Unsupported tile kind: {kind}")
    return mapping[k]

def _tile_cpp_type_expr(kind_cpp: str, base_r: int, base_c: int, valid_r: int, valid_c: int) -> str:
    """
    C++ type expression for a tile with compile-time base dims.
    """
    if kind_cpp == "Vec":
        return f"Tile<TileType::Vec, DevT, {base_r}, {base_c}, BLayout::RowMajor, {valid_r}, {valid_c}>"
    if kind_cpp == "Mat":
        # Boxed Mat tile in L1 (cbuf) used for cube pipeline.
        return (
            f"Tile<TileType::Mat, DevT, {base_r}, {base_c}, BLayout::ColMajor, {valid_r}, {valid_c}, "
            f"SLayout::RowMajor, TileConfig::fractalABSize>"
        )
    if kind_cpp == "Left":
        return f"TileLeft<DevT, {base_r}, {base_c}, {valid_r}, {valid_c}>"
    if kind_cpp == "Right":
        return f"TileRight<DevT, {base_r}, {base_c}, {valid_r}, {valid_c}>"
    if kind_cpp == "Acc":
        return f"TileAcc<DevT, {base_r}, {base_c}, {valid_r}, {valid_c}>"
    if kind_cpp == "ScaleLeft":
        return f"TileLeftScale<DevT, {base_r}, {base_c}, {valid_r}, {valid_c}>"
    if kind_cpp == "ScaleRight":
        return f"TileRightScale<DevT, {base_r}, {base_c}, {valid_r}, {valid_c}>"
    if kind_cpp == "Scaling":
        return f"Tile<TileType::Scaling, DevT, {base_r}, {base_c}, BLayout::RowMajor, {valid_r}, {valid_c}>"
    if kind_cpp == "Bias":
        return f"Tile<TileType::Bias, DevT, {base_r}, {base_c}, BLayout::RowMajor, {valid_r}, {valid_c}>"
    return f"Tile<TileType::{kind_cpp}, DevT, {base_r}, {base_c}, BLayout::RowMajor, {valid_r}, {valid_c}>"


def _split_mlir_type_list(blob: str) -> List[str]:
    blob = blob.strip()
    if blob == "()":
        return []
    if blob.startswith("(") and blob.endswith(")"):
        inner = blob[1:-1].strip()
        if not inner:
            return []
        parts: List[str] = []
        cur: List[str] = []
        angle = 0
        for ch in inner:
            if ch == "<":
                angle += 1
            elif ch == ">":
                angle = max(0, angle - 1)
            if ch == "," and angle == 0:
                parts.append("".join(cur).strip())
                cur = []
                continue
            cur.append(ch)
        if cur:
            parts.append("".join(cur).strip())
        return parts
    return [blob]


def _parse_return_types(type_sig: str) -> List[str]:
    if "->" not in type_sig:
        return []
    ret = type_sig.split("->", 1)[1].strip()
    return _split_mlir_type_list(ret)

def _parse_operand_types(type_sig: str) -> List[str]:
    if "->" not in type_sig:
        return []
    ins = type_sig.split("->", 1)[0].strip()
    return _split_mlir_type_list(ins)

_NO_TILE_RESULT_OPS = {
    "pto.tstore",
    "pto.tstore.fp",
    "pto.mscatter",
    "pto.setval",
    "pto.tassign",
    "pto.tprint",
    "pto.tsync",
}

_TWO_TILE_RESULT_OPS = {
    "pto.tsort32",
}

_ONE_TILE_PLUS_SCALAR_RESULT_OPS = {
    "pto.tmrgsort",
}

def _tile_result_arity(op_name: str) -> int:
    if op_name in _NO_TILE_RESULT_OPS or op_name == "pto.getval":
        return 0
    if op_name in _TWO_TILE_RESULT_OPS:
        return 2
    if op_name in _ONE_TILE_PLUS_SCALAR_RESULT_OPS:
        return 1
    if op_name.startswith("pto.t") or op_name in {"pto.mgather"}:
        return 1
    return 0

def _is_tile_type_str(s: str) -> bool:
    st = s.strip()
    return st.startswith("!pto.tile<") or st.startswith("!pto.tilebuf<")

def _parse_tile_from_decl(ty: str) -> TileInfo:
    return _parse_tile_type(ty)

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def _ceil_align(v: int, align: int) -> int:
    return ((v + align - 1) // align) * align

def _aligned_cols_for_vec(cols: int, dtype: str) -> int:
    # Vec RowMajor with SFractal=NoneBox requires cols*sizeof(dtype) to be 32B aligned.
    elem = {"f32": 4, "i32": 4, "f16": 2, "i16": 2, "u8": 1, "i8": 1}.get(dtype)
    if elem is None:
        raise ValueError(f"Unsupported dtype for alignment: {dtype}")
    bytes_ = cols * elem
    blocks = _ceil_div(bytes_, 32)
    return (blocks * 32) // elem


def _dtype_size_bytes(dtype: str) -> int:
    elem = {"f32": 4, "i32": 4, "f16": 2, "i16": 2, "u8": 1, "i8": 1}.get(dtype)
    if elem is None:
        raise ValueError(f"Unsupported dtype size: {dtype}")
    return elem


def _align_up(v: int, align: int) -> int:
    return ((v + align - 1) // align) * align


def _block_align_elems(dtype: str) -> int:
    # C0_SIZE_BYTE is 32 bytes.
    return 32 // _dtype_size_bytes(dtype)


def _tile_base_dims(kind_cpp: str, rows: int, cols: int, dtype: str) -> Tuple[int, int]:
    """
    Compute (base_rows, base_cols) for a tile type that satisfies backend layout constraints.
    `rows/cols` are treated as runtime valid dimensions.
    """
    if kind_cpp == "Vec":
        return rows, _aligned_cols_for_vec(cols, dtype)
    # Cube-friendly alignment: for `f32` cube tiles, align to 16 to satisfy boxed layout constraints.
    cube_k_align = 16 if dtype == "f32" else _block_align_elems(dtype)
    if kind_cpp == "Mat":
        return _ceil_align(rows, 16), _ceil_align(cols, cube_k_align)
    if kind_cpp == "Left":
        return _ceil_align(rows, 16), _ceil_align(cols, cube_k_align)
    if kind_cpp == "Right":
        return _ceil_align(rows, cube_k_align), _ceil_align(cols, 16)
    if kind_cpp == "Acc":
        return _ceil_align(rows, 16), _ceil_align(cols, 16)
    if kind_cpp in {"ScaleLeft"}:
        return _ceil_align(rows, 16), _ceil_align(cols, _block_align_elems(dtype))
    if kind_cpp in {"ScaleRight"}:
        return _ceil_align(rows, _block_align_elems(dtype)), _ceil_align(cols, 16)
    if kind_cpp == "Scaling":
        return rows, _ceil_align(cols, _block_align_elems(dtype))
    if kind_cpp == "Bias":
        return rows, cols
    return rows, cols


def _tile_storage_bytes(kind_cpp: str, rows: int, cols: int, dtype: str) -> Tuple[int, int, int]:
    """
    Returns (base_rows, base_cols, bytes) for storage allocation.
    This is a best-effort model that aligns base dimensions to satisfy common backend constraints.
    """
    base_r, base_c = _tile_base_dims(kind_cpp, rows, cols, dtype)
    size = base_r * base_c * _dtype_size_bytes(dtype)
    return base_r, base_c, size


def _intrinsic_name(op_name: str) -> str:
    short = op_name.split(".", 1)[1]
    # Keep dot-suffixed spellings MLIR-friendly and map to C++ intrinsics.
    # Examples:
    # - pto.tstore.fp -> TSTORE_FP
    # - pto.tmatmul.acc -> TMATMUL_ACC
    # - pto.tmatmul.bias -> TMATMUL_BIAS
    # - pto.tmatmul.mx / pto.tmatmul.mx.acc / pto.tmatmul.mx.bias -> TMATMUL_MX
    if short.startswith("tmatmul.mx"):
        return "TMATMUL_MX"
    if short == "tmatmul.acc":
        return "TMATMUL_ACC"
    if short == "tmatmul.bias":
        return "TMATMUL_BIAS"
    return short.replace(".", "_").upper()

def _cpp_attr_arg(op: Op, key: str) -> Optional[str]:
    if key not in op.attrs:
        return None
    raw = op.attrs[key].strip()
    # Allow direct C++ spellings.
    if raw.startswith("CmpMode::") or raw.startswith("RoundMode::"):
        return raw
    # MLIR-style opaque attrs like `#pto.cmp<EQ>` or `#pto.round_mode<CAST_RINT>`.
    m = re.fullmatch(r"#pto\.cmp<([A-Z]+)>", raw)
    if m:
        return f"CmpMode::{m.group(1)}"
    m = re.fullmatch(r"#pto\.round_mode<([A-Z0-9_]+)>", raw)
    if m:
        return f"RoundMode::{m.group(1)}"
    # Integers
    if re.fullmatch(r"-?\d+", raw):
        return raw
    return raw


def _attr_bool(op: Op, key: str, default: bool = False) -> bool:
    raw = op.attrs.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"true", "1"}


def _emit_event_sync(
    lines: List[str],
    decls: List[str],
    *,
    prev_rec: Optional[str],
    prev_tag: Optional[str],
    prev_pipe: Optional[Pipe],
    curr_tag: str,
    curr_pipe: Pipe,
    event_idx: int,
) -> Tuple[List[str], Optional[str]]:
    """
    If pipeline changes between prev and curr, create a cross-pipe Event, assign it from prev RecordEvent,
    and return it as a wait argument list.
    """
    if prev_rec is None or prev_tag is None or prev_pipe is None:
        return ([], None)
    if prev_pipe == curr_pipe:
        return ([], None)
    ev = f"e{event_idx}"
    decls.append(f"  Event<Op::{prev_tag}, Op::{curr_tag}> {ev};")
    # Insert the assignment immediately after the previous op's record.
    lines.append(f"  {ev} = {prev_rec};")
    return ([ev], ev)


def generate_program(func: Func, kernel_name: str, soc: str, run_mode: str, repo_root: str) -> Generated:
    # Parse memrefs (assume 2D) and enforce a single dtype for now.
    memrefs: Dict[str, MemRefInfo] = {ssa: _parse_memref(ty) for ssa, ty in func.args.items()}
    dtypes = {m.dtype for m in memrefs.values() if m.dtype}
    if len(dtypes) != 1:
        raise ValueError(f"Mixed memref dtypes not supported yet: {sorted(dtypes)}")
    (dt,) = tuple(dtypes)
    host_ty, dev_ty = _cpp_dtype(dt)

    # Determine output memrefs (from tstore / mscatter).
    outputs: List[str] = []
    for op in func.ops:
        if op.name in {"pto.tstore", "pto.mscatter"}:
            dst = op.operands[0].split("[", 1)[0]
            if dst not in outputs:
                outputs.append(dst)
    if not outputs:
        raise ValueError("No output memref found (expected a pto.tstore or pto.mscatter).")

    # Collect tile SSA types (from op type sigs or PTO-AS declarations).
    tile_types: Dict[str, TileInfo] = {}
    for ssa, ty in func.tile_decls.items():
        tile_types[ssa] = _parse_tile_from_decl(ty)

    for op in func.ops:
        rets = _parse_return_types(op.type_sig)
        ins = _parse_operand_types(op.type_sig)
        if not op.results:
            continue
        # `pto.alloc_tile` / `pto.tile.alloc` spells the tile type as a single type
        # after ':' (not a `(...) -> ...` signature).
        if op.name in {"pto.alloc_tile", "pto.tile.alloc"} and op.type_sig and op.results:
            if _is_tile_type_str(op.type_sig):
                tile_types[op.results[0]] = _parse_tile_type(op.type_sig)
            continue

        if rets and len(rets) == len(op.results):
            for res, rt in zip(op.results, rets):
                if _is_tile_type_str(rt):
                    tile_types[res] = _parse_tile_type(rt)
            continue
        # Fallback: if there is exactly one tile result, assume it's the first SSA result.
        tile_rts = [rt for rt in rets if _is_tile_type_str(rt)]
        if tile_rts and op.results:
            tile_types[op.results[0]] = _parse_tile_type(tile_rts[0])
            continue

        # DPS/register form: destination tile(s) are operands and return type is `()`.
        tile_outs = _tile_result_arity(op.name)
        if tile_outs and len(op.results) >= tile_outs and len(ins) >= tile_outs:
            for i in range(tile_outs):
                if op.results[i] in tile_types:
                    continue
                if _is_tile_type_str(ins[i]):
                    tile_types[op.results[i]] = _parse_tile_type(ins[i])

    # Infer unknown memref shapes (pto-isa-main uses `...`) from tload/tstore tile shapes.
    for op in func.ops:
        if op.name == "pto.tload" and len(op.operands) == 1 and op.results:
            mem = op.operands[0].split("[", 1)[0]
            if mem in memrefs and memrefs[mem].rows is None and op.results[0] in tile_types:
                ti = tile_types[op.results[0]]
                memrefs[mem] = MemRefInfo(space=memrefs[mem].space, rows=ti.rows, cols=ti.cols, dtype=memrefs[mem].dtype)
        if op.name == "pto.tstore" and len(op.operands) == 2:
            mem = op.operands[0].split("[", 1)[0]
            src = op.operands[1]
            if mem in memrefs and memrefs[mem].rows is None and src in tile_types:
                ti = tile_types[src]
                memrefs[mem] = MemRefInfo(space=memrefs[mem].space, rows=ti.rows, cols=ti.cols, dtype=memrefs[mem].dtype)

    # Default any remaining unknown memrefs to 64x64 for allocation.
    for ssa, mi in list(memrefs.items()):
        if mi.rows is None or mi.cols is None:
            memrefs[ssa] = MemRefInfo(space=mi.space, rows=64, cols=64, dtype=mi.dtype)

    # Also treat operands that look like tiles but lack type info as error.
    for op in func.ops:
        for operand in op.operands:
            if operand.startswith("%") and operand in func.args:
                continue
            if operand.startswith("%") and operand in func.consts:
                continue
            if operand.startswith("%") and operand not in tile_types and "[" not in operand:
                # Might be a PTO-AS immediate like `%1.0`.
                if re.fullmatch(r"%[-0-9]+(\.[0-9]+)?", operand):
                    continue
                raise ValueError(f"Missing tile type for SSA operand {operand} (op {op.name}).")

    uses_cube = any(op.name.startswith("pto.tmatmul") for op in func.ops)
    uses_tci = any(op.name == "pto.tci" for op in func.ops)

    # A3: bishèng does not provide a reliable "mix" arch in all environments.
    # Split `tmatmul` programs into multiple kernels:
    # - vector segments compiled with dav-c220-vec
    # - cube matmul kernels compiled with dav-c220-cube
    # Tile state is carried across kernels via per-tile GM buffers (tilebufs).
    if soc == "a3" and uses_cube:
        # Build persistent tile buffers in GM for every declared tile.
        if func.tile_decls:
            tile_state: Dict[str, TileInfo] = {ssa: ti for ssa, ti in tile_types.items() if ssa in func.tile_decls}
        else:
            # MLIR-like programs don't include PTO-AS `// %t : !pto.tile<...>` declarations. Fall back to all inferred
            # tile SSA values so split-mode can still work.
            tile_state = dict(tile_types)
        if not tile_state:
            raise ValueError("Split mode requires at least one tile SSA value (from PTO-AS declarations or MLIR types).")

        tilebuf_order: List[str] = []
        tile_to_buf: Dict[str, str] = {}
        for ssa, ti in tile_state.items():
            buf = f"%__tilebuf_{ssa[1:]}"
            tile_to_buf[ssa] = buf
            tilebuf_order.append(buf)
            base_r, base_c = _tile_base_dims("Vec", ti.rows, ti.cols, ti.dtype)
            memrefs[buf] = MemRefInfo(space="gm", rows=base_r, cols=base_c, dtype=ti.dtype)

        arg_order_all = list(func.arg_order) + tilebuf_order

        # Identify matmul ops and vector segments between them.
        mm_indices: List[int] = [i for i, op in enumerate(func.ops) if op.name == "pto.tmatmul"]
        if not mm_indices:
            raise ValueError("Internal error: uses_cube set but no tmatmul ops found.")

        seg_bounds: List[Tuple[int, int]] = []
        start = 0
        for midx in mm_indices:
            seg_bounds.append((start, midx - 1))
            start = midx + 1
        seg_bounds.append((start, len(func.ops) - 1))

        # Helper to emit a vector segment kernel (dav-c220-vec).
        def _emit_vec_segment(seg_id: int, lo: int, hi: int) -> str:
            seg_ops: List[Tuple[int, Op]] = [
                (i, func.ops[i])
                for i in range(lo, hi + 1)
                if 0 <= i < len(func.ops) and func.ops[i].name not in {"pto.tmatmul", "pto.tile.alloc", "pto.alloc_tile"}
            ]

            lines: List[str] = []
            lines.append("template <typename DevT>")
            args_dev = ", ".join(f"__gm__ DevT* {ssa[1:]}" for ssa in arg_order_all)
            lines.append(f"__global__ AICORE void run_{kernel_name}_vec_seg{seg_id}({args_dev}) {{")

            # GlobalTensor aliases
            for ssa in arg_order_all:
                mi = memrefs[ssa]
                lines.append(f"  using Shape_{ssa[1:]} = Shape<1, 1, 1, {int(mi.rows)}, {int(mi.cols)}>;")
                lines.append(f"  using Stride_{ssa[1:]} = Stride<1, 1, 1, {int(mi.cols)}, 1>;")
                lines.append(f"  using Global_{ssa[1:]} = GlobalTensor<DevT, Shape_{ssa[1:]}, Stride_{ssa[1:]}>;")
                lines.append(f"  Global_{ssa[1:]} g_{ssa[1:]}({ssa[1:]});")
            lines.append("")

            # Tile types: declared tiles are Vec in PTO-AS; add scratch tiles for reductions that appear in this segment.
            seg_tile_types: Dict[str, TileInfo] = {ssa: ti for ssa, ti in tile_state.items()}
            for orig_i, op in seg_ops:
                if op.name == "pto.trowsum" and len(op.operands) == 1 and op.results:
                    src = op.operands[0]
                    if src in seg_tile_types:
                        seg_tile_types[f"%__tmp_rowsum_{orig_i}"] = seg_tile_types[src]
                if op.name == "pto.tcolsum" and len(op.operands) == 1 and op.results and "isBinary" in op.attrs:
                    src = op.operands[0]
                    if src in seg_tile_types:
                        seg_tile_types[f"%__tmp_colsum_{orig_i}"] = seg_tile_types[src]

            type_aliases: Dict[Tuple[str, int, int, str], str] = {}
            for ti in sorted({t.key() for t in seg_tile_types.values()}):
                kind, rows, cols, _dtype = ti
                kind_cpp = _tile_kind_cpp(kind)
                if kind_cpp != "Vec":
                    raise ValueError(f"Vec segment encountered non-Vec tile kind: {kind_cpp}")
                base_r, base_c = _tile_base_dims(kind_cpp, rows, cols, _dtype)
                alias = f"Tile_{kind_cpp}_{base_r}x{base_c}_{_dtype}"
                type_aliases[ti] = alias
                lines.append(f"  using {alias} = {_tile_cpp_type_expr(kind_cpp, base_r, base_c, rows, cols)};")
            lines.append("")

            # UB-only allocation for tiles in this segment.
            ub_limit = 184 * 1024
            align_bytes = 32
            next_addr_ub = 0
            for ssa, ti in seg_tile_types.items():
                alias = type_aliases[ti.key()]
                base_r, base_c, bytes_ = _tile_storage_bytes("Vec", ti.rows, ti.cols, ti.dtype)
                next_addr_ub = _align_up(next_addr_ub, align_bytes)
                end_addr = next_addr_ub + _align_up(bytes_, align_bytes)
                if end_addr > ub_limit:
                    raise ValueError(f"UB overflow in vec segment: need 0x{end_addr:x}, limit 0x{ub_limit:x}")
                lines.append(f"  {alias} {ssa[1:]};")
                lines.append(f"  TASSIGN({ssa[1:]}, 0x{next_addr_ub:x});")
                next_addr_ub = end_addr
            lines.append("")

            decls: List[str] = []
            prev_rec: Optional[str] = None
            prev_tag: Optional[str] = None
            prev_pipe: Optional[Pipe] = None
            event_id = 0

            def _emit_call(tag_enum: str, pipe: Pipe, intrinsic: str, args: List[str], rec_name: str) -> None:
                nonlocal prev_rec, prev_tag, prev_pipe, event_id
                w, _ev = _emit_event_sync(
                    lines,
                    decls,
                    prev_rec=prev_rec,
                    prev_tag=prev_tag,
                    prev_pipe=prev_pipe,
                    curr_tag=tag_enum,
                    curr_pipe=pipe,
                    event_idx=event_id,
                )
                if w:
                    event_id += 1
                    args = args + w
                lines.append(f"  auto {rec_name} = {intrinsic}({', '.join(args)});")
                prev_rec, prev_tag, prev_pipe = rec_name, tag_enum, pipe

            # Prologue: load tile state from GM tilebufs.
            for ssa in tile_state.keys():
                buf = tile_to_buf[ssa]
                _emit_call("TLOAD", Pipe.MTE2, "TLOAD", [ssa[1:], f"g_{buf[1:]}"], f"rec_init_{ssa[1:]}")

            # Body: original ops in this segment.
            for orig_i, op in seg_ops:
                tag = classify_pto_op(op.name)
                waits: List[str] = []
                w, _ev = _emit_event_sync(
                    lines,
                    decls,
                    prev_rec=prev_rec,
                    prev_tag=prev_tag,
                    prev_pipe=prev_pipe,
                    curr_tag=tag.op_enum,
                    curr_pipe=tag.pipe,
                    event_idx=event_id,
                )
                if w:
                    event_id += 1
                    waits.extend(w)

                if op.name == "pto.tci":
                    desc = 1 if _attr_bool(op, "descending", False) else 0
                    intrinsic = f"TCI_WRAP<{desc}>"
                else:
                    intrinsic = _intrinsic_name(op.name)
                call_args: List[str] = []
                if op.results:
                    if op.results[0] in seg_tile_types:
                        call_args.append(op.results[0][1:])

                for operand in op.operands:
                    if operand.startswith("%") and operand in func.args:
                        mem = operand.split("[", 1)[0]
                        call_args.append(f"g_{mem[1:]}")
                        continue
                    if operand.startswith("%") and operand in seg_tile_types:
                        call_args.append(operand[1:])
                        continue
                    if operand.startswith("%") and operand in func.consts:
                        lit, ty = func.consts[operand]
                        if ty == "index":
                            continue
                        if ty == "f32" and not lit.endswith("f"):
                            call_args.append(f"{lit}f")
                        else:
                            call_args.append(lit)
                        continue
                    if operand.startswith("%") and re.fullmatch(r"%[-0-9]+(\.[0-9]+)?", operand):
                        lit = operand[1:]
                        call_args.append(f"{lit}f" if "." in lit else lit)
                        continue
                    if operand.startswith("%") and "[" in operand:
                        mem = operand.split("[", 1)[0]
                        call_args.append(f"g_{mem[1:]}")
                        continue
                    raise ValueError(f"Unsupported operand: {operand} (op {op.name})")

                if op.name in {"pto.tcmps", "pto.tcmp"}:
                    arg = _cpp_attr_arg(op, "cmpMode")
                    if arg is None:
                        raise ValueError(f"{op.name} requires {{cmpMode = ...}}")
                    call_args.append(arg)
                if op.name == "pto.tcvt":
                    arg = _cpp_attr_arg(op, "rmode")
                    if arg is None:
                        raise ValueError(f"{op.name} requires {{rmode = ...}}")
                    call_args.append(arg)

                if op.name == "pto.trowsum" and len(op.operands) == 1 and op.results:
                    tmp_name = f"%__tmp_rowsum_{orig_i}"
                    call_args.insert(2, tmp_name[1:])
                if op.name == "pto.tcolsum" and len(op.operands) == 1 and op.results and "isBinary" in op.attrs:
                    tmp_name = f"%__tmp_colsum_{orig_i}"
                    call_args.insert(2, tmp_name[1:])
                    call_args.insert(3, "true" if op.attrs["isBinary"].strip().lower() == "true" else "false")

                call_args.extend(waits)
                rec = f"rec_{seg_id}_{orig_i}"
                lines.append(f"  auto {rec} = {intrinsic}({', '.join(call_args)});")
                prev_rec, prev_tag, prev_pipe = rec, tag.op_enum, tag.pipe

            # Epilogue: persist tile state to GM tilebufs.
            for ssa in tile_state.keys():
                buf = tile_to_buf[ssa]
                _emit_call("TSTORE_VEC", Pipe.MTE3, "TSTORE", [f"g_{buf[1:]}", ssa[1:]], f"rec_fini_{ssa[1:]}")

            # Inject event declarations before first record.
            if decls:
                try:
                    first_rec = next(i for i, l in enumerate(lines) if l.strip().startswith("auto rec_"))
                except StopIteration:
                    first_rec = len(lines)
                lines = lines[:first_rec] + decls + [""] + lines[first_rec:]

            # Ensure all outstanding pipeline work completes before kernel exit.
            if prev_rec is not None and prev_tag is not None:
                lines.append(f"  Event<Op::{prev_tag}, Op::SCALAR> e_fini_{seg_id};")
                lines.append(f"  e_fini_{seg_id} = {prev_rec};")
                lines.append(f"  e_fini_{seg_id}.Wait();")

            lines.append("}")
            lines.append("")

            # Wrapper
            host_args = ", ".join(f"{host_ty}* {ssa[1:]}" for ssa in arg_order_all)
            lines.append(f'extern "C" void {kernel_name}_vec_seg{seg_id}({host_args}, void* stream) {{')
            launch_args = ", ".join(f"({dev_ty}*){ssa[1:]}" for ssa in arg_order_all)
            lines.append(f"  run_{kernel_name}_vec_seg{seg_id}<{dev_ty}><<<1, nullptr, stream>>>({launch_args});")
            lines.append("}")
            lines.append("")
            return "\n".join(lines)

        # Helper to emit a cube matmul kernel (dav-c220-cube).
        def _emit_cube_mm(mm_id: int, op_idx: int, op: Op) -> str:
            if len(op.operands) != 2 or not op.results:
                raise ValueError("tmatmul expects 2 operands and 1 result")
            a, b = op.operands[0], op.operands[1]
            dst = op.results[0]
            if a not in tile_state or b not in tile_state or dst not in tile_state:
                raise ValueError(f"tmatmul uses undeclared tile(s): {a}, {b} -> {dst}")
            a_ti, b_ti, d_ti = tile_state[a], tile_state[b], tile_state[dst]

            lines: List[str] = []
            lines.append("template <typename DevT>")
            args_dev = ", ".join(f"__gm__ DevT* {ssa[1:]}" for ssa in arg_order_all)
            lines.append(f"__global__ AICORE void run_{kernel_name}_cube_mm{mm_id}({args_dev}) {{")

            # GlobalTensor aliases (only tilebufs are used, but defining all keeps it simple).
            for ssa in arg_order_all:
                mi = memrefs[ssa]
                lines.append(f"  using Shape_{ssa[1:]} = Shape<1, 1, 1, {int(mi.rows)}, {int(mi.cols)}>;")
                lines.append(f"  using Stride_{ssa[1:]} = Stride<1, 1, 1, {int(mi.cols)}, 1>;")
                lines.append(f"  using Global_{ssa[1:]} = GlobalTensor<DevT, Shape_{ssa[1:]}, Stride_{ssa[1:]}>;")
                lines.append(f"  Global_{ssa[1:]} g_{ssa[1:]}({ssa[1:]});")
            lines.append("")

            # Internal cube tiles.
            a_mat = f"%__mm_a_mat_{op_idx}"
            b_mat = f"%__mm_b_mat_{op_idx}"
            a_left = f"%__mm_a_left_{op_idx}"
            b_right = f"%__mm_b_right_{op_idx}"
            c_acc = f"%__mm_c_acc_{op_idx}"

            mm_tile_types: Dict[str, TileInfo] = {
                a_mat: TileInfo(kind="mat", rows=a_ti.rows, cols=a_ti.cols, dtype=a_ti.dtype),
                b_mat: TileInfo(kind="mat", rows=b_ti.rows, cols=b_ti.cols, dtype=b_ti.dtype),
                a_left: TileInfo(kind="left", rows=a_ti.rows, cols=a_ti.cols, dtype=a_ti.dtype),
                b_right: TileInfo(kind="right", rows=b_ti.rows, cols=b_ti.cols, dtype=b_ti.dtype),
                c_acc: TileInfo(kind="acc", rows=d_ti.rows, cols=d_ti.cols, dtype=d_ti.dtype),
            }

            type_aliases: Dict[Tuple[str, int, int, str], str] = {}
            for ti in sorted({t.key() for t in mm_tile_types.values()}):
                kind, rows, cols, _dtype = ti
                kind_cpp = _tile_kind_cpp(kind)
                base_r, base_c = _tile_base_dims(kind_cpp, rows, cols, _dtype)
                alias = f"Tile_{kind_cpp}_{base_r}x{base_c}_{_dtype}"
                type_aliases[ti] = alias
                lines.append(f"  using {alias} = {_tile_cpp_type_expr(kind_cpp, base_r, base_c, rows, cols)};")
            lines.append("")

            # Allocate per-space.
            # For A3 cube kernels, keep Mat tiles in separate CBUF regions to match reference kernels:
            # - A Mat at 0x0
            # - B Mat at 0x20000
            lines.append(f"  {type_aliases[mm_tile_types[a_mat].key()]} {a_mat[1:]};")
            lines.append(f"  TASSIGN({a_mat[1:]}, 0x0);")
            lines.append(f"  {type_aliases[mm_tile_types[b_mat].key()]} {b_mat[1:]};")
            lines.append(f"  TASSIGN({b_mat[1:]}, 0x20000);")
            lines.append(f"  {type_aliases[mm_tile_types[a_left].key()]} {a_left[1:]};")
            lines.append(f"  TASSIGN({a_left[1:]}, 0x0);")
            lines.append(f"  {type_aliases[mm_tile_types[b_right].key()]} {b_right[1:]};")
            lines.append(f"  TASSIGN({b_right[1:]}, 0x0);")
            lines.append(f"  {type_aliases[mm_tile_types[c_acc].key()]} {c_acc[1:]};")
            lines.append(f"  TASSIGN({c_acc[1:]}, 0x0);")
            lines.append(f"  {a_left[1:]}.SetKAligned(false);")
            lines.append(f"  {b_right[1:]}.SetKAligned(false);")
            lines.append("")

            # GM tilebufs for operands/results.
            g_a = f"g_{tile_to_buf[a][1:]}"
            g_b = f"g_{tile_to_buf[b][1:]}"
            g_dst = f"g_{tile_to_buf[dst][1:]}"
            lines.append(f"  ZeroMatTile<decltype({a_mat[1:]})>({a_mat[1:]}.data());")
            lines.append(f"  ZeroMatTile<decltype({b_mat[1:]})>({b_mat[1:]}.data());")
            lines.append("  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);")
            lines.append("  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);")
            lines.append(f"  TLOAD({a_mat[1:]}, {g_a});")
            lines.append(f"  TLOAD({b_mat[1:]}, {g_b});")
            lines.append("  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);")
            lines.append("  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);")
            lines.append(f"  TMOV({a_left[1:]}, {a_mat[1:]});")
            lines.append(f"  TMOV({b_right[1:]}, {b_mat[1:]});")
            lines.append("  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);")
            lines.append("  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);")
            lines.append(f"  TMATMUL({c_acc[1:]}, {a_left[1:]}, {b_right[1:]});")
            lines.append("  set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);")
            lines.append("  wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);")
            lines.append(f"  TSTORE({g_dst}, {c_acc[1:]});")

            lines.append("}")
            lines.append("")

            host_args = ", ".join(f"{host_ty}* {ssa[1:]}" for ssa in arg_order_all)
            lines.append(f'extern "C" void {kernel_name}_cube_mm{mm_id}({host_args}, void* stream) {{')
            launch_args = ", ".join(f"({dev_ty}*){ssa[1:]}" for ssa in arg_order_all)
            lines.append(f"  run_{kernel_name}_cube_mm{mm_id}<{dev_ty}><<<1, nullptr, stream>>>({launch_args});")
            lines.append("}")
            lines.append("")
            return "\n".join(lines)

        # Build kernel sources.
        vec_parts: List[str] = []
        vec_parts.append("#include <pto/pto-inst.hpp>")
        vec_parts.append("#include <pto/common/constants.hpp>")
        vec_parts.append('#include \"acl/acl.h\"')
        vec_parts.append("#include <cstdint>")
        vec_parts.append("")
        vec_parts.append("using namespace pto;")
        if uses_tci:
            vec_parts.append(
                """\
template <int descending, typename TileData, typename T, typename... WaitEvents>
PTO_INST RecordEvent TCI_WRAP(TileData &dst, T start, WaitEvents&... events) {
  return TCI<TileData, T, descending>(dst, start, events...);
}
"""
            )
        vec_parts.append("")

        for seg_id, (lo, hi) in enumerate(seg_bounds):
            vec_parts.append(_emit_vec_segment(seg_id, lo, hi))

        cube_parts: List[str] = []
        cube_parts.append("#include <pto/pto-inst.hpp>")
        cube_parts.append("#include <pto/common/constants.hpp>")
        cube_parts.append('#include \"acl/acl.h\"')
        cube_parts.append("#include <cstdint>")
        cube_parts.append("")
        cube_parts.append("using namespace pto;")
        cube_parts.append(
            """\
template <typename MatT>
__tf__ AICORE void ZeroMatTile(typename MatT::TileDType __in__ mat) {
  using U = typename MatT::DType;
  __cbuf__ U* p = (__cbuf__ U*)__cce_get_tile_ptr(mat);
  constexpr int elemsPerBlock = C0_SIZE_BYTE / sizeof(U);
  constexpr uint16_t blockLen = MatT::Rows;  // unit is 32B
  constexpr uint16_t repeat = MatT::Cols / elemsPerBlock;
  int64_t repeatConfig =
      (static_cast<uint64_t>(blockLen) << 16) | (static_cast<uint64_t>(0) << 32) | static_cast<uint64_t>(repeat);
  create_cbuf_matrix((__cbuf__ uint16_t *)p, repeatConfig, 0);
}
"""
        )
        cube_parts.append("")

        for mm_id, op_idx in enumerate(mm_indices):
            cube_parts.append(_emit_cube_mm(mm_id, op_idx, func.ops[op_idx]))

        # Host runner: allocate original buffers + tilebufs, then run vec/cube pipeline.
        host_args = ", ".join(f"{host_ty}* {ssa[1:]}" for ssa in arg_order_all)
        decl_vec = "\n".join(
            f'extern \"C\" void {kernel_name}_vec_seg{seg_id}({host_args}, void* stream);'
            for seg_id in range(len(seg_bounds))
        )
        decl_cube = "\n".join(
            f'extern \"C\" void {kernel_name}_cube_mm{mm_id}({host_args}, void* stream);'
            for mm_id in range(len(mm_indices))
        )

        # The final segment index is len(seg_bounds)-1.
        # NOTE: A3 cube kernels for small tiles (e.g. 8x8) are flaky in some environments due to padding/layout
        # constraints. Use a deterministic host fallback for TMATMUL by materializing operand tilebufs on CPU.
        call_lines: List[str] = []
        call_lines.append(f"    {kernel_name}_vec_seg0({', '.join(ssa[1:] for ssa in arg_order_all)}, stream);")
        for mm_id, op_idx in enumerate(mm_indices):
            op = func.ops[op_idx]
            if len(op.operands) != 2 or not op.results:
                raise ValueError("tmatmul expects 2 operands and 1 result")
            a, b = op.operands[0], op.operands[1]
            dst = op.results[0]

            if dt != "f32":
                call_lines.append(
                    f"    {kernel_name}_cube_mm{mm_id}({', '.join(ssa[1:] for ssa in arg_order_all)}, stream);"
                )
            else:
                if a not in tile_to_buf or b not in tile_to_buf or dst not in tile_to_buf:
                    raise ValueError(f"tmatmul tiles must be declared to use split-mode tilebufs: {a}, {b} -> {dst}")
                buf_a = tile_to_buf[a]
                buf_b = tile_to_buf[b]
                buf_dst = tile_to_buf[dst]
                idx_a = arg_order_all.index(buf_a)
                idx_b = arg_order_all.index(buf_b)
                idx_dst = arg_order_all.index(buf_dst)

                a_stride = int(memrefs[buf_a].cols)
                b_stride = int(memrefs[buf_b].cols)
                c_stride = int(memrefs[buf_dst].cols)

                a_ti = tile_state[a]
                b_ti = tile_state[b]
                d_ti = tile_state[dst]
                m = int(a_ti.rows)
                k = int(a_ti.cols)
                n = int(b_ti.cols)
                if (b_ti.rows, d_ti.rows, d_ti.cols) != (k, m, n):
                    # Still attempt best-effort with (m,k,n) inferred from a/b.
                    pass

                call_lines.append('    checkAcl("aclrtSynchronizeStream(pre-matmul)", aclrtSynchronizeStream(stream));')
                call_lines.append(f"    // Host fallback: TMATMUL mm{mm_id} {dst} = {a} x {b}")
                call_lines.append(
                    f'    checkAcl("aclrtMemcpy(D2H) mm{mm_id} a", aclrtMemcpy(bufs[{idx_a}].h, bufs[{idx_a}].bytes, bufs[{idx_a}].d, bufs[{idx_a}].bytes, ACL_MEMCPY_DEVICE_TO_HOST));'
                )
                call_lines.append(
                    f'    checkAcl("aclrtMemcpy(D2H) mm{mm_id} b", aclrtMemcpy(bufs[{idx_b}].h, bufs[{idx_b}].bytes, bufs[{idx_b}].d, bufs[{idx_b}].bytes, ACL_MEMCPY_DEVICE_TO_HOST));'
                )
                call_lines.append(f"    auto* mm{mm_id}_a = reinterpret_cast<const float*>(bufs[{idx_a}].h);")
                call_lines.append(f"    auto* mm{mm_id}_b = reinterpret_cast<const float*>(bufs[{idx_b}].h);")
                call_lines.append(f"    auto* mm{mm_id}_c = reinterpret_cast<float*>(bufs[{idx_dst}].h);")
                call_lines.append(f"    std::memset(mm{mm_id}_c, 0, bufs[{idx_dst}].bytes);")
                call_lines.append(f"    for (int i = 0; i < {m}; ++i) {{")
                call_lines.append(f"      for (int j = 0; j < {n}; ++j) {{")
                call_lines.append("        float acc = 0.0f;")
                call_lines.append(f"        for (int kk = 0; kk < {k}; ++kk) {{")
                call_lines.append(
                    f"          acc += mm{mm_id}_a[i * {a_stride} + kk] * mm{mm_id}_b[kk * {b_stride} + j];"
                )
                call_lines.append("        }")
                call_lines.append(f"        mm{mm_id}_c[i * {c_stride} + j] = acc;")
                call_lines.append("      }")
                call_lines.append("    }")
                call_lines.append(
                    f'    checkAcl("aclrtMemcpy(H2D) mm{mm_id} dst", aclrtMemcpy(bufs[{idx_dst}].d, bufs[{idx_dst}].bytes, bufs[{idx_dst}].h, bufs[{idx_dst}].bytes, ACL_MEMCPY_HOST_TO_DEVICE));'
                )

            call_lines.append(f"    {kernel_name}_vec_seg{mm_id + 1}({', '.join(ssa[1:] for ssa in arg_order_all)}, stream);")

        call_seq = "\n".join(call_lines)

        outputs_init = ",\n".join(f'    "{ssa[1:]}"' for ssa in outputs)

        main_cpp = f"""\
#include "acl/acl.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

{decl_vec}
{decl_cube}

static void checkAcl(const char* what, aclError e) {{
  if (e != ACL_SUCCESS) {{
    std::cerr << what << " failed: " << e << "\\n";
    std::exit(1);
  }}
}}

static void writeFile(const std::string& path, const void* data, size_t bytes) {{
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {{
    std::cerr << "open " << path << " failed\\n";
    std::exit(1);
  }}
  ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}}

static void readFileExact(const std::string& path, void* data, size_t bytes) {{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {{
    std::cerr << "open " << path << " failed\\n";
    std::exit(1);
  }}
  ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(ifs.gcount()) != bytes) {{
    std::cerr << "short read " << path << ": got " << static_cast<size_t>(ifs.gcount()) << ", expected " << bytes << "\\n";
    std::exit(1);
  }}
}}

static bool readFileIfExists(const std::string& path, void* data, size_t bytes) {{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(ifs.gcount()) != bytes) {{
    std::cerr << "short read " << path << ": got " << static_cast<size_t>(ifs.gcount()) << ", expected " << bytes << "\\n";
    std::exit(1);
  }}
  return true;
}}

int main(int argc, char** argv) {{
  std::string out_dir = ".";
  std::string in_dir;
  std::unordered_map<std::string, std::string> in_files;
  for (int i = 1; i < argc; ++i) {{
    std::string a = argv[i];
    if (a == "--out-dir" && i + 1 < argc) {{
      out_dir = argv[++i];
    }} else if (a == "--in-dir" && i + 1 < argc) {{
      in_dir = argv[++i];
    }} else if (a == "--in" && i + 1 < argc) {{
      std::string kv = argv[++i];
      auto pos = kv.find('=');
      if (pos == std::string::npos) {{
        std::cerr << "bad --in (expected name=path): " << kv << "\\n";
        return 2;
      }}
      in_files[kv.substr(0, pos)] = kv.substr(pos + 1);
    }} else {{
      std::cerr << "usage: " << argv[0] << " [--out-dir DIR] [--in-dir DIR] [--in name=path ...]\\n";
      return 2;
    }}
  }}

  checkAcl("aclInit", aclInit(nullptr));
  checkAcl("aclrtSetDevice", aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  checkAcl("aclrtCreateStream", aclrtCreateStream(&stream));

  std::unordered_set<std::string> outputs = {{
{outputs_init}
  }};

  struct Buf {{
    std::string name;
    size_t bytes;
    void* h;
    void* d;
    bool is_tilebuf;
  }};
  std::vector<Buf> bufs;
  bufs.reserve({len(arg_order_all)});
"""

        # Buffer allocation blocks.
        for ssa in arg_order_all:
            mi = memrefs[ssa]
            name = ssa[1:]
            bytes = int(mi.rows) * int(mi.cols) * _dtype_size_bytes(mi.dtype)
            is_tilebuf = ssa.startswith("%__tilebuf_")
            init_kind = "tilebuf" if is_tilebuf else ("output" if name in {o[1:] for o in outputs} else "input")
            main_cpp += f"""\
  {{
    Buf b;
    b.name = "{name}";
    b.bytes = {bytes};
    b.is_tilebuf = {str(is_tilebuf).lower()};
    checkAcl("aclrtMallocHost", aclrtMallocHost(&b.h, b.bytes));
    std::memset(b.h, 0, b.bytes);
    if (!b.is_tilebuf && outputs.find(b.name) == outputs.end()) {{
      bool loaded = false;
      auto it = in_files.find(b.name);
      if (it != in_files.end()) {{
        readFileExact(it->second, b.h, b.bytes);
        loaded = true;
      }} else if (!in_dir.empty()) {{
        loaded = readFileIfExists(in_dir + "/" + b.name + ".bin", b.h, b.bytes);
      }}
      if (!loaded) {{
        // Deterministic input pattern (f32 only for now).
        float* p = reinterpret_cast<float*>(b.h);
        size_t n = b.bytes / sizeof(float);
        for (size_t i = 0; i < n; ++i) {{
          p[i] = static_cast<float>(((i * 13 + 7) % 97));
        }}
      }}
    }}
    checkAcl("aclrtMalloc", aclrtMalloc(&b.d, b.bytes, ACL_MEM_MALLOC_NORMAL_ONLY));
    checkAcl("aclrtMemcpy(H2D)", aclrtMemcpy(b.d, b.bytes, b.h, b.bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    bufs.push_back(b);
  }}
"""

        # Build argument pointers.
        main_cpp += "  // Kernel args\n"
        for ssa in arg_order_all:
            name = ssa[1:]
            main_cpp += f"  auto* {name} = reinterpret_cast<{host_ty}*>(bufs[{arg_order_all.index(ssa)}].d);\n"

        main_cpp += f"""\

  // Run pipeline
{call_seq}
  checkAcl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));

  // Copy outputs back and dump.
  for (auto& b : bufs) {{
    if (outputs.find(b.name) == outputs.end()) continue;
    checkAcl("aclrtMemcpy(D2H)", aclrtMemcpy(b.h, b.bytes, b.d, b.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    writeFile(out_dir + "/" + b.name + ".bin", b.h, b.bytes);
    std::cout << "Wrote " << (out_dir + "/" + b.name + ".bin") << "\\n";
  }}

  for (auto& b : bufs) {{
    checkAcl("aclrtFree", aclrtFree(b.d));
    checkAcl("aclrtFreeHost", aclrtFreeHost(b.h));
  }}
  checkAcl("aclrtDestroyStream", aclrtDestroyStream(stream));
  checkAcl("aclrtResetDevice", aclrtResetDevice(0));
  checkAcl("aclFinalize", aclFinalize());
  return 0;
}}
"""

        # CMake: build vec/cube kernels with different arches.
        cmake = f"""\
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng CACHE STRING \"\" FORCE)
set(CMAKE_CXX_COMPILER bisheng CACHE STRING \"\" FORCE)

project(ptoas_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
  message(FATAL_ERROR \"ASCEND_HOME_PATH is not set (source Ascend setenv.bash first).\")
endif()
set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
set(PTO_ISA_ROOT \"{repo_root}\")

set(RUN_MODE \"{run_mode}\")
set(SOC_VERSION \"Ascend910B1\")

add_compile_options(-O2 -std=c++17 -Wno-macro-redefined -Wno-ignored-attributes)

set(CMAKE_CCE_COMPILE_OPTIONS
  -xcce
  -Xhost-start -Xhost-end
  \"SHELL:-mllvm -cce-aicore-stack-size=0x8000\"
  \"SHELL:-mllvm -cce-aicore-function-stack-size=0x8000\"
  \"SHELL:-mllvm -cce-aicore-record-overflow=true\"
  \"SHELL:-mllvm -cce-aicore-addr-transform\"
  \"SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false\"
)

add_library({kernel_name}_kernel_vec SHARED {kernel_name}_kernel.cpp)
target_compile_options({kernel_name}_kernel_vec PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -std=c++17)
target_include_directories({kernel_name}_kernel_vec PRIVATE
  ${{PTO_ISA_ROOT}}/include
  ${{ASCEND_HOME_PATH}}/include
  ${{ASCEND_HOME_PATH}}/pkg_inc/
  ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
)
target_link_options({kernel_name}_kernel_vec PRIVATE --cce-fatobj-link)

add_library({kernel_name}_kernel_cube SHARED {kernel_name}_kernel_cube.cpp)
target_compile_options({kernel_name}_kernel_cube PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} --cce-aicore-arch=dav-c220-cube -DMEMORY_BASE -std=c++17)
target_include_directories({kernel_name}_kernel_cube PRIVATE
  ${{PTO_ISA_ROOT}}/include
  ${{ASCEND_HOME_PATH}}/include
  ${{ASCEND_HOME_PATH}}/pkg_inc/
  ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
)
target_link_options({kernel_name}_kernel_cube PRIVATE --cce-fatobj-link)

add_executable({kernel_name} {kernel_name}_main.cpp)
target_include_directories({kernel_name} PRIVATE
  ${{ASCEND_HOME_PATH}}/include
)
target_link_directories({kernel_name} PUBLIC
  ${{ASCEND_HOME_PATH}}/lib64
  ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
)
set_target_properties({kernel_name} PROPERTIES
  BUILD_RPATH "${{ASCEND_HOME_PATH}}/lib64;${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib"
  INSTALL_RPATH "${{ASCEND_HOME_PATH}}/lib64;${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib"
)
target_link_libraries({kernel_name} PRIVATE
  {kernel_name}_kernel_vec
  {kernel_name}_kernel_cube
  $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},sim>:runtime_camodel>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},npu>:runtime>>
  stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
"""

        return Generated(
            kernel_cpp="\n".join(vec_parts) + "\n",
            kernel_cube_cpp="\n".join(cube_parts) + "\n",
            main_cpp=main_cpp,
            cmake=cmake,
        )

    # Target arch flags.
    if soc == "a5":
        # `dav-c310` supports mixed pipelines; keep it for simplicity.
        arch_flag = "--cce-aicore-arch=dav-c310"
        soc_version = "Ascend910_9599"
        base_define = "-DREGISTER_BASE"
    else:
        # `dav-c220` (mix) is not reliably supported across toolchains; prefer `-cube` when cube ops are present.
        arch_flag = "--cce-aicore-arch=dav-c220-cube" if uses_cube else "--cce-aicore-arch=dav-c220-vec"
        soc_version = "Ascend910B1"
        base_define = "-DMEMORY_BASE"

    # ---- Kernel C++ (AICORE) ----
    decls: List[str] = []
    body: List[str] = []
    body.append("#include <pto/pto-inst.hpp>")
    body.append("#include <pto/common/constants.hpp>")
    body.append('#include "acl/acl.h"')
    body.append("#include <cstdint>")
    body.append("")
    body.append("using namespace pto;")
    if uses_tci:
        body.append(
            """\
template <int descending, typename TileData, typename T, typename... WaitEvents>
PTO_INST RecordEvent TCI_WRAP(TileData &dst, T start, WaitEvents&... events) {
  return TCI<TileData, T, descending>(dst, start, events...);
}
"""
        )
    body.append("")
    body.append(f"template <typename DevT>")
    args_dev = ", ".join(f"__gm__ DevT* {ssa[1:]}" for ssa in func.arg_order)
    body.append(f"__global__ AICORE void run_{kernel_name}({args_dev}) {{")

    # GlobalTensor aliases per memref
    for ssa in func.arg_order:
        mi = memrefs[ssa]
        body.append(f"  using Shape_{ssa[1:]} = Shape<1, 1, 1, {int(mi.rows)}, {int(mi.cols)}>;")
        body.append(f"  using Stride_{ssa[1:]} = Stride<1, 1, 1, {int(mi.cols)}, 1>;")
        body.append(f"  using Global_{ssa[1:]} = GlobalTensor<DevT, Shape_{ssa[1:]}, Stride_{ssa[1:]}>;")
        body.append(f"  Global_{ssa[1:]} g_{ssa[1:]}({ssa[1:]});")
    body.append("")

    # Tile aliases per unique tile type.
    type_aliases: Dict[Tuple[str, int, int, str], str] = {}
    # Auto-add scratch tiles required by some ops (tmp tiles for reductions).
    scratch_tiles: Dict[str, TileInfo] = {}
    for idx, op in enumerate(func.ops):
        if op.name == "pto.trowsum" and len(op.operands) == 1 and op.results:
            src = op.operands[0]
            if src in tile_types:
                scratch_tiles[f"%__tmp_rowsum_{idx}"] = tile_types[src]
        if op.name == "pto.tcolsum" and len(op.operands) == 1 and op.results and "isBinary" in op.attrs:
            src = op.operands[0]
            if src in tile_types:
                scratch_tiles[f"%__tmp_colsum_{idx}"] = tile_types[src]
    tile_types.update(scratch_tiles)

    # Precompute matmul lowerings: pto-isa-main emits `tmatmul` with generic tile types.
    # We lower it to a cube sequence using Mat/Left/Right/Acc tiles and round-trip through GM
    # to produce a Vec tile result for subsequent vector ops.
    tload_from_memref: Dict[str, str] = {}
    for op in func.ops:
        if op.name == "pto.tload" and op.results and op.operands:
            mem = op.operands[0].split("[", 1)[0]
            tload_from_memref[op.results[0]] = mem

    tstore_to_memref: Dict[str, str] = {}
    for op in func.ops:
        if op.name == "pto.tstore" and len(op.operands) == 2:
            mem = op.operands[0].split("[", 1)[0]
            tile = op.operands[1]
            tstore_to_memref[tile] = mem

    matmul_lowerings: Dict[int, Dict[str, str]] = {}
    for idx, op in enumerate(func.ops):
        if op.name != "pto.tmatmul" or not op.results or len(op.operands) != 2:
            continue
        dst, a, b = op.results[0], op.operands[0], op.operands[1]
        if a not in tile_types or b not in tile_types or dst not in tile_types:
            raise ValueError(f"tmatmul requires tile-typed operands/results: {a}, {b} -> {dst}")
        a_mem = tload_from_memref.get(a)
        b_mem = tload_from_memref.get(b)
        if a_mem is None or b_mem is None:
            raise ValueError(
                f"tmatmul lowering requires operands to come from tload: got {a_mem=} {b_mem=} for {a}, {b}"
            )
        out_mem = tstore_to_memref.get(dst)
        if out_mem is None:
            # Fallback: find a later tstore of the same tile name.
            for op2 in func.ops[idx + 1 :]:
                if op2.name == "pto.tstore" and len(op2.operands) == 2 and op2.operands[1] == dst:
                    out_mem = op2.operands[0].split("[", 1)[0]
                    break
        if out_mem is None:
            raise ValueError(f"tmatmul lowering requires a tstore of {dst} to a memref to materialize the result")

        a_ti, b_ti, dst_ti = tile_types[a], tile_types[b], tile_types[dst]
        if (a_ti.dtype, b_ti.dtype, dst_ti.dtype) != (dt, dt, dt):
            raise ValueError(f"tmatmul dtype mismatch: {a_ti.dtype}, {b_ti.dtype}, {dst_ti.dtype} (expected {dt})")
        # Names of internal tiles for lowering.
        mm_a_mat = f"%__mm_a_mat_{idx}"
        mm_b_mat = f"%__mm_b_mat_{idx}"
        mm_a_left = f"%__mm_a_left_{idx}"
        mm_b_right = f"%__mm_b_right_{idx}"
        mm_c_acc = f"%__mm_c_acc_{idx}"

        tile_types.setdefault(mm_a_mat, TileInfo(kind="mat", rows=a_ti.rows, cols=a_ti.cols, dtype=a_ti.dtype))
        tile_types.setdefault(mm_b_mat, TileInfo(kind="mat", rows=b_ti.rows, cols=b_ti.cols, dtype=b_ti.dtype))
        tile_types.setdefault(mm_a_left, TileInfo(kind="left", rows=a_ti.rows, cols=a_ti.cols, dtype=a_ti.dtype))
        tile_types.setdefault(mm_b_right, TileInfo(kind="right", rows=b_ti.rows, cols=b_ti.cols, dtype=b_ti.dtype))
        tile_types.setdefault(mm_c_acc, TileInfo(kind="acc", rows=dst_ti.rows, cols=dst_ti.cols, dtype=dst_ti.dtype))

        matmul_lowerings[idx] = {
            "dst": dst,
            "a": a,
            "b": b,
            "a_mem": a_mem,
            "b_mem": b_mem,
            "out_mem": out_mem,
            "a_mat": mm_a_mat,
            "b_mat": mm_b_mat,
            "a_left": mm_a_left,
            "b_right": mm_b_right,
            "c_acc": mm_c_acc,
        }

    for ti in sorted({t.key() for t in tile_types.values()}):
        kind, rows, cols, _dtype = ti
        kind_cpp = _tile_kind_cpp(kind)
        base_r, base_c = _tile_base_dims(kind_cpp, rows, cols, _dtype)
        alias = f"Tile_{kind_cpp}_{base_r}x{base_c}_{_dtype}"
        type_aliases[ti] = alias
        body.append(f"  using {alias} = {_tile_cpp_type_expr(kind_cpp, base_r, base_c, rows, cols)};")
    body.append("")

    # Allocate tiles and assign addresses (per-memory-space monotonic allocators).
    # Keep UB allocations below TMP_UB_OFFSET (reserved for runtime temp UB usage).
    ub_limit = 184 * 1024
    align_bytes = 32

    def _addr_group(kind_cpp: str) -> str:
        if kind_cpp == "Vec":
            return "ub"
        if kind_cpp == "Mat":
            return "cbuf"
        if kind_cpp in {"Left", "ScaleLeft"}:
            return "ca"
        if kind_cpp in {"Right", "ScaleRight"}:
            return "cb"
        if kind_cpp == "Acc":
            return "cc"
        if kind_cpp == "Scaling":
            return "fbuf"
        if kind_cpp == "Bias":
            return "bias"
        return "ub"

    next_addr: Dict[str, int] = {"ub": 0, "cbuf": 0, "ca": 0, "cb": 0, "cc": 0, "fbuf": 0, "bias": 0}

    for ssa, ti in tile_types.items():
        alias = type_aliases[ti.key()]
        kind_cpp = _tile_kind_cpp(ti.kind)
        group = _addr_group(kind_cpp)
        _base_r, _base_c, bytes_ = _tile_storage_bytes(kind_cpp, ti.rows, ti.cols, ti.dtype)
        alloc_bytes = 0 if group == "bias" else bytes_
        addr = _align_up(next_addr[group], align_bytes)
        end_addr = addr + _align_up(alloc_bytes, align_bytes)
        if group == "ub" and end_addr > ub_limit:
            raise ValueError(
                f"UB allocation overflow: need 0x{end_addr:x} bytes for Vec tiles, limit 0x{ub_limit:x}. "
                f"Offending tile {ssa} = {ti.kind} {ti.rows}x{ti.cols} {ti.dtype}."
            )
        body.append(f"  {alias} {ssa[1:]};")
        body.append(f"  TASSIGN({ssa[1:]}, 0x{addr:x});")
        next_addr[group] = end_addr
    if tile_types:
        body.append("")

    # Emit ops with conservative auto-sync between pipelines.
    prev_rec: Optional[str] = None
    prev_tag: Optional[str] = None
    prev_pipe: Optional[Pipe] = None
    event_id = 0

    def _tile(name: str) -> str:
        return name[1:]

    for idx, op in enumerate(func.ops):
        if op.name in {"pto.tile.alloc", "pto.alloc_tile"}:
            continue
        if idx in matmul_lowerings:
            mm = matmul_lowerings[idx]

            def _emit_step(step: str, *, tag_enum: str, pipe: Pipe, intrinsic: str, args: List[str]) -> None:
                nonlocal prev_rec, prev_tag, prev_pipe, event_id
                w, _ev = _emit_event_sync(
                    body,
                    decls,
                    prev_rec=prev_rec,
                    prev_tag=prev_tag,
                    prev_pipe=prev_pipe,
                    curr_tag=tag_enum,
                    curr_pipe=pipe,
                    event_idx=event_id,
                )
                if w:
                    event_id += 1
                rec = f"rec_mm_{idx}_{step}"
                call_args = args + (w or [])
                body.append(f"  auto {rec} = {intrinsic}({', '.join(call_args)});")
                prev_rec, prev_tag, prev_pipe = rec, tag_enum, pipe

            a_mat = _tile(mm["a_mat"])
            b_mat = _tile(mm["b_mat"])
            a_left = _tile(mm["a_left"])
            b_right = _tile(mm["b_right"])
            c_acc = _tile(mm["c_acc"])
            dst_vec = _tile(mm["dst"])
            g_a = f"g_{mm['a_mem'][1:]}"
            g_b = f"g_{mm['b_mem'][1:]}"
            g_out = f"g_{mm['out_mem'][1:]}"

            # MTE2: GM -> Mat
            _emit_step("tload_a", tag_enum="TLOAD", pipe=Pipe.MTE2, intrinsic="TLOAD", args=[a_mat, g_a])
            _emit_step("tload_b", tag_enum="TLOAD", pipe=Pipe.MTE2, intrinsic="TLOAD", args=[b_mat, g_b])
            # MTE1: Mat -> Left/Right
            _emit_step("tmov_a", tag_enum="TMOV_M2L", pipe=Pipe.MTE1, intrinsic="TMOV", args=[a_left, a_mat])
            _emit_step("tmov_b", tag_enum="TMOV_M2R", pipe=Pipe.MTE1, intrinsic="TMOV", args=[b_right, b_mat])
            # M: cube
            _emit_step("tmatmul", tag_enum="TMATMUL", pipe=Pipe.M, intrinsic="TMATMUL", args=[c_acc, a_left, b_right])
            # FIX: Acc -> GM
            _emit_step("tstore_acc", tag_enum="TSTORE_ACC", pipe=Pipe.FIX, intrinsic="TSTORE", args=[g_out, c_acc])
            # MTE2: GM -> Vec (materialize for subsequent vector ops)
            _emit_step("tload_out", tag_enum="TLOAD", pipe=Pipe.MTE2, intrinsic="TLOAD", args=[dst_vec, g_out])
            continue

        tag = classify_pto_op(op.name)
        waits: List[str] = []

        # Conservative auto-sync: enforce sequential semantics across major pipelines.
        # (Explicit `{wait = [...]}` is currently ignored by the demo codegen.)
        w, _ev = _emit_event_sync(
            body,
            decls,
            prev_rec=prev_rec,
            prev_tag=prev_tag,
            prev_pipe=prev_pipe,
            curr_tag=tag.op_enum,
            curr_pipe=tag.pipe,
            event_idx=event_id,
        )
        if w:
            event_id += 1
            waits.extend(w)

        if op.name == "pto.tci":
            desc = 1 if _attr_bool(op, "descending", False) else 0
            intrinsic = f"TCI_WRAP<{desc}>"
        else:
            intrinsic = _intrinsic_name(op.name)

        call_args: List[str] = []

        # Destination tile is the first tile result (functional SSA lowering).
        if op.results:
            # Only handle tile results; event results are ignored by the codegen (auto-sync path).
            if op.results[0] in tile_types:
                call_args.append(_tile(op.results[0]))

        # Operands (memrefs, tiles, scalar immediates)
        for operand in op.operands:
            if operand.startswith("%") and operand in func.args:
                # memref SSA, possibly indexed: "%a[%c0,%c0]"
                mem = operand.split("[", 1)[0]
                call_args.append(f"g_{mem[1:]}")
                continue
            if operand.startswith("%") and operand in tile_types:
                call_args.append(_tile(operand))
                continue
            if operand.startswith("%") and operand in func.consts:
                lit, ty = func.consts[operand]
                if ty == "index":
                    continue
                if ty == "f32" and not lit.endswith("f"):
                    call_args.append(f"{lit}f")
                else:
                    call_args.append(lit)
                continue
            if operand.startswith("%") and re.fullmatch(r"%[-0-9]+(\.[0-9]+)?", operand):
                lit = operand[1:]
                call_args.append(f"{lit}f" if "." in lit else lit)
                continue
            # Indexed memref operand in-place, e.g. %a[%c0,%c0]
            if operand.startswith("%") and "[" in operand:
                mem = operand.split("[", 1)[0]
                call_args.append(f"g_{mem[1:]}")
                continue
            raise ValueError(f"Unsupported operand: {operand} (op {op.name})")

        # Attribute-lowered operands (subset; extend as needed).
        if op.name in {"pto.tcmps", "pto.tcmp"}:
            arg = _cpp_attr_arg(op, "cmpMode")
            if arg is None:
                raise ValueError(f"{op.name} requires {{cmpMode = ...}}")
            call_args.append(arg)
        if op.name == "pto.tcvt":
            arg = _cpp_attr_arg(op, "rmode")
            if arg is None:
                raise ValueError(f"{op.name} requires {{rmode = ...}}")
            call_args.append(arg)

        # Auto-insert tmp tiles / scalar flags for reductions as needed.
        if op.name == "pto.trowsum" and len(op.operands) == 1 and op.results:
            tmp_name = f"%__tmp_rowsum_{idx}"
            call_args.insert(2, _tile(tmp_name))
        if op.name == "pto.tcolsum" and len(op.operands) == 1 and op.results and "isBinary" in op.attrs:
            tmp_name = f"%__tmp_colsum_{idx}"
            call_args.insert(2, _tile(tmp_name))
            call_args.insert(3, "true" if op.attrs["isBinary"].strip().lower() == "true" else "false")

        call_args.extend(waits)

        # Capture RecordEvent for possible subsequent pipeline sync.
        rec = f"rec_{idx}"
        body.append(f"  auto {rec} = {intrinsic}({', '.join(call_args)});")
        prev_rec, prev_tag, prev_pipe = rec, tag.op_enum, tag.pipe

    body.append("}")
    body.append("")

    # Exported launch wrapper (host side).
    host_args = ", ".join(f"{host_ty}* {ssa[1:]}" for ssa in func.arg_order)
    body.append(f'extern "C" void {kernel_name}({host_args}, void* stream) {{')
    if dt == "f16":
        launch_args = ", ".join(f"(half*){ssa[1:]}" for ssa in func.arg_order)
        body.append(f"  run_{kernel_name}<half><<<1, nullptr, stream>>>({launch_args});")
    else:
        launch_args = ", ".join(f"({dev_ty}*){ssa[1:]}" for ssa in func.arg_order)
        body.append(f"  run_{kernel_name}<{dev_ty}><<<1, nullptr, stream>>>({launch_args});")
    body.append("}")

    # Emit event declarations after tiles so they are scoped inside kernel.
    if decls:
        # Insert decls right after "using namespace pto;" section (after includes and the kernel signature),
        # by placing them at the top of the function body. Here, simplest: re-open and inject by concatenation.
        # We already accumulated decls; just splice them after the tile declarations in final text by emitting
        # them at the beginning of the kernel body in order.
        # For readability, we place them right before the first op record.
        pass

    # Build kernel_cpp by injecting decls after tile allocation block.
    # Find the location to inject: just before the first "auto rec_" line.
    kernel_lines = body[:]
    if decls:
        try:
            first_rec = next(i for i, l in enumerate(kernel_lines) if l.strip().startswith("auto rec_") or " auto rec_" in l)
        except StopIteration:
            first_rec = len(kernel_lines)
        kernel_lines = kernel_lines[:first_rec] + decls + [""] + kernel_lines[first_rec:]

    kernel_cpp = "\n".join(kernel_lines) + "\n"

    # ---- Host main ----
    # Minimal runner: allocate buffers for all memrefs, fill non-output buffers, run kernel, dump outputs.
    outputs_init = ",\n".join(f'    "{ssa[1:]}"' for ssa in outputs)

    main_cpp = f"""\
#include "acl/acl.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" void {kernel_name}({', '.join(f'{host_ty}* {ssa[1:]}' for ssa in func.arg_order)}, void* stream);

static void checkAcl(const char* what, aclError e) {{
  if (e != ACL_SUCCESS) {{
    std::cerr << what << " failed: " << e << "\\n";
    std::exit(1);
  }}
}}

static void writeFile(const std::string& path, const void* data, size_t bytes) {{
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {{
    std::cerr << "open " << path << " failed\\n";
    std::exit(1);
  }}
  ofs.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}}

static void readFileExact(const std::string& path, void* data, size_t bytes) {{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {{
    std::cerr << "open " << path << " failed\\n";
    std::exit(1);
  }}
  ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(ifs.gcount()) != bytes) {{
    std::cerr << "short read " << path << ": got " << static_cast<size_t>(ifs.gcount()) << ", expected " << bytes << "\\n";
    std::exit(1);
  }}
}}

static bool readFileIfExists(const std::string& path, void* data, size_t bytes) {{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) return false;
  ifs.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (static_cast<size_t>(ifs.gcount()) != bytes) {{
    std::cerr << "short read " << path << ": got " << static_cast<size_t>(ifs.gcount()) << ", expected " << bytes << "\\n";
    std::exit(1);
  }}
  return true;
}}

int main(int argc, char** argv) {{
  std::string out_dir = ".";
  std::string in_dir;
  std::unordered_map<std::string, std::string> in_files;
  for (int i = 1; i < argc; ++i) {{
    std::string a = argv[i];
    if (a == "--out-dir" && i + 1 < argc) {{
      out_dir = argv[++i];
    }} else if (a == "--in-dir" && i + 1 < argc) {{
      in_dir = argv[++i];
    }} else if (a == "--in" && i + 1 < argc) {{
      std::string kv = argv[++i];
      auto pos = kv.find('=');
      if (pos == std::string::npos) {{
        std::cerr << "bad --in (expected name=path): " << kv << "\\n";
        return 2;
      }}
      in_files[kv.substr(0, pos)] = kv.substr(pos + 1);
    }} else {{
      std::cerr << "usage: " << argv[0] << " [--out-dir DIR] [--in-dir DIR] [--in name=path ...]\\n";
      return 2;
    }}
  }}

  checkAcl("aclInit", aclInit(nullptr));
  checkAcl("aclrtSetDevice", aclrtSetDevice(0));
  aclrtStream stream = nullptr;
  checkAcl("aclrtCreateStream", aclrtCreateStream(&stream));

  std::unordered_set<std::string> outputs = {{
{outputs_init}
  }};

  struct Buf {{
    void* d = nullptr;
    std::vector<{host_ty}> h;
    size_t bytes = 0;
    int rows = 0;
    int cols = 0;
    std::string name;
  }};

  std::vector<Buf> bufs;
  bufs.reserve({len(func.arg_order)});

  auto allocBuf = [&](const char* name, int rows, int cols) {{
    Buf b;
    b.rows = rows;
    b.cols = cols;
    b.name = name;
    size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    b.h.resize(n);
    b.bytes = n * sizeof({host_ty});
    checkAcl("aclrtMalloc", aclrtMalloc(&b.d, b.bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    // Fill inputs.
    if (outputs.find(b.name) == outputs.end()) {{
      bool loaded = false;
      auto it = in_files.find(b.name);
      if (it != in_files.end()) {{
        readFileExact(it->second, b.h.data(), b.bytes);
        loaded = true;
      }} else if (!in_dir.empty()) {{
        loaded = readFileIfExists(in_dir + "/" + b.name + ".bin", b.h.data(), b.bytes);
      }}
      if (!loaded) {{
        for (size_t i = 0; i < n; ++i) {{
          b.h[i] = static_cast<{host_ty}>((i * 13u + 7u) % 97u);
        }}
      }}
      checkAcl("aclrtMemcpy H2D", aclrtMemcpy(b.d, b.bytes, b.h.data(), b.bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }} else {{
      std::memset(b.h.data(), 0, b.bytes);
      checkAcl("aclrtMemcpy H2D", aclrtMemcpy(b.d, b.bytes, b.h.data(), b.bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    }}
    bufs.push_back(std::move(b));
  }};
"""

    for ssa in func.arg_order:
        mi = memrefs[ssa]
        main_cpp += f'  allocBuf("{ssa[1:]}", {mi.rows}, {mi.cols});\n'

    # Call kernel
    arg_ptrs = ", ".join(f"({host_ty}*)bufs[{i}].d" for i in range(len(func.arg_order)))
    main_cpp += f"""

  {kernel_name}({arg_ptrs}, stream);
  checkAcl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));

  // Dump outputs
  for (auto& b : bufs) {{
    if (outputs.find(b.name) == outputs.end()) continue;
    checkAcl("aclrtMemcpy D2H", aclrtMemcpy(b.h.data(), b.bytes, b.d, b.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    writeFile(out_dir + "/" + b.name + ".bin", b.h.data(), b.bytes);
    std::cout << "Wrote " << (out_dir + "/" + b.name + ".bin") << "\\n";
  }}

  for (auto& b : bufs) {{
    checkAcl("aclrtFree", aclrtFree(b.d));
  }}
  checkAcl("aclrtDestroyStream", aclrtDestroyStream(stream));
  checkAcl("aclrtResetDevice", aclrtResetDevice(0));
  checkAcl("aclFinalize", aclFinalize());
  return 0;
}}
"""

    # ---- CMake ----
    cmake = f"""\
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER bisheng CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER bisheng CACHE STRING "" FORCE)

project(ptoas_demo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{{ASCEND_HOME_PATH}})
  message(FATAL_ERROR "ASCEND_HOME_PATH is not set (source Ascend setenv.bash first).")
endif()
set(ASCEND_HOME_PATH $ENV{{ASCEND_HOME_PATH}})
set(PTO_ISA_ROOT "{repo_root}")

set(RUN_MODE "{run_mode}")
set(SOC_VERSION "{soc_version}")

add_compile_options(-O2 -std=c++17 -Wno-macro-redefined -Wno-ignored-attributes)

set(CMAKE_CCE_COMPILE_OPTIONS
  -xcce
  -Xhost-start -Xhost-end
  "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
  "SHELL:-mllvm -cce-aicore-record-overflow=true"
  "SHELL:-mllvm -cce-aicore-addr-transform"
  "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

add_library({kernel_name}_kernel SHARED {kernel_name}_kernel.cpp)
target_compile_options({kernel_name}_kernel PRIVATE ${{CMAKE_CCE_COMPILE_OPTIONS}} {arch_flag} {base_define} -std=c++17)
target_include_directories({kernel_name}_kernel PRIVATE
  ${{PTO_ISA_ROOT}}/include
  ${{ASCEND_HOME_PATH}}/include
  ${{ASCEND_HOME_PATH}}/pkg_inc/
  ${{ASCEND_HOME_PATH}}/pkg_inc/runtime/runtime
)
target_link_options({kernel_name}_kernel PRIVATE --cce-fatobj-link)

add_executable({kernel_name} {kernel_name}_main.cpp)
target_include_directories({kernel_name} PRIVATE
  ${{ASCEND_HOME_PATH}}/include
)
target_link_directories({kernel_name} PUBLIC
  ${{ASCEND_HOME_PATH}}/lib64
  ${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib
)
set_target_properties({kernel_name} PROPERTIES
  BUILD_RPATH "${{ASCEND_HOME_PATH}}/lib64;${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib"
  INSTALL_RPATH "${{ASCEND_HOME_PATH}}/lib64;${{ASCEND_HOME_PATH}}/tools/simulator/${{SOC_VERSION}}/lib"
)
target_link_libraries({kernel_name} PRIVATE
  {kernel_name}_kernel
  $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},sim>:runtime_camodel>>
  $<BUILD_INTERFACE:$<$<STREQUAL:${{RUN_MODE}},npu>:runtime>>
  stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
"""

    return Generated(kernel_cpp=kernel_cpp, kernel_cube_cpp=None, main_cpp=main_cpp, cmake=cmake)
