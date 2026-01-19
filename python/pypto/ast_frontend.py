from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ir import Function, Module, Operation, Value
from .mlir_printer import print_module
from .types import MemRefType, TileBufType, TileType


def kernel(fn: Callable[..., Any]) -> Callable[..., Any]:
    setattr(fn, "_pypto_is_kernel", True)
    return fn


@dataclass(frozen=True)
class KernelSpec:
    fn: Callable[..., Any]
    name: str
    args: Sequence[Value]


def load_kernels_from_file(path: str | Path) -> List[Callable[..., Any]]:
    import runpy

    ns = runpy.run_path(str(path))
    out: List[Callable[..., Any]] = []
    for v in ns.values():
        if callable(v) and getattr(v, "_pypto_is_kernel", False):
            out.append(v)
    return out


def _mlir_const_index(name: str, value: int) -> Operation:
    res = Value(name=f"%{name}", type_str="index")
    return Operation(
        results=[res],
        op="arith.constant",
        operands=[str(value)],
        attrs={},
        type_sig="index",
    )

def _mlir_const(name: str, literal: str, ty: str) -> Operation:
    res = Value(name=f"%{name}", type_str=ty)
    return Operation(
        results=[res],
        op="arith.constant",
        operands=[literal],
        attrs={},
        type_sig=ty,
    )


def _require_type(obj: object, kind: type, what: str) -> object:
    if not isinstance(obj, kind):
        raise TypeError(f"{what} must be a {kind.__name__}, got {type(obj).__name__}")
    return obj


def _arg_values_from_annotations(fn: Callable[..., Any]) -> List[Value]:
    sig = inspect.signature(fn)
    args: List[Value] = []
    for name, p in sig.parameters.items():
        if p.annotation is inspect._empty:
            raise TypeError(f"Kernel arg '{name}' must be annotated with MemRefType")
        ann = p.annotation
        if isinstance(ann, str):
            ann = eval(ann, fn.__globals__, {})
        _require_type(ann, MemRefType, f"Kernel arg '{name}' annotation")
        args.append(Value(name=f"%{name}", type_str=ann.mlir()))
    return args


def _extract_kernel_ast(fn: Callable[..., Any]) -> ast.FunctionDef:
    src = inspect.getsource(fn)
    mod = ast.parse(src)
    fdefs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if not fdefs:
        raise ValueError("Failed to locate function AST")
    return fdefs[0]


def _is_call_to(node: ast.AST, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == name:
        return True
    if isinstance(node.func, ast.Name) and node.func.id == name:
        return True
    return False


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    raise TypeError(f"Unsupported name node: {ast.dump(node)}")


def _operand_str(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return f"%{node.id}"
    raise TypeError(f"Unsupported operand: {ast.dump(node)}")


def _tile_type_for_call(
    fn: Callable[..., Any], call: ast.Call, arg_types: Dict[str, MemRefType], created_tiles: Dict[str, TileBufType]
) -> TileBufType:
    if isinstance(call.func, ast.Attribute):
        op = call.func.attr
    elif isinstance(call.func, ast.Name):
        op = call.func.id
    else:
        raise TypeError("Unsupported call")

    if op == "tload":
        mem = _name_of(call.args[0])
        return arg_types[mem].tilebuf()
    if op in {"texpands", "tci"}:
        like = None
        for kw in call.keywords:
            if kw.arg == "like":
                like = kw.value
                break
        if like is None:
            raise TypeError(f"{op} requires a keyword argument like=<memref or tile> for type inference")
        if not isinstance(like, ast.Name):
            raise TypeError("like= must be a simple name")
        name = like.id
        if name in arg_types:
            return arg_types[name].tilebuf()
        if name in created_tiles:
            return created_tiles[name]
        raise TypeError(f"Unknown like target: {name}")

    if op == "tmatmul":
        a = created_tiles[_name_of(call.args[0])]
        b = created_tiles[_name_of(call.args[1])]
        return TileBufType(rows=a.rows, cols=b.cols, dtype=a.dtype)

    if op == "trowsum":
        a = created_tiles[_name_of(call.args[0])]
        return TileBufType(rows=a.rows, cols=1, dtype=a.dtype)

    if op == "tcolsum":
        a = created_tiles[_name_of(call.args[0])]
        return TileBufType(rows=1, cols=a.cols, dtype=a.dtype)

    unary_same = {"tabs", "tneg", "tnot", "texp", "tlog", "trelu", "tsqrt", "trecip", "trsqrt"}
    if op in unary_same:
        src0 = _name_of(call.args[0])
        return created_tiles[src0]

    binary_same = {
        "tadd",
        "tsub",
        "tmul",
        "tdiv",
        "tmax",
        "tmin",
        "tand",
        "tor",
        "txor",
        "trowexpanddiv",
        "trowexpandsub",
        "trowexpandadd",
        "trowexpandmul",
        "trowexpandmax",
        "trowexpandmin",
    }
    if op in binary_same:
        src0 = _name_of(call.args[0])
        return created_tiles[src0]
    raise ValueError(f"Unsupported op for tile type inference: {op}")


def _emit_ops_from_body(fn: Callable[..., Any], fdef: ast.FunctionDef, args: Sequence[Value]) -> List[Operation]:
    arg_types: Dict[str, MemRefType] = {}
    for a in args:
        raw = fn.__annotations__[a.name[1:]]
        ann = eval(raw, fn.__globals__, {}) if isinstance(raw, str) else raw
        _require_type(ann, MemRefType, f"Kernel arg '{a.name[1:]}' annotation")
        arg_types[a.name[1:]] = ann
    created_tiles: Dict[str, TileBufType] = {}
    ops: List[Operation] = []
    const_pool: Dict[Tuple[str, str], str] = {}
    const_types: Dict[str, str] = {}
    const_idx = 0

    ops.append(_mlir_const_index("c0", 0))

    def emit_scalar(node: ast.AST) -> Tuple[str, str]:
        nonlocal const_idx
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or node.value is None:
                raise TypeError("Unsupported scalar literal")
            if isinstance(node.value, int):
                lit = str(int(node.value))
                ty = "i32"
            elif isinstance(node.value, float):
                lit = repr(float(node.value))
                ty = "f32"
            else:
                raise TypeError(f"Unsupported scalar literal: {type(node.value).__name__}")
            key = (lit, ty)
            if key in const_pool:
                ssa = const_pool[key]
                return (ssa, const_types[ssa])
            name = f"c{const_idx}"
            const_idx += 1
            ops.append(_mlir_const(name, lit, ty))
            ssa = f"%{name}"
            const_pool[key] = ssa
            const_types[ssa] = ty
            return (ssa, ty)
        if isinstance(node, ast.Name):
            raise TypeError("Named scalar variables are not supported yet; use a literal like 1.0")
        raise TypeError(f"Unsupported scalar operand: {ast.dump(node)}")

    def _eval_const_bool(node: ast.AST) -> Optional[bool]:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return bool(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            v = _eval_const_bool(node.operand)
            return (not v) if v is not None else None
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            a = node.left
            b = node.comparators[0]
            if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                try:
                    if isinstance(node.ops[0], ast.Eq):
                        return a.value == b.value
                    if isinstance(node.ops[0], ast.NotEq):
                        return a.value != b.value
                    if isinstance(node.ops[0], ast.Lt):
                        return a.value < b.value
                    if isinstance(node.ops[0], ast.LtE):
                        return a.value <= b.value
                    if isinstance(node.ops[0], ast.Gt):
                        return a.value > b.value
                    if isinstance(node.ops[0], ast.GtE):
                        return a.value >= b.value
                except TypeError:
                    return None
        return None

    def emit_stmt_list(stmts: Sequence[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise TypeError("Only simple assignments are supported")
                lhs = stmt.targets[0].id
                rhs = stmt.value
                if not isinstance(rhs, ast.Call):
                    raise TypeError("Only calls are supported on RHS")

                if _is_call_to(rhs, "tload"):
                    mem = _operand_str(rhs.args[0])
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    ops.append(
                        Operation(
                            results=[],
                            op="pto.tload",
                            operands=[
                                f"ins({mem}[%c0, %c0] : {arg_types[mem[1:]].mlir()})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue

                if isinstance(rhs.func, ast.Name):
                    op = rhs.func.id
                elif isinstance(rhs.func, ast.Attribute):
                    op = rhs.func.attr
                else:
                    op = ""

                if op in {
                    "tabs",
                    "tneg",
                    "tnot",
                    "texp",
                    "tlog",
                    "trelu",
                    "tsqrt",
                    "trecip",
                    "trsqrt",
                    "tadd",
                    "tsub",
                    "tmul",
                    "tdiv",
                    "tmax",
                    "tmin",
                    "tand",
                    "tor",
                    "txor",
                    "trowexpanddiv",
                    "trowexpandsub",
                    "trowexpandadd",
                    "trowexpandmul",
                    "trowexpandmax",
                    "trowexpandmin",
                }:
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    operands = [_operand_str(a) for a in rhs.args]
                    ops.append(
                        Operation(
                            results=[],
                            op=f"pto.{op}",
                            operands=[
                                f"ins({', '.join(operands)} : {', '.join(created_tiles[_name_of(a)].mlir() for a in rhs.args)})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue

                if op == "tmatmul":
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    a0 = _name_of(rhs.args[0])
                    a1 = _name_of(rhs.args[1])
                    ta0 = created_tiles[a0].mlir()
                    ta1 = created_tiles[a1].mlir()
                    ops.append(
                        Operation(
                            results=[],
                            op="pto.tmatmul",
                            operands=[
                                f"ins({_operand_str(rhs.args[0])}, {_operand_str(rhs.args[1])} : {ta0}, {ta1})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue

                if op in {"trowsum", "tcolsum"}:
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    src0 = _name_of(rhs.args[0])
                    src_ty = created_tiles[src0].mlir()
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    ops.append(
                        Operation(
                            results=[],
                            op=f"pto.{op}",
                            operands=[
                                f"ins({_operand_str(rhs.args[0])} : {src_ty})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue

                if op in {"tadds", "tsubs", "tmuls", "tdivs"}:
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    src0 = _operand_str(rhs.args[0])
                    scalar, scalar_ty = emit_scalar(rhs.args[1])
                    src0_ty = created_tiles[_name_of(rhs.args[0])].mlir()
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    ops.append(
                        Operation(
                            results=[],
                            op=f"pto.{op}",
                            operands=[
                                f"ins({src0}, {scalar} : {src0_ty}, {scalar_ty})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue

                if op in {"texpands", "tci"}:
                    tile_ty = _tile_type_for_call(fn, rhs, arg_types, created_tiles)
                    tile_v = Value(name=f"%{lhs}", type_str=tile_ty.mlir())
                    scalar, scalar_ty = emit_scalar(rhs.args[0])
                    attrs: Dict[str, object] = {}
                    if op == "tci":
                        for kw in rhs.keywords:
                            if kw.arg == "descending":
                                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                                    raise TypeError("tci descending= must be a bool literal")
                                attrs["descending"] = bool(kw.value.value)
                    if lhs not in created_tiles:
                        created_tiles[lhs] = tile_ty
                        ops.append(
                            Operation(
                                results=[tile_v],
                                op="pto.alloc_tile",
                                operands=[],
                                attrs={},
                                type_sig=f"{tile_ty.mlir()}",
                            )
                        )
                    elif created_tiles[lhs] != tile_ty:
                        raise TypeError(f"Tile buffer {lhs} type changed from {created_tiles[lhs]} to {tile_ty}")
                    ops.append(
                        Operation(
                            results=[],
                            op=f"pto.{op}",
                            operands=[
                                f"ins({scalar} : {scalar_ty})",
                                f"outs({tile_v.name} : {tile_ty.mlir()})",
                            ],
                            attrs=attrs,
                            type_sig=None,
                        )
                    )
                    continue

                raise ValueError(f"Unsupported call: {ast.dump(rhs)}")

            if isinstance(stmt, ast.For):
                if not isinstance(stmt.target, ast.Name):
                    raise TypeError("Only `for <name> in range(...)` is supported")
                it = stmt.iter
                if not isinstance(it, ast.Call) or not isinstance(it.func, ast.Name) or it.func.id != "range":
                    raise TypeError("Only `for ... in range(...)` loops are supported")
                args = it.args
                if not all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in args):
                    raise TypeError("range(...) bounds must be integer literals for now")
                ints = [int(a.value) for a in args]  # type: ignore[arg-type]
                if len(ints) == 1:
                    r = range(ints[0])
                elif len(ints) == 2:
                    r = range(ints[0], ints[1])
                elif len(ints) == 3:
                    r = range(ints[0], ints[1], ints[2])
                else:
                    raise TypeError("range() supports 1..3 integer args")
                for _ in r:
                    emit_stmt_list(stmt.body)
                continue

            if isinstance(stmt, ast.If):
                v = _eval_const_bool(stmt.test)
                if v is None:
                    raise TypeError("Only compile-time constant if-conditions are supported for now")
                emit_stmt_list(stmt.body if v else stmt.orelse)
                continue

            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                if _is_call_to(call, "tstore"):
                    mem = _operand_str(call.args[0])
                    src = _operand_str(call.args[1])
                    src_name = _name_of(call.args[1])
                    mem_ty = arg_types[mem[1:]]
                    tile_ty = created_tiles[src_name]
                    ops.append(
                        Operation(
                            results=[],
                            op="pto.tstore",
                            operands=[
                                f"ins({src} : {tile_ty.mlir()})",
                                f"outs({mem}[%c0, %c0] : {mem_ty.mlir()})",
                            ],
                            attrs={},
                            type_sig=None,
                        )
                    )
                    continue
                raise ValueError("Only pto.tstore is supported as a statement")

            if isinstance(stmt, ast.Return):
                return

            raise TypeError(f"Unsupported statement: {ast.dump(stmt)}")

    emit_stmt_list(fdef.body)

    ops.append(Operation(results=[], op="builtin.return", operands=[]))
    return ops


def emit_mlir(fn: Callable[..., Any]) -> str:
    if not getattr(fn, "_pypto_is_kernel", False):
        raise TypeError("Function must be decorated with @pypto.kernel")
    args = _arg_values_from_annotations(fn)
    fdef = _extract_kernel_ast(fn)
    ops = _emit_ops_from_body(fn, fdef, args)
    m = Module(funcs=[Function(name="main", args=args, ops=ops)])
    return print_module(m)


def emit_mlir_file(fn: Callable[..., Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(emit_mlir(fn), encoding="utf-8")
    return out_path
