from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


_RE_TILE = re.compile(r"!pto\.(?:tile|tilebuf)<(?:(?P<kind>\w+),)?(?P<r>\d+)x(?P<c>\d+)x(?P<dt>\w+)>")
_RE_CONST = re.compile(
    r"^(?P<ssa>%[A-Za-z_][\w\.]*)\s*=\s*arith\.constant\s+(?P<lit>(?:[-0-9\.]+|true|false))\s*:\s*(?P<ty>\w+)$"
)
_RE_OP = re.compile(
    r"^(?:(?P<lhs>[%A-Za-z0-9_\s,\.]+)\s*=\s*)?(?P<op>(?:pto\.)?[A-Za-z_][\w\.]*)\s*(?P<rest>.*)$"
)


def _dtype_to_numpy(dt: str) -> np.dtype:
    if dt == "f32":
        return np.float32
    if dt == "i32":
        return np.int32
    if dt == "i16":
        return np.int16
    if dt == "u8":
        return np.uint8
    if dt == "i8":
        return np.int8
    raise ValueError(f"Unsupported dtype for sim: {dt}")


@dataclass
class SimState:
    scalars: Dict[str, float]
    tiles: Dict[str, np.ndarray]

def _split_operands_top_level(s: str) -> List[str]:
    parts: List[str] = []
    cur: List[str] = []
    angle = 0
    square = 0
    paren = 0
    brace = 0
    for ch in s:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "[":
            square += 1
        elif ch == "]":
            square = max(0, square - 1)
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        if ch == "," and angle == 0 and square == 0 and paren == 0 and brace == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _normalize_op(op: str) -> str:
    if op.startswith("pto.") or op.startswith("arith.") or op.startswith("builtin."):
        return op
    return f"pto.{op}"


def _parse_tile_type(type_sig: str) -> Tuple[int, int, str]:
    m = _RE_TILE.search(type_sig)
    if not m:
        raise ValueError(f"Failed to parse tile type from: {type_sig}")
    return (int(m["r"]), int(m["c"]), m["dt"])


def _operand_base_name(operand: str) -> str:
    s = operand.strip()
    if "[" in s:
        return s.split("[", 1)[0].strip()
    return s

def _parse_memref_access(operand: str) -> Tuple[str, Optional[str], Optional[str]]:
    s = operand.strip()
    if "[" not in s or "]" not in s:
        return (s, None, None)
    base, tail = s.split("[", 1)
    idxs, _rest = tail.split("]", 1)
    parts = _split_operands_top_level(idxs)
    row = parts[0].strip() if len(parts) >= 1 else None
    col = parts[1].strip() if len(parts) >= 2 else None
    return (base.strip(), row, col)

def _strip_type_annot(token: str) -> str:
    t = token.strip()
    if " : " in t:
        return t.split(" : ", 1)[0].strip()
    if ":" in t:
        # `x: type` inside ins/outs.
        return t.split(":", 1)[0].strip()
    return t

def _extract_paren_group(text: str, kw: str) -> Optional[str]:
    # Returns the string inside `kw(...)`, or None.
    i = text.find(kw + "(")
    if i < 0:
        return None
    j = i + len(kw) + 1
    depth = 1
    out: List[str] = []
    while j < len(text) and depth > 0:
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        j += 1
    return "".join(out).strip()

def _parse_ins_outs(rest: str) -> Optional[Tuple[List[str], List[str]]]:
    ins_blob = _extract_paren_group(rest, "ins")
    outs_blob = _extract_paren_group(rest, "outs")
    if ins_blob is None or outs_blob is None:
        return None
    # Drop the type-list part after ':' inside the ins/outs group.
    if " : " in ins_blob:
        ins_vals = ins_blob.split(" : ", 1)[0].strip()
    elif ":" in ins_blob:
        ins_vals = ins_blob.split(":", 1)[0].strip()
    else:
        ins_vals = ins_blob
    if " : " in outs_blob:
        outs_vals = outs_blob.split(" : ", 1)[0].strip()
    elif ":" in outs_blob:
        outs_vals = outs_blob.split(":", 1)[0].strip()
    else:
        outs_vals = outs_blob
    ins = [_strip_type_annot(x) for x in _split_operands_top_level(ins_vals)]
    outs = [_strip_type_annot(x) for x in _split_operands_top_level(outs_vals)]
    return (ins, outs)


def _resolve_scalar(state: SimState, token: str) -> float:
    t = token.strip()
    if t == "true":
        return 1.0
    if t == "false":
        return 0.0
    if t in state.scalars:
        return float(state.scalars[t])
    if t.startswith("%") and t[1:] and (t[1].isdigit() or t[1] == "-" or t[1] == "."):
        return float(t[1:])
    return float(t)


def simulate_mlir_module(mlir_text: str, inputs: Dict[str, np.ndarray], dump: Tuple[str, ...] = ()) -> SimState:
    """
    Minimal CPU simulator for a PTO tile-op subset.

    Accepts both:
    - MLIR-ish spelling: `pto.tadd`, `pto.tload`, ...
    - PTO-AS spelling: `tadd`, `tload`, ...

    `inputs` maps function arg names (without '%') to numpy arrays.
    """
    state = SimState(scalars={}, tiles={})
    out_arrays: Dict[str, np.ndarray] = dict(inputs)

    scf_for_re = re.compile(
        r"^scf\.for\s+(?P<iv>%[A-Za-z_][\w\.]*)\s*=\s*(?P<lb>\S+)\s+to\s+(?P<ub>\S+)\s+step\s+(?P<step>\S+)\s*\{\s*$"
    )
    scf_if_re = re.compile(r"^scf\.if\s+(?P<cond>[^\s]+)(?:\s*->\s*\([^)]*\))?\s*\{\s*$")
    scf_else_re = re.compile(r"^\}\s*else\s*\{\s*$")

    def _dump_tile(name: str, tile: np.ndarray) -> None:
        if name in dump:
            print(f"[pypto sim] dump {name}:\n{tile}")

    def run_block(block_lines: List[str]) -> None:
        i = 0
        while i < len(block_lines):
            raw = block_lines[i]
            line = raw.strip()

            if line.startswith("//") or line.startswith(";") or line.startswith("#"):
                i += 1
                continue
            if not line or line.startswith("module") or line.startswith("func.func") or line == "{":
                i += 1
                continue
            if line == "return":
                i += 1
                continue

            # Structured control flow (minimal scf.for simulation).
            mfor = scf_for_re.match(line)
            if mfor:
                iv = mfor["iv"]
                lb_tok, ub_tok, step_tok = mfor["lb"], mfor["ub"], mfor["step"]

                # Find matching closing brace, respecting nesting.
                depth = 1
                j = i + 1
                while j < len(block_lines) and depth > 0:
                    s = block_lines[j].strip()
                    if s.endswith("{"):
                        depth += 1
                    if s == "}":
                        depth -= 1
                    j += 1
                if depth != 0:
                    raise ValueError(f"Unmatched scf.for braces starting at: {line}")

                body = block_lines[i + 1 : j - 1]
                lb = int(_resolve_scalar(state, lb_tok))
                ub = int(_resolve_scalar(state, ub_tok))
                step = int(_resolve_scalar(state, step_tok))
                old_iv = state.scalars.get(iv)
                for v in range(lb, ub, step):
                    state.scalars[iv] = int(v)
                    run_block(body)
                if old_iv is None:
                    state.scalars.pop(iv, None)
                else:
                    state.scalars[iv] = old_iv

                i = j
                continue

            # Structured control flow (minimal scf.if simulation).
            mif = scf_if_re.match(line)
            if mif:
                cond_tok = mif["cond"]
                cond = bool(int(_resolve_scalar(state, cond_tok)))

                # Parse then-region until matching `}` (or `} else {`) at depth 1.
                depth = 1
                j = i + 1
                while j < len(block_lines):
                    s = block_lines[j].strip()
                    if depth == 1 and scf_else_re.match(s):
                        break
                    if depth == 1 and s == "}":
                        break
                    depth += s.count("{")
                    depth -= s.count("}")
                    j += 1
                if j >= len(block_lines):
                    raise ValueError(f"Unmatched scf.if braces starting at: {line}")
                then_body = block_lines[i + 1 : j]

                # Optional else-region.
                if j < len(block_lines) and scf_else_re.match(block_lines[j].strip()):
                    depth = 1
                    k = j + 1
                    while k < len(block_lines):
                        s = block_lines[k].strip()
                        if depth == 1 and s == "}":
                            break
                        depth += s.count("{")
                        depth -= s.count("}")
                        k += 1
                    if k >= len(block_lines):
                        raise ValueError(f"Unmatched scf.if else braces starting at: {line}")
                    else_body = block_lines[j + 1 : k]
                    run_block(then_body if cond else else_body)
                    i = k + 1
                    continue

                # No else.
                run_block(then_body if cond else [])
                i = j + 1
                continue

            if line == "}":
                i += 1
                continue

            mconst = _RE_CONST.match(line)
            if mconst:
                ssa = mconst["ssa"]
                lit = mconst["lit"]
                ty = mconst["ty"]
                if ty == "index":
                    state.scalars[ssa] = int(float(_resolve_scalar(state, lit)))
                elif ty.startswith("i"):
                    state.scalars[ssa] = int(float(_resolve_scalar(state, lit)))
                elif ty.startswith("f"):
                    state.scalars[ssa] = float(_resolve_scalar(state, lit))
                else:
                    raise ValueError(f"Unsupported constant type: {ty}")
                i += 1
                continue

            mop = _RE_OP.match(line)
            if not mop:
                i += 1
                continue

            lhs_blob = (mop.group("lhs") or "").strip()
            lhs0: Optional[str] = None
            if lhs_blob:
                lhs0 = lhs_blob.split(",", 1)[0].strip()
            op = _normalize_op(mop.group("op"))
            rest = (mop.group("rest") or "").strip()

            type_sig = ""
            if " : " in rest and ("ins(" not in rest and "outs(" not in rest):
                rest, type_sig = rest.split(" : ", 1)
                rest = rest.strip()
                type_sig = type_sig.strip()
            elif rest.startswith(":"):
                type_sig = rest[1:].strip()
                rest = ""

            if "{" in rest and "}" in rest:
                before, after = rest.split("{", 1)
                _attr_blob, tail = after.split("}", 1)
                rest = (before + " " + tail).strip()

            ins_outs = _parse_ins_outs(rest) if rest else None
            if ins_outs is not None:
                ins_ops, outs_ops = ins_outs
                if op in {"pto.tstore", "pto.mscatter"}:
                    operands = [outs_ops[0]] + ins_ops
                else:
                    operands = outs_ops + ins_ops
            else:
                operands = _split_operands_top_level(rest.split(":", 1)[0].strip()) if rest else []

            def _tile_dst_and_srcs() -> Tuple[str, List[str]]:
                if lhs0 is not None:
                    return (lhs0, operands)
                if not operands:
                    raise ValueError(f"Expected destination tile operand: {line}")
                return (operands[0], list(operands[1:]))

            if op in {"pto.tile.alloc", "pto.alloc_tile"}:
                if lhs0 is None:
                    raise ValueError(f"{op} requires an SSA result")
                if not type_sig:
                    raise ValueError(f"{op} requires a type signature: {line}")
                r, c, dt = _parse_tile_type(type_sig)
                t = np.zeros((r, c), dtype=_dtype_to_numpy(dt))
                state.tiles[lhs0] = t
                _dump_tile(lhs0, t)
                i += 1
                continue

            if op == "arith.cmpi":
                if lhs0 is None:
                    raise ValueError(f"arith.cmpi requires an SSA result: {line}")
                ops_ = _split_operands_top_level(rest.split(":", 1)[0].strip()) if rest else []
                if len(ops_) != 3:
                    raise ValueError(f"arith.cmpi expects 3 operands (pred, a, b), got {ops_}: {line}")
                pred, a_tok, b_tok = [x.strip() for x in ops_]
                a = _resolve_scalar(state, a_tok)
                b = _resolve_scalar(state, b_tok)
                if pred in {"eq"}:
                    v = a == b
                elif pred in {"ne"}:
                    v = a != b
                elif pred in {"slt", "ult"}:
                    v = a < b
                elif pred in {"sle", "ule"}:
                    v = a <= b
                elif pred in {"sgt", "ugt"}:
                    v = a > b
                elif pred in {"sge", "uge"}:
                    v = a >= b
                else:
                    raise ValueError(f"Unsupported arith.cmpi predicate '{pred}': {line}")
                state.scalars[lhs0] = 1 if v else 0
                i += 1
                continue

            if op == "pto.tload":
                if lhs0 is not None:
                    if len(operands) != 1:
                        raise ValueError(f"tload expects 1 operand (legacy SSA form), got {operands}")
                    dst = lhs0
                    mem = operands[0]
                else:
                    if len(operands) != 2:
                        raise ValueError(f"tload expects 2 operands (DPS/register form), got {operands}")
                    dst, mem = operands[0], operands[1]
                if dst not in state.tiles:
                    if not type_sig:
                        raise ValueError(f"tload requires either an allocated dst tile or a type signature: {line}")
                    r, c, dt = _parse_tile_type(type_sig)
                    state.tiles[dst] = np.zeros((r, c), dtype=_dtype_to_numpy(dt))
                tile = state.tiles[dst]
                r, c = int(tile.shape[0]), int(tile.shape[1])
                mem_base, row_tok, col_tok = _parse_memref_access(mem)
                mem_name = _operand_base_name(mem_base)
                src = out_arrays[mem_name[1:]]
                ro = int(_resolve_scalar(state, row_tok)) if row_tok is not None else 0
                co = int(_resolve_scalar(state, col_tok)) if col_tok is not None else 0
                view = src[ro : ro + r, co : co + c]
                if view.shape != (r, c):
                    t = np.zeros((r, c), dtype=tile.dtype)
                    t[: view.shape[0], : view.shape[1]] = view.astype(tile.dtype, copy=False)
                else:
                    t = np.array(view, copy=True).astype(tile.dtype, copy=False)
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op == "pto.tstore":
                if len(operands) != 2:
                    raise ValueError(f"tstore expects 2 operands, got {operands}")
                a0, a1 = operands
                if a0 in state.tiles:
                    src = a0
                    dst = a1
                else:
                    dst = a0
                    src = a1
                dst_base, row_tok, col_tok = _parse_memref_access(dst)
                dst_name = _operand_base_name(dst_base)
                src_tile = state.tiles[src]
                ro = int(_resolve_scalar(state, row_tok)) if row_tok is not None else 0
                co = int(_resolve_scalar(state, col_tok)) if col_tok is not None else 0
                arr = out_arrays[dst_name[1:]]
                r, c = int(src_tile.shape[0]), int(src_tile.shape[1])
                arr[ro : ro + r, co : co + c] = src_tile.reshape((r, c))
                i += 1
                continue

            if op in {"pto.tadd", "pto.tsub", "pto.tmul", "pto.tdiv"}:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 2:
                    raise ValueError(f"{op} expects 2 operands, got {srcs}")
                a, b = srcs
                f = {
                    "pto.tadd": lambda x, y: x + y,
                    "pto.tsub": lambda x, y: x - y,
                    "pto.tmul": lambda x, y: x * y,
                    "pto.tdiv": lambda x, y: x / y,
                }[op]
                t = f(state.tiles[a], state.tiles[b])
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op == "pto.tmatmul":
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 2:
                    raise ValueError(f"tmatmul expects 2 operands, got {srcs}")
                a, b = srcs
                t = state.tiles[a] @ state.tiles[b]
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op in {"pto.tmax", "pto.tmin"}:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 2:
                    raise ValueError(f"{op} expects 2 operands, got {srcs}")
                a, b = srcs
                t = (np.maximum if op == "pto.tmax" else np.minimum)(state.tiles[a], state.tiles[b])
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op in {"pto.tabs", "pto.tneg", "pto.tnot"}:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 1:
                    raise ValueError(f"{op} expects 1 operand, got {srcs}")
                (a,) = srcs
                f = {"pto.tabs": np.abs, "pto.tneg": lambda x: -x, "pto.tnot": lambda x: np.bitwise_not(x)}[op]
                t = f(state.tiles[a])
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op in {"pto.texp", "pto.tlog", "pto.tsqrt", "pto.trsqrt", "pto.trecip", "pto.trelu"}:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 1:
                    raise ValueError(f"{op} expects 1 operand, got {srcs}")
                (a,) = srcs
                x = state.tiles[a].astype(np.float32, copy=False)
                if op == "pto.texp":
                    t = np.exp(x)
                elif op == "pto.tlog":
                    t = np.log(x)
                elif op == "pto.tsqrt":
                    t = np.sqrt(x)
                elif op == "pto.trsqrt":
                    t = 1.0 / np.sqrt(x)
                elif op == "pto.trecip":
                    t = 1.0 / x
                else:
                    t = np.maximum(x, 0.0)
                state.tiles[dst] = t.astype(state.tiles[a].dtype, copy=False)
                _dump_tile(dst, state.tiles[dst])
                i += 1
                continue

            if op in {"pto.tadds", "pto.tsubs", "pto.tmuls", "pto.tdivs"}:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 2:
                    raise ValueError(f"{op} expects 2 operands, got {srcs}")
                a, b = srcs
                if a in state.tiles:
                    tile = state.tiles[a]
                    scalar = _resolve_scalar(state, b)
                    if op == "pto.tadds":
                        t = tile + scalar
                    elif op == "pto.tsubs":
                        t = tile - scalar
                    elif op == "pto.tmuls":
                        t = tile * scalar
                    else:
                        t = tile / scalar
                else:
                    scalar = _resolve_scalar(state, a)
                    tile = state.tiles[b]
                    if op == "pto.tadds":
                        t = scalar + tile
                    elif op == "pto.tsubs":
                        t = scalar - tile
                    elif op == "pto.tmuls":
                        t = scalar * tile
                    else:
                        t = scalar / tile
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op == "pto.texpands":
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 1:
                    raise ValueError(f"texpands expects 1 operand, got {srcs}")
                (s,) = srcs
                scalar = _resolve_scalar(state, s)
                if dst in state.tiles:
                    r, c = state.tiles[dst].shape
                    t = np.full((int(r), int(c)), scalar, dtype=state.tiles[dst].dtype)
                else:
                    if not type_sig:
                        raise ValueError(f"texpands requires either an allocated dst tile or a type signature: {line}")
                    r, c, dt = _parse_tile_type(type_sig)
                    t = np.full((r, c), scalar, dtype=_dtype_to_numpy(dt))
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op == "pto.trowsum":
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 1:
                    raise ValueError(f"trowsum expects 1 operand, got {srcs}")
                (a,) = srcs
                t = np.sum(state.tiles[a], axis=1, keepdims=True)
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op == "pto.tcolsum":
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 1:
                    raise ValueError(f"tcolsum expects 1 operand, got {srcs}")
                (a,) = srcs
                t = np.sum(state.tiles[a], axis=0, keepdims=True)
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            if op in {
                "pto.trowexpanddiv",
                "pto.trowexpandsub",
                "pto.trowexpandadd",
                "pto.trowexpandmul",
                "pto.trowexpandmax",
                "pto.trowexpandmin",
            }:
                dst, srcs = _tile_dst_and_srcs()
                if len(srcs) != 2:
                    raise ValueError(f"{op} expects 2 operands, got {srcs}")
                a, b = srcs
                x = state.tiles[a]
                y = state.tiles[b]
                if op == "pto.trowexpanddiv":
                    t = x / y
                elif op == "pto.trowexpandsub":
                    t = x - y
                elif op == "pto.trowexpandadd":
                    t = x + y
                elif op == "pto.trowexpandmul":
                    t = x * y
                elif op == "pto.trowexpandmax":
                    t = np.maximum(x, y)
                else:
                    t = np.minimum(x, y)
                state.tiles[dst] = t
                _dump_tile(dst, t)
                i += 1
                continue

            i += 1

    lines = [l.rstrip("\n") for l in mlir_text.splitlines()]
    run_block(lines)
    return state
