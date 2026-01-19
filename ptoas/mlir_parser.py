from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PtoType:
    kind: str  # "tile" | "memref" | "event" | "index"
    raw: str
    rows: Optional[int] = None
    cols: Optional[int] = None
    dtype: Optional[str] = None
    space: Optional[str] = None


@dataclass(frozen=True)
class Op:
    results: Sequence[str]
    name: str
    operands: Sequence[str]
    wait: Sequence[str]
    attrs: Dict[str, str]
    type_sig: str  # may be "" (PTO-AS lines can omit type sig)


@dataclass(frozen=True)
class Func:
    # Args are memrefs only (tile SSA values are declared separately in PTO-AS, or inferred from op type sigs in MLIR form).
    args: Dict[str, str]  # "%input" -> type string (lookup)
    arg_order: List[str]  # preserves memref argument order
    tile_decls: Dict[str, str]  # "%x" -> "!pto.tile<...>"
    consts: Dict[str, Tuple[str, str]]  # "%c0" or "%1.0" -> (literal, type)
    ops: List[Op]


_RE_FUNC = re.compile(r"func\.func\s+@(?P<name>[A-Za-z_][\w\.]*)\((?P<args>.*)\)\s*\{")
_RE_ARG = re.compile(r"(?P<ssa>%[A-Za-z_][\w\.]*)\s*:\s*(?P<ty>.+)")
_RE_OP = re.compile(
    r"^(?:(?P<lhs>[%\w\s,]+)\s*=\s*)?(?P<op>pto\.[A-Za-z_][\w\.]*)\s+(?P<rest>.*)$"
)
_RE_WAIT = re.compile(r"\{[^}]*wait\s*=\s*\[(?P<list>[^\]]*)\][^}]*\}")
_RE_CONST = re.compile(r"^(?P<ssa>%[A-Za-z_][\w\.]*)\s*=\s*arith\.constant\s+(?P<lit>[-0-9\.]+)\s*:\s*(?P<ty>\w+)$")
_RE_DECL = re.compile(r"^//\s*(?P<ssa>%[A-Za-z_][\w\.]*)\s*:\s*(?P<ty>.+?)\s*$")
_RE_PTOAS_ASSIGN = re.compile(r"^(?P<lhs>%[A-Za-z_][\w\.]*)\s*=\s*(?P<op>[A-Za-z_][\w\.]*)\s+(?P<rest>.*)$")

def _split_commas_top_level(s: str) -> List[str]:
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

def _strip_type_annot(token: str) -> str:
    t = token.strip()
    if " : " in t:
        return t.split(" : ", 1)[0].strip()
    if ":" in t:
        return t.split(":", 1)[0].strip()
    return t

def _extract_paren_group(text: str, kw: str) -> Optional[str]:
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

def _parse_ins_outs(operands_part: str) -> Optional[Tuple[List[str], List[str]]]:
    ins_blob = _extract_paren_group(operands_part, "ins")
    outs_blob = _extract_paren_group(operands_part, "outs")
    if ins_blob is None or outs_blob is None:
        return None
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
    ins = [_strip_type_annot(x) for x in _split_commas_top_level(ins_vals)]
    outs = [_strip_type_annot(x) for x in _split_commas_top_level(outs_vals)]
    return (ins, outs)

def _normalize_op_name(name: str) -> str:
    if name.startswith("pto."):
        return name
    return f"pto.{name}"

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
    if op_name in _NO_TILE_RESULT_OPS:
        return 0
    if op_name == "pto.getval":
        return 0
    if op_name in _TWO_TILE_RESULT_OPS:
        return 2
    if op_name in _ONE_TILE_PLUS_SCALAR_RESULT_OPS:
        return 1
    # Default: most PTO tile ops produce a single tile.
    if op_name.startswith("pto.t") or op_name in {"pto.mgather"}:
        return 1
    return 0

def _normalize_dps_to_ssa(func: Func) -> Func:
    """
    Normalize a mix of SSA-style and DPS/register-style PTO into a canonical form:
    - Tile-producing ops expose their destination tile(s) in `Op.results`.
    - DPS/register-style ops use destination tile(s) as leading operands; those are
      moved into results and removed from operands.

    This keeps downstream tooling (codegen, analyzers) simple while allowing
    multiple surface syntaxes.
    """
    norm_ops: List[Op] = []
    for op in func.ops:
        tile_outs = _tile_result_arity(op.name)
        if tile_outs == 0:
            norm_ops.append(op)
            continue

        # If the op already has enough SSA results, keep it as-is.
        if len(op.results) >= tile_outs:
            norm_ops.append(op)
            continue

        if len(op.operands) < tile_outs:
            raise ValueError(f"{op.name} expects {tile_outs} destination tile operand(s): {op}")

        dsts = list(op.operands[:tile_outs])
        rest = list(op.operands[tile_outs:])
        results = dsts + list(op.results)
        norm_ops.append(
            Op(results=results, name=op.name, operands=rest, wait=op.wait, attrs=op.attrs, type_sig=op.type_sig)
        )
    return Func(
        args=func.args,
        arg_order=func.arg_order,
        tile_decls=func.tile_decls,
        consts=func.consts,
        ops=norm_ops,
    )


def parse_pto_mlir(path: str | Path) -> Func:
    text = Path(path).read_text(encoding="utf-8")
    lines = [l.strip() for l in text.splitlines()]

    args: Dict[str, str] = {}
    arg_order: List[str] = []
    tile_decls: Dict[str, str] = {}
    consts: Dict[str, Tuple[str, str]] = {}
    ops: List[Op] = []
    in_func = False

    for line in lines:
        if line.startswith("//") or line.startswith(";") or line.startswith("#"):
            continue
        if not line or line.startswith("module") or line == "{" or line == "}":
            continue

        mfunc = _RE_FUNC.search(line)
        if mfunc:
            in_func = True
            arg_blob = mfunc["args"].strip()
            if arg_blob:
                parts: List[str] = []
                cur: List[str] = []
                angle = 0
                for ch in arg_blob:
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

                for part in parts:
                    part = part.strip()
                    marg = _RE_ARG.fullmatch(part)
                    if not marg:
                        raise ValueError(f"Bad arg syntax: {part}")
                    args[marg["ssa"]] = marg["ty"].strip()
                    arg_order.append(marg["ssa"])
            continue

        if not in_func:
            continue
        if line == "return":
            break
        mconst = _RE_CONST.match(line)
        if mconst:
            consts[mconst["ssa"]] = (mconst["lit"], mconst["ty"])
            continue

        mop = _RE_OP.match(line)
        if not mop:
            continue

        lhs = mop.group("lhs")
        results: List[str] = []
        if lhs:
            results = [x.strip() for x in lhs.split(",")]

        rest = mop.group("rest")
        wait: List[str] = []
        mw = _RE_WAIT.search(rest)
        if mw:
            wait = [x.strip() for x in mw["list"].split(",") if x.strip()]

        attrs: Dict[str, str] = {}
        # IMPORTANT: `ins(...)` / `outs(...)` assembly uses `:` inside the
        # parentheses for operand type annotations. Do not treat that as the
        # MLIR trailing type signature.
        if " : " in rest and ("ins(" not in rest and "outs(" not in rest):
            operands_part, type_sig = rest.split(" : ", 1)
        elif rest.startswith(":"):
            operands_part, type_sig = ("", rest[1:].strip())
        else:
            # Custom-assembly ops (e.g. `ins(...) outs(...)`) may omit the MLIR
            # trailing type signature.
            operands_part, type_sig = (rest, "")

        # Parse an optional MLIR-style attribute dict `{...}` in the operand section.
        attr_blob = ""
        if "{" in operands_part and "}" in operands_part:
            before, after = operands_part.split("{", 1)
            attr_blob, _tail = after.split("}", 1)
            operands_part = before.strip()
            # Parse key = value pairs (as raw strings) for codegen.
            for item in _split_commas_top_level(attr_blob):
                if "=" not in item:
                    continue
                k, v = [x.strip() for x in item.split("=", 1)]
                attrs[k] = v
        else:
            operands_part = operands_part.strip()

        ins_outs = _parse_ins_outs(operands_part)
        if ins_outs is not None:
            ins_ops, outs_ops = ins_outs
            # Normalize to our internal representation:
            # - Tile-producing ops: `results = outs`, `operands = ins`
            # - Stores/scatters: `operands = [outs0] + ins`
            op_name = mop.group("op")
            if op_name in {"pto.tstore", "pto.mscatter", "pto.tstore.fp"}:
                results = []
                operands = [outs_ops[0]] + ins_ops
            else:
                results = list(outs_ops) + results
                operands = ins_ops
        else:
            operands = _split_commas_top_level(operands_part)

        ops.append(
            Op(results=results, name=mop.group("op"), operands=operands, wait=wait, attrs=attrs, type_sig=type_sig.strip())
        )

    if not args:
        raise ValueError("No func.func args parsed (expected a single func.func @main)")
    return _normalize_dps_to_ssa(Func(args=args, arg_order=arg_order, tile_decls=tile_decls, consts=consts, ops=ops))


def parse_pto_as(path: str | Path) -> Func:
    """
    Parse PTO-AS as emitted by `pto-isa-main/examples/output_pto/*.pto`.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [l.rstrip() for l in text.splitlines()]

    memrefs: Dict[str, str] = {}
    memref_order: List[str] = []
    tile_decls: Dict[str, str] = {}
    consts: Dict[str, Tuple[str, str]] = {}
    ops: List[Op] = []

    for raw in lines:
        line = raw.strip()
        m = _RE_DECL.match(line)
        if not m:
            continue
        ssa = m["ssa"]
        ty = m["ty"].strip()
        if ty.startswith("!pto.memref<"):
            if ssa not in memrefs:
                memrefs[ssa] = ty
                memref_order.append(ssa)
        elif ty.startswith("!pto.tile<") or ty.startswith("!pto.tilebuf<"):
            tile_decls[ssa] = ty

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("//") or line.startswith(";") or line.startswith("#"):
            continue

        results: List[str] = []
        op_name = ""
        rest = ""

        massign = _RE_PTOAS_ASSIGN.match(line)
        if massign:
            results = [massign["lhs"]]
            op_name = massign["op"]
            rest = massign["rest"].strip()
        else:
            parts = line.split(None, 1)
            op_name = parts[0]
            rest = parts[1].strip() if len(parts) > 1 else ""

        # Optional type signature after ':' (not required e.g. for tstore in pto-isa-main).
        type_sig = ""
        if " : " in rest:
            rest, type_sig = rest.split(" : ", 1)
            type_sig = type_sig.strip()

        # Optional `{...}` attrs.
        attrs: Dict[str, str] = {}
        if "{" in rest and "}" in rest:
            before, after = rest.split("{", 1)
            attr_blob, tail = after.split("}", 1)
            rest = (before + " " + tail).strip()
            for item in _split_commas_top_level(attr_blob):
                if "=" not in item:
                    continue
                k, v = [x.strip() for x in item.split("=", 1)]
                attrs[k] = v

        operands = _split_commas_top_level(rest) if rest else []
        name = _normalize_op_name(op_name)

        # Normalize operand order for tstore: want (memref, tile) for TSTORE(Global, Tile).
        if name == "pto.tstore" and len(operands) == 2:
            a, b = operands[0].strip(), operands[1].strip()
            if "[" in b:
                operands = [b, a]

        # Capture `%1.0`-style immediates in scalar ops.
        if name in {"pto.tadds", "pto.tsubs", "pto.tmuls", "pto.tdivs"} and len(operands) == 2 and type_sig:
            parts = [p.strip() for p in type_sig.split(",")]
            if len(parts) >= 2:
                scalar_ty = parts[1]
                imm = operands[1].strip()
                if imm.startswith("%") and imm not in consts and imm not in tile_decls and imm not in memrefs:
                    consts[imm] = (imm[1:], scalar_ty)

        # Discover memrefs from operands if not declared.
        for operand in operands:
            if operand.startswith("%") and "[" in operand:
                ssa = operand.split("[", 1)[0]
                if ssa not in memrefs:
                    memrefs[ssa] = "!pto.memref<gm,...,f32>"
                    memref_order.append(ssa)

        ops.append(Op(results=results, name=name, operands=operands, wait=[], attrs=attrs, type_sig=type_sig))

    if not memrefs:
        raise ValueError("No memrefs declared or discovered in PTO-AS file")
    return _normalize_dps_to_ssa(Func(args=memrefs, arg_order=memref_order, tile_decls=tile_decls, consts=consts, ops=ops))


def parse_pto(path: str | Path) -> Func:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("//") or s.startswith("#") or s.startswith(";"):
            continue
        if s.startswith("module") or s.startswith("func.func"):
            return parse_pto_mlir(path)
        return parse_pto_as(path)
    raise ValueError("Empty .pto file")
