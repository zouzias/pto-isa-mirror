from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

from .ir import Function, Module, Operation, Value


def _print_attr_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_print_attr_value(x) for x in v) + "]"
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float)):
        return str(v)
    raise TypeError(f"Unsupported attr value: {type(v)}")


def _print_attr_dict(attrs: Mapping[str, object]) -> str:
    if not attrs:
        return ""
    parts = [f"{k} = {_print_attr_value(v)}" for k, v in attrs.items()]
    return " {" + ", ".join(parts) + "}"


def _print_results(results: Sequence[Value]) -> str:
    if not results:
        return ""
    names = ", ".join(v.name for v in results)
    return f"{names} = "


def _print_operands(operands: Sequence[str]) -> str:
    if not operands:
        return ""
    # `ins(...) outs(...)` style: keep as space-separated groups.
    if operands[0].startswith("ins(") or operands[0] == "ins":
        return " ".join(operands)
    return ", ".join(operands)


def print_op(op: Operation) -> str:
    operands = _print_operands(op.operands)
    line = _print_results(op.results) + op.op
    if operands:
        line += " " + operands
    line += _print_attr_dict(op.attrs)
    if op.type_sig:
        line += " : " + op.type_sig
    return line


def print_func(fn: Function) -> str:
    args = ", ".join(f"{a.name}: {a.type_str}" for a in fn.args)
    lines: List[str] = []
    lines.append(f"  func.func @{fn.name}({args}) {{")
    for op in fn.ops:
        if op.op == "builtin.return":
            lines.append("    return")
        else:
            lines.append("    " + print_op(op))
    lines.append("  }")
    return "\n".join(lines)


def print_module(m: Module) -> str:
    lines: List[str] = []
    lines.append("module {")
    for fn in m.funcs:
        lines.append(print_func(fn))
    lines.append("}")
    return "\n".join(lines) + "\n"
