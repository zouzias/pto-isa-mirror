from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Value:
    name: str  # includes leading '%'
    type_str: str


@dataclass(frozen=True)
class Operation:
    results: Sequence[Value]
    op: str  # e.g. "pto.tadd"
    operands: Sequence[str]  # SSA names and/or indexed forms like "%a[%c0,%c0]"
    attrs: Dict[str, object] = field(default_factory=dict)
    type_sig: Optional[str] = None


@dataclass
class Function:
    name: str
    args: Sequence[Value]
    ops: List[Operation]


@dataclass
class Module:
    funcs: List[Function]
