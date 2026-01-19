from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Pipe(str, Enum):
    MTE1 = "mte1"
    MTE2 = "mte2"
    MTE3 = "mte3"
    V = "v"
    M = "m"
    S = "s"
    FIX = "fix"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OpTag:
    """
    Event-tag used for Event<Op::SrcTag, Op::DstTag>.

    This is intentionally *coarser* than the full instruction set:
    not every PTO intrinsic has a dedicated Op enum tag, but the runtime
    provides generic tags like VECTOR for vector pipeline ops.
    """

    op_enum: str  # e.g. "TLOAD" | "VECTOR" | "TSTORE_VEC" | "TMATMUL"
    pipe: Pipe


def classify_pto_op(op_name: str) -> OpTag:
    """
    Classify a `pto.*` op into an event-tag and pipeline bucket.

    This enables safe, conservative auto-sync between the major pipelines
    without requiring per-instruction tags for the full ISA.
    """
    if not op_name.startswith("pto."):
        raise ValueError(f"Expected dialect op like 'pto.tadd', got: {op_name}")
    short = op_name.split(".", 1)[1]

    # Memory pipeline ops
    if short in {"tload", "mgather"} or short.startswith("tload."):
        return OpTag("TLOAD", Pipe.MTE2)
    if short in {"tstore", "mscatter"} or short.startswith("tstore."):
        return OpTag("TSTORE_VEC", Pipe.MTE3)

    # Cube pipeline (best-effort)
    if short.startswith("tmatmul"):
        return OpTag("TMATMUL", Pipe.M)

    # Default: vector pipeline
    return OpTag("VECTOR", Pipe.V)
