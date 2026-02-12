#!/usr/bin/env python3
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

"""Pipeline timeline extraction from hardware logs.

Parses start/end instruction logs from cube/vector cores and correlates them
with buffer addresses from device_addrs.toml. Emits CSV/JSON with per-
instruction timing, buffer hits, and coarse op-class (load/compute/store).
Optionally aggregates per buffer/op-class.

Typical use:
    python pipeline_log_analysis.py \
        --device-addrs ../TFATest.case_float_H_128_S0_64_S1_128/device_addrs.toml \
        --cube-start ../TFATest.case_float_H_128_S0_64_S1_128/core0.cubecore0.instr_popped_log.dump \
        --cube-end   ../TFATest.case_float_H_128_S0_64_S1_128/core0.cubecore0.instr_log.dump \
        --vec-start  ../TFATest.case_float_H_128_S0_64_S1_128/core0.veccore0.instr_popped_log.dump \
        --vec-end    ../TFATest.case_float_H_128_S0_64_S1_128/core0.veccore0.instr_log.dump \
        --out-csv timeline.csv --out-json timeline.json

Notes
- Start logs (*.instr_popped_log.dump) carry the issue timestamps; end logs
  (*.instr_log.dump) carry completion timestamps. This script aligns them by
  line order. If your logs differ, extend the matcher to align by PC/opcode.
- Buffer correlation is best-effort: if any extracted address falls within a
  buffer range from device_addrs.toml, that buffer name is attached.
- Op classification is heuristic via opcode substrings; adjust lists below
  (LOAD_OPS, STORE_OPS, COMPUTE_OPS) to match your target ISA mnemonics.
"""

import argparse
import csv
import json
import math
import re
import sys
import bisect
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Simple representation for SVG tasks derived from timeline entries
@dataclass
class SvgTask:
    name: str
    core: str
    tile: int
    start: int
    end: int

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

# Heuristic opcode buckets; extend as needed for your traces.
LOAD_OPS = (
    "LOAD_",  # LOAD_L1_TO_DST_*, LOAD_2D, etc.
    "LD_",    # LD_XD_XN_IMM, LDP_XI_XJ_XN, etc.
    "MOV_OUT_TO_L1",  # MOV_OUT_TO_L1_MULTI_*
)

STORE_OPS = (
    "STORE_",  # generic store pattern
    "ST_",     # ST_XD_XN_IMM, STI_XN_IMM, etc.
    "MOV_L1_TO_OUT",
)

COMPUTE_OPS = (
    "MMAD",         # cube matmul
    # Vector compute families we care to surface
    "VMAX",
    "VCMAX",
    "VCGMAX",    
    "VADD",
    "VSUB",
    "VBRCB",
    "VMULS",
    "VEXP",
    "VCADD",
    "VCGADD",    
    "VCONV",
    "VDIV",
    "VMUL",
)

ADDR_REGEX = re.compile(r"(?:XN|XM|XD|XT|Src|Dst|BASE|Addr|SPR)(?::[A-Za-z0-9]+)?[:=]0x([0-9a-fA-F]+)")
TS_REGEX = re.compile(r"\[(\d+)\]")
PC_REGEX = re.compile(r"PC:\s*0x([0-9a-fA-F]+)")
# Capture the first token after ")" (pipeline), and later refine to the actual mnemonic
OP_REGEX = re.compile(r"\)\s*([A-Z0-9_]+)")
# Capture mnemonic after one or more parenthesized metadata blocks
# e.g. "... CUBE : (Binary: ...) (ID: ...) MMAD ..."
OPCODE_AFTER_BINARY_REGEX = re.compile(r"\)\s*[A-Z0-9_]+\s*:\s*(?:\([^)]*\)\s*)+([A-Z][A-Z0-9_]+)")
ID_REGEX = re.compile(r"\b(?:id|instr_id)\s*[:=]\s*(\d+)\b", re.IGNORECASE)
FIX_L0C_TO_DST_REGEX = re.compile(r"FIX_L0C_TO_DST")
MOV_UB_TO_L1_REGEX = re.compile(r"MOV_UB_TO_L1")

@dataclass
class Instr:
    ts_start: int
    ts_end: Optional[int]
    pc: Optional[int]
    pipeline: str
    opcode: str
    buffer: Optional[str]
    global_buffer: Optional[str]
    buffer_addr: Optional[int]
    op_class: str  # load/compute/store/other
    stage: Optional[str]
    addresses: List[int]
    line_start: int
    line_end: Optional[int]
    core: str  # cube or vector
    instr_id: Optional[int] = None
    stage_name: Optional[str] = None


def load_device_addrs(path: Path) -> Dict[str, Tuple[int, int]]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    result: Dict[str, Tuple[int, int]] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        addr = entry.get("addr")
        size = entry.get("size_bytes")
        if isinstance(addr, str) and addr.startswith("0x") and isinstance(size, (int, float)):
            try:
                result[name] = (int(addr, 16), int(size))
            except ValueError:
                continue
    return result


def _file_contains(path: Optional[Path], pattern: re.Pattern[str]) -> bool:
    if not path or not path.exists():
        return False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if pattern.search(line):
                return True
    return False


def detect_fifo_mode(cube_end: Path, vec_end: Path, vec_mte3_issque: Optional[Path]) -> int:
    has_fix_l0c = _file_contains(cube_end, FIX_L0C_TO_DST_REGEX)
    has_mov_ub_to_l1 = _file_contains(vec_end, MOV_UB_TO_L1_REGEX) or _file_contains(vec_mte3_issque, MOV_UB_TO_L1_REGEX)
    if has_fix_l0c and has_mov_ub_to_l1:
        return 1
    if has_fix_l0c:
        return 2
    return 0


def classify_op(opcode: str, core: str, pipeline: str = "", line_text: str = "", fifo_mode: int = 0) -> str:
    # Control/flag plumbing is synchronization, not memory or compute
    if any(tag in line_text for tag in ("SET_CROSS_CORE", "SET_FLAG", "WAIT_FLAG", "WAIT_FLAG_DEVI")):
        return "sync"

    pipe = pipeline

    # Surface known compute ops early to avoid mislabeling as FIXP store
    if opcode.startswith("RV_") or pipeline.startswith("RVEC"):
        return "compute"
    if any(opcode.startswith(prefix) for prefix in COMPUTE_OPS):
        return "compute"
    if core == "cube" and (opcode.startswith("MMAD") or pipeline.startswith("CUBE")):
        return "compute"

    # Core/pipeline-based overrides for memory ops
    if core == "cube":
        if pipe.startswith("MTE2"):
            return "load"  # cube TLOAD
        if pipe.startswith("MTE1") and fifo_mode in (1, 2):
            return "load"
        # FIXP pipeline is treated as store (cube TSTORE / L0C->UB)
        if pipe.startswith("FIXP"):
            return "store"
    if core == "vector":
        if pipe.startswith("MTE2"):
            return "load"  # vector TLOAD
        if pipe.startswith("MTE3"):
            return "store"  # vector TSTORE / UB->L1 in mode1

    return "other"


def parse_log(path: Path, core: str) -> List[Dict]:
    events: List[Dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            ts_match = TS_REGEX.search(line)
            pc_match = PC_REGEX.search(line)
            op_match = OP_REGEX.search(line)
            if not ts_match or not op_match:
                continue
            ts = int(ts_match.group(1))
            pc = int(pc_match.group(1), 16) if pc_match else None
            pipeline_token = op_match.group(1)
            opcode = pipeline_token
            # If pipeline token, grab the real mnemonic after the binary blob
            op2 = OPCODE_AFTER_BINARY_REGEX.search(line)
            if op2:
                opcode = op2.group(1)
            # Filter: drop SCALAR entirely unless it is WAIT_FLAG_DEVI (treated as sync)
            if pipeline_token == "SCALAR":
                if "WAIT_FLAG_DEVI" not in line:
                    continue
            # For FC instructions, keep only WAIT_FLAG_DEVI; drop others.
            if pipeline_token == "FC" and "WAIT_FLAG_DEVI" not in line:
                continue
            # Ignore FIXP MOV_SPR_XN plumbing (no memory touch, noisy)
            if pipeline_token == "FIXP" and (opcode.startswith("MOV_SPR_XN") or "MOV_SPR_XN" in line):
                continue
            # Ignore MOVEMASK utility ops (no memory touch, noisy)
            if "MOVEMASK" in line:
                continue
            # Ignore MTE1 pipe instructions (internal buffer plumbing)
            if pipeline_token == "MTE1" or opcode.startswith("MTE1"):
                continue
            # Ignore BAR synchronization markers
            if "BAR" in line:
                continue
            addrs = [int(m, 16) for m in ADDR_REGEX.findall(line)]
            id_match = ID_REGEX.search(line)
            instr_id = int(id_match.group(1)) if id_match else None
            events.append({
                "ts": ts,
                "pc": pc,
                "pipeline": pipeline_token,
                "opcode": opcode,
                "addresses": addrs,
                "line_text": line,
                "line": idx,
                "core": core,
                "id": instr_id,
            })
    return events


def match_start_end(start_events: List[Dict], end_events: List[Dict]) -> List[Tuple[Dict, Dict]]:
    # First pass: pair by instruction id when present
    start_by_id: Dict[int, List[Dict]] = {}
    end_by_id: Dict[int, List[Dict]] = {}
    for ev in start_events:
        if ev.get("id") is not None:
            start_by_id.setdefault(ev["id"], []).append(ev)
    for ev in end_events:
        if ev.get("id") is not None:
            end_by_id.setdefault(ev["id"], []).append(ev)

    pairs: List[Tuple[Dict, Dict]] = []
    used_start = set()
    used_end = set()
    unmatched_start_ids: List[Dict] = []

    for instr_id, s_list in start_by_id.items():
        e_list = end_by_id.get(instr_id, [])
        if not e_list:
            unmatched_start_ids.extend(s_list)
            continue
        for s_ev, e_ev in zip(s_list, e_list):
            pairs.append((s_ev, e_ev))
            used_start.add(id(s_ev))
            used_end.add(id(e_ev))

    unmatched_end_ids = [ev for ev in end_events if ev.get("id") is not None and id(ev) not in used_end]
    if unmatched_start_ids or unmatched_end_ids:
        sys.stderr.write(
            f"[warn] unmatched events with id after id-based pairing: start={len(unmatched_start_ids)} end={len(unmatched_end_ids)}\n"
        )

    # Second pass: pair remaining events (only those lacking ids) in order
    rem_start = [ev for ev in start_events if ev.get("id") is None and id(ev) not in used_start]
    rem_end = [ev for ev in end_events if ev.get("id") is None and id(ev) not in used_end]
    if len(rem_start) != len(rem_end):
        min_len = min(len(rem_start), len(rem_end))
        sys.stderr.write(
            f"[warn] unmatched start/end after id pairing: start={len(rem_start)} end={len(rem_end)}; truncating to {min_len}\n"
        )
        rem_start = rem_start[:min_len]
        rem_end = rem_end[:min_len]

    pairs.extend(zip(rem_start, rem_end))
    return pairs


def map_buffer(opcode: str, addrs: List[int], line_text: str, buf_map: Dict[str, Tuple[int, int]]) -> Tuple[Optional[str], Optional[int]]:
    # Prefer XN address for MOV_SRC_TO_DST_ALIGN (captures Src:OUT, Dst:UB transfers)
    if opcode.startswith("MOV_SRC_TO_DST_ALIGN"):
        xn_match = re.search(r"XN(?::[A-Za-z0-9]+)?\s*[:=]\s*0x([0-9a-fA-F]+)", line_text)
        if xn_match:
            xn_addr = int(xn_match.group(1), 16)
            for name, (base, size) in buf_map.items():
                if base <= xn_addr < base + size:
                    return name, xn_addr
    # Fallback: any captured address that hits a buffer range
    for a in addrs:
        for name, (base, size) in buf_map.items():
            if base <= a < base + size:
                return name, a
    return None, None


def infer_stage(
    buffer: Optional[str],
    opcode: str,
    op_class: str,
    line_text: str,
    global_buffer: Optional[str],
    core: str,
    fifo_mode: int = 0,
    pc: Optional[int] = None,
) -> Optional[str]:
    # Only infer stage for load/store; compute handled later
    if op_class not in ("load", "store"):
        return None

    # Correlate stage based on buffers/opcodes/core
    buf_name = (global_buffer or buffer or "").lower()
    haystack = " ".join(filter(None, [buf_name, opcode.lower(), line_text.lower()]))

    qk_bufs = {"q_device", "k_device", "qk_device", "qk_tile", "qk_tile_fifo"}
    pv_bufs = {"v_device", "p_tile_fifo", "pv_tile", "pv_tile_fifo"}

    # Prefer explicit compute op mapping for cube MMAD (compute_qk)
    if core == "cube" and op_class == "compute" and opcode.startswith("MMAD"):
        return "compute_qk"

    if core == "cube":
        if buf_name in pv_bufs:
            return "compute_pv"
        if buf_name in qk_bufs:
            return "compute_qk"
        if fifo_mode in (1, 2) and "fix_l0c_to_dst" in haystack:
            # PC clusters observed in current A5 MODE1 traces:
            # qk path: 0x...b248 / 0x...b30c / 0x...b50c
            # pv path: 0x...b37c
            if pc is not None:
                low = pc & 0xFFF
                if low in (0x248, 0x30C, 0x50C):
                    return "compute_qk"
                if low in (0x37C,):
                    return "compute_pv"
            return "compute_qk"

    if core == "vector":
        if fifo_mode == 1 and "mov_ub_to_l1" in haystack:
            return "compute_p"
        if op_class == "store" and (
            "dst:out" in line_text.lower()
            or "mov_src_to_dst_align" in haystack
            or "o_out" in haystack
        ):
            return "compute_gu"
        if any(token in haystack for token in ("qk_tile", "qk_tile_fifo", "p_tile", "p_tile_fifo")):
            return "compute_p"

    # GU stage: pv_tile loads
    if buf_name in {"pv_tile", "pv_tile_fifo"}:
        return "compute_gu"

    return None


def infer_stage_compute(instrs: List[Instr], fifo_mode: int = 0) -> None:
    def cube_stage_from_pc(pc: Optional[int]) -> Optional[str]:
        if pc is None:
            return None
        low = pc & 0xFFF
        # PC clusters observed in current A5 MODE1 traces:
        # qk MMAD: 0x...b20c / 0x...b2e8 / 0x...b48c
        # pv MMAD: 0x...b5f0
        if low in (0x20C, 0x2E8, 0x48C):
            return "compute_qk"
        if low in (0x5F0,):
            return "compute_pv"
        return None

    def vector_stage_from_pc_mode12(pc: Optional[int]) -> Optional[str]:
        if pc is None:
            return None
        low = pc & 0xF00
        if low in (0xE00, 0xF00):
            return "compute_p"
        if low in (0xC00, 0xD00):
            return "compute_gu"
        return None

    """Infer compute stage by bracketing with prior staged load and next staged store on the same core.

    If a compute op lacks a stage, we look at the most recent load (with stage) before it
    and the next store (with stage) after it on the same core. If both stages match, use it.
    Otherwise prefer the previous load's stage, then the next store's stage.
    """
    cube_qk_load_bufs = {"k_device"}
    cube_qk_store_bufs = {"qk_tile_fifo"}
    cube_pv_load_bufs = {"p_tile_fifo"}
    cube_pv_store_bufs = {"pv_tile_fifo"}

    vec_p_load_bufs = {"qk_tile_fifo"}
    vec_p_store_bufs = {"p_tile_fifo"}

    def buf_kind(buf: Optional[str], core: str) -> Optional[str]:
        name = (buf or "").lower()
        if core == "cube":
            if name in cube_qk_load_bufs or name in cube_qk_store_bufs:
                return "compute_qk"
            if name in cube_pv_load_bufs or name in cube_pv_store_bufs:
                return "compute_pv"
            return None
        if core == "vector":
            if name in vec_p_load_bufs or name in vec_p_store_bufs:
                return "compute_p"
            if name:
                return "compute_gu"
            return None
        return None

    # Build vector compute_p windows: each compute_p load pairs with the next compute_p store.
    vec_p_loads = [ins for ins in instrs if ins.core == "vector" and ins.op_class == "load" and ins.stage == "compute_p"]
    vec_p_stores = [ins for ins in instrs if ins.core == "vector" and ins.op_class == "store" and ins.stage == "compute_p"]
    vec_p_loads.sort(key=lambda i: i.ts_start)
    vec_p_stores.sort(key=lambda i: i.ts_start)
    vec_p_windows: List[Tuple[int, int]] = []
    for ld, st in zip(vec_p_loads, vec_p_stores):
        vec_p_windows.append(((ld.ts_end or ld.ts_start), st.ts_start))
    if fifo_mode == 1 and not vec_p_windows and vec_p_stores:
        prev = 0
        for st in vec_p_stores:
            vec_p_windows.append((prev, st.ts_start))
            prev = st.ts_end or st.ts_start

    cube_stores = sorted(
        [ins for ins in instrs if ins.core == "cube" and ins.op_class == "store" and ins.stage],
        key=lambda i: i.ts_start,
    )

    next_store_kind: List[Optional[str]] = [None] * len(instrs)
    next_store_kind_by_core: Dict[str, Optional[str]] = {}

    # Reverse pass: nearest future store kind per core
    for i in range(len(instrs) - 1, -1, -1):
        ins = instrs[i]
        next_store_kind[i] = next_store_kind_by_core.get(ins.core)
        if ins.op_class == "store":
            kind = buf_kind(ins.global_buffer, ins.core)
            if kind:
                next_store_kind_by_core[ins.core] = kind

    # Forward pass: track last load kind per core, require load-before-store ordering
    last_load_kind_by_core: Dict[str, Optional[str]] = {}
    for i, ins in enumerate(instrs):
        if ins.core == "vector" and ins.op_class == "compute" and not ins.stage:
            if fifo_mode in (1, 2):
                by_pc = vector_stage_from_pc_mode12(ins.pc)
                if by_pc:
                    ins.stage = by_pc
                    continue
            if any(start <= ins.ts_start <= end for start, end in vec_p_windows):
                ins.stage = "compute_p"
            else:
                ins.stage = "compute_gu"
            continue

        if ins.op_class == "load" and ins.stage:
            kind = buf_kind(ins.global_buffer, ins.core)
            if kind:
                last_load_kind_by_core[ins.core] = kind
            continue

        if ins.op_class == "compute" and not ins.stage:
            if ins.core == "cube" and fifo_mode in (1, 2):
                by_pc = cube_stage_from_pc(ins.pc)
                if by_pc:
                    ins.stage = by_pc
                    continue

            prev_kind = last_load_kind_by_core.get(ins.core)
            next_kind = next_store_kind[i]

            # Only accept a bracket when load comes before and store after
            if prev_kind and next_kind and prev_kind == next_kind:
                ins.stage = prev_kind
                continue

            if ins.core == "cube":
                comp_end = ins.ts_end or ins.ts_start
                next_store = next((st for st in cube_stores if st.ts_start > comp_end), None)
                if next_store and next_store.stage:
                    ins.stage = next_store.stage
                    continue

            # No matching bracket; default per core
            ins.stage = "compute_gu" if ins.core == "vector" else "Unknown"


def assign_stage_names(instrs: List[Instr], fifo_mode: int = 0) -> None:
    """Assign looped stage names based on load completion order per stage.

    Rules:
    - compute_qk: increment loop after seeing both q_device and k_device loads for that loop.
    - compute_pv: increment loop after seeing both p_tile_fifo and v_device loads for that loop.
    - compute_p: each qk_tile_fifo load starts a new loop.
    - compute_gu: each pv_tile_fifo load starts a new loop.
    Compute/store ops take the most recently completed loop for their stage.
    """

    cfg = {
        "compute_qk": {
            "abbr": "qk",
            "trigger_loads": {"k_device"},  # q is reused; k load drives loop advance
            "label_loads": {"q_device", "k_device"},  # label q loads too
            "require_all": False,
        },
        "compute_pv": {
            "abbr": "pv",
            "trigger_loads": {"p_tile_fifo"},  # v_device load does not gate loop advance
            "label_loads": {"p_tile_fifo", "v_device"},
            "require_all": False,
        },
        "compute_p": {
            "abbr": "p",
            "trigger_loads": {"qk_tile_fifo"},
            "label_loads": {"qk_tile_fifo"},
            "require_all": False,
        },
        "compute_gu": {
            "abbr": "gu",
            "trigger_loads": {"pv_tile_fifo"},
            "label_loads": {"pv_tile_fifo"},
            "require_all": False,
        },
    }

    state: Dict[Tuple[str, str], Dict[str, object]] = {}

    for ins in instrs:
        stg = ins.stage
        if stg not in cfg:
            continue

        info = cfg[stg]
        abbr = info["abbr"]
        triggers = info["trigger_loads"]
        labels = info.get("label_loads", triggers)
        require_all = info["require_all"]
        key = (ins.core, stg)
        ctx = state.setdefault(key, {"current": 0, "seen": set(), "last_done": None, "store_idx": 0})

        buf_name = (ins.global_buffer or ins.buffer or "").lower()

        if fifo_mode in (1, 2):
            if "seq_idx" not in ctx:
                ctx["seq_idx"] = 0
            if "last_comp_end" not in ctx:
                ctx["last_comp_end"] = None
            seq_idx = int(ctx["seq_idx"])

            if stg == "compute_gu" and ins.op_class == "compute":
                last_comp_end = ctx.get("last_comp_end")
                if isinstance(last_comp_end, int) and ins.ts_start - last_comp_end > 200:
                    seq_idx += 1
                    ctx["seq_idx"] = seq_idx
                ctx["last_comp_end"] = ins.ts_end or ins.ts_start

            ins.stage_name = f"{abbr}{seq_idx}"
            if ins.op_class == "store":
                ctx["seq_idx"] = seq_idx + 1
            continue

        if ins.op_class == "load":
            if buf_name in labels:
                ins.stage_name = f"{abbr}{ctx['current']}"

            if buf_name in triggers:
                assign_idx = ctx["current"]
                ctx["seen"].add(buf_name)

                if (not require_all) or (triggers.issubset(ctx["seen"])):
                    ctx["last_done"] = assign_idx
                    ctx["current"] = assign_idx + 1
                    ctx["seen"] = set()
            continue

        # For compute, use per-stage compute order
        if ins.op_class == "compute":
            if ins.core == "cube":
                comp_idx = ctx.get("comp_idx", 0)
                ins.stage_name = f"{abbr}{comp_idx}"
                ctx["comp_idx"] = comp_idx + 1
            elif stg == "compute_p":
                comp_idx = ctx.get("store_idx", 0)  # tie vector compute_p to current store loop id
                ins.stage_name = f"{abbr}{comp_idx}"
            elif ctx["last_done"] is not None:
                ins.stage_name = f"{abbr}{ctx['last_done']}"

        # Store: label strictly by store order within the stage
        if ins.op_class == "store":
            ins.stage_name = f"{abbr}{ctx['store_idx']}"
            ctx["store_idx"] += 1


def _parse_stage_name(stage_name: str) -> Tuple[str, int]:
    base = "".join(ch for ch in stage_name if ch.isalpha())
    num_part = "".join(ch for ch in stage_name if ch.isdigit())
    idx = int(num_part) if num_part else -1
    return base or stage_name, idx


def svg_tasks_from_instrs(instrs: List[Instr]) -> List[SvgTask]:
    """Aggregate timeline entries into SVG tasks.

    Multiple loads within the same stage loop are emitted separately (load0, load1, ...)
    so pv0/qk0 multi-loads render as distinct blocks.
    """

    def core_for(stage: str, fallback: str) -> str:
        if stage in ("compute_qk", "compute_pv"):
            return "cube"
        if stage in ("compute_p", "compute_gu"):
            return "vector"
        return fallback or "vector"

    agg: Dict[Tuple[str, ...], Dict[str, object]] = {}
    load_seq: Dict[Tuple[str, int, str], int] = {}
    p_store_starts = sorted(
        ins.ts_start
        for ins in instrs
        if ins.stage == "compute_p" and ins.op_class == "store" and ins.pipeline.upper().startswith("MTE3")
    )
    pv_store_starts = sorted(
        ins.ts_start
        for ins in instrs
        if ins.stage == "compute_pv" and ins.op_class == "store"
    )

    def remap_idx(base: str, ts_start: int, idx: int) -> int:
        # In MODE1/MODE2, align PV/GU tiles by upstream ready boundaries.
        if base == "pv" and p_store_starts:
            return max(0, bisect.bisect_right(p_store_starts, ts_start) - 1)
        if base == "gu" and pv_store_starts:
            return max(0, bisect.bisect_right(pv_store_starts, ts_start) - 1)
        return idx

    for ins in instrs:
        if not ins.stage_name:
            continue
        base, idx = _parse_stage_name(ins.stage_name)
        idx = remap_idx(base, ins.ts_start, idx)
        if idx < 0:
            continue

        stage_label = f"{base}{idx}"

        if ins.op_class == "compute":
            key = ("compute", stage_label)
            name = f"{stage_label}_comp"
        elif ins.op_class in ("load", "store"):
            kind = ins.op_class
            unit = (ins.pipeline or "UNK").upper()
            if kind == "load":
                load_kind = "load"
                if base == "pv":
                    buf_name = (ins.global_buffer or ins.buffer or "").lower()
                    if "v_device" in buf_name:
                        load_kind = "vload"
                    elif "p_tile" in buf_name:
                        load_kind = "pload"
                seq_key = (base, idx, load_kind)
                seq = load_seq.setdefault(seq_key, 0)
                key = (f"{base}_{load_kind}{seq}_{unit}", idx, kind)
                name = f"{base}_{load_kind}{seq}_{unit}"
                load_seq[seq_key] = seq + 1
            else:
                key = (base, idx, "store", unit)
                name = f"{base}_store_{unit}"
        else:
            continue

        entry = agg.setdefault(
            key,
            {
                "start": ins.ts_start,
                "end": ins.ts_end or ins.ts_start,
                "core": core_for(ins.stage or "", ins.core),
                "name": name,
                "idx": idx,
            },
        )
        entry["start"] = min(entry["start"], ins.ts_start)
        entry["end"] = max(entry["end"], ins.ts_end or ins.ts_start)

    tasks: List[SvgTask] = []
    for info in agg.values():
        tasks.append(
            SvgTask(
                name=str(info.get("name")),
                core=str(info.get("core") or "vector"),
                tile=int(info.get("idx", -1)),
                start=int(info["start"]),
                end=int(info["end"]),
            )
        )

    return sorted(tasks, key=lambda t: (t.start, t.tile, t.name))


def render_svg(tasks: List[SvgTask], slot_width: int, output: Path, divisor: int = 10) -> None:
    if not tasks:
        return

    div = max(1, divisor)

    rows = {"qk": 60, "p": 120, "pv": 180, "gu": 240, "cube": 320, "vector": 380}
    height = 440
    tmax_raw = max(t.end for t in tasks)
    tmax = math.ceil(tmax_raw / div)
    width = int(120 + slot_width * tmax + 40)

    colors = {
        "qk": ("#1f77b4", "#1a4f7a"),
        "p": ("#ff9933", "#c66f0a"),
        "pv": ("#2ca25f", "#1f7a47"),
        "gu": ("#9467bd", "#6c3f8b"),
    }

    def base(name: str) -> str:
        first = name.split("_")[0]
        return first.rstrip("0123456789") or first

    def color_for(name: str) -> Tuple[str, str]:
        return colors.get(base(name), ("#888", "#555"))

    mini_h = 12
    mini_gap = 4

    def y_for(task: SvgTask) -> float:
        band_top = rows.get(base(task.name), rows.get(task.core, 320))
        suffix = task.name.split("_")[1] if "_" in task.name else "comp"
        if suffix.startswith(("load", "vload", "pload")):
            offset = 0
        elif suffix.startswith("comp"):
            offset = mini_h + mini_gap
        else:  # store or other
            offset = 2 * (mini_h + mini_gap)
        return band_top + offset

    def rect(task: SvgTask) -> str:
        x = 120 + slot_width * (task.start // div)
        w = slot_width * max(task.end // div - task.start // div, 1)
        y = y_for(task)
        fill, stroke = color_for(task.name)
        label = f"{task.name}({task.tile})"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    svg_parts: List[str] = []
    svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_parts.append(f'  <rect width="{width}" height="{height}" fill="#fff" />')
    svg_parts.append(
        "  <defs>\n"
        "    <marker id=\"arrowhead\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#333\" />\n"
        "    </marker>\n"
        "    <marker id=\"arrowhead-intra\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n"
        "      <polygon points=\"0 0 10 3.5 0 7\" fill=\"#1b75d1\" />\n"
        "    </marker>\n"
        "  </defs>"
    )
    svg_parts.append('  <text x="20" y="28" font-family="Arial" font-size="16" font-weight="bold" fill="#222">FA Pipeline Schedule (from log)</text>')

    svg_parts.append('  <g stroke="#f2f2f2" stroke-width="1">')
    for t in range(0, tmax + 1, 10):
        x = 120 + slot_width * t
        svg_parts.append(f'    <line x1="{x}" y1="40" x2="{x}" y2="280" />')
    svg_parts.append('  </g>')

    svg_parts.append('  <text x="20" y="80" font-family="Arial" font-size="13" fill="#222">compute_qk (cube)</text>')
    svg_parts.append('  <text x="20" y="140" font-family="Arial" font-size="13" fill="#222">compute_p (vector)</text>')
    svg_parts.append('  <text x="20" y="200" font-family="Arial" font-size="13" fill="#222">compute_pv (cube)</text>')
    svg_parts.append('  <text x="20" y="260" font-family="Arial" font-size="13" fill="#222">compute_gu (vector)</text>')
    svg_parts.append('  <text x="20" y="340" font-family="Arial" font-size="13" fill="#222">cube timeline (qk+pv)</text>')
    svg_parts.append('  <text x="20" y="400" font-family="Arial" font-size="13" fill="#222">vector timeline (p+gu)</text>')

    for task in tasks:
        svg_parts.append(rect(task))

    def rect_at(task: SvgTask, y_row: int) -> str:
        y = y_row + (y_for(task) - rows.get(base(task.name), rows.get(task.core, 320)))
        x = 120 + slot_width * (task.start // div)
        w = slot_width * max(task.end // div - task.start // div, 1)
        fill, stroke = color_for(task.name)
        label = f"{task.name}({task.tile})"
        return (
            f'  <rect x="{x}" y="{y}" width="{w}" height="{mini_h}" fill="{fill}" '
            f'rx="3" stroke="{stroke}" stroke-width="1" />\n'
            f'  <text x="{x + 4}" y="{y + mini_h - 3:.1f}" font-family="Arial" font-size="10" fill="#fff">{label}</text>'
        )

    for task in sorted(tasks, key=lambda t: (t.start, t.end)):
        if task.core == "cube":
            svg_parts.append(rect_at(task, rows["cube"]))
        else:
            svg_parts.append(rect_at(task, rows["vector"]))

    def find_pref(name: str, tile: int) -> List[SvgTask]:
        return sorted([tk for tk in tasks if tk.tile == tile and tk.name.startswith(name)], key=lambda t: t.start)

    def find_first(name: str, tile: int) -> Optional[SvgTask]:
        res = find_pref(name, tile)
        return res[0] if res else None

    def find_first_any(names: List[str], tile: int, min_start: Optional[int] = None) -> Optional[SvgTask]:
        for name in names:
            cands = find_pref(name, tile)
            if min_start is not None:
                cands = [c for c in cands if c.start >= min_start]
            if cands:
                t = cands[0]
                return t
        return None

    def adjust_overlap(y1: float, y2: float) -> Tuple[float, float]:
        # Keep arrows aligned to box centers without vertical staggering.
        return y1, y2

    def pick_dep(store_name: str, load_name: str, tile: int) -> Tuple[Optional[SvgTask], Optional[SvgTask]]:
        stores = find_pref(store_name, tile)
        loads = find_pref(load_name, tile)
        if not stores or not loads:
            return None, None
        store = max(stores, key=lambda t: t.end)  # last store in the stage
        load_candidates = [ld for ld in loads if ld.start >= store.end]
        load = load_candidates[0] if load_candidates else None
        return store, load

    for tile in range(max(t.tile for t in tasks) + 1):
        qk_store, p_load_dep = pick_dep("qk_store", "p_load0", tile)  # compute_qk store -> compute_p load
        if qk_store and not p_load_dep:
            p_load_dep = find_first_any(["p_load1", "p_comp"], tile, min_start=qk_store.end)
        if qk_store and p_load_dep and p_load_dep.start >= qk_store.end:
            x1 = 120 + slot_width * (((qk_store.start + qk_store.end) / 2) / div)
            x2 = 120 + slot_width * (p_load_dep.start / div)
            y1 = y_for(qk_store) + mini_h / 2
            y2 = y_for(p_load_dep) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1:.1f} L{x1:.1f} {y2-4:.1f} L{x2:.1f} {y2-4:.1f}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        p_src = find_first_any(["p_store", "p_comp"], tile)
        pv_load_dep = find_first_any(["pv_pload0", "pv_pload1", "pv_load0", "pv_comp"], tile, min_start=(p_src.end if p_src else None))
        if p_src and pv_load_dep and pv_load_dep.start >= p_src.end:
            x1 = 120 + slot_width * (((p_src.start + p_src.end) / 2) / div)
            x2 = 120 + slot_width * (pv_load_dep.start / div)
            y1 = y_for(p_src) + mini_h / 2
            y2 = y_for(pv_load_dep) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )
        pv_store, gu_load = pick_dep("pv_store", "gu_load", tile)  # compute_pv store -> compute_gu load
        if pv_store and not gu_load:
            gu_load = find_first_any(["gu_load0", "gu_comp"], tile, min_start=pv_store.end)
        if pv_store and gu_load and gu_load.start >= pv_store.end:
            x1 = 120 + slot_width * (((pv_store.start + pv_store.end) / 2) / div)
            x2 = 120 + slot_width * (gu_load.start / div)
            y1 = y_for(pv_store) + mini_h / 2
            y2 = y_for(gu_load) + mini_h / 2
            y1, y2 = adjust_overlap(y1, y2)
            svg_parts.append(
                f'  <path d="M{x1:.1f} {y1} L{x1:.1f} {y2-4} L{x2} {y2-4}" stroke="#333" stroke-width="1.5" fill="none" marker-end="url(#arrowhead)" />'
            )

        for base_name in ("qk", "p", "pv", "gu"):
            if base_name == "pv":
                load = find_first_any(["pv_pload", "pv_load", "pv_vload"], tile)
            else:
                load = find_first(f"{base_name}_load", tile)
            comp = find_first(f"{base_name}_comp", tile)
            store = find_first(f"{base_name}_store", tile)
            if load and comp and load.start != load.end and comp.start != comp.end:
                x1 = 120 + slot_width * (load.end / div)
                x2 = 120 + slot_width * (comp.start / div)
                y1 = y_for(load) + mini_h / 2
                y2 = y_for(comp) + mini_h / 2
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )
            if comp and store and comp.start != comp.end and store.start != store.end:
                x1 = 120 + slot_width * (comp.end / div)
                x2 = 120 + slot_width * (store.start / div)
                y1 = y_for(comp) + mini_h / 2
                y2 = y_for(store) + mini_h / 2
                svg_parts.append(
                    f'  <path d="M{x1:.1f} {y1} L{x2} {y2}" stroke="#1b75d1" stroke-width="1.2" fill="none" marker-end="url(#arrowhead-intra)" />'
                )

    svg_parts.append('</svg>')

    with output.open("w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

def to_instrs(paired: List[Tuple[Dict, Dict]], buf_map: Dict[str, Tuple[int, int]], fifo_mode: int = 0) -> List[Instr]:
    instrs: List[Instr] = []
    for start_ev, end_ev in paired:
        opcode = start_ev["opcode"]
        pipeline_token = start_ev.get("pipeline", opcode)
        op_class = classify_op(
            opcode,
            start_ev.get("core", "unknown"),
            pipeline_token,
            start_ev.get("line_text", ""),
            fifo_mode=fifo_mode,
        )
        buf, buf_addr = map_buffer(opcode, start_ev["addresses"], start_ev.get("line_text", ""), buf_map)
        stage = infer_stage(
            buf,
            opcode,
            op_class,
            start_ev.get("line_text", ""),
            buf,
            start_ev.get("core", ""),
            fifo_mode=fifo_mode,
            pc=start_ev.get("pc"),
        )
        instrs.append(
            Instr(
                ts_start=start_ev["ts"],
                ts_end=end_ev.get("ts"),
                pc=start_ev.get("pc"),
                pipeline=pipeline_token,
                opcode=opcode,
                buffer=buf,
                global_buffer=buf,
                buffer_addr=buf_addr,
                op_class=op_class,
                stage=stage,
                addresses=start_ev["addresses"],
                line_start=start_ev["line"],
                line_end=end_ev.get("line"),
                core=start_ev.get("core", "unknown"),
                instr_id=start_ev.get("id"),
            )
        )
    return instrs


def write_csv(instrs: List[Instr], path: Path) -> None:
    fieldnames = ["core", "ts_start", "ts_end", "pipeline", "opcode", "op_class", "stage", "stage_name", "buffer", "global_buffer", "buffer_addr", "addresses", "pc", "line_start", "line_end", "instr_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for instr in instrs:
            row = asdict(instr)
            row["addresses"] = ",".join(hex(a) for a in instr.addresses)
            row["pc"] = hex(instr.pc) if instr.pc is not None else ""
            row["buffer_addr"] = hex(instr.buffer_addr) if instr.buffer_addr is not None else ""
            writer.writerow(row)


def write_json(instrs: List[Instr], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in instrs], f, indent=2)


def aggregate(instrs: List[Instr]) -> List[Dict]:
    buckets: Dict[Tuple[str, str, str], Dict[str, int]] = {}
    for ins in instrs:
        key = (ins.core, ins.buffer or "unknown", ins.op_class)
        agg = buckets.setdefault(key, {"start": ins.ts_start, "end": ins.ts_end or ins.ts_start})
        agg["start"] = min(agg["start"], ins.ts_start)
        agg["end"] = max(agg["end"], ins.ts_end or ins.ts_start)
    summary = []
    for (core, buf, op_class), val in buckets.items():
        summary.append({"core": core, "buffer": buf, "op_class": op_class, "start": val["start"], "end": val["end"], "duration": val["end"] - val["start"]})
    return sorted(summary, key=lambda x: (x["core"], x["buffer"], x["start"]))


def write_aggregate_csv(summary: List[Dict], path: Path) -> None:
    fieldnames = ["core", "buffer", "op_class", "start", "end", "duration"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Parse TFA pipeline logs into timelines")
    ap.add_argument("--device-addrs", required=True, type=Path, help="Path to device_addrs.toml")
    ap.add_argument("--cube-start", required=True, type=Path, help="cube core start log (*.instr_popped_log.dump)")
    ap.add_argument("--cube-end", required=True, type=Path, help="cube core end log (*.instr_log.dump)")
    ap.add_argument("--vec-start", required=True, type=Path, help="vector core start log")
    ap.add_argument("--vec-end", required=True, type=Path, help="vector core end log")
    ap.add_argument("--vec-mte3-issque", type=Path, default=None, help="Optional vec mte3 issue queue dump for mode auto-detection")
    ap.add_argument("--fifo-mode", type=str, default="auto", choices=["auto", "0", "1", "2"], help="FIFO mode: auto/0/1/2")
    ap.add_argument("--out-csv", type=Path, default=Path("timeline.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("timeline.json"))
    ap.add_argument("--out-agg", type=Path, default=Path("timeline_agg.csv"))
    ap.add_argument("--out-svg", type=Path, default=None, help="Optional SVG timeline output rendered from parsed log")
    ap.add_argument("--svg-divisor", type=int, default=100, help="Divide timestamps by this factor for SVG scaling (e.g., 10 -> 10 cycles per unit)")
    args = ap.parse_args()

    fifo_mode = detect_fifo_mode(args.cube_end, args.vec_end, args.vec_mte3_issque) if args.fifo_mode == "auto" else int(args.fifo_mode)
    print(f"[info] fifo_mode={fifo_mode}")

    buf_map = load_device_addrs(args.device_addrs)
    cube_start = parse_log(args.cube_start, core="cube")
    cube_end = parse_log(args.cube_end, core="cube")
    vec_start = parse_log(args.vec_start, core="vector")
    vec_end = parse_log(args.vec_end, core="vector")

    cube_instrs = to_instrs(match_start_end(cube_start, cube_end), buf_map, fifo_mode=fifo_mode)
    vec_instrs = to_instrs(match_start_end(vec_start, vec_end), buf_map, fifo_mode=fifo_mode)
    instrs = sorted(cube_instrs + vec_instrs, key=lambda x: (x.ts_start, x.core))

    infer_stage_compute(instrs, fifo_mode=fifo_mode)
    assign_stage_names(instrs, fifo_mode=fifo_mode)

    write_csv(instrs, args.out_csv)
    write_json(instrs, args.out_json)
    write_aggregate_csv(aggregate(instrs), args.out_agg)

    if args.out_svg:
        tasks = svg_tasks_from_instrs(instrs)
        render_svg(tasks, slot_width=8, output=args.out_svg, divisor=args.svg_divisor)
        print(f"Wrote {args.out_svg}")

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_agg}")


if __name__ == "__main__":
    main()
