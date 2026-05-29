#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the CSA (sparse_attn) family.

Mirrors HCA/Compressor/scripts/case_utils.py. Each per-leaf gen_data.py
imports load_csa_case() from this file.
"""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <CSA>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_csa_case(idx: int = 0) -> Dict[str, int]:
    """Return the idx-th case from the family JSON manifest, or fall back to
    the historical hardcoded default if the manifest is absent.

    The default MUST match DEFAULT_CASES in generate_cases.py.
    """
    manifest = _manifest_path()
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if 0 <= idx < len(payload):
            c = payload[idx]
            c.setdefault("num_blocks", (c["s"] + c["block"] - 1) // c["block"])
            return c
    # Fallback: DEFAULT_CASES[0] in generate_cases.py
    return {
        "name": "case_H4_D32_S64_B32",
        "h": 4, "d": 32, "s": 64, "block": 32, "num_blocks": 2,
    }
