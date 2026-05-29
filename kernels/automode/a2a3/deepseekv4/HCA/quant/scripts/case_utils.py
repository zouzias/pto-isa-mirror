#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the quant family.
Mirrors HCA/Compressor/scripts/case_utils.py."""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <quant>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_quant_case(idx: int = 0) -> Dict[str, int]:
    """Return the idx-th case from the family JSON manifest, or fall back
    to the historical hardcoded default if the manifest is absent.

    The default MUST match DEFAULT_CASES[0] in generate_cases.py.
    """
    manifest = _manifest_path()
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if 0 <= idx < len(payload):
            return payload[idx]
    # Fallback: DEFAULT_CASES[0] in generate_cases.py
    return {
        "name": "case_M32_N128_B128_I1",
        "m": 32, "n": 128, "block_size": 128, "inplace": 1,
    }
