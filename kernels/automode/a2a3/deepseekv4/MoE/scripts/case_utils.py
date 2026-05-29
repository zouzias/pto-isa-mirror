#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the DeepSeek-V4 MoE family.

Each per-leaf gen_data.py imports load_dsmoe_case() from this file. The
fallback case must mirror DEFAULT_CASES[0] in generate_cases.py.
"""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <MoE>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_dsmoe_case(idx: int = 0) -> Dict[str, int]:
    """Return the idx-th case from the family JSON manifest, or fall back to
    the historical hardcoded default if the manifest is absent.

    The default MUST match DEFAULT_CASES[0] in generate_cases.py.
    """
    manifest = _manifest_path()
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if 0 <= idx < len(payload):
            return payload[idx]
    return {
        "name": "case_T64_D128_I256_E8_K2_V1024",
        "t": 64,
        "dim": 128,
        "inter_dim": 256,
        "n_routed": 8,
        "n_activated": 2,
        "vocab": 1024,
    }
