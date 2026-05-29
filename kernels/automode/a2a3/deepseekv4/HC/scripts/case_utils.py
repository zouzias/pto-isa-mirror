#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the HC (Hyper-Connections) family.

Mirrors CSA/scripts/case_utils.py and HCA/Compressor/scripts/case_utils.py.
Each per-leaf gen_data.py imports load_hc_case() from this file.
"""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <HC>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_hc_case(idx: int = 0) -> Dict[str, int]:
    """Return the idx-th case from the family JSON manifest, or fall back to
    the historical hardcoded default if the manifest is absent.

    The default MUST match DEFAULT_CASES in generate_cases.py.
    """
    manifest = _manifest_path()
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if 0 <= idx < len(payload):
            c = payload[idx]
            c.setdefault("mix_hc", (2 + c["hc_mult"]) * c["hc_mult"])
            c.setdefault("eps", 1e-6)
            return c
    # Fallback: DEFAULT_CASES[0] in generate_cases.py
    return {
        "name": "case_B1_S16_D64_HC4_I4",
        "b": 1, "s": 16, "dim": 64, "hc_mult": 4, "iters": 4,
        "mix_hc": 24, "eps": 1e-6,
    }
