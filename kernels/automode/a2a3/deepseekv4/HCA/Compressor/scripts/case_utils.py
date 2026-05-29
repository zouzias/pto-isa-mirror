#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the Compressor family. Mirrors MoE/scripts/case_utils.py."""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <Compressor>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_compressor_case(idx: int = 0) -> Dict[str, int]:
    """Return the idx-th case from the family JSON manifest, or fall back
    to the historical hardcoded default if the manifest is absent.

    The default MUST match DEFAULT_CASES in generate_cases.py — see the
    "Shape-source pitfall" rule in docs_for_ai/kernel_test_guidance.md §2.1.
    """
    manifest = _manifest_path()
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if 0 <= idx < len(payload):
            c = payload[idx]
            c.setdefault("coff", 2 if c.get("overlap", 0) else 1)
            return c
    # Fallback: DEFAULT_CASES[0] in generate_cases.py
    return {
        "name": "case_B1_S128_D128_HD64_RD16_R4_O1",
        "b": 1, "s": 128, "dim": 128, "head_dim": 64, "rope_dim": 16,
        "compress_ratio": 4, "overlap": 1, "coff": 2,
    }
