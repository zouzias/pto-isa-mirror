#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for the Indexer family.
Mirrors HCA/Compressor/scripts/case_utils.py."""
import json
from pathlib import Path
from typing import Dict


def _manifest_path() -> Path:
    # <Indexer>/build/generated_cases.json
    return Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"


def load_indexer_case(idx: int = 0) -> Dict[str, int]:
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
        "name": "case_B1_S64_T32_D128_QLR128_H8_HD64_K16",
        "b": 1, "s": 64, "t": 32, "dim": 128,
        "q_lora_rank": 128, "n_heads": 8,
        "head_dim": 64, "index_topk": 16,
    }
