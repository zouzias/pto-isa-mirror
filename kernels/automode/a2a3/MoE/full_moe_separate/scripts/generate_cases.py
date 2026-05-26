#!/usr/bin/env python3
# coding=utf-8

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "generate_cases.py"), run_name="__main__")
