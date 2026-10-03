#!/usr/bin/env python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(relative):
    """Load the complete production module, including its real dependencies."""
    name = "host_regression_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("relative", ["tests/run_cpu.py", "tests/run_costmodel.py", "tests/script/cpu_bfloat16.py"])
@pytest.mark.parametrize("family", ["g++-14", "clang++-18", "aarch64-linux-gnu-g++-14"])
def test_explicit_compiler_keeps_installation(tmp_path, monkeypatch, relative, family):
    module = load_module(relative)
    derive = getattr(module, "derive_cc_from_cxx", None) or module._derive_cc_from_cxx
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    selected.mkdir()
    other.mkdir()
    cc_name = family.replace("g++", "gcc").replace("clang++", "clang")
    for path in [selected / family, selected / cc_name, other / cc_name, other / "gcc", other / "clang"]:
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(other))
    assert derive(str(selected / family)) == str(selected / cc_name)
    (selected / cc_name).unlink()
    assert derive(str(selected / family)) == str(other / cc_name)


def test_unknown_compiler_does_not_guess():
    module = load_module("tests/run_cpu.py")
    assert module._derive_cc_from_cxx("/missing/unknown") is None
