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


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ("Suite.alpha:Suite.beta-Suite.beta", ["Suite.alpha"]),
        ("-Suite.beta", ["Suite.alpha", "Other.gamma"]),
        ("Suite.*", ["Suite.alpha", "Suite.beta"]),
        ("", []),
    ],
)
def test_gtest_filter_applies_both_halves(tmp_path, monkeypatch, pattern, expected):
    module = load_module("tests/script/run_st.py")
    source = tmp_path / "testcase" / "fixture" / "main.cc"
    source.parent.mkdir(parents=True)
    source.write_text("TEST_F(Suite, alpha) {}\nTEST_F(Suite, beta) {}\nTEST_F(Other, gamma) {}\n")
    monkeypatch.chdir(tmp_path)
    assert module.list_gtest_cases("fixture", pattern) == expected
