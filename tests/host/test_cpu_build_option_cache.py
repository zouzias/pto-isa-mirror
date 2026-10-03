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
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    "bf16,trace,cached_bf16,cached_trace,expected",
    [
        (True, False, "OFF", "OFF", True),
        (False, True, "OFF", "OFF", True),
        (False, False, "ON", "OFF", True),
        (False, False, "OFF", "ON", True),
        (True, True, "ON", "ON", False),
        (False, False, "OFF", "OFF", False),
    ],
)
def test_build_cache_checks_runtime_generation_options(tmp_path, bf16, trace, cached_bf16, cached_trace, expected):
    module = load_module("tests/run_cpu.py")
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    binary = build / "bin" / ("host_fixture.exe" if os.name == "nt" else "host_fixture")
    binary.write_text("existing test binary")
    (build / "CMakeCache.txt").write_text(
        "TEST_CASE:STRING=host_fixture\n"
        + f"PTO_CPU_SIM_ENABLE_BF16:BOOL={cached_bf16}\n"
        + f"PTO_CPU_SIM_TRACE_MODE:BOOL={cached_trace}\n"
    )
    args = SimpleNamespace(
        testcase="host_fixture",
        build_type="Release",
        no_build=False,
        rebuild=False,
        clean=False,
        enable_bf16=bf16,
        trace_mode=trace,
    )
    assert module.determine_need_build(args, tmp_path, build) is expected
    args.no_build = True
    assert module.determine_need_build(args, tmp_path, build) is False
