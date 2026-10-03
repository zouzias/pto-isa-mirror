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
import subprocess
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


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX cp/rm commands")
@pytest.mark.parametrize("testcase,nranks", [("host_fixture", 1), ("host_fixture_ccu", 2)])
def test_zero_executed_comm_runs_fail_real_cli(tmp_path, testcase, nranks):
    script = tmp_path / "tests" / "script" / "run_st.py"
    script.parent.mkdir(parents=True)
    script.write_bytes((ROOT / "tests/script/run_st.py").read_bytes())
    target = tmp_path / "tests/npu/a5/comm/st"
    data = target / "testcase" / testcase / "gen_data.py"
    data.parent.mkdir(parents=True)
    data.write_text("from pathlib import Path\nPath('host_generated').write_text('ok')\n")
    (target / "build").mkdir()
    command = [sys.executable, str(script), "-r", "npu", "-v", "a5", "-t", "comm/" + testcase, "-w", "-n", str(nranks)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert (target / "build/host_generated").read_text() == "ok"
    assert result.returncode != 0, result.stdout + result.stderr
    assert "No comm ST runs executed" in result.stdout
    assert "All 0" not in result.stdout


def test_ordinary_no_device_environment_setup_remains_available():
    module = load_module("tests/script/run_st.py")
    assert module.set_env_variables("npu", "Ascend910B1") is None
