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


@pytest.mark.skipif(os.name == "nt", reason="requires Bash")
@pytest.mark.parametrize("folder", ["ascend home", "ascend_home"])
def test_source_accepts_actual_installation_path(tmp_path, monkeypatch, folder):
    module = load_module("tests/script/run_st.py")
    home = tmp_path / folder
    (home / "bin").mkdir(parents=True)
    (home / "bin" / "setenv.bash").write_text("export PTO_HOST_FIXTURE=environment_loaded\n")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    monkeypatch.delenv("PTO_HOST_FIXTURE", raising=False)
    module.set_env_variables("sim", "Ascend910B1")
    assert os.environ.get("PTO_HOST_FIXTURE") == "environment_loaded"


@pytest.mark.skipif(os.name == "nt", reason="requires Bash")
def test_failed_environment_script_does_not_continue(tmp_path, monkeypatch):
    module = load_module("tests/script/run_st.py")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "setenv.bash").write_text("exit 7\n")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    with pytest.raises(subprocess.CalledProcessError) as error:
        module.set_env_variables("sim", "Ascend910B1")
    assert error.value.returncode == 7
