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
import os
import subprocess
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


@pytest.mark.parametrize("relative", ["tests/run_cpu.py", "tests/run_costmodel.py"])
@pytest.mark.parametrize("bf16", [False, True])
def test_prebuilt_entrypoint_never_probes_compiler(monkeypatch, relative, bf16):
    module = load_module(relative)
    args = SimpleNamespace(no_build=True, demo=None, demo_only=False, enable_bf16=bf16, cxx=None, cc=None)
    monkeypatch.setattr(module, "parse_arguments", lambda: args)
    monkeypatch.setattr(module, "setup_environment", lambda options: None)
    monkeypatch.setattr(module, "log_build_info", lambda options, cxx, cc: None)

    def forbidden(*arguments):
        raise RuntimeError("unexpected compiler probe")

    monkeypatch.setattr(module, "detect_compilers", forbidden)
    if hasattr(module, "resolve_bf16_compiler_pair"):
        monkeypatch.setattr(module, "resolve_bf16_compiler_pair", forbidden)
    # Boundary fixture isolates actual toolchain and test-launch processes.
    monkeypatch.setattr(module, "run_test_mode", lambda options, root, cxx, cc: 0 if (cxx, cc) == (None, None) else 7)
    assert module.main() == 0


@pytest.mark.parametrize("relative", ["tests/run_cpu.py", "tests/run_costmodel.py"])
def test_demo_still_requires_compilers(monkeypatch, relative):
    module = load_module(relative)
    args = SimpleNamespace(no_build=True, demo="gemm", demo_only=False, enable_bf16=False, cxx=None, cc=None)
    monkeypatch.setattr(module, "parse_arguments", lambda: args)
    monkeypatch.setattr(module, "setup_environment", lambda options: None)
    monkeypatch.setattr(module, "log_build_info", lambda options, cxx, cc: None)
    monkeypatch.setattr(module, "detect_compilers", lambda cxx, cc: ("selected-cxx", "selected-cc"))
    monkeypatch.setattr(module, "run_demo_mode", lambda options, root, cxx, cc: 0 if cxx == "selected-cxx" else 7)
    assert module.main() == 0


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX executable fixture")
def test_prebuilt_cpu_binary_runs_with_no_compiler(tmp_path):
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    executable = build / "bin/host_fixture"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=str(tmp_path))
    environment.pop("CXX", None)
    environment.pop("CC", None)
    command = [
        sys.executable,
        str(ROOT / "tests/run_cpu.py"),
        "--no-build",
        "--no-install",
        "--no-gen",
        "--testcase",
        "host_fixture",
        "--build-dir",
        str(build),
    ]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
