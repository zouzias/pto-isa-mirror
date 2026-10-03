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


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe fixture")
@pytest.mark.parametrize("code,output", [(7, ""), (1, ""), (1, "unexpected")])
def test_failed_probe_cannot_report_ready(tmp_path, code, output):
    module = load_module("docs/mkdocs/check_mkdocs.py")
    executable = tmp_path / "python-probe"
    executable.write_text(f"#!/bin/sh\nprintf '%s\n' '{output}'\nprintf 'probe failed' >&2\nexit {code}\n")
    executable.chmod(0o755)
    with pytest.raises(RuntimeError, match="Dependency probe failed"):
        module._missing_modules_for(executable)


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX executable probe fixture")
@pytest.mark.parametrize("code,output,expected", [(0, "", []), (1, "mkdocs", ["mkdocs"])])
def test_valid_probe_protocol_is_retained(tmp_path, code, output, expected):
    module = load_module("docs/mkdocs/check_mkdocs.py")
    executable = tmp_path / "python-probe"
    executable.write_text(f"#!/bin/sh\nprintf '%s\n' '{output}'\nexit {code}\n")
    executable.chmod(0o755)
    assert module._missing_modules_for(executable) == expected
