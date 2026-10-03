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

import numpy as np
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


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("value", ["1e-11", "0.123456789123", "-0.0", "1e20", "1.0"])
def test_generated_coefficients_keep_the_input_float(platform, value):
    module = load_module(f"include/pto/costmodel/{platform}/formula_costmodel/gen_formula_params_header.py")
    literal = module._fmt_float(value)
    assert float(literal) == float(value)
    assert np.signbit(float(literal)) == np.signbit(float(value))


@pytest.mark.parametrize("platform", ["a2a3", "a5"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e309"])
def test_nonfinite_coefficients_are_rejected(platform, value):
    module = load_module(f"include/pto/costmodel/{platform}/formula_costmodel/gen_formula_params_header.py")
    with pytest.raises(ValueError, match="finite"):
        module._fmt_float(value)
