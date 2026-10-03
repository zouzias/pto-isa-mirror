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


@pytest.mark.parametrize("relative", ["tests/cpu/st/utils.py", "tests/script/cpu_bfloat16.py"])
def test_bf16_preserves_nan_and_sign(relative):
    module = load_module(relative)
    convert = module.NumExt._float32_to_bfloat16_bits if hasattr(module, "NumExt") else module.float32_to_bfloat16_bits
    bits = np.array([0x7F800001, 0x7FFFFFFF, 0xFF800001, 0xFFFFFFFF], dtype=np.uint32)
    decoded = (convert(bits.view(np.float32)).astype(np.uint32) << np.uint32(16)).view(np.float32)
    assert np.isnan(decoded).all()
    np.testing.assert_array_equal(np.signbit(decoded), np.signbit(bits.view(np.float32)))


@pytest.mark.parametrize("relative", ["tests/cpu/st/utils.py", "tests/script/cpu_bfloat16.py"])
def test_bf16_finite_rounding_and_infinity(relative):
    module = load_module(relative)
    convert = module.NumExt._float32_to_bfloat16_bits if hasattr(module, "NumExt") else module.float32_to_bfloat16_bits
    bits = np.array([0x3F808000, 0x3F818000, 0x7F800000, 0xFF800000], dtype=np.uint32)
    np.testing.assert_array_equal(convert(bits.view(np.float32)), [0x3F80, 0x3F82, 0x7F80, 0xFF80])
