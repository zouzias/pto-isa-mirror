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

import json
from pathlib import Path


DEFAULT_CASE = {
    "name": "case_T256_H64_F64_E32_K1",
    "t": 256,
    "h": 64,
    "f": 64,
    "e": 32,
    "topk": 1,
}


def load_moe_case() -> dict:
    case_path = Path(__file__).resolve().parent.parent / "build" / "generated_cases.json"
    if not case_path.exists():
        return dict(DEFAULT_CASE)
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not cases:
        raise ValueError(f"No cases found in {case_path}")
    return cases[0]
