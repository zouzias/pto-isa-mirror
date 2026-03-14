#!/usr/bin/python3
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

import os
import pathlib

import torch
import torch_npu


def _candidate_so_paths():
    pkg_path = pathlib.Path(__file__).resolve().parent
    repo_root = pkg_path.parent

    env_path = os.getenv("PTO_MOE_GROUPED_FFN_SO_PATH")
    if env_path:
        yield pathlib.Path(env_path)

    yield pkg_path / "lib" / "libop_extension.so"
    yield repo_root / "build" / "lib" / "libop_extension.so"
    yield repo_root / "build" / "libop_extension.so"
    yield repo_root / "build_manual" / "lib" / "libop_extension.so"


def _load_opextension_so():
    checked = []
    for so_path in _candidate_so_paths():
        so_path = so_path.resolve()
        checked.append(str(so_path))
        if so_path.is_file():
            torch.ops.load_library(os.fspath(so_path))
            return

    raise FileNotFoundError(
        "Could not locate libop_extension.so. Checked: " + ", ".join(checked)
    )
