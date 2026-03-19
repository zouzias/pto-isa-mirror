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
import numpy as np


def histogram_counts_u8(byte_values):
    return np.bincount(byte_values, minlength=256).astype(np.uint32)


def histogram_counts_u8_manual(byte_values):
    counts = np.zeros(256, dtype=np.uint32)
    flat_values = np.asarray(byte_values, dtype=np.uint8).reshape(-1)
    print("input_u8_values_hex:", [f"0x{int(value):02x}" for value in flat_values])
    for value in flat_values:
        counts[int(value)] += 1
    return counts


def cumulative_histogram_asc(byte_values):
    counts = histogram_counts_u8(byte_values)
    return np.cumsum(counts, dtype=np.uint32)


def verify_counts_against_cumulative(byte_values, cumulative_hist):
    counts = histogram_counts_u8_manual(byte_values)
    expected_cumulative = np.cumsum(counts, dtype=np.uint32)
    if not np.array_equal(expected_cumulative, cumulative_hist):
        diff_idx = np.flatnonzero(expected_cumulative != cumulative_hist)
        first_diff = int(diff_idx[0]) if diff_idx.size > 0 else -1
        raise ValueError(
            f"cumulative histogram verification failed at bin {first_diff}: "
            f"expected {expected_cumulative[first_diff]}, got {cumulative_hist[first_diff]}"
        )
    return counts


def get_k_index(cumulative_hist_asc, k):
    if cumulative_hist_asc.size == 0:
        return 0
    total = cumulative_hist_asc[-1]
    cumulative_hist_desc = total - np.concatenate(([0], cumulative_hist_asc[:-1])).astype(np.uint32)
    valid_bins = np.flatnonzero(cumulative_hist_desc >= k)
    if valid_bins.size == 0:
        return 0
    return int(valid_bins[-1])


def gen_golden_histogram(case_name, param):
    """Generate golden data for THistogram.

    THistogram is generated per row.

    - Input shape: `[rows, cols]`
    - Output shape: `[rows, 256]`
    - Each row stores an ascending cumulative histogram.

    Golden behavior:
    - MSB mode: output per-row ascending cumulative MSB histogram.
    - LSB mode:
      1. Build per-row ascending cumulative MSB histogram.
      2. Derive one `k_index` per row from the top-k tail of that histogram.
      3. Keep only row values whose MSB byte equals that row's `k_index`.
      4. Build the per-row ascending cumulative LSB histogram on that subset.
    """
    rows, cols = param.rows, param.cols

    src = np.random.randint(0, 65536, size=(rows, cols), dtype=np.uint16)
    msb_bytes = ((src >> 8) & 0xFF).astype(np.uint8)
    lsb_bytes = (src & 0xFF).astype(np.uint8)

    golden = np.zeros((rows, 256), dtype=np.uint32)
    idx = np.zeros(rows, dtype=np.uint8)
    msb_counts_all_rows = np.zeros((rows, 256), dtype=np.uint32)
    msb_cumulative_all_rows = np.zeros((rows, 256), dtype=np.uint32)
    lsb_counts_all_rows = np.zeros((rows, 256), dtype=np.uint32)
    lsb_cumulative_all_rows = np.zeros((rows, 256), dtype=np.uint32)
    selected_mask_all_rows = np.zeros((rows, cols), dtype=np.uint8)
    selected_lsb_values_padded = np.zeros((rows, cols), dtype=np.uint8)
    selected_lsb_lengths = np.zeros(rows, dtype=np.uint32)

    if param.MSBorLSB == "MSB":
        for row in range(rows):
            msb_counts_all_rows[row] = verify_counts_against_cumulative(msb_bytes[row], cumulative_histogram_asc(msb_bytes[row]))
            msb_cumulative_all_rows[row] = np.cumsum(msb_counts_all_rows[row], dtype=np.uint32)
            golden[row] = msb_cumulative_all_rows[row]
            lsb_counts_all_rows[row] = histogram_counts_u8(lsb_bytes[row])
            lsb_cumulative_all_rows[row] = np.cumsum(lsb_counts_all_rows[row], dtype=np.uint32)
    else:
        for row in range(rows):
            msb_counts_all_rows[row] = verify_counts_against_cumulative(msb_bytes[row], cumulative_histogram_asc(msb_bytes[row]))
            row_msb_hist = np.cumsum(msb_counts_all_rows[row], dtype=np.uint32)
            msb_cumulative_all_rows[row] = row_msb_hist
            k_index = get_k_index(row_msb_hist, param.k)
            idx[row] = np.uint8(k_index)

            selected_mask = msb_bytes[row] == k_index
            selected_mask_all_rows[row] = selected_mask.astype(np.uint8)
            selected_lsb_bytes = lsb_bytes[row][selected_mask]
            selected_lsb_lengths[row] = np.uint32(selected_lsb_bytes.size)
            if selected_lsb_bytes.size > 0:
                selected_lsb_values_padded[row, :selected_lsb_bytes.size] = selected_lsb_bytes

            lsb_counts_all_rows[row] = verify_counts_against_cumulative(selected_lsb_bytes, cumulative_histogram_asc(selected_lsb_bytes))
            lsb_cumulative_all_rows[row] = np.cumsum(lsb_counts_all_rows[row], dtype=np.uint32)
            golden[row] = lsb_cumulative_all_rows[row]

    src.tofile("input.bin")
    idx.tofile("idx.bin")
    golden.tofile("golden.bin")
    msb_bytes.tofile("msb_bytes_rows.bin")
    lsb_bytes.tofile("lsb_bytes_rows.bin")
    msb_counts_all_rows.tofile("msb_counts_rows.bin")
    msb_cumulative_all_rows.tofile("msb_cumulative_rows.bin")
    lsb_counts_all_rows.tofile("lsb_counts_rows.bin")
    lsb_cumulative_all_rows.tofile("lsb_cumulative_rows.bin")
    selected_mask_all_rows.tofile("selected_mask_rows.bin")
    selected_lsb_values_padded.tofile("selected_lsb_values_rows.bin")
    selected_lsb_lengths.tofile("selected_lsb_lengths.bin")

    return src, golden


class THISTOGRAMParams:
    def __init__(self, rows, cols, MSBorLSB="MSB", k=2):
        self.rows = rows
        self.cols = cols
        self.MSBorLSB = MSBorLSB
        self.k = k

def generate_case_name(param):
    if param.MSBorLSB == "MSB":
        return f"THISTOGRAMTest.case_{param.rows}x{param.cols}_{param.MSBorLSB}"
    else:
        return f"THISTOGRAMTest.case_{param.rows}x{param.cols}_{param.MSBorLSB}_k{param.k}"


if __name__ == "__main__":
    np.random.seed(42)
    
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    testcases_dir = os.path.join(script_dir, "testcases")

    # Ensure the testcases directory exists
    if not os.path.exists(testcases_dir):
        os.makedirs(testcases_dir)

    case_params_list = [
        THISTOGRAMParams(2, 128, "MSB"),
        THISTOGRAMParams(4, 64, "MSB"),
        THISTOGRAMParams(8, 128, "MSB"),
        THISTOGRAMParams(1, 256, "MSB"),
        THISTOGRAMParams(4, 256, "MSB"),
        THISTOGRAMParams(2, 100, "MSB"),
        THISTOGRAMParams(2, 128, "LSB", 108),
        THISTOGRAMParams(4, 64, "LSB", 52),
        THISTOGRAMParams(8, 128, "LSB", 104),
        THISTOGRAMParams(1, 256, "LSB", 210),
        THISTOGRAMParams(4, 256, "LSB", 220),
        THISTOGRAMParams(2, 100, "LSB", 82),
    ]

    for param in case_params_list:
        case_name = generate_case_name(param)
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_histogram(case_name, param)
        os.chdir(original_dir)
