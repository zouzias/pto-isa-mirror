#!/usr/bin/env python3
"""二进制文件对比工具 - 支持 int8/uint8/float/float16/int32 类型"""

import os
import sys
import argparse
import numpy as np

GOLDEN_PATH = "tests/npu/a2a3/src/st/build/TStoreAcc2gmTest.case_ndc1hwc0_vector_1/golden.bin"
OUTPUT_PATH = "tests/npu/a2a3/src/st/build/TStoreAcc2gmTest.case_ndc1hwc0_vector_1/output_z.bin"

# 数据类型映射: 名称 -> (numpy dtype, txt保存格式)
DTYPE_MAP = {
    "int8":    (np.int8,    "%d"),
    "uint8":   (np.uint8,   "%d"),
    "float":   (np.float32, "%.6f"),
    "float16": (np.float16, "%.4f"),
    "int32":   (np.int32,   "%d"),
}


def read_bin(filepath: str, dtype: np.dtype) -> np.ndarray:
    """读取指定类型的二进制文件"""
    with open(filepath, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype)


def compare(golden_path: str, output_path: str, dtype_name: str):
    np_dtype, fmt = DTYPE_MAP[dtype_name]
    golden = read_bin(golden_path, np_dtype)
    output = read_bin(output_path, np_dtype)

    lines = []
    lines.append(f"数据类型:   {dtype_name}")
    lines.append(f"Golden  元素数: {len(golden)}")
    lines.append(f"Output  元素数: {len(output)}")
    lines.append("")

    if len(golden) == 0 and len(output) == 0:
        lines.append("两个文件均为空，完全一致。")
        print("\n".join(lines))
        return True

    min_len = min(len(golden), len(output))
    if len(golden) != len(output):
        lines.append(f"⚠ 文件大小不一致! Golden={len(golden)}, Output={len(output)}, 取较小值 {min_len} 进行对比")
        lines.append("")

    # 浮点类型使用近似比较
    if np.issubdtype(np_dtype, np.floating):
        mismatch_arr = ~np.isclose(golden[:min_len], output[:min_len], atol=1e-3, rtol=1e-3)
    else:
        mismatch_arr = golden[:min_len] != output[:min_len]

    mismatch_count = int(np.sum(mismatch_arr))
    match_count = min_len - mismatch_count
    match_rate = match_count / min_len * 100 if min_len > 0 else 0

    lines.append(f"不匹配数: {mismatch_count}")
    lines.append(f"匹配率:   {match_rate:.2f}%")

    print("\n".join(lines))

    # 将 golden 和 output 的数值保存为 txt 文件（与 bin 同目录）
    out_dir = os.path.dirname(golden_path)
    golden_txt = os.path.join(out_dir, "golden.txt")
    output_txt = os.path.join(out_dir, "output_z.txt")
    np.savetxt(golden_txt, golden, fmt=fmt)
    np.savetxt(output_txt, output, fmt=fmt)
    print(f"\nGolden 已保存到: {golden_txt}")
    print(f"Output 已保存到: {output_txt}")

    return mismatch_count == 0 and len(golden) == len(output)


def main():
    parser = argparse.ArgumentParser(description="对比两个二进制文件的精度")
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="int8",
        help="数据类型 (默认: int8)",
    )
    args = parser.parse_args()

    ok = compare(GOLDEN_PATH, OUTPUT_PATH, args.dtype)
    print()
    if ok:
        print("✅ 对比通过: 两个文件完全一致")
    else:
        print("❌ 对比失败: 存在不一致")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
