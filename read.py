import numpy as np
import os


def run(path, g_name, o_name, shape, dtype, max_print=100):
    """
    Args:
        max_print: 最多打印的不一致元素个数，防止差异太多时刷屏
    """
    g_path = os.path.join(path, g_name)
    o_path = os.path.join(path, o_name)

    golden = np.fromfile(g_path, dtype=dtype).reshape(shape)
    output = np.fromfile(o_path, dtype=dtype).reshape(shape)

    # 获取所有不一致元素的索引
    diff_indices = np.where(golden != output)
    diff_count = diff_indices[0].size

    print(f"diff_count: {diff_count}")
    print(f"ele nums:   {golden.size}")

    if diff_count == 0:
        print("✅ All elements match!")
        return

    # 限制打印数量，避免差异过多时输出爆炸
    n_show = min(diff_count, max_print)
    print(f"\n❌ First {n_show} mismatches (of {diff_count} total):")
    print(f"{'Index':<25} {'Golden':>15} {'Output':>15} {'Diff':>15}")
    print("-" * 75)

    for i in range(n_show):
        # 从每个维度取出第i个不一致点的坐标
        idx = tuple(dim[i] for dim in diff_indices)
        g_val = golden[idx]
        o_val = output[idx]
        print(f"{str(idx):<25} {g_val:>15} {o_val:>15} {g_val - o_val:>15}")

    if diff_count > max_print:
        print(f"... and {diff_count - max_print} more mismatches not shown.")


if __name__ == "__main__":
    golden_path = (
        r"/mnt/workspace/gitCode/qq_53648788/pto-isa/tests/npu/a5/src/st/build/TCOLEXPANDTest.case_int64_1_4_128_128"
    )
    golden_name, output_name = "golden.bin", "output.bin"
    # shape = (1)
    golden_shape = (4, 128)
    # output_shape = (4, 16384)

    golden_dtype = np.int64

    run(golden_path, golden_name, output_name, golden_shape, golden_dtype)
