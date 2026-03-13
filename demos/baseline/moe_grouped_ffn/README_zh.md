# Grouped MoE FFN Stage-1 算子示例

## 概览

本示例新增了一个基于 PTO 的 grouped MoE FFN 自定义算子，并通过 `torch_npu` 暴露为 `torch.ops.npu.pto_moe_grouped_ffn`。

当前版本刻意聚焦在 routed MoE 中最热的那一段：

- 输入 token 已经按 expert 连续打包
- 两个 expert 投影（`gate` 和 `up`）由 PTO cube kernel 完成
- host 侧再融合 `SiLU(gate) * up`，返回中间激活
- Python 侧补了 autograd fallback，因此反传仍可用，只有前向走 PTO kernel

最终的 down projection 还没有一起塞进 kernel。这样做是为了先把最重的 grouped GEMM 路径做通，并给后续融合预留清晰扩展位。

## 支持的 AI 处理器

- A2/A3

## 当前固定配置

第一版针对下面这组 demo profile 做了定型：

- hidden size：`4096`
- intermediate size：`1920`
- 输入类型：`bfloat16`
- 输出类型：`float32`
- 权重布局：`DN`，物理存储为 `[num_experts, 1920, 4096]`

`x` 必须已经按 expert 打包好。`group_offsets` 是长度为 `num_experts + 1` 的前缀和数组。

## 性能设计

kernel 采用 worklist 驱动的 grouped tiling：

- 一次 launch 覆盖所有 expert 和所有输出 `N` tile
- 每个 work item 负责一个 `[rows<=64, 192]` 输出 tile
- 通过动态 `validM` 处理变长 expert token 数，避免整批 padding
- cube 路径沿用了现有高性能 GEMM 的 `TLOAD -> TEXTRACT -> TMATMUL -> TSTORE` 双缓冲结构

这套骨架适合作为下一步继续压性能的基础，包括：

- 单次 launch 同时累计 gate/up 两路输出
- kernel 内融合 `SiLU * up`
- grouped down projection
- 小 expert batch 场景下更激进的 work packing

## 构建与运行

```bash
cd demos/baseline/moe_grouped_ffn
python -m venv virEnv
source virEnv/bin/activate
python3 -m pip install -r requirements.txt

export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa

rm -rf build op_extension.egg-info
python3 setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

cd test
python3 test.py
```

## 算子签名

```python
torch.ops.npu.pto_moe_grouped_ffn(x, gate_weight_dn, up_weight_dn, group_offsets) -> Tensor
```

- `x`：`[total_tokens, 4096]`，`bfloat16`
- `gate_weight_dn`：`[num_experts, 1920, 4096]`，`bfloat16`
- `up_weight_dn`：`[num_experts, 1920, 4096]`，`bfloat16`
- `group_offsets`：`[num_experts + 1]`，整型前缀和
- 返回：`[total_tokens, 1920]`，`float32`
