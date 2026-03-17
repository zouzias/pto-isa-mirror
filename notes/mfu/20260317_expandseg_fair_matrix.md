# 2026-03-17 `expandable_segments` 公平矩阵记录

## 目标

在同一代码、同一 checkpoint、同一窗口下，判断
`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`
到底是不是一个“普适提升 MFU”的开关，还是只对特定链路有效。

本轮固定原则：

1. 所有结果统一使用 `6503-6802` 窗口重算。
2. 除显式对比项外，不改模型配置。
3. 不把单个成功案例直接包装成通用优化，先做公平矩阵。

## 固定对比面

- 模型与数据：`qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850`
- checkpoint：
  `/home/llx/pto-isa/tmp_ckpts/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_load6500_sparse`
- 对比窗口：`6503-6802`
- 统一环境：
  `USE_FUSED_ROTARY_POS_EMB=1`

## 运行命令

### 1. `native_base + expandable_segments`

命令：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True LAUNCH_PRESET=native_base TRAIN_ITERS=6802 RUN_ID=20260317_native_base_expandseg_long /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_native_base_expandseg_long/train.log`

### 2. `mc2 + expandable_segments`

命令：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True LAUNCH_PRESET=mc2 TRAIN_ITERS=6802 RUN_ID=20260317_mc2_expandseg_long /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_mc2_expandseg_long/train.log`

### 3. `pto_mc2 + expandable_segments`

命令：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_expandseg_long /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_expandseg_long/train.log`

### 4. `pto_mc2 + defer_probs + expandable_segments`

这组复用了同日已完成的长跑结果，详细过程见：

- `/home/llx/pto-isa/notes/mfu/20260317_pto_mc2_defer_probs_revival.md`

日志：

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_fused_unpermute_memfix_long_expandseg/train.log`

## 统一重算结果

按同一解析脚本重算 `6503-6802` 平均值：

| 配置 | ms/iter | TFLOP/s/GPU | tokens/s/device | MFU |
| --- | ---: | ---: | ---: | ---: |
| `native_base + expandable_segments` | `2189.58` | `118.8310` | `7492.0970` | `37.1343%` |
| `mc2 + expandable_segments` | `2066.83` | `125.8610` | `7935.4097` | `39.3317%` |
| `pto_mc2 + expandable_segments` | `2057.40` | `126.4033` | `7969.6027` | `39.5014%` |
| `pto_mc2 + defer_probs + expandable_segments` | `2028.66` | `128.1743` | `8081.3787` | `40.0549%` |

## 关键差值

### 1. `mc2 + expandable_segments` 相对 `native_base + expandable_segments`

- `-122.75 ms/iter`
- `+7.0300 TFLOP/s/GPU`
- `+443.3127 tokens/s/device`
- `+2.1974 MFU`

### 2. `pto_mc2 + expandable_segments` 相对 `mc2 + expandable_segments`

- `-9.43 ms/iter`
- `+0.5423 TFLOP/s/GPU`
- `+34.1930 tokens/s/device`
- `+0.1697 MFU`

### 3. `pto_mc2 + defer_probs + expandable_segments` 相对 `pto_mc2 + expandable_segments`

- `-28.74 ms/iter`
- `+1.7710 TFLOP/s/GPU`
- `+111.7760 tokens/s/device`
- `+0.5535 MFU`

### 4. `pto_mc2 + defer_probs + expandable_segments` 相对 `native_base + expandable_segments`

- `-160.92 ms/iter`
- `+9.3433 TFLOP/s/GPU`
- `+589.2817 tokens/s/device`
- `+2.9206 MFU`

## 结论

本轮公平矩阵说明的不是“`expandable_segments` 本身有多强”，而是两件更重要的事：

1. `expandable_segments` 不是通用加速开关。
2. 它对 `native_base` 和 clean `mc2` 基本没有收益，甚至略有回退。
3. 它对 clean `pto_mc2` 也只有极小幅度改善。
4. 它真正有价值的地方，是把 `pto_mc2 + defer_probs` 这条本来有收益但不够稳的链路稳定化，并把收益完整释放出来。

换句话说：

- `expandable_segments` 单独看，不值得包装成“全局优化”。
- 但如果目标是保住当前最优 MFU，它是 `pto_mc2 + defer_probs` 方案的必要组成部分。

## 对默认策略的建议

当前更合理的默认化方式不是：

- “所有 preset 一律加 `expandable_segments`”

而是：

- 先把它作为 `pto_mc2 + defer_probs` 这条实验线的标准环境记录下来
- 后续如果要默认化，应该默认化的是“组合”，不是单独默认化 allocator env

## 与前一篇记录的关系

前一篇记录解决的是：

- `defer_probs` 如何从功能性失败，变成可稳定跑完并获得收益

这一篇补的是：

- 这组收益是否经得起公平矩阵对比

两篇合起来，才构成今天这条优化线的完整结论。
