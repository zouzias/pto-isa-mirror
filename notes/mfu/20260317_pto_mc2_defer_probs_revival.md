# 2026-03-17 `pto_mc2 + defer_probs` 复活实验记录

## 目标

在已有 clean `pto_mc2` 基线上，继续挖 `MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1` 这条链路是否还有 MFU 空间。

本轮关注的问题不是“要不要继续扫 env”，而是：

1. 先让 `defer_probs` 从功能性失败变成可运行。
2. 再判断它是否能在同窗口下稳定提升 MFU。
3. 如果不能稳定提升，及时删除相关试验代码，避免代码腐化。

## 固定对比面

- 模型与数据：`qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850`
- checkpoint：`/home/llx/pto-isa/tmp_ckpts/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_load6500_sparse`
- 对比窗口：`6503-6802`
- 基线日志：
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_clean_pto_mc2_fusedrope_long/train.log`

按同一解析脚本重算，clean `pto_mc2` 在 `6503-6802` 窗口上的平均值为：

- `2057.76 ms/iter`
- `126.38 TFLOP/s/GPU`
- `7968.07 tokens/s/device`
- `39.4934% MFU`

## 实验过程

### 1. 原始 `defer_probs` 直接失败

- 命令：
  `LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_defer_probs_long MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- 结果：
  首步即失败，报 `507035` / `vector core exception`。
- 日志：
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_long/train.log`

判断：

- `defer_probs` 的核心思路未必错，更像是延后到 `token_unpermutation()` 后碰到了不合适的 unpermute 实现。

### 2. 修复方向一：让 `defer_probs` 走 fused unpermute

代码位置：

- `/home/llx/MindSpeed/mindspeed/core/transformer/moe/moe_feature/mc2moe/legacy_a2a_token_dispatcher.py`

改动点：

- `defer_probs` 下不再强制关闭 `moe_permute_fusion`
- 让 `token_unpermutation()` 直接走 fused unpermute

验证：

- smoke 日志：
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_fused_unpermute_smoke/train.log`
- 结果：
  功能性问题消失，`6502-6504` 可以稳定跑到 `39%+`

但问题：

- 长跑转成了 OOM，不再是 vector core 异常。

### 3. 修复方向二：补一个最小内存修复

继续修改同一文件：

- `defer_probs` 下，传给 unpermute 的 `probs` 临时对齐到 `hidden_states.dtype`
- unpermute 调用后，立刻清掉 dispatcher 上多余的 `self.probs` / `self.routing_map` 引用

探针验证：

- 命令：
  `LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6525 RUN_ID=20260317_pto_mc2_defer_probs_fused_unpermute_memfix_probe MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`
- 日志：
  `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_fused_unpermute_memfix_probe/train.log`
- 结果：
  可以稳定穿过此前 OOM 的 `6520`，`6520-6525` 维持在 `40.0%` 左右

但问题：

- 直接放大到完整窗口仍有概率 OOM，说明代码层面虽然减压了，但还不够稳。

### 4. 修复方向三：只加官方 allocator env，不再加新代码

最后一轮只加环境变量：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

命令：

- `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6802 RUN_ID=20260317_pto_mc2_defer_probs_fused_unpermute_memfix_long_expandseg MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1 /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_0/20260317_pto_mc2_defer_probs_fused_unpermute_memfix_long_expandseg/train.log`

现象：

- `reserved` 显著下降，不再出现中段 OOM
- `6503-6802` 全窗口稳定跑完
- steady-state 长时间维持在 `40.0% ~ 40.2%`

## 最终结果

按同一解析脚本重算，`6503-6802` 平均值为：

- `2028.66 ms/iter`
- `128.17 TFLOP/s/GPU`
- `8081.38 tokens/s/device`
- `40.0549% MFU`

相对 clean `pto_mc2` 的净提升：

- `-29.10 ms/iter`
- `+1.80 TFLOP/s/GPU`
- `+113.31 tokens/s/device`
- `+0.5615 MFU`

## 结论

今天这条线的有效组合不是单独某一个开关，而是三件事一起成立：

1. `defer_probs`
2. `defer_probs` 下改走 fused unpermute，并补最小内存修复
3. `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

缺其中任何一环，都不够稳：

- 原始版本：功能性失败
- 只修 fused unpermute：长跑 OOM
- 只修代码不加 allocator env：长跑仍有概率 OOM

## 本轮保留项

保留的代码：

- `MindSpeed` 中 `legacy_a2a_token_dispatcher.py` 的最终修复

保留的归因基础设施：

- `pto-isa` wrapper 记录 `PYTORCH_NPU_ALLOC_CONF`

已清理的无价值产物：

- 本轮失败实验产生的 `extra-info/` dump 目录已删除

## 下一步建议

如果后续要把这条结果变成默认可复现流程，我建议按以下顺序推进：

1. 先把 `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` 纳入正式实验矩阵记录
2. 再决定是否把它设为某个 preset 的默认环境
3. 最后再看要不要继续把 `defer_probs` 的内存占用往下打，争取摆脱对 allocator env 的依赖
