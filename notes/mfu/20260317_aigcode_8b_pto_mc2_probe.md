# 2026-03-17 `aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317` 探针记录

## 目标

确认这套 8B 配置在当前最强 `pto_mc2` 路径上是否可运行，以及从 0 开始短跑时的 MFU 大致落点。

## 固定条件

- 配置名：`aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317`
- 启动方式：`LAUNCH_PRESET=pto_mc2`
- 从 0 开始：
  `LOAD_CHECKPOINT=0`
  `SAVE_CHECKPOINT=0`
- 当前 `pto_mc2` 默认已包含：
  - `MINDSPEED_MOE_MC2_DEFER_PROBS_TO_UNPERMUTE=1`
  - `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`

## 实验 1：沿用老脚本的 `EP=4`

命令：

- `LAUNCH_PRESET=pto_mc2 EXP_NAME=aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317 LOAD_CHECKPOINT=0 SAVE_CHECKPOINT=0 TRAIN_ITERS=30 RUN_ID=20260317_aigcode_8b_pto_mc2_from_scratch_30step /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317_0/20260317_aigcode_8b_pto_mc2_from_scratch_30step/train.log`

结果：

- 没跑到首个 iteration
- 失败点是 `npu_alltoallv_gmm`
- 关键报错：
  `epWorldSize[4] should be 8 16 32 64 128`

结论：

- 当前 MC2 主算子 `aclnnAlltoAllvGroupedMatMul` 不支持 `EP=4`
- 所以这套 8B 参数在“老脚本原样 + pto_mc2”下并不能直接跑

## 实验 2：改成 `EP=8`

命令：

- `LAUNCH_PRESET=pto_mc2 EXP_NAME=aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317 LOAD_CHECKPOINT=0 SAVE_CHECKPOINT=0 TRAIN_ITERS=20 EXPERT_MODEL_PARALLEL_SIZE=8 RUN_ID=20260317_aigcode_8b_pto_mc2_from_scratch_ep8_20step /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317_0/20260317_aigcode_8b_pto_mc2_from_scratch_ep8_20step/train.log`

启动快照：

- `/home/llx/pto-isa/logs_native_resume/aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317_0/20260317_aigcode_8b_pto_mc2_from_scratch_ep8_20step/launch.sh`

结果：

- 可以稳定跑完 `20` 步
- 日志打印总参数量：
  - transformer block: `7.38B`
  - embedding: `1.25B`
  - total: `8.63B`

### 单步结果

- 第 `20` 步：
  - `4793.0 ms/iter`
  - `103.9 TFLOP/s/GPU`
  - `5127.5 tokens/s/device`
  - `32.46% MFU`

### 短窗口均值

按统一脚本重算：

- `iteration 3-20` 平均：
  - `4507.17 ms/iter`
  - `110.77 TFLOP/s/GPU`
  - `5468.08 tokens/s/device`
  - `34.62% MFU`

- `iteration 10-20` 平均：
  - `4678.31 ms/iter`
  - `106.45 TFLOP/s/GPU`
  - `5254.92 tokens/s/device`
  - `33.27% MFU`

## 关于 `fused rotary` 的回退警告

运行中出现了：

- `Setting apply_rope_fusion to false because its implementation is not included in Apex`

这个 warning 指向的是：

- `apply_rope_fusion`
- 也就是 Megatron/Apex/TE 那条 fused rope 路径

它不等价于：

- `use_fused_rotary_pos_emb`
- 也就是 MindSpeed/NPU 侧的 fused rotary 开关

当前 patch 逻辑是：

1. 先接管 `megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb`
2. 如果 `config.apply_rope_fusion` 打开但 Apex fused rope 不在环境里，就先打印 warning 并回退
3. 随后仍会落到 `_apply_rotary_pos_emb_bshd`
4. 而 `_apply_rotary_pos_emb_bshd` 又被 MindSpeed patch 过，如果 `args.use_fused_rotary_pos_emb=True`，会调用 NPU fused rope

因此：

- 这个 warning 更像是“TE/Apex fused rope 不可用”
- 不足以单独证明 MindSpeed 的 NPU fused rope 没生效

## 当前结论

1. 对这套 8B 参数，`pto_mc2` 不是完全不能用，而是不能和 `EP=4` 一起用。
2. 当前最强 `pto_mc2` 路径在这套模型上能跑的前提是：至少改成 `EP=8`。
3. 在 `EP=8`、从 0 开始的短跑里，这套 8B 配置的 MFU 目前落在 `33% ~ 35%`，明显低于此前 1B clean best 的 `39%+`。
4. 主要矛盾更像是：
   - 这套 8B MoE 形状与 MC2 主算子约束不完全匹配
   - 当前 PTO 只优化了局部 reorder，小于 `npu_alltoallv_gmm` 主链热点
   - 不能把 `apply_rope_fusion` 的 warning 误判成这次 MFU 偏低的主因
