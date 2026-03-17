# 2026-03-17 `aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317`：`native_base` vs `pto_mc2`

## 目标

回答一个直接问题：

- 这套 8B 参数如果坚持老脚本的 `EP=4`，是不是干脆不用 MC2，走 `native_base` 反而更好？

## 固定条件

- 配置名：`aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317`
- 从 0 开始：
  - `LOAD_CHECKPOINT=0`
  - `SAVE_CHECKPOINT=0`
- 统一短跑窗口：`20` 步
- 模型结构沿用此前确认过的参数：
  - `num_layers=8`
  - `hidden_size=4096`
  - `ffn_hidden_size=8192`
  - `num_experts=16`
  - `topk=4`
  - `seq_len=6144`
  - `moe_ffn_hidden_size=4096`
  - `moe_shared_expert_intermediate_size=4096`
  - `n_shared_experts=1`

## 对比对象

### 1. `native_base`，保留老脚本风格的 `EP=4`

命令：

- `LAUNCH_PRESET=native_base EXP_NAME=aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317 LOAD_CHECKPOINT=0 SAVE_CHECKPOINT=0 TRAIN_ITERS=20 RUN_ID=20260317_aigcode_8b_native_base_from_scratch_20step /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317_0/20260317_aigcode_8b_native_base_from_scratch_20step/train.log`

### 2. `pto_mc2`，为满足当前算子约束改成 `EP=8`

命令：

- `LAUNCH_PRESET=pto_mc2 EXP_NAME=aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317 LOAD_CHECKPOINT=0 SAVE_CHECKPOINT=0 TRAIN_ITERS=20 EXPERT_MODEL_PARALLEL_SIZE=8 RUN_ID=20260317_aigcode_8b_pto_mc2_from_scratch_ep8_20step /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh`

日志：

- `/home/llx/pto-isa/logs_native_resume/aigcode_8b_bf16_6k_jamba_moe_16npu_cann850_0317_0/20260317_aigcode_8b_pto_mc2_from_scratch_ep8_20step/train.log`

补充：

- `EP=4 + pto_mc2` 跑不起来，失败记录见：
  `/home/llx/pto-isa/notes/mfu/20260317_aigcode_8b_pto_mc2_probe.md`

## 结果

### 单步末尾

- `native_base EP=4`
  - iteration 20: `3563.5 ms / 139.7 TFLOP/s/GPU / 6896.7 tokens/s/device / 43.66% MFU`

- `pto_mc2 EP=8`
  - iteration 20: `4793.0 ms / 103.9 TFLOP/s/GPU / 5127.5 tokens/s/device / 32.46% MFU`

### 去掉冷启动后的均值

#### `iteration 3-20`

- `native_base EP=4`
  - `3444.44 ms`
  - `144.66 TFLOP/s/GPU`
  - `7140.86 tokens/s/device`
  - `45.2056% MFU`

- `pto_mc2 EP=8`
  - `4507.17 ms`
  - `110.77 TFLOP/s/GPU`
  - `5468.08 tokens/s/device`
  - `34.6161% MFU`

差值：

- `native_base EP=4` 相对 `pto_mc2 EP=8`
  - `-1062.73 ms`
  - `+33.89 TFLOP/s/GPU`
  - `+1672.78 tokens/s/device`
  - `+10.5895 MFU`

#### `iteration 10-20`

- `native_base EP=4`
  - `3508.59 ms`
  - `141.92 TFLOP/s/GPU`
  - `7005.59 tokens/s/device`
  - `44.3491% MFU`

- `pto_mc2 EP=8`
  - `4678.31 ms`
  - `106.45 TFLOP/s/GPU`
  - `5254.92 tokens/s/device`
  - `33.2664% MFU`

## 结论

对这套 8B 配置，当前答案很直接：

1. 如果坚持 `EP=4`，那就不能走当前 `pto_mc2` 的 MC2 主链。
2. 在当前仓和当前算子约束下，可运行且更强的是 `native_base EP=4`。
3. 这不是说 `pto_mc2` 永远更差，而是这套模型形状和当前 MC2 主算子约束不匹配：
   - `EP=4` 直接不支持
   - 改成 `EP=8` 后，每卡 local experts 变少，主链饱满度下降
4. 所以这组参数下，`native_base` 的确比当前可运行的 `pto_mc2` 更合适。

## 现阶段建议

如果目标是“这套 8B 先拿到高 MFU”，优先级应该是：

1. 先以 `native_base EP=4` 作为当前最优可运行基线
2. 再决定要不要为了接入 MC2，重设计一套更适合 `EP=8` 的 8B 形状
3. 不要再拿 `EP=4 + pto_mc2` 继续硬跑，当前算子约束下这条路走不通
