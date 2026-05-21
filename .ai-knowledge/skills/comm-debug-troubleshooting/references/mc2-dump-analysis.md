# MC2 异常 Dump 分析

## 概述

MC2 算子（特别是 MoE V2 及以上版本）支持异常 dump，在算子异常时自动导出状态数据用于事后分析。

## Dump 数据位置

默认路径：`$ASCEND_WORK_PATH/extra-info/data-dump/`

## 分析工具

工具路径：`reference/ops-transformer/mc2/tools/dump_analysis/`

### 输入参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `TARGET_DIR` | dump 数据目录 | `/path/to/data-dump/` |
| `TOOL_PATH` | CANN 安装路径 | `/usr/local/Ascend/ascend-toolkit/latest` |
| `SOC_VERSION` | 芯片版本 | `910_93` 或 `950` |
| `PROFILING_PATH` | Profiling 数据（可选） | `/path/to/profiling/` |

### 运行分析

```bash
cd mc2/tools/dump_analysis/
python analyze.py --target_dir $TARGET_DIR --tool_path $TOOL_PATH --soc $SOC_VERSION
```

## 分析输出

### Win Region 分析

Win Region 是 MoE 算子的状态内存区（约最后 1MB），包含：
- Expert 数量和分布
- EP world size
- 每个 core 的 0/1 状态标志

### 输出文件

| 文件 | 内容 |
|------|------|
| `win_status_list` | 每个 core 的 dispatch/combine 状态 |
| `win_all_card_data` | 所有卡的 Win 数据 |
| `win_analysis_error` | 异常分析结果 |
| `win_data` | 详细 Win 数据 |
| `win_all_card_expandidx` | 全卡 expert index |
| `profiling_all_data` | 全量 profiling 数据 |
| `profiling_avg_data` | 平均 profiling（含 jitter 分析） |

## 典型问题诊断

### 问题 1：Dispatch 卡住

**特征**：`dispatch count = combine count + 1`

**分析**：
- Dispatch 多执行了一次但 Combine 未跟上
- 检查 Win status 中哪些 core 还在等待

### 问题 2：Expert 数量不匹配

**特征**：分析报告 Error，不同 rank 的 expert 数量不一致

**修复**：确认 MoE 配置中所有 rank 的 expert 分配一致

### 问题 3：GlobalBS 不一致

**特征**：分析报告 Warning

**修复**：确认所有 rank 看到的全局 batch size 相同

### 问题 4：某 Core 状态全 0

**含义**：该 core 未参与计算

**可能原因**：
- 负载不均导致该 core 无工作
- Core 分配逻辑错误

## Profiling Jitter 分析

分析工具会计算各 rank 的 Duration 和 `aiv_*` 时间与均值的偏差：
- 偏差 > ±50%：标记为异常
- 用于发现 straggler（最慢的 rank）

## 注意事项

- `SOC_VERSION` 必须与实际芯片匹配，否则解析不完整
- MoE V1 不支持异常 dump，仅 V2 及以上
