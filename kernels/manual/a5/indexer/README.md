# Indexer 开发与定位记录（MXFP8 GEMM -> ReLU -> scale -> reduce -> TopK）

本文档记录 `kernels/manual/a5/indexer` 的实现与定位经验，重点沉淀：
- 目标算子与约束
- TopK 关键实现（PTO-ISA 5-phase）
- 真实踩坑与根因
- 当前可复现的验证基线

---

## 1) 目标与规格

目标流程（Lighting Indexer）：
1. `Q[2,64,128] x K[N,128]`（MXFP8 GEMM）
2. `ReLU`
3. 乘 `post_scale[2,64]`（head 维广播）
4. 沿 head 维 `reduce_sum`，得到 `score[2,N]`
5. `TopK`（当前 smoke 用 `N=1024, K=512`；最终目标 `N=131072, K=2048`）

约束：
- `post_scale` 可为负数
- 输出 index 为 `uint32`
- TopK tie-break 允许非稳定（不要求严格保序）

---

## 2) 代码结构

核心文件：
- `kernels/manual/a5/indexer/indexer_mxfp8_kernel.cpp`
- `kernels/manual/a5/indexer/main.cpp`
- `kernels/manual/a5/indexer/scripts/gen_data.py`
- `kernels/manual/a5/indexer/scripts/radix_topk_golden_stats_bf16.py`

按阶段入口：
- `LaunchIndexerMatmul`
- `LaunchIndexerPostProcess`
- `LaunchIndexerTopK`
- `LaunchIndexerMxfp8`（all-in-one）

`main.cpp` 运行方式：
- `./indexer_mxfp8 1`：matmul
- `./indexer_mxfp8 2`：postprocess
- `./indexer_mxfp8 3`：topk
- `./indexer_mxfp8 all`（或不传参数）

---

## 3) TopK 实现要点（当前版本）

TopK 输入是 `score_bf16`（`uint16` bit pattern），先做 ordered-key 映射：
- 负数：`~bits`
- 非负：`bits ^ 0x8000`
- 指令组合：`TNOT + TXORS + TCMPS + TSEL`

5-phase（参考 `topk_ub`）：
1. `THISTOGRAM(BYTE_1)` 得到 MSB 统计
2. 选 MSB winner，并计算 `remainK`
3. 以 winner-MSB 过滤后做 `THISTOGRAM(BYTE_0)`
4. 选 LSB winner，`TSHLS + TOR` 拼 16-bit 阈值
5. `TGATHER(GT/EQ) + TCONCAT + TSTORE` 输出 index

---

## 4) 关键问题与最终定位结论

### 4.1 同步问题（已修复）

现象：
- `score` 阶段出现大量 `act->0` 或明显异常

根因：
- `TLOAD/TSTORE` 与向量计算之间缺少反向依赖/同步

修复：
- 在关键路径补 `set_flag/wait_flag`
- 特别是 `score` 输出前后增加 `PIPE_V <-> PIPE_MTE3` 同步

---

### 4.2 TopK 输入异常：`VNOT` 前出现大量 0（已定位并修复）

现象：
- `CHISTV2` 输入出现大量 `0x8080`
- `VNOT` 输入有很多 `0x0000`
- 与 `score_bf16` 文件不一致

证据链：
- 在 `veccore0.instr_log.dump` 定位 `MOV_SRC_TO_DST_ALIGNv2`（TopK `TLOAD`）
- 在 `veccore0.ub.wr_log.dump` 看到写入范围是 `0x0 ~ 0x7fc`（2048 B，正好 1024 个 bf16）
- 但后续 `RV_VLDI` 序列会读到 `0x800` 段，出现读写窗口错位

结论：
- 问题不是 TopK 逻辑本身，而是输入窗口/读写对齐错误导致读到未覆盖 UB 区域

---

### 4.3 `float -> bf16` 语义问题（已修复）

现象：
- 使用 `TCVT(..., RoundMode::CAST_RINT)` 后，`topk` 大面积偏差（值级别严重漂移）

根因：
- `CAST_RINT` 是数值四舍六入五成双，不等价原先用于 key 的截断语义

修复策略：
- 统一改为 PTO-ISA `TCVT(..., RoundMode::CAST_TRUNC)`
- 替换所有 scalar `__builtin_bit_cast(... ) >> 16` 路径
- `ScoreBf16Tile`/`ScoreBf16Global` 使用 `bfloat16_t` tile/global，保持向量转换和存储一致

备注：
- 这里的目标是保持 TopK key 语义稳定，而不是追求数值 round-to-nearest

---

## 5) Host 校验策略（当前）

`main.cpp` 的 TopK 校验分两层：
1. 先做逐元素顺序比较（deterministic）
2. 若失败，再做排序后集合比较（multiset）

因此会出现状态：
- `topk multiset test success`
- `topk idx order differs, but sorted idx set matches`

这表示：
- 选中的样本集合正确
- 顺序与 deterministic 规则不一致（可接受，前提是业务允许 non-stable tie-break）

---

## 6) 当前可复现状态（2026-05-05）

配置：
- `INDEXER_TEST_N=1024`
- `INDEXER_TOPK=512`
- `batch=2`

`./indexer_mxfp8 all` 结果：
- `matmul test success`
- `score test success`
- `topk multiset test success`
- `topk idx order differs, but sorted idx set matches`

结论：
- 全流程（matmul + postprocess + topk）已跑通
- TopK 集合正确，顺序差异符合 non-deterministic tie-break 预期

---

## 7) 回归与定位建议

1. 先看 `process 1/2/3` 分阶段结果，再跑 `all`
2. 出现 TopK 异常时，优先核对：
   - `TLOAD` 写入窗口 vs 后续 `VLD` 读取窗口
   - `float->bf16` 转换模式是否仍为 `CAST_TRUNC`
3. 保留并使用以下日志做证据链：
   - `build/core0.veccore0.instr_log.dump`
   - `build/core0.veccore0.ub.wr_log.dump`
   - `build/core0.veccore0.ub.rd_log.dump`
   - `build/core0.veccore0.rvec_pv.dump`
   - `build/core0.biu.bwif.wr_log.dump`

---

## 8) 常用命令

在 `kernels/manual/a5/indexer`：

```bash
# 生成 smoke 数据
INDEXER_TEST_N=1024 INDEXER_TOPK=512 ../../../../.venv-indexer-sim/bin/python scripts/gen_data.py

# 运行环境
source /usr/local/Ascend/cann_9b2/cann/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend910_9599/lib:$LD_LIBRARY_PATH

# 分阶段
./build/indexer_mxfp8 1
./build/indexer_mxfp8 2
./build/indexer_mxfp8 3

# 全流程
./build/indexer_mxfp8 all

# BF16 radix 统计
INDEXER_TEST_N=1024 INDEXER_TOPK=512 python3 scripts/radix_topk_golden_stats_bf16.py
```
