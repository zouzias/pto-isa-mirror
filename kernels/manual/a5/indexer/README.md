# Indexer 开发与定位记录（MXFP8 GEMM -> ReLU -> scale -> reduce -> TopK）

本文档记录 `kernels/manual/a5/indexer` 从需求对齐到当前可运行状态的完整过程，重点覆盖：
- 目标算子定义
- 代码结构演进
- 关键编译/运行问题与修复
- TopK 定位结论与当前状态

---

## 1) 目标与规格

目标实现「真实 Lighting Indexer」流程：

1. `Q[2,64,128] x K[131072,128]`（MXFP8 GEMM）
2. `ReLU`
3. 乘 `post_scale[2,64]`（按 head 广播）
4. 沿 head 维 `reduce_sum`，得到 `score[2,131072]`
5. TopK（`K=2048`，仅输出 index，`uint32`）

补充约束：
- `post_scale` 允许负值
- TopK 允许非稳定 tie-break（算法层面）
- host 侧最初使用 deterministic 规则做黄金对比（`key desc, idx asc`）

---

## 2) 代码落地总览

核心文件：
- `kernels/manual/a5/indexer/indexer_mxfp8_kernel.cpp`
- `kernels/manual/a5/indexer/main.cpp`
- `kernels/manual/a5/indexer/scripts/gen_data.py`
- `kernels/manual/a5/indexer/scripts/radix_topk_golden_stats_bf16.py`

功能拆分：
- `LaunchIndexerMatmul`：GEMM 阶段
- `LaunchIndexerPostProcess`：ReLU + scale + reduce_sum（AIV）
- `LaunchIndexerTopK`：TopK 阶段
- `LaunchIndexerMxfp8`：全流程串接

`main.cpp` 支持按阶段运行：
- `./indexer_mxfp8 1`：matmul
- `./indexer_mxfp8 2`：postprocess
- `./indexer_mxfp8 3`：topk
- `./indexer_mxfp8`：all

---

## 3) 开发演进（按问题驱动）

### 3.1 基础打通阶段

- 先完成 MXFP8 matmul 基础路径，参考 `matmul_mxfp8_performance`。
- 将后处理从 host 搬到 device（AIV），并输出 `score` 与 `score_bf16`。
- 将 TopK 接入到 device kernel，输出 `output_idx.bin`。

### 3.2 关键正确性问题与修复

1. **输入/输出尺寸不一致**
   - 现象：`file size larger than buffer size`
   - 修复：统一 `M/N/TopK` 配置，重生数据，修正 host 端 file size 计算。

2. **TLOAD/TSTORE 与向量计算同步问题**
   - 现象：中间结果全 0 或明显错误
   - 修复：在关键点补 `set_flag/wait_flag` 反向依赖与管线同步。

3. **TopK scalar 版本卡死/行为异常**
   - 修复：重构成 5-phase PTO 风格实现，参考 `topk_ub`。

4. **`TCVT` 编译失败（`TypeGet<__bf16>` / `vlds`）**
   - 触发点：在当前 mixed 编译配置里直接走 `TCVT.hpp`
   - 处理：
     - `TopKFromScoreKernel` 和 `TCvt.hpp` 相关路径改为 `__DAV_VEC__` 下编译
     - 非 vec 路径提供 stub，避免链接/符号缺失
   - 结论：该问题本质是编译路径与 `TCVT` 类型支持不匹配，不是 TopK 算法逻辑错误。

5. **大 N 下 UB 越界**
   - 现象：`ub_addr_overflow`，例如访问到 `0x40000` 以上
   - 根因：TopK kernel 使用了 `1 x kLength` 的大 tile UB 映射，小 N 思路直接套到 `N=131072`
   - 处理：调回小 N smoke（`N=1024, TopK=512`）继续定位算法正确性。

### 3.3 TopK 关键重构点

- 输入改为 `bf16` key 路线（`score_bf16`）
- 增加 ordered-key 映射：
  - 负数：`~bits`
  - 非负：`bits ^ 0x8000`
  - 指令：`TNOT + TXORS + TCMPS + TSEL`
- 5-phase 流程：
  1. `THISTOGRAM(BYTE_1)`（MSB）
  2. 选 MSB winner + remainK
  3. `THISTOGRAM(BYTE_0)`（在 winner-MSB 条件下）
  4. 选 LSB winner，拼阈值（`TSHLS + TOR`）
  5. `TGATHER(GT/EQ) + TCONCAT + TSTORE`

---

## 4) 定位方法与证据链

主要日志：
- `build/core0.biu.bwif.wr_log.dump`
- `build/core0.veccore0.ub.wr_log.dump`
- `build/core0.veccore0.ub.rd_log.dump`
- `build/core0.veccore0.rvec_pv.dump`
- `build/core0.veccore0.instr_log.dump`

辅助脚本：
- `scripts/radix_topk_golden_stats_bf16.py`
  - 用 `golden_score.bin` 推导 BF16 ordered key
  - 输出 MSB/LSB winner、remainK、packed_threshold、GT/EQ 索引集合

关键定位结论：
- `bwif.wr_log` 与 `radix_topk_golden_stats_bf16.py` 对齐，说明前半段 radix 决策链条正确。
- `topk multiset` 通过而 `topk idx` 失败，说明问题集中在 index 输出顺序（而非阈值/样本集合错误）。

---

## 5) Host 对比逻辑演进

文件：`main.cpp`

现状：
1. 先做原始 `idx` 逐元素比较（deterministic）
2. 若失败，再按 batch 内排序后比较集合

新增输出语义：
- `topk idx order differs, but sorted idx set matches`

这用于区分：
- 算法集合正确但顺序不同（目前状态）
- 真正选错索引集合

---

## 6) 当前状态（最新）

当前 smoke 配置：
- `INDEXER_TEST_N=1024`
- `INDEXER_TOPK=512`
- `batch=2`

运行 `process 3` 结果：
- `topk multiset test success`
- `topk idx order differs, but sorted idx set matches`

结论：
- TopK 样本集合与 golden 一致
- 顺序与 deterministic golden 不一致（属于 tie/order 规则差异）

---

## 7) 后续建议

1. 若业务允许 non-deterministic tie-break：
   - 保留“集合一致”为通过标准
2. 若必须与 deterministic golden 完全一致：
   - 在 kernel 末段对 `GT/EQ` 合并后的索引做稳定排序规则对齐（`key desc, idx asc`）
3. 若要回到大 N（131072）：
   - TopK 必须改为 chunk/stream UB 方案，不能继续使用 `1 x kLength` 大 tile 直铺 UB

---

## 8) 常用命令

在 `kernels/manual/a5/indexer`：

```bash
# 生成小 N 数据
INDEXER_TEST_N=1024 INDEXER_TOPK=512 ../../../../.venv-indexer-sim/bin/python scripts/gen_data.py

# 运行 topk-only
source /usr/local/Ascend/cann_9b2/cann/set_env.sh
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/Ascend910_9599/lib:$LD_LIBRARY_PATH
./build/indexer_mxfp8 3

# 打印 BF16 radix 黄金统计
INDEXER_TEST_N=1024 INDEXER_TOPK=512 python3 scripts/radix_topk_golden_stats_bf16.py
```
