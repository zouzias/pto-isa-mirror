# A5 TopK — 本地 UB radix 变体（`topk_ub`）

本目录与上游仓库里的 **`kernels/manual/a5/topk`** 脚手架**分开维护**，便于在 `git pull cann/pto-isa` 时保留你的实验实现（单次 GM 载入、全宽直方图/GATHER、`TCONCAT` 等）。

- 上游参考：[`../topk/README_zh.md`](../topk/README_zh.md)
- 构建与运行与 `topk` 相同，可执行文件名为 **`topk_ub`**：

```bash
cd kernels/manual/a5/topk_ub
bash run.sh -r sim -v Ascend950PR_9599
```

- 数据由本目录下 `scripts/gen_data.py` 生成，输出在 **`input/`、`output/`**（与 `../topk` 互不覆盖）。

## TGATHER EQ 分支与缓冲区大小说明

- `TGather.hpp` 中 `PTO_A5_TGATHER_B16_EQ_USE_VSCATTER` 默认开启（`1`）。
- 非 vscatter 路径下，`TGATHER<EQ>` worst case 可产生 **N** 个索引，EQ 缓冲区须按 **`N × 4` 字节** 规划。
- vscatter 路径下 EQ tile 可收成 **TopK**，但全等于阈值的用例仍建议在 golden 侧覆盖。

## 性能分析（A5 sim, 2026-05-19）

分析对象：`build/OPPROF_20260519153609_JHDTPDFLGZWDKNLC`（`core0.veccore0`）。

- **trace 时间**：`simulator/trace.json` 中各 VF 的 **`dur`（µs）**。
- **SIMD 周期**：`dump/core0.veccore0.instr_log.dump` 中 `PUSHQ … VF` 的 **`vf_real_execute_time`**（cycle-accurate）。
- **IPC**：`trace.json` 落在该 VF 时间窗内的 **`RV_*` 事件数 ÷ vf_real_execute_time**（用于阶段对比；多 VF 流水重叠时 trace 窗会略放大计数）。
- **PTO-ISA 列**：与 `draft.cpp` 五个 `Phase*` 及 `draft_topk_radix_ub.md` §2 对齐；**RV→PTO** 为模拟器底层指令簇到 pto 语义的映射。

**总览**：kernel trace 墙钟跨度约 **26.2 µs**；耗时主导为 **Phase5 `TGATHER<EQ>`（VF24）** 与两段直方图 **VF02 / VF14**。

### 按 Phase 聚合（VF ↔ Phase ↔ PTO-ISA）

| Phase | `draft.cpp` 函数 | VF 范围 | trace 耗时 (µs) | vf_real 周期 | IPC | 主要 PTO-ISA |
|---|---|---|---:|---:|---:|---|
| **1** | `Phase1_LoadAndHistogramMsb` | VF01–VF02 | 1.358 | 2417 | 1.50 | `TASSIGN` / `TLOAD` / `THISTOGRAM<BYTE_1>` / `TMOV` |
| **2** | `Phase2_WinnerMsbAndRemainK` | VF03–VF13 | 0.838 | 851 | 0.44 | `TCMPS` / `TCI` / `TSELS` / `TROWMIN` / `TGATHER` / `TSUB` |
| **3** | `Phase3_HistogramLsb` | VF14 | 1.464 | 2609 | 1.48 | `TCVT` / `THISTOGRAM<BYTE_0>` / `TMOV` |
| **4** | `Phase4_WinnerLsbRemainKAndPackedThresholdTor` | VF15–VF22 | 0.485 | 447 | 0.50 | `TCMPS` / `TSELS` / `TROWMIN` / `TGATHER` / `TCVT` / `TSHLS` / `TOR` |
| **5** | `Phase5_TgatherGtEqTconcatAndStore` | VF23–VF25 | 20.088 | 35680 | 0.52 | `TGATHER<GT>` / `TGATHER<EQ>` / `TCONCAT_IMPL` / `TSTORE` |

逐 VF 的 IPC 与 PTO-ISA 见下一章 [**各 VF 的 IPC 与 PTO-ISA 对照表**](#各-vf-的-ipc-与-pto-isa-对照表)。

### 关键热点（回归时优先看）

| VF | Phase | PTO-ISA | 说明 |
|---|---|---|---|
| **VF24** | 5 | `TGATHER<EQ>` | 单 VF 占 **~65%** trace 时间；`RV_VSQZ` + `RV_VSCATTER` 为主 |
| **VF23** | 5 | `TGATHER<GT>` | 次热点；`RV_VCMP_GT` + `RV_VLD` |
| **VF02** | 1 | `THISTOGRAM<BYTE_1>` | MSB 直方图；与 VF14 对称 |
| **VF14** | 3 | `THISTOGRAM<BYTE_0>` | LSB 直方图（带 MSB idx 过滤） |
| VF07–VF08 | 2 | `TROWMIN` / `TGATHER` | Phase2 控制流碎、vf_real 小，IPC 偏低属正常 |

更细的 UB 布局与 log 读法见 [`draft_topk_radix_ub.md`](draft_topk_radix_ub.md) §4。

## 各 VF 的 IPC 与 PTO-ISA 对照表

本表单独列出每个 VF 的 **IPC** 及其在 `draft.cpp` 中对应的 **PTO-ISA**（与上一节 Phase 聚合、热点表使用同一套 `OPPROF_20260519153609` 数据）。

**IPC 计算**：`IPC =`（`trace.json` 中该 VF 时间窗内的 `RV_*` 事件数）`÷ vf_real_execute_time`（`instr_log` 中 `PUSHQ … VF` 的 cycle-accurate 值）。

| VF | Phase | PC | trace (µs) | vf_real (cycle) | **IPC** | **PTO-ISA**（本 VF） |
|:---:|---:|---|---:|---:|---:|---|
| VF01 | 1 | `0x10d0d0dc` | 0.031 | 56 | **0.38** | `TASSIGN`、`set_flag` / `wait_flag`（MTE2↔V） |
| VF02 | 1 | `0x10d0d158` | 1.327 | 2361 | **1.53** | `TLOAD`、`TEXPANDS`、`THISTOGRAM<BYTE_1>`、`TMOV` |
| VF03 | 2 | `0x10d0d200` | 0.074 | 68 | **0.60** | `TCMPS`(GE)、`TCI`、`TSELS` |
| VF04 | 2 | `0x10d0d24c` | 0.062 | 63 | **0.68** | `TCI`、`TSELS` |
| VF05 | 2 | `0x10d0d328` | 0.080 | 119 | **0.64** | `TROWMIN`、`TGATHER` → `msbWinnerSaved` |
| VF06 | 2 | `0x10d0d3c4` | 0.068 | 55 | **0.36** | `TROWMIN`、`TSUB`、`TCMPS`、`TSEL` |
| VF07 | 2 | `0x10d0d454` | 0.142 | 214 | **0.36** | `TCMPS`、`TSEL`、`TGATHER` → `msbWinnerBin` |
| VF08 | 2 | `0x10d0d494` | 0.117 | 55 | **0.29** | `TGATHER` → `msbWinnerBin` |
| VF09 | 2 | `0x10d0d4d8` | 0.042 | 33 | **0.36** | `TEXPANDS`、`TGATHER`(C[w])、`TCMPS`、`TSEL` |
| VF10 | 2 | `0x10d0d528` | 0.059 | 55 | **0.29** | `TSEL`、`TSUB` → `remainKTile` |
| VF11 | 2 | `0x10d0d5a4` | 0.079 | 101 | **0.37** | `TSUB`、pipe 同步 |
| VF12 | 2 | `0x10d0d628` | 0.070 | 52 | **0.31** | `TEXPANDS`、`TCMPS`(EQ)、`TSEL`（`cwFix`，winner==0） |
| VF13 | 2 | `0x10d0d6d4` | 0.045 | 36 | **0.53** | `TSUB`（`remainK`）、收尾 |
| VF14 | 3 | `0x10d0d754` | 1.464 | 2609 | **1.48** | `TCVT`、`THISTOGRAM<BYTE_0>`、`TMOV` |
| VF15 | 4 | `0x10d0d800` | 0.073 | 68 | **0.62** | `TCMPS`(GT)、`TCI`、`TSELS` |
| VF16 | 4 | `0x10d0d84c` | 0.057 | 63 | **0.70** | `TCI`、`TSELS` |
| VF17 | 4 | `0x10d0d91c` | 0.080 | 119 | **0.65** | `TROWMIN`、`TGATHER` → `lsbWinnerBin` |
| VF18 | 4 | `0x10d0d98c` | 0.068 | 55 | **0.31** | `TROWMIN`、`TGATHER` |
| VF19 | 4 | `0x10d0da0c` | 0.042 | 34 | **0.29** | `TCVT`、`TSHLS`、`TCVT` |
| VF20 | 4 | `0x10d0dab8` | 0.052 | 36 | **0.36** | `TOR`、`set_flag` / `wait_flag`（V↔S） |
| VF21 | 4 | `0x10d0db38` | 0.056 | 34 | **0.29** | `TASSIGN`（`packedThrU`） |
| VF22 | 4 | `0x10d0dbe8` | 0.057 | 38 | **0.29** | `TASSIGN`、同步 |
| VF23 | 5 | `0x10d0dc70` | 2.910 | 5174 | **0.99** | **`TGATHER<GT>`**（全宽 N） |
| VF24 | 5 | `0x10d0dd04` | 16.882 | 30236 | **0.44** | **`TGATHER<EQ>`**（全宽 N，vscatter） |
| VF25 | 5 | `0x10d0dd4c` | 0.296 | 270 | **0.58** | **`TCONCAT_IMPL`**、`set_flag` / `wait_flag`（V↔MTE3）、**`TSTORE`** |

**读表提示**

- **IPC 高（≈1.5）**：VF02、VF14 — 直方图主循环，`THISTOGRAM` 吞吐接近饱和。
- **IPC 低（≈0.3–0.7）**：VF03–VF13、VF15–VF22 — winner / `remainK` 路径，标量与 predicate 占比高，`vf_real` 偏小。
- **trace 长但 IPC 低**：VF24 — `TGATHER<EQ>` 耗时长（`RV_VSQZ` + `RV_VSCATTER`），IPC 约 0.44。
- **IPC 接近 1**：VF23 — `TGATHER<GT>`，向量比较 + 压缩为主。
