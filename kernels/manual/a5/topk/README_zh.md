# A5 TopK 算子（框架）

本目录为 **Ascend A5** 上的 TopK 示例工程；**设备侧**写在 **`draft.cpp`**，与 `kernels/manual/a2a3/topk` 的排序归并路线不同。

与 **`kernels/manual/a5/topk_ub`** 对齐：**共五个 `Phase*`**，所有 PTO 指令（含 `TASSIGN` / `TLOAD` 等）只出现在这些 Phase 中。本工程为 **分块从 GM 读入**（N = 8192，每块 2048 列，共 4 tile）；`topk_ub` 为 **整段 key 在 UB**、整段比较 `TGATHER`。

## 当前用例

- 输入：**`[1, 8192]`** `uint16` key（`input/keys.bin`）
- 输出：**Top-512** 个下标（按 **uint16 全序**取最大的 512 个 key；**不要求**对输出下标排序）
- 校验：host 上比较「`keys[out[i]]` 的多重集合」是否与 golden 一致（`scripts/gen_data.py` 生成 `golden_topk_multiset.bin`）

## 流程概要（`draft.cpp` 与五个 Phase）

针对 **uint16 可排序 key** 的 **Radix-Select TopK**（2 字节）：

1. **Phase1** — UB `TASSIGN`，按 tile 从 GM `TLOAD`，`THISTOGRAM<HistByte::BYTE_1>`（MSB）累加成 `chistMSB`
2. **Phase2** — MSB 胜出（`TCMPS` / `TCI` / `TSELS`），`WinnerBinU8` 路径；`winner==0` 时强制 `C[-1]=0`；`TGATHER` + `TSUB` 得 `remainK`
3. **Phase3** — `TCVT` 写 `idxFilter`；再按 tile `TLOAD`，`THISTOGRAM<BYTE_0>`（LSB，带 MSB 过滤）累加成 `chistLSB`
4. **Phase4** — LSB 胜出（`TCMPS` `GT` vs `remainK` tile），`TOR` 得到 **16 位 packed 阈值**（`kRemainUbOut`）
5. **Phase5** — 每 tile **比较 `TGATHER`（GT/EQ）**；每 tile **六参 `TCONCAT_IMPL`（`NeetCntDstIdx`）** 累加到 `gtSeg`/`eqSeg`（`idx*Out` → `TMOV` → `idx*Acc`）；最后五参合并 + `TSTORE`

## 数据生成

```bash
python3 scripts/gen_data.py --seed <int>      # 随机 key
python3 scripts/gen_data.py --const 0x1234  # 全同值（走 EQ 路径）
```

## 目录与构建相关

- `CMakeLists.txt` 使用 **`BEFORE` 本仓库 `include/`**，使 `pto-isa` 头文件优先于 `ASCEND_HOME_PATH` 中的旧版头，与 `topk_ub` 一致
- `scripts/radix_topk_golden_stats.py` — 打印理论 MSB/LSB winner 与 GT|EQ 数量

```
kernels/manual/a5/topk/
├── CMakeLists.txt
├── draft.cpp
├── main.cpp
├── scripts/gen_data.py
├── run.sh
├── README.md
└── README_zh.md
```

## 可参考的仓库内代码

- **同算法 UB 全宽版：** `kernels/manual/a5/topk_ub/`
- A5 `TMRGSORT`：`tests/npu/a5/src/st/testcase/tmrgsort/`
- A5 `TSORT32` / `TGATHER`：`tests/npu/a5/src/st/testcase/tsort32/`、`tgather/`
- A2/A3 TopK 示例（仅作思路参考）：`kernels/manual/a2a3/topk/`

## 构建与运行

需已 `source` CANN 的 `set_env.sh`，**`bisheng` 在 `PATH` 中**，`ASCEND_HOME_PATH` 已设置。

**仿真**（**`Ascend950PR_9599`**，勿用 `Ascend310P*`）：

```bash
cd kernels/manual/a5/topk
bash run.sh -r sim -v Ascend950PR_9599
```

**真机**：

```bash
bash run.sh -r npu -v <你的 A5 板卡对应 SOC 字符串>
```

## 回归（8K）

在 `kernels/manual/a5/topk` 下：

```bash
source $ASCEND_HOME_PATH/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$ASCEND_HOME_PATH/aarch64-linux/lib64:$LD_LIBRARY_PATH
cd build && make -j16 && cd ..
for s in 1241200609 2203936584 191132090; do
  python3 scripts/gen_data.py --seed $s && cd build && ./topk | grep RESULT; cd ..
done
python3 scripts/gen_data.py --const 0x1234 && cd build && ./topk | grep RESULT
```

单次仿真约 **6–10 min**（8K 为 32 tile）；应全部为 **`RESULT: PASS`**。

## 性能（8K 仿真，seed `1241200609`）

数据目录：`perf/`（`msprof op simulator ./topk`，`Ascend950PR_9599`，`core0.veccore0`）。

| 文件 | 说明 |
|------|------|
| `perf/trace_8k_seed1241200609_veccore0.json` | Chrome trace — **当前**（`kTileCols=2048`，256 列切片） |
| `perf/phase_perf_8k_seed1241200609.json` | Phase VF / PMU 汇总 — **当前**（由 `build/` dump 解析） |
| `perf/trace_*_tile256.json` | 归档 trace — `kTileCols=256`、32 个 GM tile（旧基线） |
| `perf/phase_perf_*_tile256.json` | 上述 tile256 运行的 Phase 汇总 |

`msprof` 后拷贝 trace 并解析（需 source CANN `set_env.sh` + 仿真器 `LD_LIBRARY_PATH`，见 `run.sh`）：

```bash
cp build/OPPROF_*/simulator/core0.veccore0/trace.json \
  perf/trace_8k_seed1241200609_veccore0.json
python3 scripts/parse_phase_perf.py --build-dir build --seed 1241200609 \
  --opprof-dir build/OPPROF_<latest> \
  -o perf/phase_perf_8k_seed1241200609.json
```

当前数据：`OPPROF_20260529101806_EXOOUYLTJYJMATAY`（2048 GM 块 + 256 列切片）。

### 分 Phase 汇总（`vf_real_execute_time`，单位 cycle）

| Phase | VF 数 | vf_real | 占 rvec | 主要 PTO |
|-------|------:|--------:|--------:|----------|
| Phase1 初始化 | 1 | 520 | 1.5% | `TASSIGN` / `TEXPANDS` |
| Phase1+3 直方图（4×8 切片×2 遍） | 224 | 14,569 | 42.2% | `TLOAD` + `THISTOGRAM` |
| Phase2+4 控制 | 20 | 1,509 | 4.4% | `TCMPS` / `TSELS` / `TOR` |
| Phase5（32 切片×3 模板） | 96 | 17,895 | 51.9% | `TGATHER` / `TCONCAT` / `TSTORE` |

**PMU**：kernel **87,221** tick；**MTE2 约 75.7%**（66,005 cyc）；**rvec 约 39.5%**；`msprof` 墙钟 **48.37 µs**。瓶颈仍在 GM `TLOAD` 与 `wait_flag`。

**Phase5 热点**（每 256 列切片 `vf_real`）：`0x10d0e348` ~221；`0x10d0e310` ~188；`0x10d0e084` ~151 cyc。

**同步**：Phase2 `TSUB`、Phase4 `TOR` 之后的手写 `PIPE_V`/`PIPE_S` fence 已去掉；8K 仿真 3 seed + `--const` 仍 **PASS**。
