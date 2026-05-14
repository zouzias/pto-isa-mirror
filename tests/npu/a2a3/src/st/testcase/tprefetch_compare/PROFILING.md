# tprefetch_compare 性能定位指南

本目录的测试用例（A / B / C / E1）回答的是 "host `pto::PTO_PREFETCH` vs device
`pto::TPREFETCH_L2` 谁更快"。如果想进一步**定位差异到底来自哪一段**（kernel
launch、session build、SQE 提交、SDMA 引擎、busy poll …），需要把 wall-clock
微基准和 `msprof` profiling 结合起来。

本文是这套 P0 实验的操作手册。

---

## 1. 实验目标

在 A 场景里观察到：

* 小数据 (≤ 1 MB)：host 路径比 device 快 ~4 μs
* 大数据 (≥ 16 MB)：device 路径比 host 快 ~5 %

要把这两个差异拆开，回答两个具体问题：

| 问题 | 用什么定位 |
|---|---|
| Q1：device 多付的 ~6 μs 启动开销里，"launch + dispatch + sync" 占多少？ | **E1 NoopKernel**（本目录 P0 实验之一） |
| Q2：host 路径和 device 路径在硬件上是不是同一条 SDMA 通道？ | **msprof**（CANN 自带 profiling 工具） |

---

## 2. 跑 E1 NoopKernel —— 量化 launch+dispatch+sync 成本

### 2.1 命令

```bash
# 跑全部场景（含 E1）
python3 tests/script/run_st.py -r npu -v a3 -t tprefetch_compare

# 只跑 E1（最快，~30 秒）
GTEST_FILTER='TPrefetchCompare.E1_NoopKernel' \
  python3 tests/script/run_st.py -r npu -v a3 -t tprefetch_compare

# CSV 写到指定目录方便后处理
TPREFETCH_COMPARE_CSV_DIR=/tmp/prefetch_csv \
GTEST_FILTER='TPrefetchCompare.E1_NoopKernel' \
  python3 tests/script/run_st.py -r npu -v a3 -t tprefetch_compare
```

### 2.2 预期输出

```
[PERF] Scenario E1 - kernel launch + dispatch + sync overhead (NoopKernel)
  Iterations:            100 (warmup=1)
  Syscnt freq assumed:   100.00 MHz
  --- end-to-end wall (the device-path "launch tax") ---
  noop kernel wall       p50=5.XXus   p5=4.XX p95=6.XX
                         (host launch + STARS dispatch + AICORE accept + sync return)
  --- supplementary: in-kernel syscnt sanity ---
  in-kernel pipe_barrier p50=0.0Xus   p5=0.0X p95=0.0X
                         (should be sub-microsecond)
  HOWTO: subtract the wall p50 above from any other scenario's
         device wall to isolate the in-kernel cost only.
  CSV: ./tprefetch_compare_scenarioE1.csv
```

### 2.3 怎么用这个数

设：
* `E1_p50`  = E1 NoopKernel wall p50
* `A_dev_p50_S` = Scenario A 在 size=S 上的 device wall p50
* `A_host_p50_S` = Scenario A 在 size=S 上的 host wall p50

则可推：

```
device_in_kernel_overhead_S
    = A_dev_p50_S − E1_p50         # 扣掉"launch+dispatch+sync"的纯 in-kernel 成本

fair_in_kernel_overhead_vs_host_S
    = (A_dev_p50_S − E1_p50) − A_host_p50_S
                                    # 注意 host 不需要 launch，所以不扣
```

**写在汇报里就长这样**：

> 100 次中位数：device 路径在 1 MB 上 wall 31 μs，扣掉 NoopKernel 测得的 5 μs
> launch+dispatch+sync 后，纯 in-kernel 成本（build session ×2 + 提交 SQE +
> SDMA 1 MB + busy poll）= **26 μs**；host 路径同 size 是 27 μs。两者 in-kernel
> 工作量基本相当，差距全部来自 device 必须额外付的 launch tax。

---

## 3. 用 msprof 看 prefetch 落到哪个引擎

`msprof` 是 CANN 自带的硬件 profiler，timeline 能区分到每一个 task 落在哪个
engine 上（AICORE / SDMA / CMO / HCCL …）。这一步的目标是**直接确认** host
路径和 device 路径在硬件层是不是同一条物理路径。

### 3.1 准备工作

```bash
# 1) 确认 msprof 在 PATH 里（CANN toolkit 自带）
which msprof
# 通常是 /usr/local/Ascend/ascend-toolkit/latest/tools/profiler/bin/msprof

# 2) 准备一个最小的运行脚本，把 host 路径 / device 路径分别跑一次
# 直接复用 GTEST_FILTER 跑 Scenario A 1MB 即可（覆盖 host + device）
```

### 3.2 抓 profile

```bash
# 输出目录建议每次实验单独一个
PROF_OUT=/tmp/prof_$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROF_OUT"

msprof --output="$PROF_OUT" \
       --application="python3 tests/script/run_st.py -r npu -v a3 -t tprefetch_compare \
                       -- --gtest_filter=TPrefetchCompare.A_EndToEnd_1MB" \
       --task-time=on \
       --aic-metrics=PipeUtilization
```

> 注意：
> * `--application` 要传**完整可执行命令**（msprof 通过 fork+exec 启动）。
> * 如果 `run_st.py` 不接受 `--gtest_filter`，可以在 shell 里 export
>   `GTEST_FILTER='TPrefetchCompare.A_EndToEnd_1MB'`，把 `--application` 简化为
>   `bash -c 'GTEST_FILTER=... python3 ...'`。
> * iteration 默认 100 太多，可以 `TPREFETCH_COMPARE_ITER=10` 让 msprof
>   跑得快一点。

### 3.3 导出可读结果

```bash
# 把二进制 profile 转成 CSV / JSON
msprof --export=on --output="$PROF_OUT"

# 关键的两份产物：
#   $PROF_OUT/PROF_*/mindstudio_profiler_output/task_time_*.csv
#   $PROF_OUT/PROF_*/mindstudio_profiler_output/op_summary_*.csv
```

### 3.4 看哪一列

打开 `task_time_*.csv`，关心这几个列：

| 列名 | 含义 |
|---|---|
| `Op Name` | 任务名（包含 `aclrtCmoAsync` / kernel 名 / `SDMA_*` 等） |
| `Task Type` | 任务类型（HCCL / KERNEL / SDMA / HOST_API …） |
| `Stream Id` | 落在哪条 stream |
| `Task Start Time(us)` / `Task Duration(us)` | 时间戳 / 时长 |
| **`Device Engine`** 或 **`Engine Name`** | **关键列**——硬件引擎名 |

**判断方法**：

```
host 路径 (aclrtCmoAsync)：
    Op Name 含 "aclrtCmoAsync" 或 "RtCmoAsync"
    → 看它的 Engine 是 "CMO" 还是 "SDMA"

device 路径 (TPREFETCH_L2 from kernel)：
    Op Name 是 ScenarioA_DevicePrefetchKernel
    → 它内部再发 SDMA SQE，会出现一行新任务
    → 看新任务的 Engine 是 "CMO" 还是 "SDMA"
```

### 3.5 三种可能结论

| 看到的现象 | 结论 |
|---|---|
| host 落 `CMO`，device 落 `SDMA` | **直接证实**两条路径用的是不同的物理硬件单元，5 % 带宽差有硬件依据。 |
| 两者都落 `SDMA`（不同 channel） | 物理引擎相同，5 % 差异只能归到 SQE 切块策略 / 调度路径，不是引擎本身的快慢。 |
| 两者都落 `CMO` | 不太可能（device 代码里明确发 `RT_STARS_SQE_TYPE_SDMA`），如出现说明 STARS 做了透明转换，需要找内部 HAS 文档进一步确认。 |

### 3.6 timeline 可视化（可选）

如果想直接看时间轴气泡，把 `$PROF_OUT/PROF_*/timeline/msprof_*.json` 拖进
[chrome://tracing](chrome://tracing) 或者 MindStudio 的 Timeline 视图。在小数据
场景下，host 路径里 prefetch 任务和后续 kernel 之间那个 "3-5 μs 调度气泡" 在
timeline 上能直接看见（两条任务条之间的空白）。

---

## 4. 输出物清单

跑完 P0 两步实验，应该手上有：

| 文件 | 用途 |
|---|---|
| `tprefetch_compare_scenarioA.csv` | 各 size 上 host / device 的 wall p5/p50/p95 |
| `tprefetch_compare_scenarioE1.csv` | NoopKernel wall p5/p50/p95（一行） |
| `$PROF_OUT/.../task_time_*.csv` | 每条 task 落到的 engine（msprof 导出） |
| `$PROF_OUT/.../timeline/msprof_*.json` | timeline 可视化（可选） |

**汇报时的可信度顺序**：

1. **CSV 实测数字**（A + E1）—— 你自己跑的，可重复，最硬。
2. **msprof engine 列**—— 工具背书的硬件归属，比讲义里"我推断"更有力。
3. **代码引用 + 调用链**（`prefetch_talk.md` §1.4）—— 解释机制。

---

## 5. 后续实验（P1 / P2，按需要再加）

参考 `prefetch_host_vs_device.md` 末尾的实验设计章节。E2 / E3 / E4 / E5 分别隔
离 transient session build、SQE 提交、host 并行隐藏、SDMA 切块影响。这次没全
做是因为 P0 的 E1 + msprof 已经能定位 90% 的解释，剩余的 P1+ 是为更细的对照实
验做准备。
