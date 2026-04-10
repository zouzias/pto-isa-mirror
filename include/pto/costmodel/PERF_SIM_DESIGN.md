# PTO-ISA Perf-Sim 设计文档

## 1. 目标

在 PTO-ISA costmodel 基础上，构建 **perf-sim** 性能仿真，提供：

- 算子级的性能预测与 trace 分析
- 基于 Pipeline 时序的泳道图可视化
- 白盒性能指标输出

---

## 2. 核心问题：算子边界界定

### 2.1 问题分析

如果在PTO-ISA仓里面实现perf-sim性能仿真，目前没有现成的方法识别算子的边界，只能识别一系列PTO ISA指令，但是无法识别是一个整算子。

PTO-ISA 当前的 trace 机制（`PtoInstrScope`）只能捕获**单条 PTO 指令**的边界：

```
TLOAD → TADD → TMUL → TSTORE    // 这4条指令构成一个算子？还是2个？没有标记
```

### 2.2 实现方案

提供 RAII 风格的 `OperatorScope` 宏，让用户在算子入口/出口显式标记：

```cpp
void my_softmax_kernel(/* ... */) {
    PERF_SIM_OPERATOR("softmax");
    // ... PTO 指令序列 ...
    TLOAD(a, src);
    TROWMAX(max_val, a);
    TSUBS(sub, a, max_val);
    TEXP(exp_val, sub);
    TROWSUM(sum_val, exp_val);
    TDIVS(output, exp_val, sum_val);
    TSTORE(dst, output);
}
```


### 2.3 数据结构扩展

在现有 `TraceState` 基础上增加算子层：

```cpp
// 新增：算子级记录
struct OperatorRecord {
    std::string name;                        // 算子名
    std::vector<PtoInstrRecord> instructions; // 包含的 PTO 指令序列
    // 运行时自动填充：
    uint64_t total_cycles = 0;
    bool supported = true;
};

// 扩展 TraceState
struct TraceState {
    // --- 原有字段 ---
    std::vector<PtoInstrRecord> executed_pto;
    std::vector<CceCallRecord> raw_cce_calls;
    std::vector<std::size_t> active_pto_stack;

    // --- perf-sim 新增 ---
    std::vector<TraceEntry> timeline;                  // 统一时间线（见 4.1.3）
    std::vector<OperatorRecord> operators;
    std::vector<std::size_t> active_operator_stack;
};

// 新增：算子级 RAII
class OperatorScope {
public:
    explicit OperatorScope(std::string_view name);
    ~OperatorScope();
    OperatorScope(const OperatorScope&) = delete;
    OperatorScope& operator=(const OperatorScope&) = delete;
};
```

### 2.4 API 定义

```cpp
namespace pto::perf_sim {

class OperatorScope { /* ... */ };

// 宏快捷方式（RAII，作用域结束自动析构）
#define PERF_SIM_OPERATOR(name)  ::pto::perf_sim::OperatorScope _op_scope(name)

} // namespace pto::perf_sim
```

---

## 3. 整体架构

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户 PTO 代码                          │
│  PERF_SIM_OPERATOR("matmul_add")                        │
│    TLOAD / TMATMUL / TADD / TSTORE                      │
├─────────────────────────────────────────────────────────┤
│                  perf-sim 框架层                          │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  trace    │  │  pipeline    │  │  reporter         │  │
│  │  扩展     │  │  scheduler   │  │  报告生成         │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  operator │  │  metrics     │  │  swimlane         │  │
│  │  scope    │  │  白盒指标    │  │  泳道图生成       │  │
│  └──────────┘  └──────────────┘  └───────────────────┘  │
├─────────────────────────────────────────────────────────┤
│                现有 costmodel 基础层                      │
│  arch_config / cce_evaluator / trace / pto_instr        │
└─────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```text
include/pto/costmodel/
├── ... (现有文件不变) ...
└── perf_sim/
    ├── operator_scope.hpp      算子边界标记
    ├── pipeline_model.hpp      Pipeline 时序建模
    ├── metrics.hpp             白盒指标采集
    ├── reporter.hpp            报告生成框架
    ├── swimlane.hpp            泳道图数据生成
    └── swimlane_types.hpp      泳道图数据结构
```

---

## 4. 模块详细设计

### 4.1 Pipeline 时序建模（pipeline_model.hpp）

现有 costmodel 只做 cycle 简单累加，不考虑 Pipeline 时序。perf-sim 需要建模 NPU 的 C&V 双核并行流水线。

#### 4.1.1 双核流水线结构

NPU AICore 由 Cube 核和 Vector 核组成，各自拥有独立的 Pipeline，通过 FFTS 硬件通道跨核同步：

```
Cube 核:   MTE2(搬入) → MTE1 → Matrix(矩阵计算) → FIX(定点转换) → MTE3(搬出)
Vector 核: MTE2(搬入) → Vector(向量计算) → MTE3(搬出)
                       ↕ FFTS 跨核同步 ↕
```

```cpp
enum class CoreKind : uint8_t { Cube, Vector };

enum class PipeStage : uint8_t {
    Scalar,     // 标量控制：set_flag, wait_flag, ffts_cross_core_sync
    MTE2,       // 搬入：copy_gm_to_ubuf, copy_gm_to_cbuf
    MTE1,       // Cube 专用：搬运到矩阵缓冲
    Vector,     // 向量计算（Vector 核）：vadd, vmul, vexp ...
    Matrix,     // 矩阵计算（Cube 核）：mad
    Fix,        // 定点转换（Cube 核）：copy_matrix_cc_to_gm
    MTE3,       // 搬出：copy_ubuf_to_gm, copy_cbuf_to_gm
};
```

已有 `CoreType` 枚举（`arch_config.hpp`）可以直接映射到 `PipeStage`。

#### 4.1.2 统一同步模型

PTO-ISA 中存在两种同步原语，但**处理逻辑完全相同**：

| 同步类型 | Producer（发信号） | Consumer（等信号） | 场景 |
|----------|-------------------|-------------------|------|
| 核内同步 | `set_flag(srcPipe, dstPipe, id)` | `wait_flag(srcPipe, dstPipe, id)` | 同核内 Pipeline 间 |
| 跨核同步 | `ffts_cross_core_sync(pipe, msg)` | `wait_flag_dev(flag_id)` | Cube ↔ Vector |

PTO 代码中有两种写法风格，但底层都产生相同的 CCE 调用：

```
写法一（手动）：                          写法二（Event<> 语法糖）：
TLOAD(...);                               evt = TLOAD(...);
set_flag(MTE2, V, id);                    → 内部也是 set_flag(MTE2, V, token)
wait_flag(MTE2, V, id);                   → 内部也是 wait_flag(MTE2, V, token)
TABS(...);                                 TADD(..., evt);
```

**perf-sim 统一基于 CCE 级别的 trace 来处理同步**，不关心上层用的是哪种写法。

当前 `arch_config.hpp` 已有的同步规则：

```cpp
FixedRule("set_flag",      CoreType::Scalar, 1),
FixedRule("wait_flag",     CoreType::Scalar, 1),
FixedRule("wait_flag_dev", CoreType::Scalar, 1),
```

需要补充：

```cpp
FixedRule("ffts_cross_core_sync", CoreType::Scalar, ???),
```

#### 4.1.3 Trace 统一时间线

当前 `TraceState` 把 PTO 指令和同步 CCE 调用**分开存储**，丢失了交错顺序：

```
executed_pto:    [TLOAD, TABS, TSTORE]           // PTO 指令
raw_cce_calls:   [set_flag, wait_flag, ...]      // 同步调用（不知道在哪些 PTO 指令之间）
```

实际执行顺序是交错的：`TLOAD → set_flag → wait_flag → TABS → set_flag → wait_flag → TSTORE`

perf-sim 需要**统一时间线**，保持 PTO 指令和同步调用的交错顺序：

```cpp
enum class TraceEntryKind : uint8_t {
    PtoInstr,       // PTO 指令（含其内部 CCE 调用）
    Sync,           // 同步调用：set_flag, wait_flag, ffts_cross_core_sync, wait_flag_dev
};

struct TraceEntry {
    TraceEntryKind kind;

    // kind == PtoInstr 时有效
    PtoInstrRecord pto_instr;

    // kind == Sync 时有效
    std::string sync_name;     // "set_flag" / "wait_flag" / "ffts_cross_core_sync" / "wait_flag_dev"
    pipe_t src_pipe = {};
    pipe_t dst_pipe = {};
    event_t event_id = 0;
};

// 扩展 TraceState
struct TraceState {
    // --- 原有字段（保留兼容）---
    std::vector<PtoInstrRecord> executed_pto;
    std::vector<CceCallRecord> raw_cce_calls;
    std::vector<std::size_t> active_pto_stack;

    // --- perf-sim 新增：统一时间线 ---
    std::vector<TraceEntry> timeline;

    // --- perf-sim 新增：算子层 ---
    std::vector<OperatorRecord> operators;
    std::vector<std::size_t> active_operator_stack;
};
```

改造后的 trace 示例：

```
timeline: [
    PtoInstr(TLOAD,  [copy_gm_to_ubuf, ...]),
    Sync(set_flag,   src=MTE2, dst=V,    id=0),
    Sync(wait_flag,  src=MTE2, dst=V,    id=0),
    PtoInstr(TABS,   [vabs, ...]),
    Sync(set_flag,   src=V,    dst=MTE3, id=0),
    Sync(wait_flag,  src=V,    dst=MTE3, id=0),
    PtoInstr(TSTORE, [copy_ubuf_to_gm, ...]),
]
```

实现方式：在现有的 `RecordCceCall` 和 `PtoInstrScope` 机制基础上，对同步类 CCE 调用额外记录一条 `TraceEntry` 到 `timeline`。不需要改变用户代码或 PTO 指令的写法。

#### 4.1.4 时序数据结构

```cpp
struct PipeEvent {
    std::string name;           // CCE 操作名 或 同步操作名
    CoreKind core;              // Cube 核 或 Vector 核
    PipeStage stage;            // 所属流水线阶段
    uint64_t start_cycle;       // 开始时刻
    uint64_t end_cycle;         // 结束时刻
    uint64_t duration;          // 持续时间
};

struct PipeTimeline {
    std::vector<PipeEvent> events;

    // 分析接口
    uint64_t TotalActiveTime(CoreKind core, PipeStage stage) const;
    uint64_t TotalBubbleTime(CoreKind core, PipeStage stage) const;
    double Utilization(CoreKind core, PipeStage stage, uint64_t total_span) const;

    // C&V 专属分析
    uint64_t CVOverlapTime() const;              // Cube 和 Vector 同时活跃的时间
    uint64_t SyncWaitTime() const;               // 同步等待引入的空闲
};
```

#### 4.1.5 时序调度算法

```cpp
class PipelineScheduler {
public:
    // 输入：算子的统一时间线
    // 输出：C&V 双核并行时序安排
    PipeTimeline Schedule(const OperatorRecord& op, const ArchConfig& arch);

private:
    // 每个 PipeStage 的完成时刻（区分 Cube / Vector 核）
    uint64_t stage_ready_[2][7] = {};   // [CoreKind][PipeStage]

    // 信号表：event_id → 信号就绪时刻
    std::unordered_map<event_t, uint64_t> signal_table_;
};
```

**核心调度逻辑（三种 entry 统一处理）：**

```
按序扫描 timeline 中的每个 TraceEntry：

┌─ kind == Sync（同步调用）─────────────────────────┐
│                                                    │
│  发信号操作（set_flag / ffts_cross_core_sync）：    │
│    duration = FindRule(sync_name).cycles            │
│    start = stage_ready_[core][Scalar]               │
│    end = start + duration                           │
│    signal_table_[event_id] = end                    │
│    stage_ready_[core][Scalar] = end                 │
│                                                    │
│  等信号操作（wait_flag / wait_flag_dev）：           │
│    duration = FindRule(sync_name).cycles            │
│    dep = signal_table_[event_id]                    │
│    start = max(stage_ready_[core][Scalar], dep)     │
│    end = start + duration                           │
│    stage_ready_[core][Scalar] = end                 │
│                                                    │
├─ kind == PtoInstr（PTO 指令）─────────────────────┤
│                                                    │
│  对该 PTO 指令内部的每条 CCE 调用：                  │
│    stage = FindRule(cce_name).stage                 │
│    core  = stage → CoreKind                        │
│    duration = EvaluateCceCall(cce, arch).cycles     │
│    start = stage_ready_[core][stage]                │
│    end = start + duration                           │
│    stage_ready_[core][stage] = end                  │
│                                                    │
│  （同步依赖已通过前面的 Sync entry 体现在            │
│    stage_ready_ 中，此处无需额外处理）               │
│                                                    │
└────────────────────────────────────────────────────┘
```

**核内同步和跨核同步使用完全相同的逻辑**，不区分。原因：

- `set_flag` / `ffts_cross_core_sync` 都是往 `signal_table_` 写入完成时刻
- `wait_flag` / `wait_flag_dev` 都是从 `signal_table_` 读出依赖时刻
- 计算操作的 `start = stage_ready_[core][stage]`，而 `stage_ready_` 已被前面的 wait 更新过
- 唯一差异是 latency 值不同（由 `arch_config` 规则配置）

**示例：核内同步**（TLOAD → TABS）

```
timeline: [TLOAD(copy_gm_to_ubuf), set_flag(MTE2→V,id=0), wait_flag(MTE2→V,id=0), TABS(vabs)]

1. copy_gm_to_ubuf:  stage=MTE2, core=Vector, start=0, end=100  → vector_stage_ready[MTE2]=100
2. set_flag:         start=0, end=1                              → signal_table_[0]=1
3. wait_flag:        dep=signal_table_[0]=1, start=max(0,1)=1    → vector_stage_ready[Scalar]=2
4. vabs:             stage=Vector, start=max(0,2)=2, end=52      → vector_stage_ready[Vector]=52
```

**示例：跨核同步**（Cube TMATMUL → Vector TADD）

```
timeline: [..., ffts_sync(PIPE_FIX,id=5), ..., wait_dev(id=5), ..., TADD(vadd)]

1. mad(Cube):        stage=Matrix, start=0, end=400               → cube_stage_ready[Matrix]=400
2. ffts_sync(Cube):  start=400, end=405                           → signal_table_[5]=405
3. wait_dev(Vector):  dep=signal_table_[5]=405, start=max(0,405)=405 → vector_stage_ready[Scalar]=407
4. vadd(Vector):     stage=Vector, start=max(0,407)=407, end=607  → vector_stage_ready[Vector]=607
```

两种场景的调度代码路径完全一致，只是操作名和 latency 值不同。

时序可视化示例（matmul + bias_add）：

```
Cycle:  0    100   200   300   400   500   600   700   800
Cube:   [TLOAD] [TMATMUL----] [TSTORE_ACC] [ffts_sync]
                                               ↓ signal_table
Vector:                                      [wait_dev] [TLOAD] [TADD] [TSTORE]
```

### 4.2 白盒指标采集（metrics.hpp）

#### 4.2.1 指标定义

```cpp
struct OperatorMetrics {
    // --- 基础性能指标 ---
    std::string operator_name;
    uint64_t total_cycles = 0;
    uint64_t compute_cycles = 0;     // Vector + Matrix
    uint64_t memory_cycles = 0;      // MTE2 + MTE3
    uint64_t scalar_cycles = 0;      // Scalar 控制
    uint64_t pipeline_span_cycles = 0; // Pipeline 全跨度

    // --- Pipe 利用率 ---
    double vector_util = 0.0;        // Vector 活跃时间 / 总跨度
    double matrix_util = 0.0;        // Matrix 活跃时间 / 总跨度
    double mte2_util = 0.0;          // MTE2 活跃时间 / 总跨度
    double mte3_util = 0.0;          // MTE3 活跃时间 / 总跨度

    // --- Compute/Memory Ratio ---
    double compute_memory_ratio = 0.0;  // compute_cycles / memory_cycles

    // --- Bubble 分析 ---
    uint64_t total_bubble_cycles = 0;
    uint64_t vector_bubble = 0;
    uint64_t matrix_bubble = 0;

    // --- C&V 协作指标 ---
    uint64_t cv_overlap_cycles = 0;       // Cube 和 Vector 同时活跃的时间
    uint64_t cv_sync_wait_cycles = 0;     // FFTS 等信号引入的空闲
    uint64_t cv_gm_transfer_cycles = 0;   // C→V / V→C GM 中转搬运开销
    double cv_overlap_ratio = 0.0;        // overlap / pipeline_span

    // --- 指令统计 ---
    uint64_t pto_instr_count = 0;
    uint64_t cce_call_count = 0;

    // --- PTO 指令级拆分 ---
    struct InstrBreakdown {
        std::string name;
        uint64_t cycles;
        double percentage;        // cycles / total_cycles * 100
    };
    std::vector<InstrBreakdown> instr_breakdown;
};

struct SimReport {
    std::string arch_name;
    std::vector<OperatorMetrics> operators;
    uint64_t total_cycles = 0;

    // 全局统计
    uint64_t total_compute_cycles = 0;
    uint64_t total_memory_cycles = 0;
    double overall_compute_memory_ratio = 0.0;
};
```

#### 4.2.2 指标计算

```cpp
OperatorMetrics ComputeMetrics(const OperatorRecord& op, const PipeTimeline& timeline);
SimReport ComputeFullReport(const TraceState& trace, const ArchConfig& arch);
```

### 4.3 泳道图数据生成（swimlane.hpp）

#### 4.3.1 输出格式

输出 Chrome Trace Event 格式（可用 `chrome://tracing` 直接查看）：

```json
{
  "traceEvents": [
    {
      "name": "TLOAD",
      "cat": "MTE2",
      "ph": "X",
      "ts": 0,
      "dur": 120,
      "pid": 0,
      "tid": 1,
      "args": { "operator": "softmax", "cce_calls": ["copy_gm_to_ubuf_align_b32"] }
    },
    {
      "name": "TROWMAX",
      "cat": "Vector",
      "ph": "X",
      "ts": 120,
      "dur": 200,
      "pid": 0,
      "tid": 2,
      "args": { "operator": "softmax" }
    }
  ],
  "displayTimeUnit": "ns"
}
```

字段映射：
- `pid` → 核编号（Cube 核=0, Vector 核=1）
- `tid` → PipeStage 枚举值（Scalar=0, MTE2=1, MTE1=2, Vector=3, Matrix=4, Fix=5, MTE3=6）
- `ts` / `dur` → Pipeline scheduler 计算的 cycle 数 × 时钟周期（ns）

示例：C&V 双核泳道布局

```
pid=0 (Cube Core)
  tid=0: Scalar   ─┐
  tid=1: MTE2     ─┤  [TLOAD]
  tid=2: MTE1     ─┤
  tid=4: Matrix   ─┤          [TMATMUL]
  tid=5: Fix      ─┤                    [TSTORE_ACC] [ffts_sync]
  tid=6: MTE3     ─┘

pid=1 (Vector Core)
  tid=0: Scalar   ─┐
  tid=1: MTE2     ─┤                                      [TLOAD]
  tid=3: Vector   ─┤                                             [TADD]
  tid=6: MTE3     ─┘                                                    [TSTORE]
```

#### 4.3.2 生成接口

```cpp
// 生成 Chrome Trace JSON 字符串
std::string GenerateSwimlaneJson(const SimReport& report,
                                  const ArchConfig& arch,
                                  uint64_t clock_period_ns = 4);

// 直接写入文件
void WriteSwimlaneJson(const std::string& filepath,
                       const SimReport& report,
                       const ArchConfig& arch,
                       uint64_t clock_period_ns = 4);
```

#### 4.3.3 可视化方式

1. **Chrome Trace**：直接用 `chrome://tracing` 打开 JSON 文件
2. **Perfetto UI**：上传 JSON 到 https://ui.perfetto.dev（离线工具，适合内部环境）
3. **脚本绘图**：提供 Python 脚本生成 PNG/HTML

### 4.4 报告生成框架（reporter.hpp）

```cpp
class PerfSimReporter {
public:
    explicit PerfSimReporter(const ArchConfig& arch);

    // 执行完整仿真流水线
    SimReport Run();

    // --- 输出接口 ---

    // 控制台文本报告
    void PrintTextReport(const SimReport& report, std::ostream& os = std::cout);

    // JSON 格式报告（包含所有指标数据）
    void WriteJsonReport(const std::string& filepath, const SimReport& report);

    // 泳道图 JSON
    void WriteSwimlane(const std::string& filepath, const SimReport& report);

    // 一键输出所有产物
    void WriteAll(const std::string& output_dir, const SimReport& report);
};
```

文本报告示例输出：

```
============================================
  PTO-ISA Perf-Sim Report  (arch: a2a3)
============================================

Operator: softmax
  Total Cycles      : 1,248
  Pipeline Span     : 1,100
  PTO Instructions  : 7
  CCE Calls         : 12

  Pipe Utilization:
    Scalar  :  2.3%  (  25 cycles)
    MTE2    : 14.5%  ( 160 cycles)
    Vector  : 48.2%  ( 530 cycles)
    MTE3    :  9.1%  ( 100 cycles)
    Bubble  : 26.0%  ( 285 cycles)

  Compute/Memory    : 2.12x
  Top Instructions:
    TROWMAX  : 280 cycles (22.4%)
    TEXP     : 250 cycles (20.0%)
    TLOAD    : 160 cycles (12.8%)

--------------------------------------------
Operator: matmul
  ...
============================================
  Summary: 2 operators, 3,848 total cycles
============================================
```

---

## 5. 使用方式

### 5.1 基本用法

```cpp
#include "pto/pto-inst.hpp"
#include "pto/costmodel/perf_sim/operator_scope.hpp"
#include "pto/costmodel/perf_sim/reporter.hpp"

using namespace pto;

void softmax_kernel(/* ... */) {
    PERF_SIM_OPERATOR("softmax");

    TLOAD(a_tile, src_tensor, /* ... */);
    TROWMAX(max_tile, a_tile);
    TSUB(sub_tile, a_tile, max_tile);
    TEXP(exp_tile, sub_tile);
    TROWSUM(sum_tile, exp_tile);
    TDIVS(result_tile, exp_tile, sum_tile);
    TSTORE(dst_tensor, result_tile, /* ... */);
}

int main() {
    softmax_kernel(/* ... */);

    // 生成报告
    perf_sim::PerfSimReporter reporter(perf_sim::GetDefaultArchConfig());
    auto report = reporter.Run();
    reporter.PrintTextReport(report);
    reporter.WriteAll("./perf_sim_output", report);

    return 0;
}
```

---

编译宏：perf-sim 功能在 `__COSTMODEL` 宏下启用，无需新增编译开关。