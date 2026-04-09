# PTO-ISA Perf-Sim 设计文档

## 1. 背景与目标

### 1.1 现状分析

**PyPTO CostModel**（位于 `/Users/wangxingkai/PycharmProjects/pypto`）提供了一套完整的性能仿真体系：

| 能力 | 实现方式 |
|------|----------|
| 性能预测 | SimSys 层次的 cycle 精确仿真，建模 L1/L2 Cache、Pipeline、Memory Access |
| 泳道图输出 | 基于仿真数据生成 `merged_swimlane.json`（Chrome Trace 格式），可视化展示多 Core、多 Pipe 的时序执行 |
| 白盒指标分析 | Pipe 利用率、Bubble 分析、任务调度等待时间、依赖解析速率等 |
| 算子边界 | TileGraph 自然划分：Subgraph → Function → TileOp，有清晰的层级结构 |

**PTO-ISA CostModel**（当前目录 `include/pto/costmodel`）已有的能力：

| 能力 | 实现方式 |
|------|----------|
| 单指令 cycle 估算 | `MAP_INSTR_IMPL` 宏 + `PtoInstrScope` + CCE trace → `arch_config` 规则查表 |
| 逐 Tile cycle 存储 | `tile.GetCycle()` 直接查询 |
| 指令级 trace | `TraceState` 记录每条 PTO 指令及其 CCE 调用序列 |
| 架构配置 | `ArchConfig` + `CceLatencyRule`，支持 A2/A3 和 A5 |

**缺失能力**（本设计需要补齐）：

1. **算子级仿真**：当前只有指令级 cycle，缺少算子层面的聚合与分析
2. **算子边界界定**：PTO-ISA 没有 TileGraph，缺少显式的算子范围标记
3. **Pipeline 建模**：当前只做 cycle 累加，缺少 Scalar → MTE2 → Vector/Matrix → MTE3 的流水线时序建模
4. **泳道图输出**：没有可视化手段
5. **白盒指标**：没有 Pipe 利用率、Bubble 分析等深度指标

### 1.2 目标

在 PTO-ISA costmodel 基础上，构建 **perf-sim** 子系统，提供：

- 算子级的性能预测与 trace 分析
- 基于 Pipeline 时序的泳道图可视化
- 白盒性能指标输出
- **解决算子边界界定这一核心问题**

---

## 2. 核心问题：算子边界界定

### 2.1 问题分析

PyPTO 有 TileGraph 做天然的算子边界划分（Subgraph → Function → TileOp 三级结构），而 PTO-ISA 是**直接用 PTO 指令编程**的，不存在中间 IR 或计算图。

PTO-ISA 当前的 trace 机制（`PtoInstrScope`）只能捕获**单条 PTO 指令**的边界：

```
TLOAD → TADD → TMUL → TSTORE    // 这4条指令构成一个算子？还是2个？没有标记
```

### 2.2 设计方案：双层标记机制

提出 **OperatorScope + PtoInstrScope** 双层 RAII 标记：

```
┌─────────────────── OperatorScope("softmax") ───────────────────┐
│                                                                 │
│  ┌─ PtoInstrScope("TLOAD") ─┐  ┌─ PtoInstrScope("TLOAD") ─┐  │
│  │  copy_gm_to_ubuf          │  │  copy_gm_to_ubuf          │  │
│  └───────────────────────────┘  └───────────────────────────┘  │
│  ┌─ PtoInstrScope("TROWMAX") ──────────────────────────────┐   │
│  │  vmax + vsub + vexp + vadd + vdiv                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌─ PtoInstrScope("TSTORE") ─┐                                 │
│  │  copy_ubuf_to_gm           │                                 │
│  └────────────────────────────┘                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.2.1 显式标记（手动标注）

提供 RAII 风格的 `OperatorScope` 宏，让用户在算子入口/出口显式标记：

```cpp
// 方式一：RAII 作用域（推荐）
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

// 方式二：手动 Begin/End（适用于跨函数场景）
PERF_SIM_BEGIN("layer_norm");
compute_mean();
compute_variance();
normalize();
PERF_SIM_END();
```

#### 2.2.2 隐式推断（自动分组）

对于没有显式标记的场景，提供基于规则的自动推断：

```cpp
// 策略：将"连续的 PTO 指令，直到遇到 TSTORE 或下一次 TLOAD 序列"视为一个算子
// 这符合 PTO 编程的典型模式：Load → Compute → Store
```

推断规则：
- 以 `TLOAD` / `TLOADCONV` 开始
- 以 `TSTORE` 结束
- 中间的计算指令归入同一算子
- 无法推断时归入 "unnamed" 组

#### 2.2.3 数据结构扩展

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

#### 2.2.4 API 定义

```cpp
namespace pto::perf_sim {

// 算子标记
void BeginOperator(std::string_view name);
void EndOperator();

// RAII 包装
class OperatorScope { /* ... */ };

// 宏快捷方式
#define PERF_SIM_OPERATOR(name)  ::pto::perf_sim::OperatorScope _op_scope(name)
#define PERF_SIM_BEGIN(name)     ::pto::perf_sim::BeginOperator(name)
#define PERF_SIM_END()           ::pto::perf_sim::EndOperator()

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

现有 costmodel 只做 cycle 简单累加，不考虑 Pipeline 时序。perf-sim 需要建模 NPU 的多级流水线。

#### 4.1.1 Pipeline 阶段定义

```cpp
enum class PipeStage : uint8_t {
    Scalar,     // 标量控制：set_flag, wait_flag, pipe_barrier
    MTE2,       // 搬入：copy_gm_to_ubuf, copy_gm_to_cbuf
    Vector,     // 向量计算：vadd, vmul, vexp ...
    Matrix,     // 矩阵计算：mad
    Fix,        // 定点转换
    MTE3,       // 搬出：copy_ubuf_to_gm, copy_cbuf_to_gm
};
```

已有 `CoreType` 枚举（`arch_config.hpp`）可以直接映射到 `PipeStage`。

#### 4.1.2 时序模型

参考 PyPTO 的 SimSys 中 PipeMachine 的建模方式，为 PTO-ISA 设计简化版：

```cpp
struct PipeEvent {
    std::string name;           // CCE 操作名
    PipeStage stage;            // 所属流水线阶段
    uint64_t start_cycle;       // 开始时刻
    uint64_t end_cycle;         // 结束时刻
    uint64_t duration;          // 持续时间
};

struct PipeTimeline {
    std::vector<PipeEvent> events;

    // 分析接口
    uint64_t TotalActiveTime(PipeStage stage) const;
    uint64_t TotalBubbleTime(PipeStage stage) const;
    double Utilization(PipeStage stage, uint64_t total_span) const;
};
```

#### 4.1.3 时序调度算法

```cpp
class PipelineScheduler {
public:
    // 输入：算子内所有 CCE 调用的 cycle 估算
    // 输出：考虑 Pipeline 并行的时序安排
    PipeTimeline Schedule(const OperatorRecord& op, const ArchConfig& arch);

private:
    // 关键简化假设（与 PyPTO 的差异）：
    // 1. 单核单算子串行执行，不做跨算子 overlap
    // 2. 同一 PipeStage 内的操作严格串行
    // 3. 不同 PipeStage 的操作按数据依赖排列
    // 4. 不建模 L2 Cache 和多核通信

    uint64_t current_cycle_[6] = {};  // 每个 stage 的当前时刻
};
```

时序计算逻辑：

```
对于算子内的每条 CCE 调用：
  1. 确定所属 PipeStage
  2. start_cycle = max(当前 stage 上次完成时刻, 前驱 stage 完成时刻)
  3. end_cycle = start_cycle + EvaluateCceCall(...).cycles
  4. 更新 current_cycle_[stage] = end_cycle
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

参考 PyPTO 的 `merged_swimlane.json` 格式，输出 Chrome Trace Event 格式（可用 `chrome://tracing` 直接查看）：

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
- `pid` → Core Index（PTO-ISA 单核场景下固定为 0）
- `tid` → PipeStage 枚举值（Scalar=0, MTE2=1, Vector=2, Matrix=3, Fix=4, MTE3=5）
- `ts` / `dur` → Pipeline scheduler 计算的 cycle 数 × 时钟周期（ns）

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
3. **脚本绘图**：提供 Python 脚本生成 PNG/HTML（参考 PyPTO 的 `draw_pipe_swim_lane.py`）

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

### 5.2 多算子仿真

```cpp
void my_network_layer(/* ... */) {
    {
        PERF_SIM_OPERATOR("matmul");
        TLOAD(a, src_a); TLOAD(b, src_b);
        TMATMUL(c, a, b);
        TSTORE(dst, c);
    }
    {
        PERF_SIM_OPERATOR("bias_add_relu");
        TLOAD(c, dst); TLOAD(bias, bias_tensor);
        TADD(d, c, bias);
        TRELU(e, d);
        TSTORE(dst, e);
    }
}
```

### 5.3 测试中使用

在现有 costmodel 测试框架基础上扩展：

```cpp
#include "pto/pto-inst.hpp"
#include "pto/costmodel/perf_sim/operator_scope.hpp"
#include "pto/costmodel/perf_sim/reporter.hpp"
#include "cost_check.hpp"

TEST(PerfSim, Softmax) {
    PERF_SIM_OPERATOR("softmax");

    // ... 构造 Tile，执行 PTO 指令 ...

    perf_sim::PerfSimReporter reporter;
    auto report = reporter.Run();

    ASSERT_EQ(report.operators.size(), 1);
    EXPECT_GT(report.operators[0].total_cycles, 0);

    reporter.WriteSwimlane("./output/softmax_swimlane.json", report);
}
```

---

## 6. 与 PyPTO CostModel 的差异

| 维度 | PyPTO CostModel | PTO-ISA Perf-Sim |
|------|----------------|-------------------|
| **算子边界** | TileGraph 自动划分 | `OperatorScope` 手动标记 + Load/Store 模式推断 |
| **仿真层次** | 多级 SimSys（Device → AICPU → Core → Pipe） | 单核单算子，PipeStage 级别 |
| **Cache 建模** | L1/L2 Cache 仿真 | 不建模 Cache（cycle 规则已隐含访存开销） |
| **多核调度** | 多 Core 并行调度 | 单核串行（PTO-ISA 单核编程模型） |
| **性能可变性** | PV Model 建模随机波动 | 确定性规则估算 |
| **优化 Pass** | 图优化、分区、约束等 | 不涉及（PTO-ISA 无计算图） |
| **输入方式** | Python JIT + TileGraph | C++ 宏 + 直接编写 PTO 指令 |

---

## 7. 实现计划

### Phase 1：算子边界 + 指标采集

1. 实现 `operator_scope.hpp`：`OperatorScope` RAII 类 + `BeginOperator/EndOperator`
2. 扩展 `TraceState`：新增 `operators` 和 `active_operator_stack`
3. 实现 `metrics.hpp`：`OperatorMetrics` 结构 + 指标计算逻辑
4. 修改 `MAP_INSTR_IMPL` / `PTO_TRACE_CALL`：在 `PtoInstrScope` 内自动关联到当前活跃的 `OperatorScope`

### Phase 2：Pipeline 时序建模

1. 实现 `pipeline_model.hpp`：`PipeStage` 映射 + `PipelineScheduler`
2. 为每个 CCE 调用计算 Pipeline 时序
3. 实现 Bubble 分析和 Pipe 利用率计算

### Phase 3：泳道图 + 报告

1. 实现 `swimlane.hpp`：Chrome Trace JSON 生成
2. 实现 `reporter.hpp`：文本报告 + JSON 报告 + 一键输出
3. 提供可视化 Python 脚本

### Phase 4：测试与验证

1. 为 `OperatorScope` 编写单元测试
2. 为 Pipeline Scheduler 编写单元测试
3. 端到端测试：softmax / matmul 等典型算子的完整仿真流程
4. 泳道图正确性验证

---

## 8. 关键设计决策

### Q1：为什么不复用 PyPTO 的 TileGraph？

PTO-ISA 的定位是**底层指令集编程**，用户直接操控 PTO 指令，没有上层计算图。引入 TileGraph 会改变 PTO-ISA 的编程范式，违背其设计初衷。`OperatorScope` 是一种**轻量级标记**，不改变编程模型，只在需要仿真时添加。

### Q2：为什么不做多核仿真？

PTO-ISA 当前的编程模型是**单核 AICore**。多核并行需要通信原语（如 `pto_comm`），属于更高层的调度问题。perf-sim 聚焦单核性能分析，多核可由上层调度器组合多个单核结果。

### Q3：Pipeline 调度的精度如何？

perf-sim 的 Pipeline 调度是**简化模型**，不考虑：
- 指令级并行（ILP）
- Cache miss 的额外延迟
- 多核竞争带宽

它提供的是**确定性的上界估算**，适合性能瓶颈定位和算法选择，不适合精确的实时性验证。

### Q4：如何处理循环和条件分支？

PTO-ISA 中常见循环结构（如 `for (int i = 0; i < N; i++) TMATMUL(...)`）。perf-sim 的 trace 会记录每次迭代的 PTO 指令，`OperatorScope` 会将所有迭代归入同一个算子。报告中的指令统计会反映实际执行次数。

---

## 9. 依赖关系

```
perf_sim/operator_scope.hpp  →  trace.hpp（扩展 TraceState）
perf_sim/pipeline_model.hpp  →  arch_config.hpp, cce_evaluator.hpp
perf_sim/metrics.hpp         →  pipeline_model.hpp, trace.hpp
perf_sim/swimlane.hpp        →  pipeline_model.hpp, metrics.hpp
perf_sim/reporter.hpp        →  所有上述模块
```

编译宏：perf-sim 功能在 `__COSTMODEL` 宏下启用，无需新增编译开关。

---

## 10. 总结

PTO-ISA perf-sim 的核心创新点是**在无计算图的指令级编程环境中实现算子级性能仿真**。通过 `OperatorScope` 手动标记 + Load/Store 模式推断的双层机制解决算子边界问题，通过 Pipeline 时序建模提供超越简单 cycle 累加的分析能力，通过 Chrome Trace 格式的泳道图输出提供直观的可视化手段。

整体设计在保持 PTO-ISA 编程模型不变的前提下，以最小侵入的方式补齐了与 PyPTO costmodel 对标的性能仿真能力。
