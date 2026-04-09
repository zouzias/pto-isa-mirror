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

基于 NPU AICore 的流水线结构设计简化时序模型：

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
    // 1. 同一 PipeStage 内的操作严格串行
    // 2. 不同 PipeStage 的操作按数据依赖排列
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