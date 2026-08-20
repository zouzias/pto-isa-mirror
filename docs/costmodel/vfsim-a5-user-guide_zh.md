# A5 VfSim Costmodel 用户指南

A5 VfSim costmodel 是 PTO-ISA 在 `__COSTMODEL` 模式下接入的 A5 向量函数性能预测模型。它不会在 NPU 上真正执行 kernel，而是在普通 C++ 进程中运行 A5 CCE mock 版本的 PTO/tileop，拦截 VF micro-op 和 `__VEC_SCOPE__` 内的 loop 结构，构造 PTO `VfInfo`，直接 lowering 为 `CanonicalVfInfo`，再调用 VfSim native C++ 模型预测 VF cycle。

## 适用场景

- 预测 A5 vector tileop 中 VF 主体的执行时间。
- 对比不同 tile shape、repeat 次数、dtype、寄存器/UB 访问模式下的 VF 性能趋势。
- 在没有 NPU 硬件或不想跑完整 NPU ST 时，快速验证 A5 tileop 的 costmodel 结果。

VfSim costmodel 的结果用于性能建模和趋势判断，不替代真实硬件 profiling。

## 基本原理

A5 VfSim costmodel 的主链路是：

```text
PTO/tileop 执行
  -> A5 CCE mock 拦截 VF micro-op
  -> LLVM pass 插桩拦截 __VEC_SCOPE__ 内 for-loop
  -> micro-op 事件 + loop 事件构造 PTO 侧 VfInfo
  -> PTO 指令结束时调用 predictVfCycles(...)
  -> PTO VfInfo 直接 lowering 为 VfSim CanonicalVfInfo
  -> VfSim native C++ 模型预测 cycle 和命中状态
  -> 将结果写回 PTO trace 和 Perf-Sim 指令记录
```

几个关键点：

- micro-op 捕获靠 CCE intrinsic mock stub。
- for-loop 结构捕获靠 LLVM pass 插桩。
- PTO `VfInfo` 是内部捕获结构；`CanonicalVfInfo` 是 native VfSim 唯一的输入接口。
- 单个 `__VEC_SCOPE__` 构造一个 `VfInfo`。
- 一个 PTO/tileop 内如果收集到多个 `VfInfo`，会逐个预测后求和返回一个标量 cycle。
- 如果没有收集到 `VfInfo`，不会调用 VfSim，保留普通 PTO costmodel 已累计的 cycle。
- VF cycle 作为 Perf-Sim 的 Vector pipeline latency，不会重复计费。

## 快速运行现有用例

A5 VfSim costmodel 单 tileop 测试位于：

```text
tests/costmodel/st_a5
```

推荐使用独立 build 目录：

```bash
cd /path/to/pto-isa

cmake -S tests/costmodel/st_a5 \
      -B /path/to/build/st_a5 \
      -DCMAKE_BUILD_TYPE=Debug \
      -DPTO_A5_LLVM_CONFIG=/path/to/llvm-config

cmake --build /path/to/build/st_a5 -j4
ctest --test-dir /path/to/build/st_a5 --output-on-failure
```

如果不显式传 `-DPTO_A5_LLVM_CONFIG=...`，构建脚本会查找 LLVM 19 的 `llvm-config` 和 `clang++`。其他 LLVM/Clang 主版本会直接报错，因为当前 loop capture ABI 只支持版本 19。

只运行一个用例示例：

```bash
ctest --test-dir /path/to/build/st_a5 \
      --output-on-failure \
      -R tadd
```

当前 `st_a5` 下的用例主要覆盖 A5 VfSim 已支持的基础 vector tileop，例如：

```text
tadd, tadds, tsub, tsubs, tmul, tmuls, tdiv,
tmin, tmins, tmax, tand, tshls, tshrs
```

此外还包括 adapter golden、memory/sync 分流、ACL Host Runtime、未修改完整 Host TADD 和配置搬移回归。

“tileop 已支持”指现有回归覆盖的具体 dtype/form；不能据此推断同名 tileop 的所有 dtype 都已命中。例如未列入支持回归的 form 会返回可观测的 `UnsupportedForm` 或 `InvalidTrace`，并使用 A5 fallback。

## 接入一个新 A5 costmodel 用例

### 1. 使用统一入口头

用户侧正常包含 PTO 统一入口即可：

```cpp
#include <pto/pto-inst.hpp>
```

不需要直接包含 VfSim 头文件。定义 `__COSTMODEL` 后，`pto/pto-inst.hpp` 会选择 costmodel 版本 PTO 指令实现；`__NPU_ARCH__=3101` 会进入 A5 CCE mock 和 VfSim 结算路径。

### 2. 使用 costmodel 编译宏

A5 VfSim costmodel 至少需要：

```text
-D__COSTMODEL
-D__NPU_ARCH__=3101
```

常用测试还会加：

```text
-DPTO_COMM_NOT_SUPPORTED
```

含义：

| 宏 | 含义 |
| --- | --- |
| `__COSTMODEL` | 启用 PTO costmodel/mock 路径。 |
| `__NPU_ARCH__=3101` | 选择 A5 costmodel 路径。 |
| `PTO_COMM_NOT_SUPPORTED` | 跳过通信相关 PTO 头；普通接入是否需要取决于用例是否包含通信指令。 |

### 3. 必须使用 Clang/LLVM pass

A5 VfSim 需要捕获 `__VEC_SCOPE__` 内的 loop 结构。micro-op 可以由 CCE mock 拦截，但 loop 需要 LLVM pass 插桩，因此推荐通过 CMake helper 构建，不建议直接用 `g++` 编译。

当前经过验证的 loop 捕获支持边界是 **Clang 19 + `-O0` + `-g` 下的规则 `for` 循环**。Clang 与用于构建 pass plugin 的 LLVM 必须同为 19 主版本。当前 pass 依赖 loop `DebugLoc` 和 preheader；缺少任一信息的 loop 可能被跳过。复杂控制流、其他优化级别、其他 Clang/LLVM 版本以及 `__VEC_SCOPE__` 析构边界外的 loop 归属尚未作为已支持场景。

普通 `g++` 编译可能出现以下问题：

- 不支持 `-fpass-plugin`。
- 捕获不到 loop 结构，`VfInfo` 不完整。
- 未链接 VfSim native 库时会出现链接错误，无法调用真实预测实现。
- 测试看似编过，但实际没有进入真实 VfSim 预测链路。

### 4. 必须编译并链接 VfSim

历史上的 PTO-ISA 大部分能力可以按 header-only 方式使用：用户只 include `pto/pto-inst.hpp`，再通过编译宏选择 costmodel 路径即可。

A5 VfSim costmodel 不完全是 header-only。真实 VfSim 预测后端包含 native C++ 实现，需要先在 PTO-ISA 构建过程中编译出 `pto_a5_vfsim`，然后将用户测试目标或 costmodel 用例链接到该库。

因此，完整接入需要同时满足：

- include PTO 统一入口：`#include <pto/pto-inst.hpp>`。
- 编译时定义 A5 costmodel 宏：`-D__COSTMODEL -D__NPU_ARCH__=3101`。
- 使用 Clang 加载 loop 插桩 pass：`-fpass-plugin=/path/to/PtoLoopTracePass.so`。
- 编译并链接 native VfSim 后端：`-lpto_a5_vfsim`。

如果只 include 头文件但没有编译/链接 `pto_a5_vfsim`，单文件 lit 或手写编译命令会出现 `predictVfCyclesWithVfSim` 未定义引用。

### 5. CMake 接入方式

A5 VfSim costmodel 的 CMake helper 在：

```text
pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake
```

最小接入示例：

```cmake
include(${PTO_ISA_ROOT}/pkg_inc/pto/costmodel/vfsim/cmake/a5_vf_mock.cmake)
_pto_a5_find_llvm()
set(CMAKE_CXX_COMPILER ${PTO_A5_CLANGXX})

project(<your_project_name> CXX)

add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)

add_executable(<your_a5_case_target> main.cpp)
target_include_directories(<your_a5_case_target> PRIVATE ${PTO_ISA_ROOT}/include)
target_enable_a5_vf_mock(<your_a5_case_target>)
```

`target_enable_a5_vf_mock(<your_a5_case_target>)` 会把 A5 VfSim costmodel 的编译、插桩和链接配置挂到用户自己的 target 上，具体包括：

- 构建并链接 `pto_a5_vfsim` native C++ 模型库。
- 构建 `PtoLoopTracePass` LLVM pass plugin。
- 为 target 添加 loop 插桩编译选项：

```text
-O0 -g -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>
```

- 为 target 加入必要的 include 和依赖。
- 在每次构建时将 VfSim JSON 完整同步到 build 目录的相对 `pkg_inc/pto/costmodel/vfsim/configs` 布局，避免复用 build 目录时残留旧配置。

该 helper 位于 internal `pkg_inc`，会与 VfSim 源码和配置一起安装。`PTO_ISA_ROOT` 应指向同时包含 `include/` 和 `pkg_inc/` 的 PTO-ISA 架构目录。外部工程只应通过该 helper 启用 A5 costmodel，不应直接依赖 VfSim native C++ API。

标准 Clang 不能直接解析 Ascend 的 `kernel<<<...>>>`。当前 A5 costmodel 接收与 CPU_SIM 相同的 Host 可编译用例：kernel launch wrapper 应直接调用 kernel 函数，不支持将含 `<<<...>>>` 的 NPU 用例源码直接交给该构建入口。costmodel 不对用户源码执行额外转换。

现有示例可参考：

```text
tests/costmodel/st_a5/CMakeLists.txt
tests/costmodel/st_a5/testcase/CMakeLists.txt
tests/costmodel/st_a5/testcase/tadd/CMakeLists.txt
```

### 6. 手写编译命令参考

不推荐手写命令；如果必须手写，需要同时处理编译和链接：

```bash
/path/to/clang++ main.cpp \
  -std=c++23 -O0 -g \
  -D__COSTMODEL \
  -D__NPU_ARCH__=3101 \
  -DPTO_COMM_NOT_SUPPORTED \
  -fpass-plugin=/path/to/PtoLoopTracePass.so \
  -I/path/to/pto-isa/include \
  -L/path/to/pto-isa/build/lib \
  -lpto_a5_vfsim \
  -Wl,-rpath,/path/to/pto-isa/build/lib \
  -o <your_a5_case_binary>
```

注意：

- `clang++` 的 LLVM 主版本需要和 pass plugin 的 LLVM 主版本一致。
- `-fpass-plugin` 指向的是构建出的 A5 loop 插桩 pass。
- `-lpto_a5_vfsim` 用于链接 VfSim native C++ 模型实现。
- 如果环境里采用 `PTO_A5_VFSIM_LINKED` 一类链接守卫宏，正式走 native VfSim 时还需要同步定义该宏；具体以当前分支实现为准。

## 写测试用例

### 1. 基本测试结构

一个最小测试通常包含：

```cpp
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

#include "common/a5_vfsim_tileop_check.hpp"

TEST(A5VfSimTAdd, Basic)
{
    pto::mocker::ResetTrace();

    // 构造 tile/global/tensor 参数，调用 PTO tileop。
    // 例如：TADD(dst, src0, src1);

    const auto cycles = pto::mocker::GetLastPtoInstrCycles();
    EXPECT_GT(cycles, 0U);
}
```

实际用例建议参考：

```text
tests/costmodel/st_a5/testcase/tadd/main.cpp
```

### 2. ResetTrace 的作用

`pto::mocker::ResetTrace()` 用来清空当前线程的 costmodel trace 状态。一个测试进程里连续跑多个 case 时，建议每个 case 开始前调用一次，避免上一轮 PTO 指令、VF 信息、pipe 状态影响下一轮。

示例：

```cpp
TEST(A5VfSim, Case1)
{
    pto::mocker::ResetTrace();
    TADD(dst, src0, src1);
    auto cycles = pto::mocker::GetLastPtoInstrCycles();
}

TEST(A5VfSim, Case2)
{
    pto::mocker::ResetTrace();
    TMUL(dst, src0, src1);
    auto cycles = pto::mocker::GetLastPtoInstrCycles();
}
```

如果不 reset，`GetTrace().executed_pto` 里可能同时包含 Case1 和 Case2 的记录，统计总指令数或总 cycle 时会串数据。

### 3. 获取预测结果

PTO 指令结束后，可以通过：

```cpp
uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
```

获取最后一条 PTO 指令的预测 cycle。

如果目标 tile 对象支持 `SetLastCycle(...)`，`MAP_INSTR_IMPL` 还会把当前 PTO 指令 cycle 注入到 tile 对象中，供测试 helper 检查。

需要区分真实命中和 fallback 时，读取结构化结果：

```cpp
const auto& result = ::pto::mocker::GetTrace().executed_pto.back().vfPrediction;
EXPECT_EQ(result.status, ::pto::mocker::vf::VfPredictionStatus::VfSimHit);
EXPECT_GT(result.vfsimHitCount, 0U);
EXPECT_EQ(result.fallbackCount, 0U);
```

当前 native VfSim 尚未建模 predicate register。`plt_b8/plt_b16/plt_b32` 会在 PTO adapter
中显式过滤，其余已支持指令仍可返回 `VfSimHit`。这种结果属于近似预测，可通过
`result.ignoredInstructionCount` 和 `result.diagnostics` 检查过滤情况；PLT 本身的周期和
predicate 数据依赖暂不计入预测。

其他已完整捕获但 native VfSim 尚无模型的 predicate、carry 或多输出指令不会被当作损坏的 trace，
而是返回 `UnsupportedForm` 并使用 A5 公式 fallback。此时 `InvalidTrace` 仍专门表示 capture 信息
缺失或相互矛盾。

也可以控制日志和配置目录：

```cpp
::pto::mocker::vf::VfPredictionOptions options;
options.logLevel = ::pto::mocker::vf::VfSimLogLevel::Summary;
options.configDir = "/path/to/vfsim-or-configs";
::pto::mocker::vf::SetVfPredictionOptions(options);
```

日志级别为 `Off`、`Errors`、`Summary` 和 `Detailed`。配置查找顺序为：`options.configDir`、`PerfSimConfig::vfsim_config_dir`、环境变量 `PTO_VFSIM_CONFIG_DIR`、可执行文件或当前目录附近的可搬移安装布局。

### 4. 检查是否捕获到期望 micro-op

`st_a5` 用例中常见 helper：

```cpp
pto::test::a5::ExpectLastBinaryVecTileOp({"vlds", "vlds", "vadd", "vsts"}, repeat);
```

它用于检查最后一条 tileop 捕获到的 VF micro-op 序列是否符合预期，避免测试只验证 cycle 大于 0，却没有确认真实进入了 VF 捕获路径。

## fallback 策略

A5 VfSim costmodel 有 fallback 机制。常见 fallback 原因包括：

- 当前 tileop 或 micro-op 不在 VfSim 支持集合内。
- `VfInfo` 构造不完整，例如缺少 loop 信息。
- 编译时没有启用 LLVM pass 插桩。
- 配置 JSON 缺失或 simulator 抛出异常。
- 当前 dtype/form 的 capture 信息不完整。

fallback 通常按：

```text
repeat times * 单指令 cycle
```

或从 A5 `vec_cycle_generated.hpp` 返回一个保守估计。A5 fallback 不读取 A2/A3 formula 参数。fallback 可以保证基础预测继续运行，但不等价于真实 VfSim native 模型预测；调用方应检查结构化状态，不能只断言 `cycle > 0`。

## 如何判断是否真的走到 VfSim

建议从以下几个方面检查：

1. 构建日志是否显示 A5 pass 插桩启用：

```text
a5_vf_mock: <target> enabled pass instrumentation (-O0 -g -fpass-plugin)
```

2. 编译命令里是否有：

```text
-fpass-plugin=...
```

3. target 是否链接了：

```text
pto_a5_vfsim
```

4. 测试是否调用 `ExpectLastVfSimHit()`，或直接断言 `VfPredictionStatus::VfSimHit`。

5. `GetLastPtoInstrCycles()` 是否随 repeat、shape、VF 结构变化而变化。

如果只用 `g++` 编译、没有 `-fpass-plugin`、没有链接 `pto_a5_vfsim`，大概率没有进入完整 VfSim native costmodel。

## 与 Perf-Sim 的关系

Perf-Sim 是 pipeline 级算子仿真，关注 AIC/AIV、MTE、CUBE、VEC、MTE3 等 pipeline 时序。

A5 VfSim costmodel 关注的是 A5 vector function 内部 VF 的预测。它可以作为 PTO costmodel 中 VF 部分的后端模型，在 PTO 指令结束时追加 VF cycle。

简单理解：

| 模型 | 关注点 |
| --- | --- |
| Perf-Sim | 整个 PTO kernel / pipeline 时序。 |
| A5 VfSim costmodel | 单个或多个 VF 结构的执行时间。 |
| fallback | VfSim 不支持或信息不足时的保守估计。 |

## 常见问题

### 为什么不能只用 g++？

因为 A5 VfSim 需要 LLVM pass 捕获 loop 结构，而 `g++` 不支持 `-fpass-plugin`。用 `g++` 可能能编过部分头文件，但无法保证构造完整 `VfInfo`，通常只能走 fallback。

### `-fpass-plugin` 是什么？

它是 Clang 的 LLVM pass 插件参数。A5 VfSim costmodel 用它加载 loop 插桩 pass，在编译期给 `__VEC_SCOPE__` 内的 loop 插入记录逻辑。

### `-lpto_a5_vfsim` 是什么？

它链接 PTO-ISA 中 vendored VfSim native C++ 模型库。真实 VfSim 预测实现不完全是 header-only，单文件 lit 编译如果不链接该库会出现 undefined reference。

### 完整 Host main 如何处理 ACL Runtime？

`include/pto/costmodel/common/aclrt_stub.hpp` 已提供初始化、设备、stream、Host/Device 分配、Memcpy/Memset、同步和释放的 Host mock。这些 API 只保证控制流进入 kernel，周期为零，不模拟真实 ACL 异步调度。`target_enable_a5_vf_mock` 会设置 costmodel stub include 路径。

### 编译报 `ld_dev` / `st_atomic` / `ATOMIC_SUM` 未定义

这些是 CCE/device intrinsic。A5 costmodel 使用专用 sync mock：VF 内 `mem_bar` 进入 VfSim，VF 外 `pipe_barrier`、event 和 SyncAll 进入 Perf-Sim，不再 include `syncall_soft.hpp` 的 device atomic 路径。

### 编译报 TLOAD/TSTORE 相关 intrinsic

真实 A5 `TLoad.hpp` / `TStore.hpp` 会引入 MTE intrinsic。A5 costmodel 通常不应直接 include 真实实现，而应使用 costmodel 专用 TLOAD/TSTORE stub 来满足模板匹配和 trace 捕获。

### 结果一直像 fallback

优先检查：

- 是否使用 Clang 而不是 g++。
- 是否带 `-D__COSTMODEL -D__NPU_ARCH__=3101`。
- 是否启用了 `-fpass-plugin`。
- 是否链接 `pto_a5_vfsim`。
- 当前 tileop / micro-op 是否在 VfSim 支持范围内。
- 测试是否真的执行了 `__VEC_SCOPE__` 路径。

### 多个 VF 怎么计算？

单个 `VfInfo` 返回一个预测 cycle。多个 `VfInfo` 时，PTO-ISA 会对每个 VF 独立调用 VfSim 预测，然后把结果求和，最终仍返回一个标量 cycle。
