# A5 CCE Mock VfSim 接口与调用链说明

本文档整理当前 PTO-ISA A5 CCE mock 接入 VfSim costmodel 的接口形式、`VfInfo` 构造来源、单 VF/多 VF 调用语义，以及 fallback 策略。

当前代码状态：

- PTO-ISA 最新相关提交：`5123a6a0 Integrate A5 VfSim costmodel`
- Vendored VfSimulator 来源分支：`vfinfo-core-api-unification`
- Vendored VfSimulator commit：`38f974a5b4e89cdb69fcd1dc02329b448d759eea`

## 1. 总体逻辑

当前主链路是：

```text
PTO/tileop 执行
  -> A5 CCE mock 拦截 VF micro-op
  -> LLVM pass 插桩拦截 __VEC_SCOPE__ 内 for-loop
  -> micro-op 事件 + loop 事件折叠成 PTO 侧 VfInfo
  -> EndPtoInstr() 调用 PredictVfCycles(...)
  -> PTO VfInfo lower 成 vfsim::VfInfo
  -> VfSim runVfInfo(...) 预测 cycle
  -> 返回 uint64_t 标量 cycle
```

要点：

- micro-op 捕获靠 CCE intrinsic mock stub。
- for-loop 结构捕获靠 LLVM pass 插桩。
- `VfInfo` 是当前 PTO 和 VfSim costmodel 之间的接口数据结构。
- 单个 `__VEC_SCOPE__` 构造一个 `VfInfo`。
- 一个 PTO/tileop 理论上可以收集一个或多个 `VfInfo`。
- 当前常见真实路径多数是条件选择一个 `__VEC_SCOPE__`，即一个 PTO/tileop 产生一个 `VfInfo`。

## 2. PTO 侧接口

PTO 侧 costmodel 调用口定义在：

- `include/pto/costmodel/a5/cce_costmodel/vf_cost.hpp`

接口有两个 overload：

```cpp
namespace pto::mocker::vf {

uint64_t PredictVfCycles(const VfInfo &vf);
uint64_t PredictVfCycles(const std::vector<VfInfo> &vfs);

}
```

语义：

- `PredictVfCycles(const VfInfo &vf)`：单个 VF，返回该 VF 的预测 cycle。
- `PredictVfCycles(const std::vector<VfInfo> &vfs)`：一个 PTO/tileop 内多个 VF，逐个 VF 独立预测，最后求和返回一个标量 cycle。

`PredictVfCycles(...)` 内部调用：

```cpp
uint64_t PredictVfCyclesWithVfSim(const std::vector<VfInfo> &vfs);
```

定义在：

- `include/pto/costmodel/a5/VfSim/VfSimCostModel.h`
- `include/pto/costmodel/a5/VfSim/VfSimCostModel.cpp`

## 3. 用户如何启用和调用

### 3.1 入口头文件

正常用户不需要直接 include VfSim 头文件。costmodel 编译时使用 PTO 统一入口：

```cpp
#include <pto/pto-inst.hpp>
```

当定义了 `__COSTMODEL` 后，`pto/pto-inst.hpp` 会选择 costmodel 侧 PTO 指令实现，进入：

- `include/pto/costmodel/pto_instr.hpp`

A5 架构下，costmodel 会进一步 include A5 CCE mock：

- `include/pto/costmodel/a5/cce_costmodel/cce_costmodel.hpp`

### 3.2 必要编译宏

A5 VfSim costmodel 需要以下宏：

```text
-D__COSTMODEL
-D__NPU_ARCH__=3101
```

说明：

- `__COSTMODEL`：启用 PTO costmodel/mock 路径。
- `__NPU_ARCH__=3101`：选择 A5 costmodel。当前代码中 `3101` 和 `3510` 都走 A5 VF 结算 gate。
- `-DPTO_COMM_NOT_SUPPORTED`：测试工程中使用，用来跳过通信相关 PTO 头；普通接入是否需要取决于编译目标是否包含通信指令。

当前 `st_a5` 测试工程使用：

```cmake
add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)
```

位置：

- `tests/costmodel/st_a5/CMakeLists.txt`

### 3.3 LLVM/Clang 编译要求

因为 loop 结构依赖 LLVM pass 插桩，所以测试 target 必须用和 `llvm-config` 主版本匹配的 `clang++` 编译。

当前封装在：

- `cmake/a5_vf_mock.cmake`

该 CMake 会：

1. 查找 `llvm-config`。
2. 查找同主版本的 `clang++`。
3. 编译 `PtoLoopTracePass.cpp` 为 LLVM pass plugin。
4. 编译 vendored VfSim native core 为 `pto_a5_vfsim` 静态库。
5. 给目标 target 增加 pass 插桩编译选项。

关键编译选项：

```text
-O0
-g
-fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>
```

当前还会加：

```text
-Wno-macro-redefined
-Wno-ignored-attributes
-include stdint.h
-include stddef.h
```

### 3.4 CMake 启用方式

最小 CMake 用法：

```cmake
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/a5_vf_mock.cmake)
_pto_a5_find_llvm()
set(CMAKE_CXX_COMPILER ${PTO_A5_CLANGXX})

project(my_a5_costmodel_test CXX)

add_definitions(-D__COSTMODEL -D__NPU_ARCH__=3101 -DPTO_COMM_NOT_SUPPORTED)

add_executable(my_case main.cpp)
target_include_directories(my_case PRIVATE ${PTO_ISA_ROOT}/include)
target_enable_a5_vf_mock(my_case)
```

`target_enable_a5_vf_mock(my_case)` 是启用 A5 VF costmodel 的关键接口。它会：

- link `pto_a5_vfsim`
- build 并依赖 `PtoLoopTracePass`
- 给 target 加 `-fpass-plugin`

当前工程示例：

- `tests/costmodel/st_a5/CMakeLists.txt`
- `tests/costmodel/st_a5/testcase/tadd/CMakeLists.txt`

当前验证工程的配置、编译、运行命令示例。以下命令假设当前 shell 中：

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
export BUILD_DIR=/path/to/build/st_a5
```

其中 `PTO_ISA_ROOT` 是 PTO-ISA 仓库根目录，`BUILD_DIR` 是任意可写 build 目录。

```bash
cd "$PTO_ISA_ROOT"

cmake -S tests/costmodel/st_a5 \
      -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DPTO_A5_LLVM_CONFIG=/path/to/llvm-config

cmake --build "$BUILD_DIR" -j4

ctest --test-dir "$BUILD_DIR" --output-on-failure
```

如果没有显式传 `-DPTO_A5_LLVM_CONFIG=...`，`cmake/a5_vf_mock.cmake` 会尝试自动查找 `llvm-config`、`llvm-config-20`、`llvm-config-19` 等，并寻找同主版本的 `clang++`。

### 3.5 入口函数和自动调用链

用户正常调用 PTO/tileop API 即可，例如：

```cpp
TADD(dst, src0, src1);
```

从用户视角看，入口函数就是 PTO/tileop 指令 API，例如 `TADD(...)`、后续接入的其他 A5 tileop API。只要该编译单元走 `__COSTMODEL` + A5 mock 编译路径，用户不需要手动调用 VfSim。

在 `__COSTMODEL` 下，PTO API 会经由 `MAP_INSTR_IMPL` 包装：

```cpp
::pto::mocker::PtoInstrScope _scope(#API);
API##_IMPL(__VA_ARGS__);
_scope.Finish();
::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));
::RecordInstrFromFirst(#API, __VA_ARGS__);
```

因此用户不需要手动调用 `PredictVfCycles`。自动调用链是：

```text
TADD(...)
  -> MAP_INSTR_IMPL(TADD, ...)
  -> PtoInstrScope
  -> TADD_IMPL 内执行 __VEC_SCOPE__
  -> ScopeSentinel 析构生成 VfInfo
  -> EndPtoInstr()
  -> vf::PredictVfCycles(pto.vf_infos)
  -> VfSim/fallback
  -> total_cycles
```

这里有三个层次的入口：

- 用户入口：`TADD(...)` 等 PTO/tileop API。
- PTO 指令结算入口：`::pto::mocker::EndPtoInstr()`，由 `PtoInstrScope::Finish()` 触发。
- VF costmodel 入口：`pto::mocker::vf::PredictVfCycles(...)`，由 `EndPtoInstr()` 在发现当前 PTO 指令包含 `vf_infos` 时调用。

`EndPtoInstr()` 的核心决策是：

```text
if 当前 PTO 指令收集到 vf_infos:
    total_cycles += vf::PredictVfCycles(vf_infos)
else:
    不调用 VfSim，保留普通 PTO/非 VF 路径已经累计的 cycle
```

注意这里是累加，不是覆盖。`EndPtoInstr()` 前会先 flush 非 vector stream 的 pending tail，混合指令中已经由 CCE/pipe trace 累计出的非 VF cycle 会保留；VF costmodel 只追加 VF 部分的预测 cycle。

### 3.6 获取预测结果

PTO 指令结束后，预测值会写入当前 trace：

```cpp
uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
```

位置：

- `include/pto/costmodel/trace.hpp`

同时，`MAP_INSTR_IMPL` 会调用：

```cpp
::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__));
```

如果第一个 tile 对象支持 `SetLastCycle(...)`，则会把当前 PTO 指令 cycle 写回 tile。

### 3.7 手动调用入口

单测或调试时也可以绕过 PTO API，手动构造 `VfInfo` 并调用：

```cpp
#include "pto/costmodel/a5/cce_costmodel/vf_cost.hpp"

uint64_t one = pto::mocker::vf::PredictVfCycles(vf_info);
uint64_t sum = pto::mocker::vf::PredictVfCycles(std::vector<VfInfo>{vf0, vf1});
```

当前主测试路径不直接手写 `VfInfo`，而是通过 PTO tileop API 触发完整 CCE mock 调用链。手写 `VfInfo`
更适合作为后续单元测试补充，用于单独覆盖 lowering、fallback、多 VF 求和等内部行为。

## 4. PTO 侧 VfInfo 结构

定义位置：

- `include/pto/costmodel/a5/cce_costmodel/vf_info.hpp`

核心结构：

```cpp
enum class MemLocation : uint8_t {
    PhyRegister,
    UB,
};

struct MemInfo {
    std::string name;
    MemLocation location = MemLocation::PhyRegister;
    std::string dtype;
};

struct VfInst {
    std::string opName;
    std::vector<MemInfo> dst;
    std::vector<MemInfo> src;
};

struct VfLoop {
    uint64_t count = 0;
    std::vector<VfNode> body;
};

struct VfInfo {
    std::string op;
    std::string shape;
    std::vector<VfNode> tree;
};
```

当前约定：

- `VfInfo` 不保存全局 dtype。
- `VfInst` 不保存 dtype。
- dtype 只在 `MemInfo` 上表达。
- `MemInfo.location` 显式表达 storage：
  - `PhyRegister`：physical register value
  - `UB`：UB/memory-side value
- `MemInfo.name` 是公开 value id，不承担 storage 编码职责。
- register/UB 判断不依赖 `V` 或 `mem` 前缀。

## 5. VfInfo 构造来源

### 5.1 PTO 指令作用域

PTO 指令通过 `PtoInstrScope` 建立计费作用域。

位置：

- `include/pto/costmodel/pto_instr.hpp`
- `include/pto/costmodel/trace.hpp`

典型宏展开：

```cpp
#define MAP_INSTR_IMPL(API, ...)                                     \
    do {                                                             \
        ::pto::mocker::PtoInstrScope _scope(#API);                   \
        API##_IMPL(__VA_ARGS__);                                     \
        _scope.Finish();                                             \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
        ::RecordInstrFromFirst(#API, __VA_ARGS__);                   \
    } while (0)
```

`PtoInstrScope` 生命周期：

```text
BeginPtoInstr(name)
  -> 执行 PTO _IMPL
  -> 期间可能进入 __VEC_SCOPE__
  -> EndPtoInstr()
```

### 5.2 micro-op 捕获

VF micro-op 由 CCE mock stub 捕获。

位置：

- `include/pto/costmodel/a5/cce_costmodel/a5_vf_stub.hpp`

记录入口：

- `capture::RecordLoad(...)`
- `capture::RecordStore(...)`
- `capture::RecordCompute(...)`
- `capture::rec(VfInst inst)`

捕获内容：

- `opName`
- `dst`
- `src`
- 每个 operand 的 `name/location/dtype`

operand 映射规则：

- `RegTensor<T>` -> `MemLocation::PhyRegister`，dtype 来自 `T`
- pointer -> `MemLocation::UB`，dtype 来自 pointer element type
- `vector_f32/vector_f16/vector_s32/...` -> `MemLocation::PhyRegister`

predicate setup 不算 micro-op：

- `plt_b8`
- `plt_b16`
- `plt_b32`

这些 stub 只返回 predicate，不写入 `VfInfo`。

### 5.3 loop 捕获

CCE mock 本身不能拦截 C++ `for` 结构，所以 loop 捕获依赖 LLVM pass。

相关位置：

- `include/pto/costmodel/a5/PtoLoopTracePass.cpp`
- `cmake/a5_vf_mock.cmake`
- `include/pto/costmodel/a5/cce_costmodel/vf_trace.hpp`

CMake 会构建 LLVM pass plugin，并给测试 target 加：

```text
-O0 -g -fpass-plugin=$<TARGET_FILE:PtoLoopTracePass>
```

pass 只对 `__VEC_SCOPE__` 内的 loop 插桩。运行时 hook：

```cpp
extern "C" void __pto_trace_loop_enter(uint64_t loopId, const char *file, int line, int col);
extern "C" void __pto_trace_loop_iter(uint64_t loopId);
extern "C" void __pto_trace_loop_exit(uint64_t loopId);
```

### 5.4 __VEC_SCOPE__ 与 VfInfo 生成

`__VEC_SCOPE__` 被定义为 RAII sentinel：

```cpp
#define __VEC_SCOPE__ \
    if (::pto::mocker::vf::ScopeSentinel _pto_vf_scope_{}; _pto_vf_scope_)
```

`ScopeSentinel` 构造时：

- 发出 `__pto_vf_scope_enter()` marker
- `trace::Arm(true)`
- 清空当前 VF operand name 映射

`ScopeSentinel` 析构时：

- 发出 `__pto_vf_scope_exit()` marker
- `trace::Arm(false)`
- 调用 `trace::BuildVfInfo(op, "")`
- 成功后把生成的 `VfInfo` push 到当前 `PtoInstrRecord.vf_infos`
- 清空当前 VF 事件流

因此：

```text
一个 __VEC_SCOPE__ -> 一个 PTO VfInfo
一个 PTO/tileop -> std::vector<VfInfo> vf_infos
```

## 6. VfInfo 折叠规则

折叠逻辑位置：

- `include/pto/costmodel/a5/cce_costmodel/vf_trace.hpp`

输入事件类型：

```cpp
enum class EvKind : uint8_t {
    LoopEnter,
    LoopIter,
    LoopExit,
    Op,
    MemBar,
};
```

折叠规则：

- `LoopEnter/LoopIter/LoopExit` 折叠为 `VfLoop`
- `Op` 折叠为 `VfInst`
- `MemBar` 折叠为 `VfMemBar`，表示 VF scope 内的 `mem_bar(...)` 节点
- loop count 取非空 body 的迭代次数
- 同一 loop 的每轮 body 必须结构一致
- 非均匀 loop 或非法事件流会导致 `BuildVfInfo` 返回失败

说明：`pipe_barrier(...)` 是 pipe/comm 同步层面的 barrier，正常不应出现在 `__VEC_SCOPE__` 内，也不作为 VfSim VF micro-op 输入。mock stub 中保留了兜底记录能力；如果它异常进入 VF trace，则按异常 barrier 节点触发 fallback。

当前失败处理：

- `BuildVfInfo` 失败时，该 VF 不会 push 到 `vf_infos`
- 这类构建失败目前不进入 `PredictVfCycles` fallback

## 7. VfSim Lowering

VfSim typed API 定义位置：

- `include/pto/costmodel/a5/VfSim/VfInfo.h`

核心结构：

```cpp
namespace vfsim {

enum class ValueStorageKind { Register, UB, Scalar };

struct ValueInfo {
  std::string valueId;
  ValueStorageKind storage = ValueStorageKind::Register;
  std::string dtype;
  std::vector<int64_t> shape;
};

struct ProgramInstNode {
  std::string op;
  std::vector<std::string> src;
  std::vector<std::string> dst;
  std::string form;
};

struct VfInfo {
  std::unordered_map<std::string, ValueInfo> values;
  std::vector<ProgramNode> body;
  std::unordered_map<std::string, int64_t> params;
  std::string defaultDtype = "fp32";
};

}
```

PTO `VfInfo` lowering 到 `vfsim::VfInfo` 的规则：

- `VfInst.opName` 转大写后写入 `ProgramInstNode.op`
- `MemInfo.name` 保留为 `ValueInfo.valueId`
- `MemInfo.location` 映射为 `ValueInfo.storage`
- `MemInfo.dtype` 映射为 `ValueInfo.dtype`
- `VfLoop.count` 映射为 `ProgramLoopNode.iters`

storage 判定：

- VfSim core 通过 `ValueStorageLookup` 读取 `ValueInfo.storage`
- 不依赖 value id 的 `V/mem` 前缀
- unroll lane 后缀会回查原始 value id 的 storage

dtype/form 推导：

- PTO 不做全局 dtype 合并
- VfSim `canonicalizeVfInfo()` 根据 src/dst value dtype 推导每条指令 `form`
- 支持 mixed dtype，例如 `fp32`、`fp16`、`f32_to_f16`

## 8. 单 VF 和多 VF 预测语义

### 8.1 单 VF

单 VF 调用：

```cpp
PredictVfCycles(const VfInfo &vf)
```

语义：

```text
PTO VfInfo
  -> lower 成一个 vfsim::VfInfo
  -> vfsim::runVfInfo(...)
  -> 返回该 VF 的 uint64_t cycle
```

### 8.2 多 VF

多 VF 调用：

```cpp
PredictVfCycles(const std::vector<VfInfo> &vfs)
```

语义：

```text
total = 0
for vf in vfs:
    total += PredictVfCycles(vf)
return total
```

当前实现不会把多个 PTO `VfInfo` 的 nodes 合并到同一个 `vfsim::VfInfo.body`。

也就是说：

- `vfsim::runVfInfo(...)` 始终针对单个 VF 结构体调用
- 多 VF 是 PTO/tileop 层面的容器语义
- 多 VF 最终返回一个标量，等于每个 VF 标量预测值之和
- 当前没有跨 VF overlap 建模

## 9. Fallback 策略

### 9.1 fallback 触发条件

单个 VF 中以下情况会触发该 VF fallback：

- `VfInst.opName` 为空
- `VfInst.dst` 为空
- 非 `VDUP` 指令 `src` 为空
- 出现 `VfMemBar`，即 VF scope 内捕获到 `mem_bar(...)` 或异常 barrier 节点
- `MemInfo.name` 为空
- 同一个 value id 的 storage 冲突
- 同一个 value id 的 dtype 冲突
- canonical 后存在 VfSim `ParamDB` 不支持的 `op + form`
- VfSim canonicalize 或 run 过程抛异常

### 9.2 fallback 计算方式

fallback 逻辑位置：

- `include/pto/costmodel/a5/VfSim/VfSimCostModel.cpp`

单 VF fallback 等价于：

```text
sum(loop_count_product * per_inst_cycle)
```

即：

```text
repeat_times * 指令 cycle 数
```

per-inst cycle 来自：

- `include/pto/costmodel/a5/cce_costmodel/vec_cycle_generated.hpp`

### 9.3 多 VF fallback 粒度

当前不是 all-or-nothing。

多 VF 中每个 VF 独立处理：

```text
if vf_i supported:
    total += VfSim(vf_i)
else:
    total += fallback(vf_i)
```

该逻辑当前在接口实现中保留。当前 `st_a5` 主测试更侧重真实 PTO tileop 入口的端到端链路；多 VF 混合
fallback 适合作为后续 direct `VfInfo` 单测补充。

## 10. 验证方式

### 10.1 测试目录

当前 A5 CCE mock + VfSim 端到端测试位于：

```text
tests/costmodel/st_a5
```

关键文件：

```text
tests/costmodel/st_a5/CMakeLists.txt
tests/costmodel/st_a5/testcase/CMakeLists.txt
tests/costmodel/st_a5/testcase/common/a5_vfsim_tileop_check.hpp
tests/costmodel/st_a5/testcase/<tileop>/CMakeLists.txt
tests/costmodel/st_a5/testcase/<tileop>/main.cpp
```

说明：

- `tests/costmodel/st_a5/CMakeLists.txt` 是独立 CMake 工程入口。
- `tests/costmodel/st_a5/testcase/CMakeLists.txt` 统一注册所有 tileop testcase。
- `common/a5_vfsim_tileop_check.hpp` 提供公共断言 helper。
- 每个 tileop 子目录下的 `main.cpp` 构造 tile、调用 PTO API，并检查最后一条 PTO 指令捕获到的 `VfInfo`。
- 每个 tileop 子目录下的 `CMakeLists.txt` 调用 `pto_a5_vfsim_tileop_test(<name>)`，该函数会调用
  `target_enable_a5_vf_mock(<name>)` 启用 LLVM pass 插桩和 VfSim native core 链接。

### 10.2 当前测试用例

当前 `tests/costmodel/st_a5/testcase/CMakeLists.txt` 注册 13 个 testcase：

| testcase | PTO API | 期望 VF micro-op 序列 |
| --- | --- | --- |
| `tadd` | `TADD` | `vlds, vlds, vadd, vsts` |
| `tadds` | `TADDS` | `vlds, vadds, vsts` |
| `tsub` | `TSUB` | `vlds, vlds, vsub, vsts` |
| `tsubs` | `TSUBS` | `vlds, vadds, vsts` |
| `tmul` | `TMUL` | `vlds, vlds, vmul, vsts` |
| `tmuls` | `TMULS` | `vlds, vmuls, vsts` |
| `tdiv` | `TDIV` | `vlds, vlds, vdiv, vsts` |
| `tmax` | `TMAX` | `vlds, vlds, vmax, vsts` |
| `tmin` | `TMIN` | `vlds, vlds, vmin, vsts` |
| `tmins` | `TMINS` | `vlds, vmins, vsts` |
| `tand` | `TAND` | `vlds, vlds, vand, vsts` |
| `tshls` | `TSHLS` | `vlds, vshls, vsts` |
| `tshrs` | `TSHRS` | `vlds, vshrs, vsts` |

当前 shape 覆盖：

- `tadd` 覆盖 `float` 的 `1x64`、`1x512`、`1x1024`、`1x2048`、`1x4096`、`1x6144`，用于检查 repeat
  次数随 shape 变化。
- 其余 testcase 当前主要覆盖 `1x512`。
- `tand/tshls/tshrs` 使用 `uint32_t`，其余当前主要使用 `float`。

### 10.3 测试断言内容

公共断言定义在：

```text
tests/costmodel/st_a5/testcase/common/a5_vfsim_tileop_check.hpp
```

核心 helper：

```cpp
ExpectLastVecTileOp(expectedBody, expectedRepeat);
ExpectLastBinaryVecTileOp(expectedBody, expectedRepeat);
```

`ExpectLastVecTileOp(...)` 检查：

- `GetLastPtoInstrCycles()` 返回值大于 0，说明 PTO 指令已经走到 cycle 结算。
- `GetTrace().executed_pto.back()` 存在，说明 PTO 指令被记录。
- 最后一条 PTO 指令捕获到且只捕获到 1 个 `VfInfo`。
- `VfInfo.tree` 顶层是 1 个 loop。
- loop depth 为 1。
- loop count 等于测试根据 shape 推导出的 repeat。
- loop body 展平后的 leaf micro-op 序列等于预期序列。

`ExpectLastBinaryVecTileOp(...)` 在上述基础上进一步检查二元向量 tileop 的 operand 捕获：

- 两条 `vlds` 的 `dst` 是 `PhyRegister`，`src` 是 `UB`。
- compute op 的前两个 `src` 分别来自两条 load 的 register `dst`。
- `vsts` 的 `dst` 是 `UB`，`src` 来自 compute op 的 register `dst`。

因此当前测试覆盖的是完整端到端链路：

```text
PTO tileop API
  -> __COSTMODEL + A5 mock 编译路径
  -> __VEC_SCOPE__
  -> LLVM pass loop 插桩
  -> CCE intrinsic stub 捕获 micro-op 和 operand
  -> vf_trace 折叠成 PTO VfInfo
  -> EndPtoInstr 调用 PredictVfCycles
  -> VfSim/fallback 返回 cycle
  -> trace 中可读取最后一条 PTO 指令 cycle 和 VfInfo
```

### 10.4 构建和运行全部测试

推荐使用仓库外的独立 build 目录，避免污染源码树。以下命令假设当前 shell 中：

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
export BUILD_DIR=/path/to/build/st_a5
```

其中：

- `PTO_ISA_ROOT`：PTO-ISA 仓库根目录。
- `BUILD_DIR`：任意可写 build 目录。

首次配置：

```bash
cd "$PTO_ISA_ROOT"

cmake -S tests/costmodel/st_a5 \
      -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DPTO_A5_LLVM_CONFIG=/path/to/llvm-config
```

如果需要显式指定 clang，可增加：

```bash
-DPTO_A5_CLANGXX=/path/to/clang++
```

要求：`PTO_A5_CLANGXX` 的 clang 主版本需要与 `PTO_A5_LLVM_CONFIG` 指向的 LLVM 主版本一致。

编译并运行全部 13 个测试：

```bash
cmake --build "$BUILD_DIR" -j4
ctest --test-dir "$BUILD_DIR" --output-on-failure
```

最近一次结果：

```text
100% tests passed, 0 tests failed out of 13
```

### 10.5 只构建或运行单个 testcase

配置阶段可通过 `TEST_CASE` 只加入一个 testcase，例如只跑 `tadd`：

```bash
export PTO_ISA_ROOT=/path/to/pto-isa
export BUILD_DIR=/path/to/build/st_a5_tadd

cd "$PTO_ISA_ROOT"

cmake -S tests/costmodel/st_a5 \
      -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Debug \
      -DPTO_A5_LLVM_CONFIG=/path/to/llvm-config \
      -DTEST_CASE=tadd

cmake --build "$BUILD_DIR" -j4
ctest --test-dir "$BUILD_DIR" --output-on-failure
```

如果已经配置了全部 testcase，也可以用 `ctest -R` 只运行某个测试：

```bash
ctest --test-dir "$BUILD_DIR" -R '^tadd$' --output-on-failure
```

### 10.6 当前没有覆盖的测试点

当前 `st_a5` 主要覆盖真实 PTO tileop 入口的单 VF 端到端链路。以下行为目前主要由代码路径保证，建议后续补充
direct `VfInfo` 单测或专门 synthetic testcase：

- 一个 PTO/tileop 内多个 `VfInfo` 的逐个预测并求和。
- supported VF 和 unsupported VF 混合时的逐 VF fallback。
- `VfMemBar` 触发 fallback。
- 非均匀 loop 或非法 loop 事件流导致 `BuildVfInfo` 失败。
- 更复杂嵌套 loop 或多层 VF loop 的折叠与 lowering。

## 11. 当前限制

- 真实 tileop 当前多数路径是条件选择一个 `__VEC_SCOPE__`，多 VF 主要是接口支持和 synthetic test 覆盖。
- 多 VF 当前逐个 VF 预测后求和，没有跨 VF overlap 建模。
- fallback cycle 表仍是占位/初版，需要后续标定。
- VF scope 内的 `VfMemBar` 当前触发该 VF fallback，尚未接入 VfSim timing；正常 VF scope 外的 `pipe_barrier(...)` 不属于 VfSim 输入。
- `BuildVfInfo` 失败时当前 VF 不进入 `vf_infos`，也不进入 `PredictVfCycles` fallback。
- VfSim 支持范围取决于 vendored `configs/isa.json` 中的 `op + form` 覆盖。
