# PTO Costmodel

`include/pto/costmodel` 是 PTO-ISA 的主机侧静态时延估算后端。它的目标不是模拟数值结果，而是在 host 上复用原始 PTO 指令实现路径，记录底层 fake CCE 调用，并给出每条 PTO 指令的 cycle 预测结果。

如果你需要的是数值正确性验证，请使用 CPU 后端；如果你要看一条 PTO 指令大致会发出哪些 CCE 调用、每个调用估多少 cycle、总延迟是多少，就看 costmodel。

## 当前能力范围

- 只支持 `__NPU_ARCH__ == 2201`，也就是当前仓库里的 A2/A3 costmodel 路径。
- live 路径已经不再依赖 deferred evaluator。
- cycle 仍然保存在 PTO trace 记录里，同时会写回到第一运行时参数（如果它支持 `SetLastCycle(float)`）。
- `include/pto/costmodel/a2a3/cce_stub.hpp` 现在只是兼容 include shim，真正生效的是 `a2a3/cce_costmodel.hpp`。

## 它是怎么工作的

当前 live 路径可以概括成下面这条链路：

```text
用户代码
  -> #include <pto/pto-inst.hpp>
  -> __COSTMODEL 选择 include/pto/costmodel/runtime_stub.hpp
  -> common/arch_select.hpp
  -> a2a3/cce_costmodel.hpp
  -> 原始 PTO A2/A3 指令头继续编译
  -> include/pto/costmodel/pto_instr.hpp 用 `MAP_INSTR_IMPL` / `MAP_INSTR_IMPL_T` 包一层 PTO API
  -> fake CCE intrinsic 在 cce_costmodel.hpp 里直接算周期并 RecordCceCall(...)
  -> trace.hpp 把这些 CCE 调用累加到当前 PTO 指令记录的 total_cycles
```

### 流程图 1：stub 入口和 live 头选择

```text
                    用户代码
                         |
                         v
              #include <pto/pto-inst.hpp>
                         |
                   定义 __COSTMODEL
                         |
                         v
        include/pto/costmodel/runtime_stub.hpp
                         |
         +---------------+-------------------+
         |               |                   |
         v               v                   v
 common/qualifiers   common/aclrt_stub   common/runtime_util
   host 限定符替身      host ACL 伪实现      host helper shim
                         |
                         v
             common/arch_select.hpp
                         |
             __NPU_ARCH__ == 2201 ?
                  | yes
                  v
        a2a3/cce_costmodel.hpp   <---- a2a3/cce_stub.hpp 仅兼容转发
                  |
                  v
        原始 PTO A2/A3 指令头继续编译
```

这个流程里所谓的 “stub” 主要分成两层：

- `runtime_stub.hpp` 和 `common/` 下的 host 兼容层，负责伪造设备编译环境
- `a2a3/cce_costmodel.hpp` 里的 fake intrinsic，负责真正记录 CCE 调用并计算 latency

### 1. PTO 指令边界由 `MAP_INSTR_IMPL` / `MAP_INSTR_IMPL_T` 标记

`include/pto/costmodel/pto_instr.hpp` 在 `__COSTMODEL` 下把每条顶层 PTO API 包成：

```cpp
#define PTO_FIRST_ARG(first, ...) first
#define PTO_TEMPLATE_ARGS(...) <__VA_ARGS__>

#define MAP_INSTR_IMPL(API, ...) \
    do { \
        ::pto::mocker::PtoInstrScope _scope(#API); \
        API##_IMPL(__VA_ARGS__); \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
    } while (0)

#define MAP_INSTR_IMPL_T(API, TEMPLATE_ARGS, ...) \
    do { \
        ::pto::mocker::PtoInstrScope _scope(#API); \
        API##_IMPL TEMPLATE_ARGS (__VA_ARGS__); \
        ::pto::mocker::InjectTileCycles(PTO_FIRST_ARG(__VA_ARGS__)); \
    } while (0)
```

也就是说，像下面这样的 PTO 调用：

```cpp
TADD(dst, src0, src1);
```

最终会在 costmodel 路径里变成：

```cpp
MAP_INSTR_IMPL(TADD, dst, src0, src1);
MAP_INSTR_IMPL_T(TSTORE, PTO_TEMPLATE_ARGS(TileData, GlobalData, AtomicType::AtomicNone), dst, src);
```

`PtoInstrScope` 负责给当前这条 PTO 指令建立一个 trace 记录边界；`InjectTileCycles(...)`
会在 `_IMPL` 之后把当前 PTO 总 cycle 写回第一运行时参数（如果该对象支持 `SetLastCycle`）。

### 流程图 2：如何捕获 PTO 指令和 CCE 指令

```text
用户调用 PTO API
  例如: TADD(dst, src0, src1)
          |
          v
MAP_INSTR_IMPL(TADD, dst, src0, src1)
          |
          v
PtoInstrScope("TADD")
  - BeginPtoInstr("TADD")
  - 在 trace.executed_pto 中建立 / 复用顶层 PTO 记录
          |
          v
执行 TADD_IMPL(...)
          |
          v
InjectTileCycles(dst)
          |
          v
原始 PTO 实现内部调用 fake CCE intrinsic
  例如: vadd(...)
          |
          v
RecordCceCall("vadd", cycles, args...)
          |
    +-----+-----------------------------+
    |                                   |
    v                                   v
追加到当前 PTO 记录的 cce_calls        累加到当前 PTO 记录的 total_cycles
          |
          v
PtoInstrScope 析构
  - EndPtoInstr()
          |
          v
得到一条完整 PTO 记录
  name = "TADD"
  cce_calls = [...]
  total_cycles = ...
```

这里有一个关键点：当前 live path 只在顶层 PTO 记录上累计总时延，不再把 cycle 写回 tile。

### 2. fake CCE intrinsic 在一个地方同时做“记录 + 估时”

`include/pto/costmodel/a2a3/cce_costmodel.hpp` 是当前唯一 live 的 A2/A3 fake intrinsic 头。

这里的每个 fake intrinsic 都直接做两件事：

1. 根据当前公式计算自己的 latency
2. 调 `::pto::mocker::RecordCceCall(name, cycles, args...)`

例如一个典型的向量指令就是这种结构：

```cpp
inline void vadd(auto dst, auto src0, auto src1, auto repeat, ...)
{
    const uint64_t cycles =
        EstimateLinearCycles(repeat, 14, 1, 18);
    ::pto::mocker::RecordCceCall("vadd", cycles, dst, src0, src1, repeat, ...);
}
```

当前已经没有 “先记 trace，再走第二遍 evaluator” 这一步了。公式在哪个 intrinsic 上，cycle 就在那个 intrinsic 上直接算出来。

### 3. `trace.hpp` 负责累计一条 PTO 指令的总时延

`include/pto/costmodel/trace.hpp` 维护当前 live trace 状态：

```cpp
struct CceCallRecord {
    std::string name;
    std::vector<Arg> args;
    uint64_t cycles = 0;
};

struct PtoInstrRecord {
    std::string name;
    std::vector<CceCallRecord> cce_calls;
    uint64_t total_cycles = 0;
};
```

当 `RecordCceCall(...)` 被调用时，它会：

- 把参数按位置顺序转成 `uint64_t` trace 值
- 追加一条 `CceCallRecord`
- 同时把 `cycles` 累加到当前 PTO 记录的 `total_cycles`

嵌套 PTO helper 调用会折叠到顶层 PTO 记录里，不会拆成多条独立顶层记录。

### 流程图 3：一条 CCE intrinsic 如何计算 latency

```text
进入 fake CCE intrinsic
  例如: vadd(...) / copy_gm_to_cbuf(...) / wait_flag(...)
             |
             v
根据指令类型准备公式输入
  - 搬运类: bytes
  - 向量类: repeat
  - 常数类: 固定 cycles
             |
             v
调用通用 helper
- EstimateBandwidthCycles(bytes, key)
- EstimateLinearCycles(repeat) 或 EstimateLinearCycles(repeat, head, slope, tail)
- EstimateConstCycles(cycles)
             |
             v
得到该 CCE 调用自己的 cycles
             |
             v
RecordCceCall(name, cycles, args...)
             |
    +--------+---------------------------+
    |                                    |
    v                                    v
记录 CceCallRecord{name, args, cycles}   PTO.total_cycles += cycles
```

所以现在的 latency 路径是“边执行 fake intrinsic，边算自己的延迟并立刻累计”，而不是执行完 PTO 后再统一评估一次。

## 时延预测公式在哪里

### 1. 架构参数

架构常量定义在 [arch_config.hpp](arch_config.hpp)：

- `frequency_hz`
- `bandwidth[PipeKey::*]`
- `kBlockBytes`

当前默认配置是 `kA2A3ArchConfig`。

### 2. 通用估算原语

通用 helper 直接定义在 [a2a3/cce_costmodel.hpp](a2a3/cce_costmodel.hpp) 里，和 fake CCE intrinsic 共用同一个头文件作用域：

- `CurrentArch()`
- `EstimateBandwidthCycles(bytes, key)`
- `EstimateLinearCycles(repeat, head = 6, slope = 2, tail = 0)`
- `EstimateConstCycles(cycles)`
- `CeilDiv(x, y)`
- `ExtractBits(value, shift, mask)`

这些 helper 本身不记录 trace，只负责把局部公式写得更直接。

- `EstimateLinearCycles(repeat)` 表示这条线性公式还是 guessed placeholder，当前默认按 `head=6, slope=2, tail=0` 计算。
- `EstimateLinearCycles(repeat, head, slope, tail)` 表示这条公式已经填入了实际数据，应继续显式保留。

### 3. 每条 CCE intrinsic 的具体公式

还是在 [a2a3/cce_costmodel.hpp](a2a3/cce_costmodel.hpp)。

常见模式有三类：

- 带宽类指令：先根据搬运字节数算 `bytes`，再走 `EstimateBandwidthCycles(...)`
- 线性类指令：按 `repeat` 走 `EstimateLinearCycles(...)`
- 常数类指令：直接走 `EstimateConstCycles(...)`

小例子：

```cpp
inline void copy_gm_to_cbuf(auto dst, auto src, auto sid, auto nBurst, auto lenBurst, auto gmGap, auto l1Gap, auto pad)
{
    const uint64_t bytes = nBurst * lenBurst * ::pto::mocker::evaluator::kBlockBytes;
    const uint64_t cycles =
        EstimateBandwidthCycles(
            bytes, ::pto::mocker::evaluator::PipeKey::GM_TO_L1);
    ::pto::mocker::RecordCceCall("copy_gm_to_cbuf", cycles, dst, src, sid, nBurst, lenBurst, gmGap, l1Gap, pad);
}
```

```cpp
inline void wait_flag(auto srcPipe, auto dstPipe, auto token)
{
    const uint64_t cycles = EstimateConstCycles();
    ::pto::mocker::RecordCceCall("wait_flag", cycles, srcPipe, dstPipe, token);
}
```

也可以把公式定位过程理解成下面这个查找顺序：

```text
想看某条 PTO 指令的 latency 从哪里来
        |
        v
先看 pto_instr.hpp 里这条 PTO API 调了哪个 *_IMPL
        |
        v
再看原始 PTO 实现内部发出了哪些 fake CCE intrinsic
        |
        v
到 a2a3/cce_costmodel.hpp 里找到对应 intrinsic
        |
        v
看该 intrinsic 里如何算 bytes / repeat / cycles
        |
        v
最后看 trace.hpp 中 RecordCceCall(...) 如何把它累计到 PTO.total_cycles
```

## 如何使用

### 1. 让 PTO 代码走到 costmodel 路径

最常见的方式是在编译时定义：

- `__COSTMODEL`
- `__NPU_ARCH__=2201`

然后像平时一样包含：

```cpp
#include <pto/pto-inst.hpp>
```

PTO 调用方式不需要改，仍然正常写：

```cpp
using TileData = Tile<TileType::Vec, float, 64, 64>;

TileData src0(64, 64), src1(64, 64), dst(64, 64);
TASSIGN(src0, 0x0);
TASSIGN(src1, 0x4000);
TASSIGN(dst, 0x8000);

TADD(dst, src0, src1);
```

### 2. 在 host 上读取结果

当前结果不再写回 tile。要看最近一条 PTO 指令的总 cycle，用：

```cpp
uint64_t cycles = ::pto::mocker::GetLastPtoInstrCycles();
```

如果你想看更细的记录，可以直接读：

```cpp
const auto &trace = ::pto::mocker::GetTrace();
```

其中：

- `trace.executed_pto.back().name` 是 PTO 指令名
- `trace.executed_pto.back().total_cycles` 是该 PTO 指令总 cycle
- `trace.executed_pto.back().cce_calls` 是它展开后的 CCE 调用列表

读取结果时的逻辑可以简单理解成：

```text
执行 PTO 指令
    |
    v
trace.executed_pto.back()
    |
    +--> name         : 这条 PTO 指令叫什么
    +--> cce_calls    : 它展开成了哪些 CCE 调用
    +--> total_cycles : 所有 CCE 调用 cycles 的直接累加
    |
    v
GetLastPtoInstrCycles()
```

如果需要清空上下文，可以调用：

```cpp
::pto::mocker::ResetTrace();
```

## 目录结构

```text
include/pto/costmodel/
├── runtime_stub.hpp            __COSTMODEL 总入口
├── pto_instr.hpp               PTO API 包装，定义 MAP_INSTR_IMPL
├── trace.hpp                   PTO / CCE trace 记录与累计
├── arch_config.hpp             架构频率与带宽参数
├── common/
│   ├── qualifiers.hpp          host 侧 qualifier 替身
│   ├── aclrt_stub.hpp          ACL runtime 伪实现
│   ├── runtime_util.hpp        host helper shim
│   └── arch_select.hpp         按 __NPU_ARCH__ 选择 live costmodel 头
└── a2a3/
    ├── cce_costmodel.hpp       A2/A3 live fake intrinsic + latency 公式
    ├── cce_stub.hpp            兼容 include shim
    └── README.md               A2/A3 细节说明
```

## 如何运行测试

costmodel 的主测试集在 `tests/costmodel/st`，是 host 上运行的 GTest。

### 1. 跑全部 costmodel ST

```bash
./tests/run_costmodel_tests.sh
```

这个脚本会遍历 `tests/costmodel/st/testcase/a2a3/` 下的 testcase，并对每个 case 走一次：

```bash
python tests/run_costmodel.py --testcase <name> --clean --verbose
```

### 2. 跑单个 testcase

```bash
python tests/run_costmodel.py --testcase tadd --clean --verbose
```

常用变体：

```bash
python tests/run_costmodel.py --testcase tadd --no-build
python tests/run_costmodel.py --testcase tadd --gtest_filter "TAdd.float_64x64"
python tests/run_costmodel.py --testcase tmatmul --build-type Debug
```

### 3. 测试里如何校验 cycle

当前 ST 不再用 `tile.GetCycle()`。公共断言在
[tests/costmodel/st/common/cost_check.hpp](../../../tests/costmodel/st/common/cost_check.hpp)：

```cpp
EXPECT_CYCLE_NEAR(profiling, accuracy);
```

它内部读取的是：

```cpp
::pto::mocker::GetLastPtoInstrCycles()
```

也就是最近一条 PTO trace 记录的 `total_cycles`。

### 4. 一个最小测试例子

```cpp
#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>
#include "cost_check.hpp"

using namespace pto;

TEST(TAdd, float_64x64)
{
    using TileData = Tile<TileType::Vec, float, 64, 64>;
    TileData src0(64, 64), src1(64, 64), dst(64, 64);

    TASSIGN(src0, 0x0);
    TASSIGN(src1, 0x4000);
    TASSIGN(dst, 0x8000);

    TADD(dst, src0, src1);

    EXPECT_CYCLE_NEAR(132.0f, 1.0f);
}
```

## 相关文件

- [a2a3/cce_costmodel.hpp](a2a3/cce_costmodel.hpp)：A2/A3 fake intrinsic 和具体 latency 公式
- [trace.hpp](trace.hpp)：PTO 记录、CCE 记录、`GetLastPtoInstrCycles()`
- [pto_instr.hpp](pto_instr.hpp)：`MAP_INSTR_IMPL` 包装点
- [tests/costmodel/README.md](../../../tests/costmodel/README.md)：测试目录与脚本入口
- [tests/costmodel/USER_GUIDE.zh-CN.md](../../../tests/costmodel/USER_GUIDE.zh-CN.md)：更完整的测试使用说明
