# pto.twait

## 概要

`pto.twait` 会阻塞，直到一个 signal 对象满足与 `int32_t` 参考值之间的比较条件。

## 语义

`TWAIT` 是 `TTEST` 的阻塞版本。

已核实的 public wrapper 接受：

- signal 对象，
- `int32_t` 比较值，
- `WaitCmp` 比较模式，
- 以及可选的等待事件 token。

从概念上看，执行会一直等待，直到实现定义的 signal 状态满足所选比较关系。

## 汇编语法

```text
pto.twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - public wrapper 将比较值类型固定为 `int32_t`。
    - 信号存储必须与所选后端实现兼容。
    - wrapper 在进入等待前会先等待所有传入事件 token。

## 与 `TNOTIFY` / `TTEST` 的关系

- `TNOTIFY` 负责产生 signal 更新；
- `TWAIT` 负责阻塞直到更新可见并满足比较条件；
- `TTEST` 提供非阻塞测试形式。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

void wait_ready(auto &signal) {
    comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
}
```
