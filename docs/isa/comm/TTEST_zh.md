# pto.ttest

## 概要

`pto.ttest` 对 signal 对象执行一次非阻塞比较测试，并返回布尔结果。

## 语义

已核实的 public wrapper 接受：

- signal 对象，
- `int32_t` 比较值，
- `WaitCmp` 比较模式，
- 以及可选的等待事件 token。

它返回后端实现给出的布尔测试结果。

## 汇编语法

```text
%result = pto.ttest %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32) -> i1
```

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`。

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST bool TTEST(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents &... events);
```

## 约束

!!! warning "约束"
    - public wrapper 将比较值类型固定为 `int32_t`。
    - 信号存储必须与所选后端实现兼容。
    - wrapper 在发起测试前会先等待所有传入事件 token。

## 与 `TWAIT` / `TNOTIFY` 的关系

- `TTEST` 是 `TWAIT` 的非阻塞对应物；
- `TNOTIFY` 通常负责 producer 侧的 signal 更新。

## 示例

```cpp
#include <pto/comm/pto_comm_inst.hpp>
using namespace pto;

bool is_ready(auto &signal) {
    return comm::TTEST(signal, 1, comm::WaitCmp::EQ);
}
```
