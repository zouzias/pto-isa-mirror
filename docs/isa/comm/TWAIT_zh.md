# TWAIT


## 简介


`TWAIT` 是阻塞等待原语：在本地信号满足比较条件之前一直等待。它通常与 `TNOTIFY` 配合使用，实现基于标志的同步。

既支持单个信号，也支持最多 5 维的信号 tensor。对 tensor 形式，要求所有元素都满足比较条件后才结束等待。

## 数学语义


单个信号：

$$ \mathrm{signal} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

信号 tensor：

$$ \forall d_0, d_1, d_2, d_3, d_4:\ \mathrm{signal}_{d_0, d_1, d_2, d_3, d_4} \;\mathtt{cmp}\; \mathrm{cmpValue} $$

其中 `cmp ∈ {EQ, NE, GT, GE, LT, LE}`。

## 汇编语法


PTO-AS 形式：

```text
twait %signal, %cmp_value {cmp = #pto.cmp<EQ>} : (!pto.memref<i32>, i32)
twait %signal_matrix, %cmp_value {cmp = #pto.cmp<GE>} : (!pto.memref<i32, MxN>, i32)
```

## C++ 内建接口


声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
template <typename GlobalSignalData, typename... WaitEvents>
PTO_INST void TWAIT(GlobalSignalData &signalData, int32_t cmpValue, WaitCmp cmp, WaitEvents&... events);
```

## 约束


!!! warning "约束"
    - `GlobalSignalData::DType` 必须为 `int32_t`
    - `signalData` 必须指向本地地址（当前 NPU）
    - 单个信号的形状为 `<1,1,1,1,1>`
    - tensor 形式由其 shape 决定等待区域，并要求所有元素满足条件

    ### 比较运算符

    | 值 | 条件 |
    | --- | --- |
    | `EQ` | `signal == cmpValue` |
    | `NE` | `signal != cmpValue` |
    | `GT` | `signal > cmpValue` |
    | `GE` | `signal >= cmpValue` |
    | `LT` | `signal < cmpValue` |
    | `LE` | `signal <= cmpValue` |

## 示例


### 等待单个信号


```cpp
void wait_for_ready(__gm__ int32_t* local_signal) {
    comm::Signal sig(local_signal);
    comm::TWAIT(sig, 1, comm::WaitCmp::EQ);
}
```

### 等待计数器达到阈值


```cpp
void wait_for_count(__gm__ int32_t* local_counter, int expected_count) {
    comm::Signal counter(local_counter);
    comm::TWAIT(counter, expected_count, comm::WaitCmp::GE);
}
```

### 与 TNOTIFY 配合


```cpp
void producer(__gm__ int32_t* remote_flag) {
    comm::Signal flag(remote_flag);
    comm::TNOTIFY(flag, 1, comm::NotifyOp::Set);
}

void consumer(__gm__ int32_t* local_flag) {
    comm::Signal flag(local_flag);
    comm::TWAIT(flag, 1, comm::WaitCmp::EQ);
}
```

## Summary
本节给出该指令/主题的核心语义与使用定位，和英文章节保持一致。

## Mechanism
本节说明执行机制与关键语义规则，细节与边界条件以英文版为准。

## Assembly Syntax
本节列出语法形态（SSA / DPS / Assembly），用于与英文页逐项对照。

## C++ Intrinsic
本节给出 C++ 内建接口入口与参数语义说明。

## Inputs
本节定义输入操作数角色、数据来源与有效区域要求。

### Comparison Operators
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## Expected Outputs
本节定义输出结果及其在有效区域内的语义保证。

## Side Effects
本节说明除结果写回外是否存在额外可观察副作用。

## Constraints
本节列出类型、布局、shape、valid-region 与 profile 相关约束。

## Exceptions
本节描述非法输入、不支持组合与验证失败行为。

## Examples
本节提供 Auto/Manual 及 AS 形式示例，便于中英文对照复现。

### Wait for Single Signal
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Wait for Signal Matrix
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

### Producer-Consumer Pattern
本节与英文同名章节对齐，后续可继续补充更细粒度中文说明。

## See Also
本节给出上下游指令与相关章节链接。
