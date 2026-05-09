# Auto 与 Manual

PTO 同时支持 Auto 和 Manual 两种编程方式，因为它们解决的是不同问题。ISA manual 记录的是共享的架构契约；两种方式在作者与工具链之间分配职责的方式如下。

## 选择路径

```text
Compiler / toolchain developer
  -> Auto 是工具需要实现的契约
  -> Manual 是工具可能为用户生成的显式形式

Kernel author
  -> 需要精确流水线控制？是：Manual
  -> 否：Auto
```

## Auto 模式

在 Auto 模式下，编译器或运行时自动插入 `TASSIGN`、`TSYNC` 和必要的数据移动操作。源码主要描述计算有效载荷。

### 源码中写什么

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add(Tile<float, 16, 16>& c,
             const GlobalTensor<float>& ga,
             const GlobalTensor<float>& gb,
             const GlobalTensor<float>& gc) {
    Tile<float, 16, 16> a, b;
    TLOAD(a, ga);
    TLOAD(b, gb);
    TADD(c, a, b);
    TSTORE(gc, c);
}
```

### 工具链补什么

```text
TASSIGN(a, @tile(slot))
TSYNC()
TLOAD(a, ga)
TSYNC()
TASSIGN(b, @tile(slot))
TSYNC()
TLOAD(b, gb)
TSYNC()
TADD(c, a, b)
TSYNC()
TSTORE(gc, c)
```

Auto 模式不会改变 PTO ISA 语义。插入的仍然是标准 PTO 操作。

### 约束

- 工具链插入的操作必须满足与显式操作相同的合法性规则
- Auto 模式假设 tile shape 和 valid region 已能在编译时确定
- 自动插入同步只能覆盖默认数据依赖，不代表替代所有手工流水线结构

## Manual 模式

在 Manual 模式下，作者显式绑定 tile 资源并管理同步。这提供了 tile 放置、双缓冲和流水重叠的精确控制。

### 源码中写什么

```cpp
#include <pto/pto-inst.hpp>
using namespace pto;

void vec_add_manual(Tile<float, 16, 16>& c,
                    const GlobalTensor<float>& ga,
                    const GlobalTensor<float>& gb) {
    Tile<float, 16, 16> a, b;
    TASSIGN(a, 0x1000);
    TASSIGN(b, 0x2000);
    TASSIGN(c, 0x3000);
    TLOAD(a, ga);
    TLOAD(b, gb);
    TSYNC();
    TADD(c, a, b);
    TSYNC();
    TSTORE(gc, c);
}
```

### 双缓冲示例

```cpp
TASSIGN(tile[0], 0x1000);
TASSIGN(tile[1], 0x2000);

TLOAD(tile[1], gm_next);
set_flag(PIPE_MTE2, PIPE_V, ID0);
wait_flag(PIPE_MTE2, PIPE_V, ID0);
TADD(c, tile[0], src[0]);
TSTORE(gm_out, c);
TSYNC();
```

## 共享契约

两种模式共享相同的 ISA 契约：

| 方面 | Auto | Manual |
| --- | --- | --- |
| PTO ISA 语义 | 相同 | 相同 |
| valid-region 规则 | 相同 | 相同 |
| movement 语义 | 相同 | 相同 |
| 同步契约 | 工具链插入 | 作者显式控制 |
| 资源绑定 | 工具链插入 | 作者显式控制 |
| tile type / layout 限制 | 相同 | 相同 |
| target profile 限制 | 相同 | 相同 |

## 不允许的情形

!!! danger "不允许的情形"
    - 把 Auto 模式写成能让非法程序“自动合法化”
    - 把 Manual 细节误写成所有 PTO 程序的默认保证
    - 把 Auto 和 Manual 写成两套不同 ISA
    - 在需要精确流水控制的代码里默认依赖自动同步

## 相关页面

- [执行代理与目标 Profile](../machine-model/execution-agents_zh.md)
- [顺序与同步](../machine-model/ordering-and-synchronization_zh.md)
- [可移植性与目标 Profile](../reference/portability-and-target-profiles_zh.md)
- [GlobalTensor 与数据搬运](./globaltensor-and-data-movement_zh.md)
