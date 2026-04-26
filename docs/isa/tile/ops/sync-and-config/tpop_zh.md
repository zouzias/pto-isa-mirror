# pto.tpop

## 指令示意图

![TPOP tile operation](../../../../figures/isa/TPOP.svg)

`pto.tpop` 属于[同步与配置指令集](../../sync-and-config_zh.md)。

## 摘要

在生产者已经提供数据后，从 pipe 或 FIFO 的消费者端弹出一个 Tile。

## 机制

`pto.tpop` 在生产者侧发布数据后，从 pipe 或 FIFO 的消费者端读取 Tile 负载。它属于 Tile 的同步/配置外壳类指令，可见效果主要体现为状态交接与顺序约束，而不是算术负载变换。

除非另有说明，语义按目标 valid region 定义；与目标相关的行为标记为实现定义。

## 语法

文本拼写由 PTO ISA 语法与操作数页面定义。

示意形式：

```text
%dst = pto.tpop %pipe : !pto.pipe<...> -> !pto.tile<...>
```

### AS Level 1（SSA）

```text
%dst = pto.tpop %pipe : (!pto.pipe<...>) -> !pto.tile<...>
```

### AS Level 2（DPS）

```text
pto.tpop ins(%pipe : !pto.pipe<...>) outs(%dst : !pto.tile_buf<...>)
```

### IR Level 1（SSA）

```text
%dst = pto.tpop %pipe : (!pto.pipe<...>) -> !pto.tile<...>
```

### IR Level 2（DPS）

```text
pto.tpop ins(%pipe : !pto.pipe<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

- `pipe`：承载排队 Tile 负载的消费者端点。
- `dst`：接收弹出负载的目标 Tile。

## 期望输出

`dst` 接收与之配对的生产者侧操作发布的 Tile 负载。

## 副作用

该操作可能建立消费者侧顺序关系、推进 FIFO 状态，并释放或回收实现定义的队列资源。

## 约束

- Tile 类型、split 模式和 pipe 方向的精确合法性依赖具体 backend。
- 程序必须将 `pto.tpop` 与兼容的生产者侧操作（如 `pto.tpush`）配对使用。
- 队列空/就绪语义在文档声明的 PTO 可见契约之外均属实现定义。

## 异常

- 非法操作数组合、不支持的数据类型、非法 layout 组合或目标画像不支持的模式，会由验证器或具体 backend 指令集拒绝。
- 即使某个 backend 当前接受超出文档合法域的写法，程序也不得依赖该行为。

## 目标画像限制

- `pto.tpop` 在 CPU 仿真与受支持的 NPU backend 上保持 PTO 可见语义，但具体支持子集可能因画像而异。
- 可移植代码只能依赖所选目标画像明确保证的类型、layout、shape 和 mode 组合。

## 示例

具体示例可参见 `docs/isa/` 与 `docs/coding/tutorials/` 中的相关内容。

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
%dst = pto.tpop %pipe : (!pto.pipe<...>) -> !pto.tile<...>
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
%dst = pto.tpop %pipe : (!pto.pipe<...>) -> !pto.tile<...>
```

### PTO Assembly Form

```text
%dst = pto.tpop %pipe : (!pto.pipe<...>) -> !pto.tile<...>
# AS Level 2（DPS）
pto.tpop ins(%pipe : !pto.pipe<...>) outs(%dst : !pto.tile_buf<...>)
```

## 相关操作 / 指令集链接

- 指令集总览：[同步与配置指令集](../../sync-and-config_zh.md)
- 上一个相关操作：[pto.tpush](./tpush_zh.md)
