# pto.tpush

## 指令示意图

![TPUSH tile operation](../../../../figures/isa/TPUSH.svg)

`pto.tpush` 属于[同步与配置指令集](../../sync-and-config_zh.md)。

## 摘要

将 Tile 推入 pipe 或 FIFO 的生产者端，用于跨阶段或跨核传输。

## 机制

`pto.tpush` 将 Tile 负载发布到 pipe 或 FIFO 的生产者端，以便后续由 `pto.tpop` 等消费者侧操作读取。它属于 Tile 的同步/配置外壳类指令，可见效果主要体现为状态交接与顺序约束，而不是算术负载变换。

除非另有说明，语义按 valid region 定义；与目标相关的行为标记为实现定义。

## 语法

文本拼写由 PTO ISA 语法与操作数页面定义。

示意形式：

```text
pto.tpush %src, %pipe : !pto.tile<...>, !pto.pipe<...> -> ()
```

### AS Level 1（SSA）

```text
pto.tpush %src, %pipe : (!pto.tile<...>, !pto.pipe<...>) -> ()
```

### AS Level 2（DPS）

```text
pto.tpush ins(%src, %pipe : !pto.tile_buf<...>, !pto.pipe<...>) outs()
```

### IR Level 1（SSA）

```text
pto.tpush %src, %pipe : (!pto.tile<...>, !pto.pipe<...>) -> ()
```

### IR Level 2（DPS）

```text
pto.tpush ins(%src, %pipe : !pto.tile_buf<...>, !pto.pipe<...>) outs()
```

## C++ 内建接口

声明于 `include/pto/common/pto_instr.hpp`。

## 输入

- `src`：要发布到 pipe 或 FIFO 的源 Tile。
- `pipe`：承载排队 Tile 负载的生产者端点。

## 期望输出

该形式主要体现为顺序或队列发布效果，不会在 pipe 或 FIFO 状态之外再引入新的负载 Tile。

## 副作用

该操作可能建立生产者到消费者的顺序关系、推进 FIFO 状态，并更新实现定义的片上通信元数据。

## 约束

- Tile 类型、split 模式和 pipe 方向的精确合法性依赖具体 backend。
- 程序应将 `pto.tpush` 与兼容的消费者侧操作（如 `pto.tpop`）配对使用。
- 除文档声明的 PTO 可见契约外，跨核 FIFO 细节均属实现定义。

## 异常

- 非法操作数组合、不支持的数据类型、非法 layout 组合或目标画像不支持的模式，会由验证器或具体 backend 指令集拒绝。
- 即使某个 backend 当前接受超出文档合法域的写法，程序也不得依赖该行为。

## 目标画像限制

- `pto.tpush` 在 CPU 仿真与受支持的 NPU backend 上保持 PTO 可见语义，但具体支持子集可能因画像而异。
- 可移植代码只能依赖所选目标画像明确保证的类型、layout、shape 和 mode 组合。

## 示例

具体示例可参见 `docs/isa/` 与 `docs/coding/tutorials/` 中的相关内容。

### Auto Mode

```text
# Auto mode: compiler/runtime-managed placement and scheduling.
pto.tpush %src, %pipe : (!pto.tile<...>, !pto.pipe<...>) -> ()
```

### Manual Mode

```text
# Manual mode: bind resources explicitly before issuing the instruction.
# Optional for tile operands:
# pto.tassign %arg0, @tile(0x1000)
pto.tpush %src, %pipe : (!pto.tile<...>, !pto.pipe<...>) -> ()
```

### PTO Assembly Form

```text
pto.tpush %src, %pipe : (!pto.tile<...>, !pto.pipe<...>) -> ()
# AS Level 2（DPS）
pto.tpush ins(%src, %pipe : !pto.tile_buf<...>, !pto.pipe<...>) outs()
```

## 相关操作 / 指令集链接

- 指令集总览：[同步与配置指令集](../../sync-and-config_zh.md)
- 下一个相关操作：[pto.tpop](./tpop_zh.md)
