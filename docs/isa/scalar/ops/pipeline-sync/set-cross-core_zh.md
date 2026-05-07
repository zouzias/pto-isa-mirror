# pto.set_cross_core

建立跨 core 的同步设置。

## 语法

### PTO 汇编形式

```text
set_cross_core %core_id, %event_id : i64, i64
```

### AS Level 1（SSA）

```mlir
pto.set_cross_core %core_id, %event_id : i64, i64
```

## 关键约束

- pipe、event 和 buffer 标识必须被当前 target profile 支持。
- 等待一个未建立事件属于非法程序。

## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
