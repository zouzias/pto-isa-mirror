# pto.wait_flag_dev

等待设备级事件条件成立。

## 语法

### PTO 汇编形式

```text
wait_flag_dev %event_id : i64
```

### AS Level 1（SSA）

```mlir
pto.wait_flag_dev %event_id : i64
```

## 关键约束

- pipe、event 和 buffer 标识必须被当前 target profile 支持。
- 等待一个未建立事件属于非法程序。

## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
