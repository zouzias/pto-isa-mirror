# pto.set_flag

从源 pipeline 向目标 pipeline 设置信号事件。

## 语法

```mlir
pto.set_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
