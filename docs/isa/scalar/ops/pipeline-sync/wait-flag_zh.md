# pto.wait_flag

等待匹配事件被生产者设置。

## 语法

```mlir
pto.wait_flag["SRC_PIPE", "DST_PIPE", "EVENT_ID"]
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
