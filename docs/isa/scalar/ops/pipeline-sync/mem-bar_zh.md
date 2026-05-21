# pto.mem_bar

建立内存可见性边界。

## 语法

```mlir
pto.mem_bar "BARRIER_TYPE"    // BARRIER_TYPE ∈ { "VV_ALL", "VST_VLD", "VLD_VST" }
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [流水线同步](../../pipeline-sync_zh.md)
