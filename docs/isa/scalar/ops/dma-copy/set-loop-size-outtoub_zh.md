# pto.set_loop_size_outtoub

配置 out-to-ub 方向 DMA 的循环大小。

## 语法

```mlir
pto.set_loop_size_outtoub %loop1_count, %loop2_count : i64, i64
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [DMA 拷贝](../../dma-copy_zh.md)
