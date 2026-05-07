# pto.set_loop1_stride_ubtoout

配置 ub-to-out 方向 DMA 的第一层 stride。

## 语法

```mlir
pto.set_loop1_stride_ubtoout %src_stride, %dst_stride : i64, i64
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [DMA 拷贝](../../dma-copy_zh.md)
