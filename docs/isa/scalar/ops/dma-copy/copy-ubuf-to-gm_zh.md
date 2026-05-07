# pto.copy_ubuf_to_gm

把数据从 Unified Buffer 回写到 Global Memory。

## 语法

```mlir
pto.copy_ubuf_to_gm %ub_src, %gm_dst,
    %sid, %n_burst, %len_burst, %reserved, %dst_stride, %src_stride
    : !pto.ptr<T, ub>, !pto.ptr<T, gm>, i64, i64, i64, i64, i64, i64
```

## 关键约束

- 当前 target profile 可能对该形式施加额外限制。


## 相关页面

- [DMA 拷贝](../../dma-copy_zh.md)
