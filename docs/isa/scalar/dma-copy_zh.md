# DMA 拷贝

这些 `pto.*` 形式配置并执行 GM↔UB 以及 UB 内部的标量侧 DMA 搬运。它们属于标量与控制指令，因为它们定义的是配置和搬运行为，而不是向量寄存器计算。

## 本指令集覆盖

- GM↔UB 传输的嵌套循环大小和 stride 配置
- GM → UB 拷贝
- UB → GM 拷贝
- UB → UB 拷贝

## per-op 页面

- `pto.set_loop_size_outtoub`
- `pto.set_loop2_stride_outtoub`
- `pto.set_loop1_stride_outtoub`
- `pto.set_loop_size_ubtoout`
- `pto.set_loop2_stride_ubtoout`
- `pto.set_loop1_stride_ubtoout`
- `pto.copy_gm_to_ubuf`
- `pto.copy_ubuf_to_gm`
- `pto.copy_ubuf_to_ubuf`

## 相关页面

- [控制与配置](./control-and-configuration_zh.md)
- [向量 DMA 路径](../vector/dma-copy_zh.md)
