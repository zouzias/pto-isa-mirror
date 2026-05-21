# PTO V3 vs MC2 v2 性能分析

## 背景

本文用于解释 `dispatch_ffn_combine_v3` 在 large case 下相比 MC2 `dispatch_ffn_combine_v2` 更快的原因。重点不是做逐行代码 diff，而是讲清楚 PTO 编程模型相对传统 AscendC + CATLASS 编程方式，在 fused MoE 场景里的表达优势和性能收益来源。

`dispatch_ffn_combine` 不是单纯 GMM，也不是单纯通信算子，而是一条融合链路：

```text
token routing
  -> dispatch / repack
  -> cross-rank remote data exchange
  -> GMM1
  -> SwiGLU + quant
  -> GMM2
  -> combine / reduce
  -> restore / unpermute
```

这类 workload 的性能瓶颈通常不只在矩阵乘本身，还包括 token 重排、跨 rank 通信、同步等待、GMM 前后 layout 转换、量化和 restore 等非 GMM 开销。

## 实测数据

测试 case：

```text
m=4097 k=128 n=128 topk=2 expert_per_rank=2 world_size=2
```

两边 logical work 相同，且结果均 PASS：

```text
input_tokens       = 8194
routed_tokens      = 16387
remote_routed_tokens = 8192
compute_flops      = 805453824
comm_bytes         = 3178496
```

性能对比：

| 指标 | PTO V3 | MC2 v2 | 变化 |
| --- | ---: | ---: | ---: |
| kernel avg | 30.39 us | 53.75 us | 约降低 43% |
| e2e avg | 99.23 us | 150.74 us | 约降低 34% |
| kernel std | 8.38 us | 26.43 us | 波动明显降低 |
| e2e std | 31.40 us | 46.38 us | 波动降低 |
| eq_compute | 26.51 TFLOPS | 14.98 TFLOPS | 约 1.77x |
| eq_comm | 97.41 GB/s | 55.07 GB/s | 约 1.77x |

因此，这个 case 的结论不是“V3 少算了”，而是：在相同 logical workload 和相同正确性边界下，PTO V3 把通信、重排、计算和同步组织得更直接，减少了中间开销和阶段等待。

## 核心结论

PTO V3 快，不是因为“把 API 名字换成 PTO”，而是因为 PTO 把 fused MoE 的通信、重排、计算、同步从传统 AscendC + CATLASS 的隐式工程拼装，变成了显式、可编排、可融合、可验证的数据流表达。

一句话概括：

```text
PTO V3 的收益来自编程模型升级：让 MoE 的真实数据流直接围绕 expert-major layout、remote window、AIC/AIV stage 和 PTO primitive 编排。
```

## 1. 传统 AscendC + CATLASS 写法的问题：算法融合，但工程表达割裂

传统 AscendC + CATLASS 写法很强，但在 fused MoE 这类复杂链路里，代码表达容易被拆成多个子系统：

- GM/UB/L1/L0 生命周期由 `GlobalTensor`、`LocalTensor`、`TQue`、`TBuf`、`DataCopy`、`LoadData`、`Fixpipe` 等组合表达。
- GMM 主体由 CATLASS policy/helper 表达。
- HCCL window / shmem 通信又是一套封装。
- routing、sort、gather 是另一组 AscendC 逻辑。
- combine、restore、unpermute 又有自己的 tensor/copy/scalar seam。

结果是：算法上是一个 fused operator，但工程上像几个子系统串接起来。

这会带来几个代价：

1. 数据什么时候 materialize 不够显式。
2. 通信和重排的边界容易变厚。
3. GMM 前后的 layout 转换成本容易藏在 glue code 中。
4. 同步点容易保守。
5. 性能优化要跨多个抽象层反推真实硬件动作。

对于 MoE large case，GMM 本身很快后，这些“非 GMM 的胶水成本”会变得非常明显。

## 2. PTO V3 的优势：把 MoE 数据流写成统一 primitive 流

PTO V3 的关键变化，是把核心动作统一收到一套 PTO primitive vocabulary 里：

```text
TLOAD / TSTORE
TCVT / TEXPANDS / TADD / TMUL / TDIV / TABS / TEXP / TROWMAX
TSORT32 / TMRGSORT / TGATHER
TGET / TPUT
TNOTIFY / TWAIT
TMATMUL / TSTORE_FP
```

这不是简单替换 API，而是把 MoE 链路改成更直接的数据流表达：

```text
routing 产生目标布局
  -> PTO sort/gather 明确重排
  -> TGET / TPUT 围绕 remote window 搬运数据
  -> PTO vector primitive 做 quant / swiglu / restore
  -> PTO matmul policy 承接主计算
  -> TNOTIFY / TWAIT 做跨 rank ready / wait
```

传统写法更像：

```text
构造 tensor
  -> DataCopy
  -> 调 CATLASS
  -> 处理 shmem
  -> 再 copy / restore
```

PTO 写法更像：

```text
从哪里取
  -> 搬到哪里
  -> 按什么 layout 算
  -> 什么时候通知对端
  -> 什么时候消费远端数据
```

这就是 PTO 的第一层收益：让 fused MoE 的真实数据流显性化。

## 3. 通信不再只是独立搬运，而是 layout 编排的一部分

MegaMoE 类优化的核心不是“通信 API 更快”这么简单，而是减少通信前后的中间重排和 materialization。

传统路径容易形成：

```text
local routing
  -> local repack
  -> communication
  -> receive buffer
  -> post-communication repack
  -> GMM input
```

PTO V3 更容易表达成：

```text
local routing / count
  -> remote window layout 按目标消费方式规划
  -> TGET / TPUT 面向 GMM / Combine 所需布局搬运
  -> GMM / Combine 直接消费更少中间态
```

也就是说，通信不是孤立阶段，而是服务于 expert-major layout 的地址编排。

这可以解释 `eq_comm` 的提升：

```text
PTO V3 eq_comm = 97.41 GB/s
MC2 v2 eq_comm = 55.07 GB/s
```

这里的 `eq_comm` 是基于 logical workload 推导的等效指标，不是硬件计数器。它说明单位 logical communication work 在 PTO V3 中被组织得更有效，通信等待、重排、同步、buffer 周转等附加成本更小。

## 4. AIC/AIV stage 边界更清楚，减少保守同步和阶段空洞

PTO V3 把主流程拆成了明确 stage：

```text
AIV:
  routing
  dispatch_gather
  swiglu
  combine
  restore

AIC:
  GMM1
  interlock
  GMM2
```

这不是单纯为了代码可读性，而是让调度关系更清楚。

MoE 链路最怕被写成保守串行：

```text
AIV 做完整个 routing
  -> AIC 才开始 GMM1
  -> AIC 做完整个 GMM1
  -> AIV 才开始 swiglu
  -> ...
```

当 stage 边界和同步关系不清楚时，工程实现容易加入过多同步点，造成 AIC/AIV 空转。

PTO V3 通过 stage facade 和 primitive-level communication/sync，把关键依赖显式化：

```text
routing/count ready
  -> dispatch_gather 推进
  -> GMM1 消费 ready 的 expert block
  -> swiglu/quant 接 GMM1 output
  -> GMM2 接 quant output
  -> combine/restore 接 GMM2 output
```

因此，AIC/AIV 的 interlock 更容易收敛到必要同步，而不是工程上保守同步。

这也能解释 kernel std 的下降：

```text
PTO V3 kernel std = 8.38 us
MC2 v2 kernel std = 26.43 us
```

std 明显降低，说明调度抖动、等待、长尾 rank 或阶段间空转更少。

## 5. PTO helper 把散落的低级动作收口成统一优化面

PTO V3 中的 `pto_vector_ops.hpp`、`pto_global_view.hpp` 等 helper，把分散在业务逻辑里的低级动作收口：

```text
DataCopy / DataCopyPad
Cast
Duplicate
Add / Mul / Div / Exp / Abs / ReduceMax
atomic add
GM view construction
tail copy
aligned copy
```

对应到统一的 PTO helper：

```text
PtoLoadVector
PtoStoreVector
PtoStoreAtomicAddVector
PtoCastVector
PtoFillVector
PtoAddVector
PtoMulVector
PtoDivVector
PtoAbsVector
PtoExpVector
PtoReduceMaxVector
```

这个改造的重点不是“多封装一层”，而是让业务代码从 AscendC API 机械细节里解耦出来，直接表达数据意图。

传统写法中，一个 routing/quant/restore 小逻辑可能混着：

```text
地址计算
tensor 构造
copy 参数
tail 处理
cast 参数
repeat 参数
同步
业务公式
```

PTO V3 后，业务代码更接近：

```text
load token slice
cast to fp32
abs
reduce max
scale
div
cast int8
store
```

这样有两个直接收益：

1. 对齐、tail、repeat、tile shape 等细节集中处理，减少局部写错或局部保守。
2. 后续优化一个 helper，就能影响多个 stage，而不是在业务代码里到处找 DataCopy/Cast/Reduce seam。

这使优化面从“散点手写 API 调参”变成“统一 primitive helper 的系统性优化”。

## 6. PTO sort/gather 更贴近 MoE routing 本质

MoE routing 的核心是：

```text
token -> expert
token topk 展开
按 expert / rank 重排
生成 GMM 可消费的连续 expert block
```

传统写法里，routing 容易变成复杂 glue code：

```text
排序
索引
计数
prefix sum
gather
copy
tail
atomic count
```

PTO V3 引入 PTO sort/gather 后，routing 更接近算法本身：

```text
TSORT32
TMRGSORT
TGATHER
```

这对 large case 很重要。`m=4097 topk=2` 时 routed token 数接近 `16387`，routing/repack 不再是小开销。当 GMM 足够快后，routing、gather、combine、restore 这些非矩阵乘部分会成为显性瓶颈。

因此，V3 large case 快不能只归因于 GMM，而是：

```text
GMM 主体没有拖后腿
+
routing / gather / remote / combine 周边开销被 PTO 化后变薄
```

这也解释了为什么 kernel avg 能从 `53.75 us` 降到 `30.39 us`：收益来自整条链路多个中间成本一起下降，而不是某一个 API 的局部微优化。

## 7. PTO 不是消灭 substrate，而是把边界显式化

需要准确说明的是：PTO V3 不是“纯 PTO kernel”。当前实现仍保留必要 substrate：

```text
kernel ABI
TPipe / TQue / TBuf 生命周期
LocalTensor / GlobalTensor substrate
LoadData / Fixpipe 的底层硬件语义
CrossCore flag
cache coherence
ACL / HCCL / MPI host bootstrap
```

PTO V3 的优势在于：它知道哪些地方必须保留 substrate，哪些地方应该上升到 PTO primitive。

所以不应该讲成：

```text
PTO 消灭了 AscendC / CATLASS。
```

更准确的表述是：

```text
PTO 把传统 AscendC / CATLASS 中最容易分散、最难审计、最影响融合调度的部分抽成统一 primitive；底层必要 substrate 仍保留，但不再主导业务表达。
```

## 对外叙事口径

这次 large case 中，PTO V3 相比 MC2 v2 的优势不是来自减少计算量；两边 logical work 一样，结果都 PASS。真正差异在于编程模型：传统 AscendC + CATLASS 写法更像把 routing、通信、GMM、combine 几个子系统用 tensor/copy/helper 拼起来，数据流边界比较厚；PTO V3 则把 load/store、sort/gather、remote get/put、notify/wait、vector math、matmul 都纳入同一套 primitive 表达，使 MoE 的真实数据流可以直接围绕 expert-major layout 和 remote window 编排。

这样通信不再只是独立搬运阶段，而是参与布局重排；AIC/AIV 阶段依赖更清晰，保守同步更少；routing/quant/restore 这些非 GMM 开销也被统一 helper 收口。最终表现就是同样 workload 下 kernel 平均时延从 `53.75 us` 降到 `30.39 us`，e2e 从 `150.74 us` 降到 `99.23 us`，同时波动明显降低。

一句话版本：

```text
PTO V3 快，不是因为“换了 API”，而是因为 PTO 让 fused MoE 的通信、重排、计算、同步从隐式拼装变成显式数据流编排；这使 large case 里的中间搬运、重排、等待和同步成本一起下降。
```
