## q1： 通信重排的理解
“通信后重排”不是指“每个专家内部还要按语义重新排序 token”。更准确地说，它是在 接收端把 AlltoAllV 的 source-rank-major 布局转换成 expert-major 布局。

举个 EP=2、每卡 2 个专家：

- rank0 有专家 A/B
- rank1 有专家 C/D
- rank0 本地发送前可能已经排成：`A0, B0, C0, D0`
- rank1 本地发送前可能已经排成：`A1, B1, C1, D1`

rank1 接收属于 C/D 的 token 后，普通 AlltoAllV 的 recv buffer 很可能是按来源 rank 拼起来：

来自 rank0: C0, D0

来自 rank1: C1, D1

recv buffer: C0, D0, C1, D1

但 GMM 需要的是每个 local expert 的 token 连续：

expert-major: C0, C1, D0, D1

所以这里的“通信后重排”更准确是：

> 对每个目标 NPU，把来自不同 source NPU 的 token，从“按 source rank 分段”重排成“按 local expert 分段”；每个 expert 内部通常再按 source rank、再按源 rank 内 token 顺序拼接。

不是为了保持序列语义，而是为了让 GMM 的输入矩阵按 expert 连续，能用 group GEMM。

关于 DP/EP：MoE 这里主要是 EP 通信，不要求不同 NPU 的输入 token 序列之间有语义关系。不同 NPU 上的 token 往往来自不同请求或不同 batch shard，彼此独立。即使同一请求的不同 token 分布在不同 rank，FFN/MoE 这一层也是逐 token 行独立计算，真正需要序列关系的是 attention，不是 MoE FFN。

所以：

- NPU 之间 token 序列通常无语义依赖。
- NPU 内 token 顺序有 bookkeeping 意义，用于最终 unpermute/combine 回原位置。
- GMM 不关心 token 原始顺序，只要求同一 expert 的 token 连续。
- Combine/unpermute 才关心原始 token index 和 topK 权重，所以要保存 `expandedRowIdx` 这类映射。

文章后面说融合方案要避免这次“通信后重排”，本质就是通过 token count 矩阵和 cumsum 直接算出远端读/写地址，让通信落到 expert-major 的最终位置。代码里对应 `tokenPerExpert`、`cumsumMM` 和 `CopyGMToGMPerToken` 这套逻辑。


## Q2:--算法做的改进就是通信前不做重排了？

算法改进不是简单地“通信前不做重排”，而是：

> 保留通信前的本地 routing/展开/量化，但重构通信地址计算，让通信直接落到 GMM 需要的密排位置，从而省掉通信后的本地重排，并让通信和 GMM 能流水并行。

文章里其实在对比三种东西：

1. 原始 `AlltoAllV` 通信前重排 → AlltoAllV → 通信后重排 → GMM
    
2. `Dispatch&Combine` 把通信前重排和发送融合： token 从 GM 到 AIV UB 后，根据 expert 直接量化并发送到目标卡。 但因为目标卡不知道各来源 rank 的 token 数，只能先写到稀疏区域：
    
    [rank0->expertC 预留 M 行][rank0->expertD 预留 M 行]
    
    [rank1->expertC 预留 M 行][rank1->expertD 预留 M 行]
    
    然后本地再密排成：
    
    [expertC 实际 tokens][expertD 实际 tokens]
    
3. MegaMoE 融合方案 它不是取消通信前所有重排，而是把 `MoeInitRoutingQuant` 放在通信前，先得到本 rank 的按 expert 展开结果和 token count。 然后通过一次 token count 同步/AllGather 得到全局矩阵：
    
    tokenPerExpert[dst_rank][src_rank][local_expert]
    
    再做 cumsum，提前知道每个 source rank 的 token 应该写到目标 expert 的哪一段。 于是通信时可以直接写/读到最终密排位置：
    
    [expertC_from_rank0][expertC_from_rank1]
    
    [expertD_from_rank0][expertD_from_rank1]
    
    这样就不用“通信后再重排”。
    

所以更准确的说法是：

Dispatch&Combine 是“边 routing 边发送，但接收端先落稀疏，再密排”。MegaMoE 是“先做本地 routing/quant/count，再根据全局 count 让通信直接生成密排 GMM 输入”。

因此通信前仍有处理，只是它的目标不是最终全局密排，而是为后续通信提供：

- 本地按 expert 的 token 数据
- `expandedRowIdx`
- per-token scale
- `localTokenPerExpert`

代码里对应：

- 通信前 routing/quant：`moe_init_routing_quant_v2`
- 全局 token count：`CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`
- cumsum 地址表：`GetCumsumForMMAIV`
- 通信直接密排：`CopyGMToGMPerToken`

一句话：改进点不是“不重排”，而是“把通信后的稀疏到密排重排，变成通信地址计算的一部分”。

## Q3：routing 这里的routing指的是啥？
这里的 `routing` 指的是 MoE token 路由整理过程，不是网络路由，也不是 gating 本身。

在 MoE 里，前面 gating/topk 已经算好了：

expertIdx[token, k] = 这个 token 第 k 个要去的专家

probs[token, k] = 这个专家的权重

`dispatch_ffn_combine` 里的 routing 是拿这些结果做数据重排：

1. 看每个 token 要发给哪些 expert。
2. 把一个 token 按 `topK` 复制/展开成多份。
3. 按 expert id 把这些 token 归类排列。
4. 记录展开后的 token 对应原始 token 位置，也就是 `expandedRowIdx`。
5. 统计每个 expert 收到多少 token，也就是 `localTokenPerExpert`。
6. W8A8 路径里顺便做动态量化，生成 per-token scale。

所以 `routing` 更准确可以理解为：

> 根据 gating 输出的 expert id，把 token 从原始 `[M, K]` 顺序整理成按 expert 分组的 dispatch 输入。

代码里对应的是：

moe_init_routing_quant_v2

输入大概是：

x: [M, K]

expertIdx: [M, topK]

输出大概是：

按 expert 展开的 token 数据

expandedRowIdx

localTokenPerExpert

per-token scale

一句话：gating 决定“去哪个专家”，routing 负责“把 token 搬到这些专家对应的位置”。


## Q4：DataAsFlag技术
这里的“通信”不是指 AllGather 获取全局 token 统计信息。

这段讲的是 原有 Dispatch&Combine 算子的 dispatch 通信算法，也就是把 token 数据从源 NPU 发送到目标 expert 所在 NPU 的那次跨卡通信。

### Data / Count / Flag 分别是什么

在 Dispatch 阶段，源 NPU 要告诉目标 NPU 三类东西：

`Data`：真正的 token 数据  
比如 rank0 有一些 token 要发给 rank1 的专家 C/D，这些 token 的 hidden 向量就是 Data。

`Count`：本次发了多少 token  
目标 NPU 需要知道每个来源 rank、每个目标 expert 收到了多少 token，才能后续重排/密排。

`Flag`：同步信号  
表示“我这批数据已经写完了，你可以读/处理了”。

传统做法可能是：

1. 发 Data

2. 发 Count

3. 发 Flag 或等同步

4. 接收端看到 Flag 后再处理 Data/Count

而 DataAsFlag 是把它们合起来：

一个报文里同时携带 Data + Count + Flag

接收端只要 check 到 Flag，就说明：

Data 已经到了

Count 也已经到了

### “半次同步”是什么意思

普通跨卡同步经常是一个 round trip：

发送方通知接收方

接收方确认/返回

这可以理解为 1 RTT。

DataAsFlag 的语义更像单向投递：

发送方把 Data/Count/Flag 一起写到接收端

接收端本地轮询 Flag

接收端看到 Flag 时，数据已经在本地可见，不需要再发一个确认回来。所以从链路时延角度只算 0.5 RTT：只有发送方向接收方这一趟，没有接收方回发送方这一趟。

### 和 AllGather token count 的区别

文章后面 MegaMoE 融合方案里提到的全局 token count 矩阵，是另一套逻辑：

tokenPerExpert[dst_rank][src_rank][expert]

它是为了提前知道每个 expert 的最终密排地址，让通信可以直接落到 GMM 输入位置。

而你引用这段说的 Dispatch&Combine，是：

源 rank 发送 token 到目标 rank 时，把 token 数据、token 数、完成标志合在一起发

所以这里的“通信”是 token 数据通信本身，不是单独的 count AllGather。

### 为什么收到 Count 后还要重排

因为目标端收到的数据通常是按“来源 rank / 稀疏预留区域”摆放的，不一定已经是 GMM 要的 expert-contiguous 布局。

收到 Count 后，目标 NPU 才知道每段实际有多少有效 token，然后可以把：

rank0->expertC 预留区

rank1->expertC 预留区

rank0->expertD 预留区

rank1->expertD 预留区

压紧成：

expertC 的连续 token

expertD 的连续 token

这就是“根据 Count 在接收端重排”。

## Q5:理解MegaMoe重排阶段的优化
理想上，只要某个专家收到一块连续 token，就可以开始这个专家的 GMM。MegaMoE 方案就是朝这个方向做的。

但文章这段说的是 Dispatch&Combine 原通信模式下做不到，原因在于它收到的数据布局不是“某个专家的一段连续 GMM 子矩阵”。

在 Dispatch&Combine 里，接收端通常先得到的是稀疏、乱序、按发送事件/来源 rank 分散的区域，例如：

recv/shmem:

rank0 -> expertC 的预留区

rank0 -> expertD 的预留区

rank1 -> expertC 的预留区

rank1 -> expertD 的预留区

...

每段预留区大小通常按最大 token 数留，实际有效 token 数要等 `Count/Flag` 到了才知道。即使某些 token 已经到了，也可能是：

expertC 的一部分在这里

expertC 的另一部分在很远的位置

中间夹着 expertD 或空洞

而 GMM 想要的是：

expertC: [C_from_rank0][C_from_rank1][C_from_rank2] 连续矩阵

expertD: [D_from_rank0][D_from_rank1][D_from_rank2] 连续矩阵

所以在原 Dispatch&Combine 模式下，不是所有专家数学上必须等全局收完才能算，而是因为数据布局和 token count 信息不满足 GMM 输入要求，接收端通常要等最后同步完成、拿到 count，再做一次本地密排，才能形成可计算矩阵。

文章“完成通信最后的同步之前，根本无法获得一个可供 matmul 计算的子矩阵”的意思是：

> 在 Dispatch&Combine 的接收布局里，通信中途没有稳定、连续、边界明确的 expert 子矩阵给 GMM 用。

MegaMoE 的改进就是让通信按 expert 粒度形成可计算子矩阵：

1. 通信前先做本地 routing/quant/count。
2. 同步 token count，算出全局 `cumsum`。
3. 通信时直接把数据搬到目标 expert 的连续位置。
4. 这样第 `i` 个 expert 的数据到齐后，`GMM_i` 就能开始。
5. 同时第 `i+1` 个 expert 的通信还能继续进行。

所以在 MegaMoE 里可以做到类似：

通信 expert0 -> GMM expert0

通信 expert1 -> GMM expert1

通信 expert2 -> GMM expert2

形成流水：

Comm expert1 与 GMM expert0 重叠

Comm expert2 与 GMM expert1 重叠

回答你的问题：

- 理论上：一个专家的数据收齐后就可以计算这个专家。
- Dispatch&Combine 原模式下：由于接收布局稀疏/乱序，通常要等最终同步 + 后重排后，才能拿到可给 GMM 的连续专家矩阵。
- MegaMoE 融合方案下：目标就是让每个专家或每个 expert block 数据一到位就能启动对应 GMM，不必等所有专家全部通信完成。

## Q: 4中通信算法
同样是“通信 + Matmul”融合，通信方向不同、谁先产生数据不同，会决定同步开销完全不一样。

先拆几个概念。

`AIC`：Cube/矩阵计算核，主要跑 Matmul/GMM。  
`AIV`：Vector/搬运/通信核，主要做 routing、copy、quant/dequant、combine。  
`GM`：本卡显存。  
`Shm`：本卡 GM 中一块可被其他 NPU 访问的共享显存窗口，可以理解成跨卡可见的 buffer。

### 远端读 vs 远端写

跨卡 p2p 通信可以有两种基本模式：

`远端读`：我主动去别的卡读数据。

rank0 AIV 从 rank1 Shm 读数据 -> 写到 rank0 GM

`远端写`：我主动把数据写到别的卡。

rank0 AIV 从 rank0 GM 读数据 -> 写到 rank1 Shm

区别是：谁发起通信。

### 先通信后计算：Dispatch-GMM

Dispatch 阶段是 token 先跨卡到专家所在卡，然后专家所在卡做 GMM。

通信生产数据 -> GMM 消费数据

也就是“先通信后计算”。

这里有两种模式：

1. Dispatch 用远端读  
    目标卡自己去读别的卡的数据。
    
    rank1 AIV 读 rank0 Shm 的 token
    
    rank1 AIC 计算 rank1 本地专家 GMM
    
    通信发起者和计算消费者都在 rank1。
    
2. Dispatch 用远端写  
    源卡把 token 写到目标卡。
    
    rank0 AIV 写 token 到 rank1 Shm
    
    rank1 AIC 计算 rank1 本地专家 GMM
    
    通信发起者在 rank0，计算消费者在 rank1。
    

### 先计算后通信：GMM-Combine

Combine 阶段是专家所在卡先算完 GMM2，然后把结果发回原 token 所在卡。

GMM 生产数据 -> 通信消费数据

也就是“先计算后通信”。

也有两种模式：

3. Combine 用远端读  
    原 token 所在卡去读专家卡的结果。
    
    rank1 AIC 算完 expert 输出，放 rank1 Shm
    
    rank0 AIV 读 rank1 Shm 回本地
    
    计算生产者在 rank1，通信发起者/消费者在 rank0。
    
4. Combine 用远端写  
    专家卡算完后，自己把结果写回原 token 所在卡。
    
    rank1 AIC 算完 expert 输出
    
    rank1 AIV 写结果到 rank0 Shm
    
    计算生产者和通信发起者都在 rank1。
    

### 为什么会有 2×2=4 种

两个维度：

数据依赖方向:

- 通信 -> 计算 Dispatch-GMM

- 计算 -> 通信 GMM-Combine

通信发起方式:

- 远端读

- 远端写

组合起来就是 4 种。

### 文章真正想表达什么

关键不是列举 4 种，而是判断哪种同步少。

如果通信发起者和计算方在同一张卡，同步可以变成卡内 AIV/AIC 同步：

同卡 AIV <-> AIC

这比较快。

如果通信发起者和计算方在不同卡，每个小块都可能需要跨卡同步：

rank0 通知 rank1

rank1 再确认数据可用

这在 decode 小 token 场景特别贵。

所以文章后面选的是：

- Dispatch-GMM 用远端读：目标专家卡自己读数据，然后本卡 AIC 计算。通信和计算都由目标卡协调。
- GMM-Combine 用远端写：专家卡算完后，自己写回原 token 卡。计算和通信都由专家卡协调。

这样在两次全局同步之间，大部分细粒度流水只需要卡内 CV 同步，不需要每个 tile 都跨卡同步。

一句话：这段是在说明为什么 MegaMoE 不随便选 p2p 读/写，而是根据“谁生产、谁消费”选择让通信发起方和计算方尽量在同一张卡，从而减少跨卡同步。

## Q: MoeInitRoutingQuant 跟重排的关系
`MoeInitRoutingQuant` 阶段只完成本卡的 routing/量化/本卡 token count，不会自己完成“全局 count/cumsum 同步到所有卡”。

更准确流程是：

1. `MoeInitRoutingQuant` 本卡完成：
    
    x_local -> 按 expert 展开/重排/量化
    
    localTokenPerExpert
    
    expandedRowIdx
    
    per-token scale
    
    写到本卡 shmem/workspace。
    
2. token count 同步 / AllGather 各卡交换自己的 `localTokenPerExpert`，形成：
    
    tokenPerExpert[dst_rank][src_rank][local_expert]
    
3. cumsum 每张目标卡根据 `tokenPerExpert` 算前缀和：
    
    cumsumMM
    
    这样就知道：
    
    rank0 给 expert C 的 token 放 gmA 哪一段
    
    rank1 给 expert C 的 token 放 gmA 哪一段
    
    ...
    
4. Dispatch 远端读 目标卡根据 `tokenPerExpert/cumsumMM`：
    
    从各 source rank 的 shmem 拉 token
    
    直接写入本卡 gmA 的 expert-contiguous 位置
    
5. GMM 某个 expert 的数据段到位后，AIC 用：
    
    gmA[expert] + w1[expert]
    
    开始 `GMM1`。
    

所以你的总结可以改成：

> `MoeInitRoutingQuant` 先产生本卡已重排/量化的 token 和本卡 token count；随后算子通过跨卡同步得到全局 token count/cumsum；目标卡在 dispatch 阶段根据这些信息直接远端读取对应 token 到 GMM 输入布局，再做 GMM。


## Q：第i个专家的GMM可以与第i+1专家的AlltoallV并行 ? 只能两两相邻？
文章里的 `第 i 个专家 GMM` 和 `第 i+1 个专家 AlltoAllV` 是在描述稳态流水关系，不是说硬件上只能相邻两个 expert 同时跑。

更准确理解是：

AIV 通信按 expert 分组推进：

comm expert0 -> comm expert1 -> comm expert2 -> ...

AIC GMM 也按 expert 分组消费：

gmm expert0 -> gmm expert1 -> gmm expert2 -> ...

当 `expert0` 的通信数据已经准备好，AIC 可以开始 `GMM expert0`；与此同时 AIV 不需要停下来，可以继续搬 `expert1` 的数据。于是文章写成：

GMM(i) 与 AlltoAllV(i+1) 并行

这是流水线里最典型的一拍错位。

代码里确实有一个按 local expert 顺序推进的结构：

- AIV 在 `DispatchAndCombine()` 里按 `groupIdx = 0 .. expertPerRank-1` 做 dispatch 搬运。
- 每完成一组，会通过 `CrossCoreSetFlag` 通知 AIC。
- AIC 在 `GMM1()` 里也按 `groupIdx` 消费，并 `CrossCoreWaitFlag` 等对应 expert 的数据 ready。

所以“专家顺序”基本是按 `groupIdx` 固定推进的，因为：

- 权重 `w1/w2` 是按 local expert index 组织的。
- `cumsumMM` 也是按 expert index 计算每个 expert 的 token 段。
- GMM 输入 `gmA` 是按 expert 连续排布的。

但这不等于只能同时存在两个任务。实际运行时可能是：

AIC 正在算 expert0

AIV 已经搬完 expert1 的一部分/全部

AIV 继续搬 expert2

取决于通信和 GMM 谁更慢、每个 expert token 数多少、同步 flag 怎么推进。

Combine 阶段反过来：

GMM2 expert i 产出结果

AIV combine 通信 expert i-1 或更早的结果

意思是 combine 通信消费的是前面已经算完的结果，所以它可以滞后一拍或多拍，与后续 GMM2 重叠。

一句话：代码中 expert 的逻辑处理顺序是按 `groupIdx` 固定的，但 `i` 和 `i+1` 只是流水错位的典型描述，不是限制只能相邻两个 expert 并行；真正依赖是“某个 expert 的数据 ready 后才能算该 expert，某个 expert 的 GMM2 ready 后才能 combine 该 expert”。

## Q：计分板
“计分板是一种硬件机制”说的是概念来源，但这篇 MegaMoE 里的实现更像是借鉴计分板思想的软件/GM 状态表方案，不是直接用了某个硬件 scoreboard 指令。

从文章后面的描述看，它确实还是轮询：

AIC 写 Status

AIV Ctrl 轮询所有 AIC Status

AIV Ctrl 计算 MinStatus

worker AIV 轮询 MinStatus

这不是硬件自动调度，而是软件实现的依赖进度管理。

区别在于它比朴素软同步少轮询很多。

### 朴素软同步

每个 worker AIV 都轮询所有 AIC：

AIV0 轮询 AIC0..AIC23

AIV1 轮询 AIC0..AIC23

AIV2 轮询 AIC0..AIC23

...

复杂度大概是：

step × AIC数 × AIV数

### 计分板式软同步

让一个 AIV Ctrl 负责轮询所有 AIC，算出当前所有 AIC 最慢进度：

min_status = min(status[AIC0], status[AIC1], ..., status[AIC23])

然后 worker AIV 只轮询这个 `min_status`：

AIV0 轮询 min_status

AIV1 轮询 min_status

AIV2 轮询 min_status

...

复杂度变成：

step × (AIC数 + AIV数)

所以它还是软件轮询，只是把“每个消费者轮询所有生产者”优化成“一个控制者汇总生产者进度，消费者只看汇总进度”。

### 为什么叫 scoreboarding

因为它维护了一个“谁完成到哪一步”的进度表：

status[AIC_i] = task_id 已完成到几

min_status = 所有生产者的安全消费边界

这和经典硬件 scoreboard 的思想类似：跟踪生产者/消费者依赖，决定哪些任务可以继续。但在这里是代码手写状态区和轮询，不是硬件自动解决。

所以更准确的说法是：

> 文章中的计分板是借鉴硬件 scoreboarding 思想的软件同步方案，通过一个控制 AIV 汇总 AIC 进度，减少 worker AIV 的轮询访存量。它不是完全硬件机制。


## Q: Dispatch-GMM 为什么多对多
通信先产生 GMM 输入，GMM 才能算。问题在于 通信按 token/来源 rank/目标 expert 搬，计算按 Matmul tile 算，二者切分维度不一致。

### Dispatch-GMM 为什么多对多

GMM1 的一个计算 tile 通常需要：

128 个 token × K hidden

也就是它要拿到一批 token 的完整输入行。

但这 128 个 token 可能来自多个 source NPU：

expert C 的 GMM tile:

token 0-30 来自 rank0

token 31-70 来自 rank1

token 71-127 来自 rank5

所以一个 AIC tile 启动前，可能要等多个 AIV/多条链路通信完成：

AIC_tile_C0 依赖:

rank0 -> C 的通信

rank1 -> C 的通信

rank5 -> C 的通信

这是“多通信生产者 -> 一个计算消费者”。

反过来，一条通信过来的 token 段也可能被多个 AIC tile 使用。比如 rank0 发给 expert C 的 token 很多，跨越多个 128 行 tile：

rank0 -> C 通信段:

tokens 0-300

它会被分到：

AIC tile0: tokens 0-127

AIC tile1: tokens 128-255

AIC tile2: tokens 256-300

这是“一个通信生产者 -> 多个计算消费者”。

合起来就是多对多。

### 为什么 Dispatch-GMM 不能简单拆成单核依赖

你可能想：让 AIV0 通信一段，AIC0 算这一段，不就行了？

问题是 GMM tile 需要的是连续完整矩阵块，而通信到达的边界不一定刚好等于 tile 边界：

通信边界: 按 source rank / token count

计算边界: 按 128×256 tile

一个 tile 可能拼了多个 rank 的 token；一个 rank 的 token 又可能跨多个 tile。边界错位，所以不能天然一对一。

### GMM-Combine 为什么可以解耦

GMM2 后，数据已经是 AIC 算出来的 tile。此时通信可以按 AIC 的 tile 结果来切：

AIC0 算出来的 N×256 结果 -> AIV0 负责搬

AIC1 算出来的 N×256 结果 -> AIV1 负责搬

也就是说，通信消费的是“某个 AIC 已经产出的那块数据”。AIV 不必等其他 AIC，只要自己对应的 AIC 完成就可以搬。

这就从多对多变成：

AIC_i -> AIV_i

或文章说的 “CV 单核间依赖”。

### 为什么会有非连续远端搬运

GMM tile 是按列块/矩阵 tile 切的，例如：

N tokens × 256 columns

但这些 token 最终要回到不同原始 NPU，可能这个 tile 里：

token 0-10 回 rank0

token 11-15 回 rank3

token 16-20 回 rank0

所以 AIV 通信时不是一整块连续发给一个远端，而是带 stride/多段非连续搬运。文章说 AIV 内存语义 DataCopy 支持带 stride，所以可以处理。

### 总结

- Dispatch-GMM：通信产出 GMM 输入。通信边界和 GMM tile 边界错位，一个 tile 依赖多段通信，一段通信服务多个 tile，所以是多对多，难解耦。
- GMM-Combine：GMM 已经产出 tile，通信可以直接按 tile 消费，每个 AIV 搬对应 AIC 的结果，所以能解耦成单核/少量核依赖。
- 因此文章说：Dispatch-GMM 用软同步；GMM-Combine 用 tile 粒度切分通信。


## Q:  三级深度融合流水掩盖
W8A8/W4A8 量化本来会带来额外的量化、反量化、SwiGLU 前后处理开销，MegaMoE 试图把这些开销塞进通信/GMM流水的空隙里，让它们“不暴露”到端到端耗时上。

关键点有 4 个。

### 1. Dispatch 阶段先量化，减少通信量

通信前做 `InitRoutingQuant`：

FP16/BF16 token

-> 按 expert 展开/重排

-> 动态量化成 INT8

-> 生成 per-token scale

这样跨卡 dispatch 传的是 INT8 token，不是 FP16/BF16 token。

收益：

通信数据量更小

shmem/GM 占用更低

代码里对应 `moe_init_routing_quant_v2`，输出到 `shmem.offsetA` 的 token 是 INT8，同时有 per-token scale。

### 2. GMM 中做 per-channel 反量化，减轻 AIV 压力

INT8 GMM 算完后，需要根据 scale 把结果还原到浮点域，或者进入后续激活/量化流程。

文章说“Cube Per-Channel 反量化”，意思是把一部分反量化逻辑放到 AIC/Cube 的矩阵计算 epilogue 里做，而不是全交给 AIV。

收益：

AIC 算 GMM 时顺手处理一部分 scale/dequant

AIV 不用承担所有后处理

代码里可以从 `BlockMmad(... gmS ...)` 和 epilogue policy 看到这类设计：GMM 调用时传入 `scale1/scale2`，后续 `BlockEpilogue` 再做 per-token dequant/SwiGLU/quant。

### 3. Combine 通信传浮点结果，不再传 INT8

Dispatch 阶段传 INT8 是为了减少输入 token 通信量。

但 Combine 阶段文章说“通信过程中采用反量化后的浮点数进行传输”，意思是 GMM2 后结果已经回到输出 dtype，比如 FP16/BF16，然后再 combine 回原 token rank。

为什么这么做？

因为最终 `out[M, K]` 本来就是浮点输出，而且 combine 还要按 `probs` 做加权求和。传浮点可以避免在 combine 侧再引入一套额外量化/反量化误差和 AIV 后处理压力。

### 4. 双缓冲把前后处理藏到通信里

量化相关前后处理包括：

per-token dequant

SwiGLU

再次量化

per-channel/per-token scale 处理

这些很多在 AIV 上做。如果串行执行，会变成：

通信

-> 量化/反量化/SwiGLU

-> 下一段通信

MegaMoE 用 double buffer 做流水：

buffer0 正在通信

buffer1 正在做前后处理

下一轮交换

于是前后处理耗时被通信掩盖。

可以理解成：

时间片 t:

AIV 通信 chunk i

AIV/或另一部分处理 chunk i-1 的 dequant/SwiGLU/quant

AIC 算 chunk i-2 的 GMM

### “三级深度融合流水”怎么理解

它说的三层大概是：

1. Cube Per-Channel 反量化 GMM/Cube 里融合部分反量化，减少独立后处理。
    
2. GMM 计算 - AlltoAll 通信重叠 AIC 算 expert i，同时 AIV 搬 expert i+1 或 combine expert i-1。
    
3. AlltoAll 通信 - 前后处理重叠 AIV 通信时，用 double buffer 把 dequant/SwiGLU/quant 等处理藏进去。
    

所以这段不是在说一个单独的新算法，而是在说：

> 量化不是免费午餐，MegaMoE 把量化、反量化、SwiGLU、通信、GMM 拆成可流水的小块，利用 AIC/AIV 并行和 double buffer，把量化开销尽量隐藏在通信和计算后面。

## Q: SwiGLU 粗-细粒度结合的激活融合方案
在讲 SwiGLU 这个 AIV 后处理怎么安排同步粒度。核心矛盾是：

> GMM 和通信需要非常细粒度流水，但 SwiGLU 本身也占 AIV，不能让它挡住 Combine 通信。

### 1. SwiGLU 为什么麻烦

MoE FFN 通常是：

GMM1 -> SwiGLU -> GMM2

在量化路径里，SwiGLU 前后还多了：

dequant -> SwiGLU -> quant

这些主要吃 AIV/Vector 资源。

而 AIV 同时还要负责通信：

Dispatch 通信

Combine 通信

dequant/SwiGLU/quant

所以 SwiGLU 和通信会抢 AIV。

### 2. 非量化场景比较容易藏

非量化时，SwiGLU 没那么重。AIV 在两次通信之间可能有空窗：

Dispatch 通信结束

AIC 做 GMM1/GMM2

Combine 通信开始前

这段空窗里可以插入 SwiGLU，把它拆成几段跑，减少暴露开销。

### 3. W4A8 场景更麻烦

W4A8 量化里前后处理更重，比如拆 int4、拼接、scale、dequant/quant 等。

如果还按非量化那种方式把 SwiGLU 拆得太细、排在 Combine 前面，就可能变成：

GMM2 已经算完

但 AIV 还在做 SwiGLU/量化处理

Combine 通信启动不了

这样会推迟 Combine AlltoAll，端到端变慢。

### 4. 他们的方案：GMM/通信细粒度，SwiGLU 粗粒度

文章说“粗-细粒度结合”：

- GMM 和 AlltoAll 通信：细粒度，每个 expert 独立同步。 因为这是收益核心，要尽量流水。
    
    expert0 通信 ready -> expert0 GMM
    
    expert1 通信 ready -> expert1 GMM
    
    ...
    
- SwiGLU：粗粒度，把多个 expert 合成一组同步。 因为 SwiGLU 有较大掩盖空间，不需要每个 expert 都单独抢同步事件。
    
    group0: expert0-7
    
    group1: expert8-11
    
    group2: expert12-13
    
    group3: expert14
    
    group4: expert15
    

这样减少同步事件数量，也避免 SwiGLU 过细导致调度碎片化。

### 5. 为什么用 `{8, 4, 2, 1, 1}`

这是“幂指数递减”的分组。

对单卡 16 个 expert：

第一组 8 个 expert

第二组 4 个 expert

第三组 2 个 expert

第四组 1 个 expert

第五组 1 个 expert

直觉是：前面专家多的时候，Combine 还没那么急，可以用大组批量做 SwiGLU；越靠后，离 Combine 关键路径越近，就把组切小，让后续 GMM2/Combine 更早启动，避免最后被大块 SwiGLU 卡住。

如果一直用大组，比如：

{8, 8}

后半段 8 个 expert 的 SwiGLU 可能太大，导致 Combine 等很久。

如果每个 expert 都单独同步：

{1,1,1,...}

同步事件太多，调度开销大，事件资源也可能不够。

所以 `{8,4,2,1,1}` 是折中：前面粗，后面细。

### 6. 代码里也有类似思想

文档里你现在看到的 `epilogueGranularity` 就是在控制 SwiGLU 分段：

GMM1 通知 AIV 做第一段 SwiGLU

AIV 做完后通知 GMM2

必要时再做第二段

当前 W8A8 主路径里看起来是较简化版本，不一定完整体现文章里 W4A8 的 `{8,4,2,1,1}` 策略，但思想一致：SwiGLU 不一定按每个 expert 细粒度同步，而是按组插入流水。

一句话：GMM/通信是收益主线，所以按 expert 细切；SwiGLU 是辅助但耗 AIV，所以按专家组粗切，且越靠近尾部越细，避免堵住 Combine 通信。

## Q: 通算并行效率
这段是在讲 MegaMoE 融合方案的一个 trade-off：为了让通信和 GMM 能流水并行，它牺牲了一点小 shape 下的通信启动效率。

### Dispatch&Combine 小 shape 为什么快

原来的 Dispatch&Combine 用远端写 + DataAsFlag：

源 NPU 把 token 数据直接写到目标 NPU

Data + Count + Flag 一起发

目标 NPU 看到 flag 时：

数据到了

count 到了

同步也基本完成了

所以它的同步开销被数据发送过程掩盖了一部分。小 batch/token 很少时，这种方式很省启动延迟。

### MegaMoE 为什么有前同步开销

MegaMoE 要让目标卡能直接把 token 拉到 GMM 输入的连续位置。为此通信前必须先知道全局 token count：

rank0 给 expertC 多少 token

rank1 给 expertC 多少 token

rank2 给 expertC 多少 token

...

所以它要先做：

1. 本地重排/量化 MoeInitRoutingQuant

2. 通信前同步/交换 token count

3. 算 cumsum 地址

4. 目标卡开始远端读 token

5. 第一个 expert token 到齐

6. 第一个 Matmul 启动

文章说“第一个 Matmul 启动前至少需要完成本地重排、通信前同步、第一个专家 token 发送”就是这个意思。

### 为什么小 batch 可能吃亏

小 batch 时，GMM 计算很短，通信量也很小。MegaMoE 的固定前置开销占比很高：

本地 routing

+ count 同步

+ 第一批远端读

后面虽然可以流水：

comm expert1 和 GMM expert0 重叠

comm expert2 和 GMM expert1 重叠

但 batch 太小时，后续可重叠的计算/通信本来就很少，掩盖收益不够抵消前同步成本。

所以小 shape 可能：

MegaMoE 前同步额外开销 > 后续流水节省

### 为什么 batch 大后收益出来

batch 大时，每个 expert token 多，GMM 和通信耗时都变长。前同步仍然是一次固定成本，但后续可以大量重叠：

GMM expert0 同时 comm expert1

GMM expert1 同时 comm expert2

GMM expert2 同时 comm expert3

...

而且省掉了一些通信后重排/多算子调度开销，所以收益明显。

### 一句话

Dispatch&Combine 更像低启动延迟方案，小 token 场景很轻；MegaMoE 是用前同步换取后续通信/GMM流水，大 batch 下能把前置成本摊薄并获得收益，小 batch 下可能不划算。


## Q:先系统比较下MegaMoE对比 Dispatch&Combine的优化点，然后再比较dispatch_gmm_combine 是否具备？横向系统比较下?
横向看应该分三层：Dispatch&Combine 是通信优化，MegaMoE 是通算融合优化，`dispatch_gmm_combine` 是 接近 MegaMoE 数据流的 SHMEM demo，但高级调度不完整。

## 总体定位

|方案|主要目标|核心特征|
|---|---|---|
|Dispatch&Combine|把 MoE dispatch/combine 通信本身做快|远端写、DataAsFlag、少同步、少重排|
|MegaMoE|把 dispatch/GMM/combine 做成计算通信流水融合|让通信结果尽早成为 GMM 可消费的连续矩阵，并减少 AIC/AIV 同步空泡|
|`dispatch_gmm_combine`|独立验证 Dispatch-GMM-Combine 链路|有 routing/quant/GMM/SwiGLU/combine/unpermute，但调度更朴素|

### 横向比较

| 优化点                           | Dispatch&Combine                       | MegaMoE 相对优化                                                    | `dispatch_gmm_combine` 是否具备                                         |
| ----------------------------- | -------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| 通信前布局                         | 通信前不完全生成目标侧 GMM-ready 矩阵，更偏边发送边写目标稀疏区域 | 把前重排/量化拆到通信前，先得到 count/cumsum，让目标侧能按 expert 连续接收                | 部分具备：有 `moe_init_routing_quant_v2` 和 count/cumsum                   |
| 通信模式                          | 远端写为主，源 rank 把 token 写到目标 rank         | Dispatch-GMM 倾向远端读，目标 expert rank 主动拉取 token；GMM-Combine 倾向远端写  | 基本具备：dispatch 远端读 token，combine 远端写结果                               |
| 同步方式                          | DataAsFlag，Data/Count/Flag 合并发送，低启动开销  | 前同步 count/cumsum，换取后续按 expert 流水；再用软同步减少全核同步                    | 只具备前同步雏形：有 count 信号和 `SyncAll`，没有完整软同步                              |
| GMM 输入连续性                     | 完成最终同步/后重排前，不容易形成可直接 GMM 的连续子矩阵        | 通信按 expert-major 目标布局组织，第一个 expert 数据到达后即可启动 GMM                | 部分具备：dispatch 后 `gmA` 按 expert 连续给 GMM                              |
| 通信计算流水                        | 主要优化通信本身，难以和 GMM 深度流水                  | Dispatch 阶段 `GMM(i)` 可与 `AllToAllV(i+1)` 并行；Combine 阶段反向流水      | 有基础 overlap：按 expert + `epilogueGranularity` 做部分交叠，但不是完整 MegaMoE 调度 |
| Dispatch-GMM 软同步 / scoreboard | 无                                      | 用软同步/scoreboard 替代粗粒度 `SyncAll`，降低 AIC/AIV 多对多等待                | 没有：代码仍大量使用 `SyncAll` 和 `CrossCoreFlag`                              |
| GMM-Combine tile 切分通信         | 无                                      | 计算后通信场景可按 AIC tile 切分，变成单 AIC 到 AIV 的依赖，支持 stride 远端搬运          | 没有完整实现：combine 按 expert/source-rank 子段写回，不是 tile 级解耦通信              |
| W8A8 量化流水                     | 可有量化通信收益，但不是完整三级流水                     | routing 中量化、Cube per-channel 反量化、GMM-AllToAll、AllToAll-前后处理三级掩盖 | 有基础 W8A8 链路：INT8 token、per-token scale、SwiGLU 后再量化                  |
| W4A8 支持                       | 不是重点                                   | 支持 W4A8/W8A8，并处理额外量化开销暴露问题                                      | 不具备：demo kernel 固定 int8 权重，没有真实 W4A8 packed int4 分派                 |
| SwiGLU/激活调度                   | 非核心                                    | 粗细粒度结合，GMM/通信细切，SwiGLU 粗切，类似 `{8,4,2,1,1}` 分组避免阻塞 combine       | 只有简化版：有 `epilogueGranularity`，但没有指数递减专家分组                           |
| 小 shape 性能                    | 有优势，DataAsFlag 启动成本低                   | 小 shape 可能吃亏，前同步开销更明显；大 batch 才体现流水收益                           | 更像验证 demo，没有完整小/大 shape 自适应策略                                       |

### 关键判断

`dispatch_gmm_combine` 已经具备 MegaMoE 的 主数据流骨架：

routing + quant

-> count/cumsum

-> dispatch

-> GMM1

-> SwiGLU + quant

-> GMM2

-> combine

-> unpermute

但它不具备 MegaMoE 文章中最关键的几类高级优化：

- 没有完整 Dispatch-GMM scoreboard 软同步。
- 没有完整 GMM-Combine tile 粒度通信解耦。
- 没有 W4A8 路径。
- 没有 W4A8/W8A8 三级流水的完整协同调度。
- 没有粗细粒度结合的 SwiGLU 指数分组调度。
- 仍然大量依赖 `SyncAll`，所以还没有真正解决文章里说的 BSP 同步空泡问题。

所以可以这样定位：`dispatch_gmm_combine` 是 MegaMoE 思路的基础 SHMEM demo / 骨架实现；`dispatch_ffn_combine` 更接近 vLLM Ascend 中面向产品化集成的版本；而文章里描述的 MegaMoE 是更完整的算法与调度体系。


Q: 系统的优化点总结
MegaMoE融合算子的主要优化点是：1 dispatch阶段 将3步合并到了2步 ，combile阶段同样 2 dispatch combine的融合算法实现采用的是远端读+计算 以及计算+远端写 3 ： 通信流水采用expert串行的方式，做1和1+1的通算掩盖 4： dispatch和combile阶段的同步方式采用计分板和Tile切分通信以防止全局同步 5：量化三级流水线优化 6：Swiglu的粗-细粒度结合的激活融合方案 这些？是这些吗？另外这些事怎么系统的串起来的？

是的，你列的基本就是 MegaMoE 融合算子的主线，但第 1 点要稍微修正一下。

准确说可以这样串：

目标：让 MoE 的 dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine

从“通信算子 + 计算算子分开跑”

变成“通信和计算细粒度流水重叠”。

## 主要优化点

1. 先解决数据布局问题
    
    - 原 Dispatch&Combine 通信效率高，但接收端数据是稀疏/乱序到达的，不适合 GMM 边收边算。
    - MegaMoE 把 `MoeInitRoutingQuant` 前置：先完成 token 重排、量化、count/cumsum 同步。
    - 这样 dispatch 后的数据可以直接按 expert 连续落到 GMM 输入矩阵里。
2. 通信方向按生产者/消费者关系选择
    
    - Dispatch-GMM：通信生产 GMM 输入，所以采用 远端读 + 本地计算。目标 expert rank 主动拉 token，拉到本地后直接 GMM。
    - GMM-Combine：GMM 生产 combine 输入，所以采用 本地计算 + 远端写。expert rank 算完后把结果写回 token 原 rank。
3. 按 expert 做流水
    
    - 不等所有 expert 通信完成再统一 GMM。
    - Dispatch 阶段可以形成：
        
        通信 expert i+1 和 GMM expert i 并行
        
    - Combine 阶段可以形成：
        
        GMM expert i 和 combine expert i-1 并行
        
    - “expert 串行”不是说一个 expert 全流程跑完再下一个，而是按 expert 顺序推进流水拍。
4. 同步优化
    
    - Dispatch-GMM 是通信先、计算后，依赖关系多对多，难拆，所以用 软同步/计分板 减少 `SyncAll` 带来的快核等慢核。
    - GMM-Combine 是计算先、通信后，依赖更容易按 tile 拆开，所以用 Tile 切分通信，把多对多依赖拆成更细的 CV 单核依赖。
    - 这两者共同目标都是避免全核 BSP 同步空泡。
5. 量化三级流水
    
    - Dispatch 前 routing 时融合量化，传 INT8，减少通信量。
    - GMM/Cube 侧承担部分 per-channel 反量化。
    - AllToAll 通信和量化前后处理用 double buffer/流水掩盖。
    - 核心目标是：量化有收益，但量化开销不要暴露在端到端时延里。
6. SwiGLU 粗细粒度结合
    
    - GMM/通信是收益主线，所以按 expert 细粒度切。
    - SwiGLU 会占 AIV 资源，如果也每 expert 细切，会消耗太多同步事件，甚至阻塞 Combine。
    - 所以 SwiGLU 采用粗粒度 expert 分组，例如 `{8,4,2,1,1}`，尽量塞进通信空隙里。

## 怎么系统串起来

可以按“先让数据能算，再让算和通信重叠，再减少同步和量化开销”理解：

1. 前置 routing/quant/count/cumsum

-> 让 dispatch 后的数据天然是 GMM-ready

2. 选择远端读/远端写

-> 让通信发起方尽量和计算生产者/消费者在同一张卡

3. 按 expert 组织流水

-> 让通信 expert i+1 掩盖 GMM expert i

4. 用 scoreboard / tile split 降同步

-> 避免流水被 SyncAll 空泡吃掉

5. 把量化和 SwiGLU 塞进流水空隙

-> 避免前后处理成为新的瓶颈

关于一机 8 卡和多机场景：这些优化大部分是 多卡 EP 通信优化，不是只针对多机。单机 8 卡也适用。多机时链路 RTT、跨机带宽、慢链路问题更明显，所以收益和调度压力更大；但算法机制本身不是“只有多机才用”。


## 它们怎么系统串起来

主线是：

先把 token 按 expert 组织

-> 让 dispatch 后的数据天然成为 GMM 可消费的连续矩阵

-> 按 expert 做通信/GMM流水

-> 用远端读/远端写减少跨卡依赖

-> 用软同步/tile切分避免 SyncAll 空泡

-> 把量化和 SwiGLU 塞进流水空隙

所以核心不是 6 个孤立优化，而是一条逻辑链：

数据布局为 expert 连续服务，expert 连续布局为 GMM 服务，GMM 和通信按 expert 流水，流水又要求更细同步，量化和 SwiGLU 再嵌入这个流水里隐藏开销。

## 程序运行时是不是一开始就按 expert 切

不是一开始就“先处理 expert1 的重排通信”。

更准确是两层：

1. routing 阶段先整体处理本 rank 的所有 token。  
    它读取所有 `expertIdx[M, topK]`，把 token 展开、按 expert 排列，统计每个 expert 的 token count。这一步是全局整理，不是只做 expert1。
    
2. 有了 count/cumsum 后，后面的 dispatch/GMM/combine 才按 local expert 顺序推进流水。  
    例如：
    

routing/quant/count/cumsum 先完成基础布局信息

然后进入 expert 流水：

dispatch expert0 token

dispatch expert1 token || GMM expert0

dispatch expert2 token || GMM expert1

...

GMM2 expert0 || combine expert0/相邻阶段

GMM2 expert1 || combine expert1/相邻阶段

所以，“按 expert 组织流水”不是程序第一行就只处理 expert1，而是 routing 先把全体 token 分类并生成 count/cumsum，之后通信和 GMM 按 expert 段流水推进。