dispatch_ffn_combine 融合算子全解（详解版）
第〇章　这个算子到底在解决什么问题
要理解它的每一个设计，得先理解它诞生的背景。

大模型里的 MoE（Mixture-of-Experts）层有几百个专家，单张卡的显存装不下全部专家的权重。工程上的做法是专家并行（EP, Expert Parallel）：把全部专家平摊到 EP 张卡上，每张卡只持有 expertPerRank 个专家的权重。

这就引出一个绕不开的矛盾：token 此刻在哪张卡，和它被 router 选中的专家在哪张卡，几乎总是对不上。 一个 token 经过 gating 会选 topK 个专家，这 topK 个专家可能散落在不同的卡上。于是一次完整的 MoE FFN，被迫拆成"通信—计算—通信"三大段：

dispatch（分发）：把每个 token 送到"它选中的专家所在的那张卡"——这是一次 All2All 通信。
FFN 计算：在专家所在卡上做两层矩阵乘 GMM1（升维）→ SwiGLU（门控激活）→ GMM2（降维）。
combine（回收）：把每个专家算出的结果送回 token 原来所在的卡，再按 topK 权重把同一 token 的多份结果加权合并——这是又一次 All2All。
如果把这几步拆成独立算子来做（传统做法），问题很严重：每步之间都要把中间结果完整写回 HBM、下一步再读回来，反复的 HBM 往返吞掉大量带宽；而且通信和计算是串行的，通信时计算单元空着、计算时通信链路空着，硬件利用率很低。

dispatch_ffn_combine 的使命就是把上面五步（dispatch → GMM1 → SwiGLU → GMM2 → combine，外加收尾的 unpermute）融进一个 kernel，达到两个目的：

消除中间 HBM 往返：中间结果尽量留在片上或对称内存，不落盘再读。
让通信藏到计算之下：把工作切成以"单个专家"为单位的小块，让"算第 i 个专家"和"搬第 i±1 个专家"在时间上重叠，把通信延迟掩盖掉。
整个 kernel 跑在昇腾 A2 上。A2 的每个计算单元由两种核组成：AIC（Cube 核，专做矩阵乘） 和 AIV（Vector 核，做向量运算和数据搬运）。这个算子做了彻底的 CV 分离：


dispatch_ffn_combine_kernel.hpp
Lines 215-228
    void operator()<AscendC::AIC>(Params const &params)
    {
        GMM1(params);
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
        GMM2(params);
    }
    void operator()<AscendC::AIV>(Params const &params)
    {
        DispatchAndCombine(params);
    }
AIC 只干两件事——GMM1 和 GMM2，中间停下来等 AIV 把 SwiGLU 做完（CrossCoreWaitFlag<0x2>(SYNCFLAGV2C)）。AIV 则包揽其余一切：路由量化、三跳 All2All 通信、SwiGLU 激活、combine、unpermute。两种核之间靠跨核 flag（CrossCoreSetFlag/WaitFlag<0x2>）握手，构成"AIV 当生产者/搬运工、AIC 当计算引擎"的流水关系。

后面所有章节，都是在拆解"这五步具体怎么切数据、怎么同步、怎么把通信藏起来"。

第一章　两个贯穿全局的"配置概念"
在进入执行流程前，先讲两个会反复出现、但容易误解的概念。

1.1 tilingKey：它是"走哪条路"的开关，不是数据大小
场景：同一个算子要面对千变万化的输入——序列长短不一、量化方式不同、专家负载有的均匀有的极端。如果为每种情况都编译一份 kernel，二进制会爆炸。CANN 的解法是：编译一份覆盖所有情况的 kernel，在运行前由 host 算出一个整数 tilingKey，device 端用它来 if/else 选择走哪条代码分支。

所以 tilingKey 是"分支路由码"，把"该走哪条实现路径"编码成一个整数。数据规模（size）只是决定这个整数取值的输入之一，它本身不代表 size。

主算子的 tilingKey 这样拼出来：


dispatch_ffn_combine_tiling.cpp
Lines 248-251
    uint64_t tilingKey = INIT_TILINGKEY;                       // 基础值 1000000
    tilingKey += info.isTransposeB ? TILINGKEY_TRANS_B : 0;    // 权重是否转置 +1
    tilingKey += info.isWeightNz ? TILINGKEY_WEIGHT_NZ : 0;    // 权重是否 NZ 格式 +10
    context->SetTilingKey(tilingKey);
之前提到的 1000010，拆开就是"int8 基础路径 + 不转置 B + 权重用 NZ 格式"。可以看到它是用十进制的不同位各编码一个开关，叠加成一个数。后面 init_routing 内部还有一套自己的 tilingKey，套路相同，只是编码的开关换成了"全load/量化模式/drop模式/单核还是多核排序"。

1.2 五层嵌套的数据切分
场景：要让几十张卡、每张卡几十个核协同算完一个巨大的 MoE FFN，数据必须被层层切碎，每一层切给不同的硬件单位。这五层是嵌套的，从粗到细：

层级	按什么切	切给谁	这一层解决的问题
① 卡（EP rank）	专家归属	EP 张卡	显存装不下全部专家
② 核	通信按源卡、计算按 tile	各 AIC / AIV 核	单核算力/带宽有限，要并行
③ group（专家）	本地专家序号	不绑定核，是流水的批次单位	让通信和计算以专家为粒度重叠
④ tile	128×256 块	轮转给各 AIC	一个专家的矩阵乘仍太大，要再切
⑤ L1/L0	128×256×512 / 128×256×128	Cube 内部多级缓存	片上 SRAM 容量有限，K 维要分批进
这张表是后面所有细节的骨架。特别注意第③层：group（专家）不是"分给某颗核"的单位，而是"数据就绪同步 + 通信计算重叠"的批次边界——这是全篇最容易误解的点，第五章会详细说。

第二章　阶段 A：init_routing —— 进 GMM 前最复杂的子系统
2.1 它要解决什么
dispatch 通信的前提是：token 必须先按目标专家排好序、密排在一起，这样发往同一个专家的 token 才是一段连续内存，才能高效地跨卡搬。同时还要把 token 量化成 int8（减少通信量，也为后面 GMM 用 int8 做准备），并记录好"排列后第几行 ↔ 原始第几个 token"的映射（combine 完了要靠它还原回去）。

这件事的输入是 expertIdx（每个 token 选了哪 topK 个专家，展开成 m*topK 个 (token, expert) 对），输出三样：

expandedX：按专家密排 + int8 量化后的 token，写进对称内存等别卡来取；
expandedRowIdx：排列后行 ↔ 原始 token 的映射；
localTokenPerExpert：每个专家收到多少 token。
核心难点是按 expert 排序。m*topK 动辄几万到几十万，单核的 UB（片上向量内存）根本装不下，必须多核分块排序再归并。所以这一步绝不是"路由量化"四个字能概括的，它是一个多 stage、含多核归并排序、各 stage 内部再切分的完整子系统。

主 kernel 这样调用它，调用前先 ApplyXActiveMask（把失效 token 的专家号改成无效值，后续会被丢弃），调用后 SyncAll 收口：


dispatch_ffn_combine_kernel.hpp
Lines 798-806
        ApplyXActiveMask(params);
        moe_init_routing_quant_v2<ElementD2>(reinterpret_cast<GM_ADDR> (params.ptrA), params.expertIdx,
        params.moeInitRoutingQuantV2Scale, params.moeInitRoutingQuantV2Offset, shmem() + peermemInfo.offsetA,
        workspaceInfo.expandedRowIdx, localTokenPerExpert, params.expertTokensBeforeCapacity,
        shmem() + peermemInfo.offsetPeerPerTokenScale,
        params.ptrWorkspace + expandedRowIdxOffset,
        &params.moeInitRoutingQuantV2TilingData, params.initRoutingQuantTilingKey);
        AscendC::SyncAll<true>();
2.2 第一层结构：4 个串行 stage，靠"销毁/重建 TPipe"复用 UB
场景：排序、统计、建映射、量化搬运，这四件事用的 UB 布局完全不同，且必须按顺序做（后一步依赖前一步结果）。如果同时分配四套 UB buffer，UB 不够用。解法是让它们串行复用整块 UB——每个 stage 用一个独立的 TPipe（UB 分配器），用完立刻 Destroy 释放，下一个 stage 再新建。


moe_init_routing_quant_v2.cpp
Lines 67-132
  // Stage 1: sort —— 按 expert 排序
  if (tilingKey == 10000 || ...) {        MoeV2SortOneCore op;   ... sortPipe.Destroy(); }
  else if (tilingKey == 10010 || ...) {   MoeV2SortMultiCore op; ... sortPipe.Destroy(); }
  // Stage 2: expertTokenOut —— 统计每专家 token 数
  // Stage 3: srcToDst —— 建立行映射 expandedRowIdx
  // Stage 4: gather + quant —— 按映射搬运并量化
四个 stage 的数据怎么一步步变过来：

expertIdx[m*topK]
 ① sort     ：把 (expert, token) 按 expert 升序排好 → 得到 sortedExpert + expandDstToSrcRow（排序后行 → 原行）
 ② count    ：扫描排好的 expert 序列，数出每个专家有多少 token → localTokenPerExpert
 ③ srcToDst ：把"排序后行→原行"反过来，得到"原行→目标行"，即 expandedRowIdx（permute 映射表）
 ④ gather+quant：按映射把原始 x 的每一行 gather 到它的目标位置，顺便 int8 量化 → expandedX 落 shmem
这里还藏着一条快路径：当 m*topK 小到单核 UB 能一次全装下时，根本不用分 stage，直接 MoeV2FullLoadQuant 一把梭（tilingKey 20000/21000）。是否走快路径、走哪条慢路径，全由 tilingKey 编码：


moe_init_routing_quant_v2_tiling.h
Lines 147-153
  if (isFullLoad) return TILING_KEY_PERF_BASE + quantMode * 1000;   // 20000/21000 全load快路径
  return TILING_KEY_BASE + quantMode * 1000 + dropPadMode * 100
       + (totalLength > sortLoopMaxElement) * 10;   // 末位 +10 ⇒ 数据太大，走多核排序
那个 +10（10000 变 10010）就是判断"数据是否超过单核排序上限"——超了就切多核排序。

2.3 第二层结构：排序本身是 VBS→VMS→SortOut 三段多核归并
场景：当 m*topK 大到单核装不下，就得用外部归并排序的思路——先让每个核各自排好自己那一段（局部有序），再把这些有序段跨核归并成全局有序。这正是数据库/大数据里经典的多路归并，搬到了 NPU 上。


moe_v2_sort_multi_core.h
Lines 366-371
__aicore__ inline void MoeV2SortMultiCore::Process() {
  InitExpertTokensGlobalMemory();
  VBSProcess();    // Vector Block Sort：核间均分 + 核内分块排序
  VMSProcess();    // Vector Merge Sort：跨核多轮 4 路归并
  SortOutProcess(); // 单核做最后归并并输出
}
VBS（块内排序）：把 m*topK 个元素按核均分，每个核负责一段；核内再按 UB 容量切成若干 loop，每个 loop 调硬件 Sort 指令排一小段。这里有个巧妙的 key 设计——把 expert id 转成 fp32 再乘 -1，用"升序 Sort + 负值"实现按 expert 排序，同时 Sort 指令的 value 槽塞进原始行号，排完就天然得到 (sortedExpert, srcRow) 对：


moe_v2_sort_multi_core.h
Lines 221-237
__aicore__ inline void MoeV2SortMultiCore::VBSProcess() {
  if (this->blockIdx < this->vbsTilingData->needCoreNum) {
    int64_t sortNum = Ceil(sortCoreLoopElements, ONE_REPEAT_SORT_NUM) * ONE_REPEAT_SORT_NUM;
    for (int64_t loop = 0; loop < sortCoreLoops - 1; loop++) {
      UBSortProcess(loop, sortCoreLoopElements, sortNum);
    }
    ...
    if (sortCoreLoops > 1) {
      OneCoreVMSProcess(sortCoreLoops, ...);   // 核内多段先归并成一条链
    }
  }
  AscendC::SyncAll();   // ← stage 间全核同步，保证所有核都排完
}
VMS（跨核归并）：VBS 后每核有一条局部有序链，但核之间还没合。VMS 做多轮归并，每轮把相邻 4 条链（MAX_MRGSORT_LIST=4，因为昇腾 MrgSort 指令一次最多归并 4 路）合成一条，listNum 每轮除以 4，直到 ≤4：


moe_v2_sort_multi_core.h
Lines 245-274
  for (; listNum > MAX_MRGSORT_LIST;) {
    currentStageNeedCoreNum = Ceil(listNum, MAX_MRGSORT_LIST);   // 本轮需要几个核
    ...
    if (this->blockIdx < currentStageNeedCoreNum - 1) {
      InitMoeMrgSort(&mrgsorter, MAX_MRGSORT_LIST, coreOffset, 0);  // 满 4 路归并
      mrgsorter.Process();
    } else if (this->blockIdx == currentStageNeedCoreNum - 1) {
      InitMoeMrgSort(&mrgsorter, remainListNum, ...);              // 尾核归并剩余
    }
    listNum = currentStageNeedCoreNum;             // 链数缩为约 1/4
    srcWsIndex = (srcWsIndex + 1) % WORK_GM_NUM;   // 乒乓切换两块 GM，避免原地读写冲突
    ...
    AscendC::SyncAll();   // ← 每轮归并后全核同步
  }
这里有三个值得注意的设计：4 叉归并树（指令限制）、两块 GM 乒乓交替（这一轮的输出当下一轮的输入）、参与核数逐轮收缩（链越来越少，干活的核越来越少）。

SortOut：链数 ≤4 时，由 0 号核做最后一次归并，同时拆出 sortedExpertIdx 和 expandDstToSrcRow 写到 GM。

如果数据本来就小（totalLength ≤ sortLoopMaxElement），则走 MoeV2SortOneCore，单核一把排完，没有这三段、也没有中间的 SyncAll。

2.4 第三层结构：每个 stage 内部还有行/列切分 + UB 双 buffer
场景：以 ④ gather+quant 为例，要把 totalLength 行、每行 cols（hidden 维，如 7168）的数据搬运并量化。行要按核分；如果一行太宽、UB 装不下一整行，列还要再切成多个 loop。


moe_init_routing_quant_v2_tiling.h
Lines 302-337
void MoeInitRoutingQuantV2TilingBase::Tiling4GatherQuant() {
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);    // ① 行按核分
  tilingData->needCoreNum = CeilDiv(totalLength, perCoreRows);
  ...
  if (rowSize + colSize < ubSize / 2) {
    SetGatherTilingData(...);                            // UB 够 → 整行一次搬
  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP_QUANT;       // 8192
    SetGatherTilingDataCols(tilingData, baseMaxCols, cols);  // ② 列切 loop
    SetGatherTilingDataRows(tilingData, perCoreRows, ...);   // ③ 行切 loop
  }
}
stage 内部还用 BUFFER_NUM=2 的双 buffer 做 CopyIn/Compute/CopyOut 流水。

小结：init_routing 是"4 stage 串行流水（串行复用 UB）+ 多核归并排序（VBS/VMS/SortOut，自带一套 SyncAll 同步体系）+ 每个 stage 内行列切分 + UB 双 buffer"的完整子系统。它内部的 SyncAll 同步和主 kernel 的跨核同步是两套独立的体系。之前把它当"一步"是严重低估。

第三章　阶段 B：第一跳 All2All —— 先把"谁发给谁多少"对齐
3.1 它要解决什么
dispatch 真正搬 token 之前有个鸡生蛋问题：本卡要从别的卡那里"取走发给本卡专家的 token"，但本卡并不知道每张源卡到底发给本卡的每个专家多少 token。不知道数量，就算不出每段数据该放在 gmA 的哪个偏移。

所以必须先做一次轻量通信：把每张卡本地统计的 [expertPerRank] 计数，AllGather 成一张全局的 [EP, expertPerRank] 通信矩阵，再算好前缀和（cumsum），后面 dispatch 就能用查表代替复杂地址计算。

3.2 怎么做：按目标卡切核 + DataAsFlag 点对点等待
这一步按目标卡把活分给 AIV 核（dstEpIdx = coreIdx; += coreNum，第 coreIdx 个核负责往第 coreIdx、coreIdx+coreNum… 张卡写）。分两步：先把本卡计数写到所有别的卡，再等所有别的卡把它们的计数写到本卡。


dispatch_ffn_combine_kernel.hpp
Lines 657-705
        for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx == params.rank) { continue; }
            ...
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);   // 给计数打标记
            copyUbToGm(dstAddress[0], tmpBuffer, ...);                   // 写到对端卡的 shmem
        }
        for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            if (dstEpIdx != params.rank) {
                int32_t intPer512 = CACHE_LINE / sizeof(int);
                for(int32_t checkIdx = 0; checkIdx < paddedExpertNumAligned; checkIdx += intPer512) {
                    __gm__ int32_t* sync_check = ... tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                    gm_signal_wait_until_ne(sync_check, 0);              // 自旋等这张源卡的数据到达
                }
                AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore); // 去掉标记还原真值
这里的同步机制是 DataAsFlag（数据落地即信号），值得单独说：跨卡通信怎么知道"对方写完了"？常规做法要额外发一个完成信号，有延迟。这里的巧法是——发送端给每个计数值加一个大偏移 0x800000（让它一定非 0），写进对端 shmem 约定的位置；接收端按 512B cache line 粒度自旋读（gm_signal_wait_until_ne(., 0)），一旦读到非 0，就说明这块数据确实被写进来了（内存序保证 cache line 是原子可见的），读完再减回 0x800000 还原真实计数。数据本身兼当了到达信号，省掉一次独立的信号通信，而且是点对点、按卡各等各的，不是全局栅栏。

收齐后每个核算自己负责那部分的前缀和写进 preSumBeforeRank，最后 SyncAll 把本卡所有 AIV 对齐，再由 0 号核算出 cumsumMM（dispatch 查表用的累计偏移）并打第一个 flag 通知 AIC 的 GMM1 可以准备了。

第四章　阶段 C：第二跳 All2All —— dispatch 远端读 gather
4.1 它要解决什么
现在偏移表有了，可以真正搬 token 了。本卡要把"全世界发给本卡 groupIdx 号专家"的 token，从各张源卡的 shmem 里读过来，密排进本地的 gmA，供 GMM1 计算。

4.2 怎么做：按源卡切核 + 按 group 流水通知 GMM1
外层循环遍历本地专家（group），内层按源卡分核去远端读：


dispatch_ffn_combine_kernel.hpp
Lines 839-864
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM((params.EP - 1) * params.expertPerRank + groupIdx);
            for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
                uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * params.expertPerRank + groupIdx)) + prevGroupSum1;
                    GM_ADDR otherRankPtr = shmem(0, dstEpIdx);            // 第 dstEpIdx 张源卡的 shmem
                    CopyGMToGMPerToken(gmA[gmOffsetA], gmPerTokenScale1[rowStart], gmRemoteA[gmOffsetPeer], rows, params.problemShape.k(), ...);
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);  // 这一组搬完了，通知 GMM1
            syncgmm1Idx ++;
        }
这里有两个关键点。

第一，gather 顺便完成了重排。 读地址 gmOffsetPeer 是源卡上的位置，写地址 gmOffsetA 是 GMM 需要的密排位置，rowStart 由 cumsumMM 查表算出。这样一边搬、一边就把数据排成了 GMM1 直接能吃的布局——"通信后重排"被折叠进了 gather 的地址映射里，不需要单独再排一遍。

第二，按 group 流水是通信掩盖的核心。 注意每搬完一个专家（一个 group）的全部 token，就 SyncAll 一次，然后 CrossCoreSetFlag 通知 AIC："groupIdx 这个专家的数据齐了，你可以开始算它的 GMM1 了。" 这样 AIC 不用等所有专家都搬完，搬好一个就能算一个——第 i 个专家的 GMM1 计算，和第 i+1 个专家的 dispatch 搬运，在时间上重叠了。这正是融合算子掩盖通信的手段。（flagId = 序号/15 是因为硬件 flag id 数量有限，用整除把逻辑序号映射到有限的物理 flag 上。）

第三，搬运本身也是流水的。 CopyGMToGMPerToken 内部用两块 UB 做 MTE2（读对端）和 MTE3（写本地）的 ping-pong，让一块在读、另一块在写：


dispatch_ffn_combine_kernel.hpp
Lines 338-350
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::DataCopy(buf, src[inputOffset], dataLen);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            ...
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
第五章　阶段 D：GMM1 —— 升维矩阵乘，以及四个藏在细节里的优化
5.1 它要解决什么 & group/tile 怎么分
GMM1 要对每个专家做 [currentM, K] × [K, N] 的矩阵乘（升维）。currentM 是这个专家收到的 token 数，由路由动态决定，所以每个专家的矩阵乘大小都不一样——这是 Grouped MatMul（一批大小不一的矩阵乘）。


dispatch_ffn_combine_kernel.hpp
Lines 451-521
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM(...);
            if (preCurrentmSum >= params.maxOutputSize) { currentM = 0; }
            else if (preCurrentmSum + currentM >= params.maxOutputSize) { currentM = params.maxOutputSize - preCurrentmSum; }
            ...
            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for(;syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx ++;
                }
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                ...
                blockMmad(gmA[...], layoutA, gmB1[...], layoutB1, gmC[...], layoutC, gmS[...], layoutScale, actualBlockShape);
            }
            if ((groupIdx + 1) == params.epilogueGranularity && (groupIdx < params.expertPerRank - 1)) {
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) { blockMmad.SynchronizeBlock(); }
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
这里必须讲清 group 和 tile 的关系，这是全篇最易错的点。 外层 for groupIdx 是每个 AIC 核都要跑的——不存在"AIC0 负责专家0、AIC1 负责专家1"这回事。真正分给各核的是 tile：每个专家的矩阵乘被 BlockScheduler 按 128×256 切成 coreLoops 个 tile，用 loopIdx += coreNum 轮转分到所有 AIC。而 CrossCoreWaitFlag 等的是 group 粒度的信号——AIV 每搬完一个专家发一次，这一次握手会放行所有 AIC 核，它们放行后各自去算属于自己的 tile。所以：数据就绪是 group 粒度（粗），计算划分是 tile 粒度（细），两者正交。 startCoreIdx = (startCoreIdx + coreLoops) % coreNum 是跨专家接力——让下一个专家的 tile 从上一个专家结束的核继续轮转，避免每个专家都从 0 号核开始、把活堆在前几个核上。

5.2 四个藏在细节里的优化
这一段还嵌着四个之前没展开、但很关键的机制，每个都对应一个具体场景：

① swizzle 遍历顺序（访存局部性优化）。 tile 的编号 loopIdx 怎么映射成 (m,n) 坐标，用的是 GemmIdentityBlockSwizzle<9,1>（见 dispatch_ffn_combine.h:249），不是简单的行优先。场景：矩阵乘里 weight 是被反复读的（M 方向每个 tile 都要读同一份 weight）。swizzle 让相邻被调度的 tile 在 N 方向聚簇，这样它们读的是同一块 weight，能命中 L2 cache，省掉重复从 HBM 读 weight。这是独立于"核分配"的一层访存调度。

② maxOutputSize 截断（容量约束）。 currentM 会被 maxOutputSize（本卡 workspace 能容纳的输出行数上限）截断，超出的部分直接 currentM=0 丢掉。场景：MoE 路由可能极不均，某个热门专家收到的 token 数爆表，但 workspace 是预分配的固定大小，装不下就只能丢——这是容量决定的硬切分约束（drop 语义）。

③ L2 Cache Hint（针对 MoE 负载不均的 cache 管理）。 当 currentM ≤ 128（一个 M-tile 都不到，说明是个冷门专家、token 很少）时，主动 CACHE_MODE_DISABLE 禁用这个 weight 的 L2 缓存。场景：weight 进 L2 的价值在于"会被后续 tile 复用"。但小 group 只有一个 M-tile，weight 读一次就换下一个专家了，根本没有复用。这时还把它塞进 L2，不仅没收益，反而会把别的真正会复用的数据挤出 L2（污染）。所以小 group 直接旁路 L2。这是利用 MoE "专家负载天差地别"特性的精细化优化——热门专家的 weight 留 L2，冷门专家的 weight 不占 L2。

④ SynchronizeBlock（异步 preload 流水的收尾排空）。 这个 BlockMmad 用的是 preload 异步策略：为了把"从 GM 搬数据进 L1"的延迟藏起来，它故意让"计算"落后"搬运" PRELOAD_STAGES 拍——先发射好几拍的搬运，再回头算最早那拍。代价是循环跑完时，最后几个 tile 的搬运发了但计算还没做（preloadCount 记着这笔"欠账"）：


block_mmad_preload_async_fixpipe_quant.hpp
Lines 266-273
    void SynchronizeBlock()
    {
        while (preloadCount > 0) {
            L1TileMmad(l1TileMmadParamsList[l1TileMmadParamsId]);   // 把欠着的计算补做完
            l1TileMmadParamsId = (l1TileMmadParamsId + 1 < PRELOAD_STAGES) ? (l1TileMmadParamsId + 1) : 0;
            --preloadCount;
        }
    }
场景：GMM1 马上要打 flag 通知 SwiGLU "你的输入好了"。但如果还有 tile 没算完，SwiGLU 就会读到半成品。所以在 Finalize 打 flag 之前，必须先 SynchronizeBlock() 把欠账全部算完、写出到 GM，保证"通知下游"发生在"数据真正全写完"之后。

Finalize 内部本身有软同步（计分板）和硬同步两条路，但计分板这条没启用（构造时传了空指针），实际走硬同步 CrossCoreSetFlag：


block_mmad_preload_async_fixpipe_quant.hpp
Lines 276-293
    void Finalize(int32_t target, int32_t flag = 0)
    {
        if (ptrSoftFlagBase_ != nullptr) { ... }   // 计分板软同步：构造传 nullptr，不走
        else {
            for(;syncGroupIdx <= target; syncGroupIdx++) {
                int32_t flagId = syncGroupIdx / 15 + flag;
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(flagId);
            }
        }
    }
5.3 为什么必须 int8：L1 容量算给你看
GMM 的 tile 是 128×256×512（M×N×K），双缓冲（l1Stages=2）。场景：A2 单个 AIC 的 L1 只有 512KB，tile 大小和精度必须凑得进去，否则编译期 static_assert 直接拦下。

精度	A tile (128×512)	B tile (512×256)	Scale	单缓冲	双缓冲	512KB?
int8	64KB	128KB	2KB	194KB	388KB	✅
fp16	128KB	256KB	—	384KB	768KB	❌ 爆
所以必须把矩阵乘做成 int8，才能让 K=512 这么大的 tile 还能双缓冲塞进 L1——这也是为什么 init_routing 阶段就要把 token 量化成 int8。

第六章　阶段 E：SwiGLU —— 两段错峰，把激活藏进 GMM 的缝里
6.1 它要解决什么
SwiGLU 是 GMM1 和 GMM2 之间的门控激活，跑在 AIV 上。场景：如果等 GMM1 把所有专家都算完，AIV 才开始做 SwiGLU，做完 GMM2 才能开工——那 AIV 做 SwiGLU 时 AIC 在干等，AIC 做 GMM 时 AIV 在干等，两种核轮流空转，流水断了。

解法是把专家分成两段，让依赖链以"半批"为单位流动：GMM1 算完前一段专家就先通知 SwiGLU 去处理，自己继续算后一段；SwiGLU 处理前一段时 GMM2 已经能用它的结果开工了。这样 GMM1 尾 / SwiGLU / GMM2 头三者错峰重叠。

切分点由 epilogueGranularity = expertPerRank - 3（专家数 ≤4 时为 -1）决定，对应 dequantSum1（第一段行数）和 dequantSum2（第二段）。

6.2 怎么做：两段，每段是对称的五步握手

dispatch_ffn_combine_kernel.hpp
Lines 917-949
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);   // 等 GMM1 第一段算完
        AscendC::SyncAll<true>();
        if (dequantSum1 > 0) {
            ...
            blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[...], gmPermutedToken[...], gmPerTokenScale2[...], params.epilogueCoreNum);  // 做第一段 SwiGLU
        }
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);  // 通知 GMM2：第一段输入好了
        if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);        // 等 GMM1 第二段
            AscendC::SyncAll<true>();
            if (dequantSum2 > 0) {
                uint32_t rowStartThisCore = dequantSum1;
                ...
                blockEpilogue1(gmC[...], shapeC, ..., coreNum);  // 做第二段 SwiGLU
            }
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);  // 通知 GMM2：第二段好了
        }
每段都是固定的五步节拍：WaitFlag(C2V) → SyncAll → 算 → SyncAll → SetFlag(V2C)。SwiGLU 内部做的是反量化 → SiLU 门控 → 重新量化成 int8（给 GMM2 用）。

补充：v4/MegaMoE 把这个两段思路进阶成了幂指数分组 {8,4,2,1,1}（前面专家分组大、越靠后越小，目的是让最后的专家尽快做完 SwiGLU、不阻塞 combine 通信）。当前 v1 是它的两段简化版，机制相同、不冲突——都是"SwiGLU 按专家组粗切，错峰塞进 GMM 流水"。

6.3 epilogue 内部的 UB 流水：开场预置与收场排空
场景：SwiGLU（以及后面的 combine）的 epilogue 内部是一个 UB ping-pong 流水，循环体结构是"先 WaitFlag 等这块 UB 空闲、再用、用完 SetFlag"。这带来两个边界问题：循环第一拍的 WaitFlag 没有"上一轮"可等，会永久死锁；循环最后几拍的 SetFlag 没人消费，残留的 flag 会污染下一个使用者。

解法就是进循环前 SetFlag() 预置、出循环后 Finalize() 排空：


block_epilogue_pertoken_row.hpp
Lines 108-123
    void SetFlag()      // 开场：给所有 UB stage 预置"空闲"信号，让第一拍 WaitFlag 能立刻通过，避免死锁
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
    void Finalize()     // 收场：把残留 flag 全部消费掉，保证最后几拍数据真落了 GM、且不污染后续
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }
类比一条流水线：SetFlag 是"开机，把所有工位标空闲，让第一个零件能进站"；Finalize 是"停机，等所有在制零件出站、清空工位"。少了开机，第一个零件进站发现工位被占（其实没人初始化）→ 死锁；少了停机，关灯时还有零件卡在线上 → 数据没写完。主流程里 blockEpilogue1.Finalize() 排空 SwiGLU，接着 blockEpilogue2/3.SetFlag() 给 combine 开场。

第七章　阶段 F/G：GMM2 与 combine 两条路
7.1 GMM2
GMM2 结构和 GMM1 完全一样（逐 group、tile swizzle 分核、startCoreIdx 接力、SynchronizeBlock 排空），只是维度变了：n2 = K、k2 = N/2（因为 SwiGLU 把宽度砍半了）。它在每个专家组的最后一轮设置 syncLoopIdx，由此对该组打组级 flag，作为 combine 的就绪信号。

7.2 combine 要解决什么 & 两条路的取舍
combine 是 dispatch 的逆操作：把 GMM2 的结果反量化后，写回每个 token 原本所在的源卡 shmem（offsetD），供 unpermute 还原。这里按规模分了两条路，是一个"负载均衡 vs 调度开销"的经典取舍：


dispatch_ffn_combine_kernel.hpp
Lines 952-959
        blockEpilogue1.Finalize();
        if (isCombineV1) { blockEpilogue2.SetFlag(); CombineV1(params, blockEpilogue2); }
        else { blockEpilogue3.SetFlag(); CombineV2(params, blockEpilogue3); }
CombineV1（大 shape，prefill）	CombineV2（小 shape ≤4096，decode）
分核	按目的卡 dstEpIdx，一次发整段 {dataRows, n2} 全宽连续块	按 tile + AIV 子核 + 16 行 sub-tile
epilogue	逐行 per-row 反量化	BlockEpilogue3 按卡带 stride 远端写
同步	每组 CrossCoreWaitFlag + SyncAll（全核栅栏）	仅 CrossCoreWaitFlag（CV 一对二），无 SyncAll
CombineV1 为什么要 SyncAll：V1 按"目的卡"重新分核，而一块发往某目的卡的连续数据，是被多颗 AIC 分散算出来的（GMM2 的 tile 散在各 AIC 上）。CrossCoreWaitFlag<0x2> 是 CV 配对机制，某颗 AIV 只能确认"和它配对的那颗 AIC 算完了"，确认不了所有 AIC。所以要靠 SyncAll 把所有 AIV 卡在同一条线上——所有 AIV 都过线，就等价于所有 AIC 都算完了，这才敢读 gmC2 往外发。注意这个 SyncAll 是 AIV↔AIV 之间的栅栏，够不到 AIC（AIC 跑的是另一份代码）；跨 AIC↔AIV 的同步只能靠 CrossCoreFlag。

CombineV2 的 Sub-Tile 是什么、为什么小 shape 才用：一个 128×256 的 GMM2 输出 tile，它的 128 行 token 可能来自不同的源卡（combine 是逆操作，行的归属是混的）。所以写回时必须按源卡把这 128 行拆成若干段，每段单独发回自己的卡；又因为每个 token 在目标卡上是 n2=7168 宽的整行、这次只发其中 256 列，所以要带 stride（行间距 = 7168）才能把这个"N行×256列、行距7168"的非连续块一条指令搬过去。这就是 Sub-Tile（"N×256 带 stride"）。

它的价值不在 DMA 大小，而在同步：因为每颗 AIV 只负责它对口那颗 AIC 算的 tile（V2 让 AIV 分活和 AIC 分活对齐），生产者-消费者是固定的一对二关系，所以只需 CV 硬同步，能彻底去掉 SyncAll 全核栅栏。而全核栅栏的空泡恰恰在小 shape/decode 最致命（decode 通信才几十 us，单次 SyncAll 就 2~3us，叠加快慢核）。所以小 shape 用 V2：虽然 sub-tile 让单条 DMA 变小（这是它的副作用，文章也承认），但省下的同步空泡远大于这个损失，净赚。反过来大 shape（prefill）通信/计算本身很长、SyncAll 被摊薄到可忽略，又想要大块连续 DMA 的高带宽，所以用 V1——大 shape 才是"不该用 sub-tile"的场景。

第八章　阶段 H：unpermute —— combine 的后重排
8.1 它要解决什么
combine 把结果写回了源卡，但此时数据还是"按专家排列"的，而且同一个 token 被 topK 个专家各算了一份，散在不同位置。unpermute 要干两件事：按 expandedRowIdx 把数据还原回原始 token 顺序；把同一 token 的 topK 份结果按 router 权重 probs 加权求和，合成最终的每 token 一行输出。


dispatch_ffn_combine_kernel.hpp
Lines 963-972
        AscendC::SyncAll<true>();
        ResetTokenPerExpert(params.EP * paddedExpertNumAligned);
        shmem.CrossRankSync();              // combine 之后唯一一次全卡 barrier
        MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum);
        KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
        kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
        kernelMoeTokenUnpermuteOp.Process();
shmem.CrossRankSync() 是一个全卡 barrier，保证所有卡的 combine 写回都完成后，再统一开始 unpermute——这是整个算子第二次、也是最后一次全卡同步。

8.2 为什么 dispatch 和 combine 的重排不对称
这是一个值得理解的设计对称性：

dispatch 侧：前重排（init_routing 把 token 按专家排好）+ gather 把"通信后重排"折叠进地址映射。dispatch 的本质是 expand（一个 token 复制 topK 份发给不同专家），这是纯地址映射，能完全折叠进通信。
combine 侧：scatter 把"计算后重排"折叠进写回地址 + 一个独立的 unpermute。combine 末尾的 topK 加权求和是 reduce（计算，不是寻址），reduce 没法折叠进通信搬运，所以必须留一个独立的 unpermute 步骤来做。
一句话：permute 能靠通信寻址"免费"完成，reduce 不行——这就是 combine 比 dispatch 多一个独立步骤的根本原因。

第九章　把所有同步点归类
经过前面的拆解，可以把这个算子里所有的同步机制按"作用范围"归成几类。它们粒度从粗到细，各管一摊：

范围	机制	用在哪、解决什么
跨卡·点对点	DataAsFlag（+0x800000）+ gm_signal_wait_until_ne	阶段 B AllGather：数据落地兼当信号，按卡各等各的
跨卡·全局	shmem.CrossRankSync()	阶段 H 前唯一的全卡 barrier
AIC↔AIV	CrossCoreSetFlag/WaitFlag<0x2>	dispatch→GMM1（组级）、GMM↔SwiGLU（SYNCFLAGC2V/V2C）：异构核流水握手
AIV↔AIV	SyncAll<true>	各阶段收口、SwiGLU 每段、CombineV1 全核栅栏
init_routing 内部	VBS/VMS/SortOut 段间 SyncAll	多核归并排序的轮次同步（独立体系）
epilogue UB 流水	SetFlag() 开场预置 / Finalize() 收场排空	防第一拍死锁、防最后一拍脏数据
preload 异步流水	SynchronizeBlock()	GMM 每 block 结束、通知下游前补算欠账
核内 pipe	SetFlag/WaitFlag<HardEvent> ping-pong	dispatch 搬运、各 epilogue/gather 内部双 buffer
值得强调的是：全卡同步只有两次（dispatch 前、unpermute 前）。这正是第〇章说的"两次全卡同步之间，整个算子退化成一个卡内 CV 融合算子"——把昂贵的跨卡同步压到最少，是这个融合算子能跑快的关键。

第十章　一个常被忽略的维度：访存局部性调度
最后单独拎出一类机制。它们既不是"切分"也不是"同步"，而是在数据已经切好、同步已经定好之后，进一步压榨 cache 和带宽的调度策略——之前的总结完全没覆盖：

机制	出处	解决什么
swizzle 遍历	GemmIdentityBlockSwizzle<9,1>	tile 非行优先遍历，让相邻 tile 复用同一 weight 的 L2
L2 Cache Hint	currentM≤128 时 CACHE_MODE_DISABLE	冷门专家 weight 只用一次，禁 L2 防污染
maxOutputSize 截断	GMM1 currentM 截断	workspace 容量上界，超出 drop
icache_preload(8)	GMM1/GMM2 入口	预热指令 cache
AlltoAll 拓扑	host level0:fullmesh;level1:pairwise	机内全连接、机间成对，匹配物理拓扑
第十一章　端到端全景图
┌──────────────────────────── 每张卡（EP rank）────────────────────────────┐
│ 原始 token                                                              │
│                                                                         │
│ A. init_routing 子系统（进 GMM 前的预处理）                              │
│    ├ 快路径 FullLoad（数据小，单核全装下）                               │
│    └ 慢路径 4 stage: sort → count → srcToDst → gather+quant             │
│         其中 sort 大数据走 VBS→VMS→SortOut 多核归并（段间 SyncAll）       │
│    产出: expandedX(int8)→shmem(offsetA), expandedRowIdx, 每专家计数      │
│    ────────────────────────────────────────────────  [SyncAll]         │
│ B. All2All① 通信矩阵 AllGather（按卡切核 + DataAsFlag 点对点等）          │
│    产出: cumsumMM / preSumBeforeRank                  [SyncAll]         │
│ C. All2All② dispatch gather（按源卡切核, cumsumMM 折叠后重排）            │
│    每搬完一个专家 ─SyncAll + SetFlag→ 通知 GMM1（按 group 流水重叠）       │
│    搬运内部 MTE2/MTE3 ping-pong                                          │
│ D. GMM1 升维（每核遍历所有 group；tile swizzle 分核；startCoreIdx 接力）  │
│    · maxOutputSize 截断   · 小 group 禁 L2   · L1 int8 双缓冲 388KB      │
│    · SynchronizeBlock 排空 preload 欠账                                  │
│    每段算完 ─Finalize/SetFlag(SYNCFLAGC2V)→ 通知 SwiGLU                  │
│ E. SwiGLU 两段错峰（epilogueGranularity；SetFlag 预置 / Finalize 排空）   │
│    每段: WaitFlag(C2V)→SyncAll→反量化+门控+重量化→SyncAll→SetFlag(V2C)    │
│    产出: gmPermutedToken                                                 │
│ F. GMM2 降维（同 D 结构, n2=K k2=N/2）─组级 flag→ 通知 combine            │
│ G. combine 写回源卡 shmem(offsetD)                                       │
│    ├ V1 大 shape: 按目的卡全宽连续 DMA + SyncAll 全核栅栏                 │
│    └ V2 小 shape: 16 行 sub-tile 带 stride + 仅 CV 同步（去同步空泡）      │
│    ──────────────────────────────────────────  [CrossRankSync 全卡]    │
│ H. unpermute（expandedRowIdx 还原顺序 + probs 加权 reduce）→ 最终 output  │
└─────────────────────────────────────────────────────────────────────────┘
第十二章　一句话收束
这个算子的全部复杂度，都是为了回答一个问题：在专家散落多卡、token 必须来回搬运的前提下，怎么把一次 MoE FFN 做得既不浪费带宽、又不让计算和通信互相等待。

它的答案是三层叠加的：

空间上，把数据按"卡→核→专家→tile→L1/L0"五层切碎，让每个硬件单位都有活干；
时间上，以"单个专家"为流水单位，让 dispatch、GMM、SwiGLU、combine 错峰重叠，把通信藏到计算之下；
正确性上，用一套分层的同步（跨卡 DataAsFlag/CrossRankSync、异构核 CrossCoreFlag、同核 SyncAll、流水级 SetFlag/Finalize/SynchronizeBlock）保证每一拍数据都在该到的时候到位，且全卡同步只用两次。
而进 GMM 之前那个被低估的 init_routing，本身就是一个"多 stage 流水 + 多核归并排序 + stage 内切分"的完整子系统——它先把混乱的 token 按专家排好、量化好，才让后面这套精密的流水有了干净的输入。

需要的话，我可以把这份详解版落盘成文档（放进仓库 DESIGN.md 旁或 .ai-knowledge），方便后续检索复用。
