# A3 vs A5 源码级特化差异总结

本文只总结 C/C++ 源码里软硬件接口使用差异，不覆盖 include 路径、CMake、环境变量、构建脚本等外围差异。

## 总体结论

A3 到 A5 的差异不是简单的目录迁移或宏替换。

- `dispatch_ffn_combine_v3` 的 A5 特化主要集中在算子内部指令/API 兼容：架构 policy、kernel ABI、PTO 指令可用性、vector barrier、L1/GM 数据搬运。
- `gemm_ar` 的 A5 特化主要集中在分布式通信和 remote GM 可见性：HCCL context ABI、MC2 tiling、remote window 布局、ready queue cache-line 隔离、DDR fence 和跨 rank signal 协议。

## dispatch_ffn_combine_v3

### 1. 架构资源 policy：AtlasA2 vs AtlasA5

A3/A2A3 侧使用 `AtlasA2` 体系：

- `ArchTag = AtlasA2`
- `MmadAtlasA2*`
- `EpilogueAtlasA2*`

A5 侧使用 `AtlasA5` 体系：

- `ArchTag = AtlasA5`
- `MmadAtlasA5*`
- `EpilogueAtlasA5*`

这会影响 L1、L0A、L0B、L0C、UB、fixpipe buffer 规划，以及 matmul preload/fixpipe quant 和 epilogue dequant/swiglu/quant 路径。A5 的资源参数也不同，例如 UB/L0C 更大，fix buffer 和 bias buffer 规划不同，因此不能直接覆盖 A5 policy 文件。

### 2. Kernel 入口 ABI 差异

A3/A2A3 kernel entry 偏 `GM_ADDR` 风格：

```cpp
extern "C" __global__ __aicore__ void dispatch_ffn_combine(
    GM_ADDR x,
    GM_ADDR w1,
    ...)
```

A5 kernel entry 使用 typed GM pointer：

```cpp
extern "C" __global__ __aicore__ void dispatch_ffn_combine(
    __gm__ uint8_t *x,
    __gm__ uint8_t *w1,
    ...)
```

对应 host launch 侧也从统一 reinterpret 为 `GM_ADDR`，变成传入 `static_cast<uint8_t *>`。

### 3. AIC/AIV stage 调度组织差异

A5 保留了单独的 stage sequence 抽象，把 mixed AIC/AIV 执行顺序显式化：

- AIC：`RunGmm1Stage` → `RunGmmInterlockStage` → `RunGmm2Stage`
- AIV：`RunRoutingStage` → `RunDispatchGatherStage` → `RunSwigluStage` → `RunCombineStage` → `RunRestoreStage`

这体现了 A5 工程对 mixed AIC/AIV interlock 和执行顺序的特化，不能只按 A3 分散 stage header 的组织方式整体覆盖。

### 4. Vector cast：`pto::TCVT` vs `AscendC::Cast`

A3/A2A3 路径中通用 vector cast 封装可以倾向走 `pto::TCVT`。

A5 上该路径会暴露 C310/A5 指令支持问题，例如 `TCVT_IMPL` 不可用或底层 cast intrinsic 不匹配。因此 A5 侧需要回退到 AscendC 基础 API：

```cpp
AscendC::Cast(dstTensor, srcTensor, ToAscendRoundMode(mode), cur);
```

同时不能假设 `pto::RoundMode` 和 `AscendC::RoundMode` enum 值一致，需要显式映射。

### 5. Vector barrier/sync：`TSYNC<TROWMAX/TMAX>` vs `PipeBarrier<PIPE_V>`

A3/A2A3 中可见类似：

```cpp
pto::TROWMAX(...);
pto::TSYNC<pto::Op::TROWMAX>();
```

或：

```cpp
pto::TMAX(...);
pto::TSYNC<pto::Op::TMAX>();
```

A5 上 single op `TSYNC` 支持范围更窄，`TROWMAX/TMAX` 这类 V pipe 操作不能照搬。A5 侧改为：

```cpp
AscendC::PipeBarrier<PIPE_V>();
```

裸 `pipe_barrier(PIPE_V)` 在 A5 上也可能触发 intrinsic 参数范围限制，优先使用 AscendC 包装接口。

### 6. L1/GM soft flag 搬运：PTO `TLOAD/TSTORE` vs `AscendC::DataCopy`

A3/A2A3 风格里 soft flag 可以用 PTO tile 搬运：

```cpp
pto::TASSIGN(tile, offset);
pto::TSTORE(dstGlobal, tile);
```

A5 上该路径有两个风险：

1. `TSTORE` 对 source tile type 有限制，source 只能是 Vec/Acc。
2. 即使改成 Vec tile，也可能触发 A5 不支持的 `copy_ubuf_to_gm_align_v2` 目标特性。

因此 A5 侧 soft flag L1/GM 搬运改用 AscendC 显式拷贝：

```cpp
AscendC::LocalTensor<int32_t> srcTensor(
    AscendC::TPosition::A1,
    srcOffset,
    cur);

AscendC::GlobalTensor<int32_t> dstGlobal;
dstGlobal.SetGlobalBuffer(dst);

AscendC::DataCopy(dstGlobal[offset], srcTensor, params);
```

这说明 A5 对某些 raw PTO GM/L1/UB copy 指令组合不是 A3 的直接超集。

### 7. `TPipe` 命名空间差异

A5 侧同时可见 `pto::TPipe` 和 `AscendC::TPipe`，裸写 `TPipe` 会产生歧义。因此 routing、unpermute 等模块里需要显式使用：

```cpp
AscendC::TPipe
```

### 8. HCCL remote window/rank 能力差异

A3/A2A3 remote window rank 上限通常是固定值，例如 32。

A5 使用 `PTO_HCCL_MAX_RANKS`，并保留 64-rank 能力，同时保留 A5 remote window layout offset，例如：

- `offsetPeerPerTokenScale`
- `offsetPeerTokenPerExpert`

这些 offset 影响 dispatch output、per-token scale、expert token count 和 signal 区域布局，不能被 A3 版本直接覆盖。

### 9. Host tiling/platform runtime 差异

A5 host tiling 支持默认平台自动识别：

```cpp
if (cfg.soc_version.empty()) {
    return platform_ascendc::PlatformAscendCManager::GetInstance();
}
return platform_ascendc::PlatformAscendCManager::GetInstance(cfg.soc_version.c_str());
```

这属于 host C++ 源码中的 platform manager 接口差异。A5 侧还增加了 `aclGetRecentErrMsg()` 展开 ACL runtime 错误，便于区分 kernel crash、非法指令、硬件不匹配等问题。

## gemm_ar

### 1. HCCL device context ABI 差异

A3/A2A3 的 `HcclDeviceContext` 主要包含：

- `workSpace`
- `workSpaceSize`
- `rankId`
- `rankNum`
- `winSize`
- `windowsIn[]`
- `windowsOut[]`

A5 的 context 在相同前缀后额外包含：

- `xnAddr`
- `ckeAddr`
- `msAddr`
- `msSize`

这说明 A5 HCCL context 不只是 remote window 地址表，还暴露/依赖 CCU 或寄存器相关尾字段。A3 更偏兼容抽象结构，A5 更贴近官方 A5 comm ST context 前缀布局。

### 2. MC2/HCCL tiling 构造接口差异

A3/A2A3 手写 MC2 tiling struct：

- `Mc2InitTilingInner`
- `Mc2cCTilingInner`
- `Mc2CommConfigV2`

并手动填：

- `version`
- `mc2HcommCnt`
- `commBlockNum`
- `devType`
- `commEngine`
- `groupName`
- `algConfig`

再调用：

```cpp
HcclAllocComResourceByTiling(commHandle, hcclStream, &tiling, &ctxPtr);
```

A5 改为官方 CANN 9.0 风格：

```cpp
AscendC::Mc2CcTilingConfig tilingConfig(group, HCCL_CMD_BATCH_WRITE, "BatchWrite=level0:fullmesh");
tilingConfig.SetCommBlockNum(...);
tilingConfig.SetCommEngine(...);
tilingConfig.GetTiling(tiling.mc2InitTiling);
tilingConfig.GetTiling(tiling.mc2CcTiling);
```

因此这里是 HCCL/MC2 resource tiling 接口层级变化，不是普通结构体字段重命名。

### 3. HCCL context 初始化路径差异

A3/A2A3 初始化逻辑：

1. MESH topology：直接 memcpy HCCL 返回的 context。
2. RING topology：走兼容路径，从 `HcclOpResParam::remoteRes` 手工抽 RDMA window 地址。

A5 初始化逻辑：

1. 优先 `InitDirectPath`，直接按 A5 HCCL context ABI 解码。
2. direct decode 失败后，如果是 MESH，走 mesh-compatible fallback。
3. 非 MESH 则 fallback 到 ring bridge。

这说明 A5 工程优先信任官方 A5 context ABI，兼容 bridge 是兜底路径。

### 4. Ready queue GM 可见性模型差异

这是 `gemm_ar` 最明显的 A5 特化。

A3/A2A3 queue payload 是紧凑 `int32_t data[1]`，enqueue/dequeue 主要依赖：

- `dcci`
- 空 asm barrier
- `pto::comm::TTEST`

A5 把每个 queue slot 扩成一个 64B cache line：

```cpp
struct alignas(64) PerBlockQueueSlot {
    int32_t tile;
    int32_t padding[15];
};
```

并确保 payload 和 `tail/count` metadata 分离到不同 cache line。

A5 enqueue 在发布 `count` 前执行：

```cpp
pipe_barrier(PIPE_ALL);
dsb(DSB_DDR);
```

A5 dequeue 在看到 `count` 后也执行 acquire fence，并最多重读 payload 8 次。

底层逻辑是：A5 上不能只靠 `dcci + TTEST` 假设 payload 和 count 的 GM 可见顺序，需要 cache-line 隔离和 DDR fence 保证“payload 先可见，count 后可见”。

### 5. 通信 kernel：PTO 通信指令一致，但 A5 memory ordering 更强

A3/A5 两边核心 RS/AG 协议都使用 PTO 通信/搬运指令：

- `pto::comm::Signal`
- `TNOTIFY`
- `TWAIT`
- `TTEST`
- `TLOAD`
- `TSTORE_IMPL<..., AtomicAdd>`
- `set_flag` / `wait_flag`

但 A5 在发布 subtile ready 或 summary signal 前增加：

```cpp
pipe_barrier(PIPE_ALL);
dsb(DSB_DDR);
```

A5 还保留了更重的跨 rank `DeviceBarrier`，其中包含：

- rank 内 arrival counter `TNOTIFY/TWAIT`
- 对其他 rank 的 remote `TNOTIFY`
- 等待其他 rank local signal
- DDR fence

所以 A5 不是换了通信算法，而是在同一个 RS/AG 协议上补强 remote GM 写入与 signal 发布之间的 ordering。

### 6. HCCL window 布局差异

A3/A2A3 从 HCCL window 起始位置直接分配：

- `reduced_output`
- `signal_matrix`

A5 先预留 4KB guard：

```cpp
WINDOW_GUARD_BYTES = 4096;
```

然后依次分配：

- `reduced_output_head_pad`
- `reduced_output`
- `signal_matrix`

这说明 A5 不希望 live RS/AG buffer 从 remote window 0 偏移直接开始，可能是为了规避 window 起始区域特殊语义、对齐或硬件访问边界问题。

### 7. `windowsIn/windowsOut` 使用完整性差异

A3/A2A3 RING 兼容路径重点是从 remote resource 中拿到 RDMA 地址，主要服务 `windowsIn` 访问。

A5 direct path 会检查本 rank `windowsIn` 和 `windowsOut`，并在 context 中同时保留二者以及 CCU 尾字段。A5 侧对双向 window 表完整性的要求更强。

### 8. Compute kernel 差异很小

`gemm_ar` 的 GEMM compute kernel 在 A3/A5 上基本仍是同一套 Cube/PTO 路径：

- `TLOAD`
- `TEXTRACT`
- `TMATMUL`
- `TSTORE`

tile 参数和主计算协议基本一致。因此 `gemm_ar` 的 A3/A5 差异重点不在 matmul 指令本身，而在通信资源、remote window、queue 可见性和跨 rank signal ordering。

## 横向对比

| 模块 | A3/A5 差异重心 | A5 特化抓手 |
| --- | --- | --- |
| `dispatch_ffn_combine_v3` | 算子内部 PTO/AscendC 指令兼容 | `AscendC::Cast`、`PipeBarrier<PIPE_V>`、`DataCopy`、`AtlasA5` policy、typed GM ABI |
| `gemm_ar` | 分布式通信和 remote GM 一致性 | A5 HCCL context ABI、官方 `Mc2CcTilingConfig`、64B queue slot、`dsb(DSB_DDR)`、window guard |

## 迁移/同步原则

1. 不能用 A3 文件整文件覆盖 A5 policy、HCCL context、window layout、queue 和通信同步代码。
2. A3 的 PTO 指令路径在 A5 上不一定可用；遇到 `TCVT`、`TSYNC`、`TSTORE` 相关错误时，优先检查 A5 是否需要 AscendC 基础 API 替代。
3. A5 的 remote GM signal 协议要特别关注 memory ordering：payload 写入、`dcci`、`pipe_barrier`、`dsb(DSB_DDR)`、signal publish 的顺序不能随意改。
4. `dispatch_ffn_combine_v3` 同步时重点保护 A5 架构 policy 和 vector/copy API 适配；`gemm_ar` 同步时重点保护 A5 HCCL/MC2 context、queue cache-line 布局和 DDR fence。
