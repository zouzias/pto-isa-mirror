# moe_token_permute_chevron 用 MIX 编译跑通的根因分析

目标：让 `moe_token_permute_chevron`（chevron `<<<>>>` 启动 + `pto_mix_st` 单文件 mix 编译）在 NPU 上 PASS，方式对齐生成的 ascendc 版本（其用的是 **SyncAll MIX**）。

> 背景：tilelang 同事基于同一前端算子（`moe_token_permute`）生成 ascendc 与 pto 两份后端实现。ascendc 版能过，pto 版（chevron 启动）失败。仓库内 `moe_token_permute_register`（handle 启动）已 PASS，`moe_token_permute_chevron`（chevron 启动）未过。

---

> **重要修正（见第七节路线2实测）**：第一版假设"缺 `F_TYPE_CROSS_CORE_SYNC` TLV"是根因——**该假设已被实测推翻**。ascendc 与 PTO 自动生成的 mix meta 都**没有** cross-core TLV。真正根因见下方"结论先行（修正版）"与第七节。

## 一、结论先行（修正版）

- `dav-c220` mix arch 下，bisheng 会把单个 `__global__`（如 `launch_kernel`）**自动拆成 `<name>_mix_aic` / `<name>_mix_aiv` 两个 device 符号**，并自动打上 `KTYPE=MIX_AIC_MAIN(4) + RATION(1:2)` 的 `.ascend.meta.<name>_mix_aic/_mix_aiv`。这与 ascendc 生成的 mix kernel meta **完全一致**。
- ascendc 版能过：它就是一个**真 MIX 1:2 kernel**——AIC 分支参与 `SyncAll`，AIV 分支 `cid=GetBlockIdx()/2` 做计算，单条 chevron `<<<16>>>`。
- pto chevron 版失败的真正原因（两条叠加）：
  1. `PTO_SYNCALL_AIV_KERNEL_META(launch_kernel)` 额外注入了一条**冲突 meta** `.ascend.meta.launch_kernel = MIX_AIV_MAIN(5)/ratio 0:1`，与编译器自动发的 `MIX_AIC(4)/1:2` 抵触；
  2. body 是**纯 AIV-only 写法**（`SYNCALL<AIVOnly>`、`cid=get_block_idx()` 未 `/2`、无 AIC 分支），与"被当 MIX 1:2 调度"不自洽。
- 对照 `RunSyncAll_mix_aiv`（PASS）：用纯 `dav-c220-vec` arch，编译器**不做 mix 拆分**，AIV-only 写法自洽 → 过。
- 因此 **cross-core TLV 不是关键**；要用 MIX 跑通 chevron，应让它成为**真 MIX kernel**（对齐 ascendc）：去掉冲突的 AIV meta、body 改 MIX 结构（AIC 参与 `SYNCALL<Mix>`、AIV `cid=block_idx/2`）、保持 `pto_mix_st`+chevron。

---

## 二、CANN 侧 ground truth

定义在 `/usr/local/Ascend/cann/aarch64-linux/asc/impl/basic_api/utils/kernel_utils_macros.h`：

```c
enum FuncMetaType {            // 函数级 TLV 类型
    F_TYPE_KTYPE = 1,             // kernel type
    F_TYPE_CROSS_CORE_SYNC = 2,   // cross core sync   <-- PTO 缺这个
    F_TYPE_MIX_TASK_RATION = 3,   // MIX core ratio
    ...
};

struct FunMetaKType        { BaseTlv head; unsigned int  ktype; };               // type=1
struct FunMetaCrossCoreType{ BaseTlv head; unsigned int  usedCrossCoreSync; };   // type=2
struct FunMetaMixCoreType  { BaseTlv head; unsigned short taskRation0, taskRation1; }; // type=3

struct FunLevelCrossCoreType { FunMetaKType ktypeMeta; FunMetaCrossCoreType crossCoreType; };
struct FunLevelMixCoreType   { FunMetaKType ktypeMeta; FunMetaMixCoreType  mixCoreType;   };

enum CrossCoreSyncType { C_TYPE_USE_SYNC = 1, C_TYPE_MAX };  // usedCrossCoreSync 取值

enum KernelType {                  // F_TYPE_KTYPE 的取值
    K_TYPE_MIX_AIC_MAIN = 4,       // v220 mix cube/vector
    K_TYPE_MIX_AIV_MAIN = 5,       // v220 mix vector/cube
    ...
};

// KERNEL_TASK_TYPE_DEFAULT 本体只是 feature 探测占位，真正发 .ascend.meta 由 AscendC 编译流水线完成
#define KERNEL_TASK_TYPE_DEFAULT(value)  ENABLE_FEATURE_FOR_COMPILE(default, value)
```

PTO 现状（`include/pto/common/kernel_meta.hpp`）：

```cpp
struct PtoMetaFunLevelMixCoreType {
    PtoMetaKType    ktypeMeta;     // F_TYPE_KTYPE
    PtoMetaMixCoreType mixCoreType; // F_TYPE_MIX_TASK_RATION
};   // <-- 没有 FunMetaCrossCoreType / F_TYPE_CROSS_CORE_SYNC
```

---

## 三、四案例对照（为什么有的过有的不过）

| 案例 | 编译 arch | meta | 启动 | 结果 | 原因 |
|---|---|---|---|---|---|
| `syncall` 参考 `RunSyncAll_mix_aiv` | `dav-c220-vec`（纯 AIV，`pto_vec_st`） | AIV ktype，无 cross-core | chevron | PASS | 纯 vec arch + AIV ktype，FFTS 按 AIV-only 建立，不需 cross-core TLV |
| ascendc moe（生成） | `dav-c220`（mix） | `KERNEL_TYPE_MIX_AIC_1_2` + **cross-core TLV**（编译器自动）+ AIC/AIV 双分支 | chevron | PASS | mix arch 下 runtime 据 cross-core TLV 建 FFTS |
| pto register moe | mix（双 ELF 经 `make_mix_register_elf.py` 拼） | MIX ktype + ration | **handle**（`rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2`） | PASS | handle 路径据 MIX ktype 建 FFTS，不依赖 cross-core TLV |
| **pto chevron moe** | `dav-c220`（mix） | **AIV ktype，无 cross-core TLV**，仅 `#if __DAV_VEC__` 单分支 | chevron | **FAIL** | mix arch + chevron 路径需要 cross-core TLV，PTO 没发 → FFTS 没建 → SYNCALL 失效 → 跨核直方图/偏移读到垃圾 |

关键差分：
- `RunSyncAll_mix_aiv`（PASS）与 pto chevron（FAIL）都没 cross-core TLV，差别在 **arch（vec vs mix）**。
- ascendc（PASS）与 pto chevron（FAIL）都是 mix arch + chevron，差别在 **是否带 cross-core TLV + 是否 MIX 结构**。

→ mix-arch chevron 这条路，cross-core TLV 是必需的。

---

## 四、生成 pto 代码（lwdpto）的两处问题

1. 没有发 meta：tilelang `codegen_ascend_pto.cc` 里有 `used_mix_syncall_` → 发 `PTO_SYNCALL_MIX_AIC_KERNEL_META(launch_kernel,1,ratio)`（注释自述 "mirror AscendC 的 KERNEL_TASK_TYPE_DEFAULT"），但本算子前端 SyncAll 未标 `Mix`，该分支未触发。
2. 没有 AIC 分支：kernel body 仅 `#if defined(__DAV_C220_VEC__)`，AIC 核不参与 SYNCALL → 即便声明 MIX 也会因 AIC 不到齐而死锁。

注：即便上面两点修好（发 MIX ktype/ration + 补 AIC 分支），由于 PTO meta 宏本身不发 `F_TYPE_CROSS_CORE_SYNC`，mix-arch chevron 路径仍可能建不起 FFTS——这是根因层面的缺口。

---

## 五、路线规划（修正后）

- **路线 2（已完成，见第七节）**：编译 chevron + 最小 ascendc 探针，解析 `.ascend.meta.*` 段。结论：cross-core TLV 不是差异点；差异是"AIV-only meta/body" vs "真 MIX kernel"。
- **路线 1（实施，修正版）**：把 chevron 改成真 MIX kernel，对齐 ascendc：
  1. 去掉 `PTO_SYNCALL_AIV_KERNEL_META(launch_kernel)`（消除冲突 meta），让 bisheng 自动发 `_mix_aic/_mix_aiv` 的 MIX 1:2 meta（或显式 `PTO_SYNCALL_MIX_AIC_KERNEL_META`，按实测取舍）；
  2. body 改 MIX：`#if __DAV_CUBE__` AIC 分支调与 AIV 等量的 `SYNCALL<Mix>`；`#if __DAV_VEC__` AIV 分支 `cid=get_block_idx()/2`、`vid=get_subblockid()`、`SYNCALL<Mix>` + 计算；
  3. 保持 `pto_mix_st`(`dav-c220`) + chevron `<<<kMoeNumCores>>>` 启动；NPU 实测。
- **路线 3（对照/兜底）**：纯 `dav-c220-vec`(`pto_vec_st`) + `_mix_aiv` 命名 + chevron，照搬 `RunSyncAll_mix_aiv`，能过但非 MIX。
- **路线 4（兜底）**：`SYNCALL<Soft>` 软件模式（PTO `kernels/manual/a2a3/moe_dispatch` 即此法），不依赖 FFTS/meta，最稳但非硬件 MIX。

## 七、路线2 实测结论（推翻 cross-core 假设）

用 `readelf`/自写解析脚本 dump 了两个编译产物的 `.ascend.meta.*` 段。

### chevron kernel（`moe_token_permute_chevron_kernel.cpp.o`，`dav-c220` mix）

外层对象与嵌套 device ELF 中共出现三个 meta 段：

| section | TLV 解码 | 来源 |
|---|---|---|
| `.ascend.meta.launch_kernel` | `KTYPE=5(MIX_AIV_MAIN)` + `RATION(0:1)` | **PTO 宏 `PTO_SYNCALL_AIV_KERNEL_META`** |
| `.ascend.meta.launch_kernel_mix_aic` | `KTYPE=4(MIX_AIC_MAIN)` + `RATION(1:2)` | **bisheng 自动** |
| `.ascend.meta.launch_kernel_mix_aiv` | `KTYPE=4(MIX_AIC_MAIN)` + `RATION(1:2)` | **bisheng 自动** |

device 符号被自动拆为 `launch_kernel_mix_aic` / `launch_kernel_mix_aiv`。

### 最小 ascendc 探针（`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `SyncAll()`）

嵌套 device ELF 中仅两个 meta 段，**均无 cross-core TLV**：

| section | TLV 解码 |
|---|---|
| `.ascend.meta.probe_mix_syncall_mix_aic` | `KTYPE=4(MIX_AIC_MAIN)` + `RATION(1:2)` |
| `.ascend.meta.probe_mix_syncall_mix_aiv` | `KTYPE=4(MIX_AIC_MAIN)` + `RATION(1:2)` |

### 关键发现

- ascendc 的 mix meta（KTYPE=4/1:2）= chevron 编译器自动发的 `_mix_aic/_mix_aiv` meta，**二者完全相同，且都没有 `F_TYPE_CROSS_CORE_SYNC`** → cross-core TLV 假设被推翻。
- 唯一多出来的是 PTO 宏注入的 `.ascend.meta.launch_kernel`（AIV-only 0:1），与编译器自动 meta 冲突。
- 叠加 chevron body 的 AIV-only 写法（`SYNCALL<AIVOnly>`、`cid` 未 `/2`），与 MIX 1:2 调度不自洽 → 失败根因。
- `RunSyncAll_mix_aiv` 用 `dav-c220-vec`，编译器不拆 mix，故 AIV-only 自洽能过。

→ 修正后的 route 1 不再补 cross-core TLV，而是把 chevron 改造成真 MIX kernel（对齐 ascendc）。

## 八、路线1 实测结论（已 PASS）

最终改动（三处）：

1. `moe_token_permute_chevron_kernel.cpp`：删除 `PTO_SYNCALL_AIV_KERNEL_META(launch_kernel)`，消除与编译器自动 meta 的冲突，依赖 bisheng 自动生成的 `launch_kernel_mix_aic/_mix_aiv`（`MIX_AIC_MAIN, 1:2`）。
2. `moe_token_permute_body.hpp`：`RunAivSyncBarrier()`（`SYNCALL<AIVOnly>`）→ `RunMixSyncBarrier()`（`SYNCALL<SyncCoreType::Mix>`，AIC/AIV 均调用）；barrier 移到 `#if __DAV_VEC__` 之外，使 AIC 分支也参与（两侧各 2 次，数量一致）。
3. 编译/启动不变：`pto_mix_st`(`dav-c220`) + `launch_kernel<<<kMoeNumCores>>>`。

关键修正点 —— **AIV 的 `cid` 不要 `/2`**：

- 先按 ascendc 写法用 `cid = get_block_idx() / 2`，结果数值系统性减半（sio 约为期望一半、仅低半数核有直方图）→ 说明本 runtime 在 `<<<16>>>` MIX 1:2 下，AIV 的 `get_block_idx()` 已是逻辑核号 `0..N-1`，`vid = get_subblockid()`(0/1) 分担 hidden。
- 改回 `cid = get_block_idx()`（不 `/2`）后：`max diff=0, err count=0`，**PASS**。
- 注：ascendc 生成代码对 AIV 做了 `cid/=2`，与本 runtime 实测的 block_idx 语义不同（可能与 ascendc 的 blockdim/hidden 切分约定有关）；以本仓 `RunSyncAll_mix_aiv` 等既有用例的语义为准，chevron MIX 下用 `get_block_idx()` 直接作 cid。

复现：

```bash
source /mnt/data/ntlab/liulei/set_env_new.sh
cd /mnt/data/ntlab/liulei/code/pto-isa-main
python3 tests/script/build_st.py -r npu -v a3 -t moe_token_permute_chevron
python3 tests/script/run_st.py -r npu -v a3 -t moe_token_permute_chevron \
    -g MoeTokenPermuteChevronTest.case_fp16_standard
# => [  PASSED  ] 1 test.
```

---

## 六、复现命令

```bash
source /mnt/data/ntlab/liulei/set_env_new.sh
cd /mnt/data/ntlab/liulei/code/pto-isa-main

# 编译（route 2 只需编译产物，不需上设备）
python3 tests/script/build_st.py -r npu -v a3 -t moe_token_permute_chevron

# 运行（route 1 验证）
python3 tests/script/run_st.py -r npu -v a3 -t moe_token_permute_chevron \
    -g MoeTokenPermuteChevronTest.case_fp16_standard
```
