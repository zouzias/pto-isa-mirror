# moe_token_permute_register 调试记录

记录 `moe_token_permute_register` ST 用例从「运行挂起」到「PASSED」过程中定位到的问题、根因与解决方案。修改集中在各用例目录内的独立副本：

- `moe_token_permute_chevron/moe_token_permute_body.hpp`、`moe_token_permute_common.hpp`
- `moe_token_permute_register/moe_token_permute_body.hpp`、`moe_token_permute_common.hpp`

固定规模参数：`num_tokens=16, topK=4, num_experts=4, actual_cores=16, chunk_size=4, hidden_size=8, out_len=64, dtype=float16, mix 1:2`。

---

## 问题总览

| # | 现象 | 根因 | 解决方案 |
|---|------|------|----------|
| 0 | 所有 syncall/register 用例集体挂起 | **环境问题**（device 0 异常占用） | 环境恢复后参考用例自动通过，非代码问题 |
| 1 | SYNCALL 处死锁挂起 | kernel 错误地解引用 `fftsAddr` | 用指针值本身作 ffts 基址，不解引用 |
| 2 | workspace 跨核数据丢失 | 相邻核共享 32B cache line，写入与 dcci 互相覆盖 | 每核 histogram 独占一条 cache line（stride=8 int32） |
| 3 | sio / perm 输出为垃圾值 | `copy_ubuf_to_gm` 32B 粒度溢写 | 改用 `*_align_b32 / *_align_b16` 精确字节写 |
| 4 | 直方图/偏移错乱（最隐蔽） | MTE2→标量、标量→MTE3 缺同步屏障 | 在标量↔DMA 边界补 `pipe_barrier(PIPE_ALL)` |

---

## 问题 0：环境导致的集体挂起（非代码问题）

### 现象
`moe_token_permute_register`、以及官方参考 `SYNCALLTest.case_aiv_only/mix_1_1/mix_1_2` 全部超时挂起。

### 定位
- 单核计算用例 `tadd`（float/int32/int16）能通过，说明 NPU 基本通路正常。
- 但**所有**依赖 SYNCALL（FFTS 跨核同步）的用例都挂，连最简单的 `case_aiv_only_all_blocks` 也挂 → 不是某个用例的代码 bug。
- 清理残留进程、确认 `npu-smi` 无进程占用、环境恢复后，三个 syncall 参考用例全部 PASS。

### 结论
集体挂起是 **device 异常占用 / 环境状态**导致，与本用例代码无关。排查 SYNCALL 类问题时应**先用官方参考用例确认环境**，再判断是否为自身代码问题。

---

## 问题 1：fftsAddr 解引用导致 SYNCALL 死锁

### 现象
环境正常后，register 用例仍挂死在 SYNCALL barrier。

### 根因
register 路径通过 `rtKernelLaunchWithHandleV2` 启动，`args[0]` 传入的是 `rtGetC2cCtrlAddr` 返回的 **ffts 地址值本身**。kernel 形参为 `__gm__ uint64_t *fftsAddrGm`，原实现写成：

```cpp
set_ffts_base_addr(*fftsAddrGm);   // 错误：把地址值当 GM 指针再解引用，读到垃圾
```

ffts 基址被设成垃圾值，FFTS 跨核同步无法工作，所有核挂在 barrier。

### 解决
对齐 syncall 参考实现（`syncall_mix_common.hpp`），直接用指针值作基址，**不解引用**：

```cpp
set_ffts_base_addr(reinterpret_cast<uint64_t>(fftsAddrGm));
```

AIC（`RunMoeTokenPermuteAicOnly`）与 AIV（`RunMoeTokenPermuteAivBody`）两处都需修正。

---

## 问题 2：workspace 相邻核共享 cache line 导致跨核写丢失

### 现象
不再挂起后，`workspace` 直方图出现整段丢核（如某核读回 `[0,0,0,0]`），后续 offsets/sio 全错。

### 根因
原布局每核 histogram 仅占 `num_experts=4` 个 int32 = **16 字节**，但：
- `copy_ubuf_to_gm` 一个 burst 写 **32 字节**；
- 相邻核 slot 间距仅 16 字节 → 写区间重叠；
- 跨核 `dcci`（按 32B cache line 回写/失效）会把同一 cache line 内邻核的数据一起覆盖。

### 定位证据
用 `syncall_tilelang_repro` 跑 6 个 variant，结论唯一：

| variant | 每核 stride | 结果 |
|---|---|---|
| baseline / sync_mix / aiv_only | 4 (16B) | **FAILED** |
| **fix_cacheline / fix_cacheline_sync_mix** | **8 (32B)** | **PASSED** |

即只有 cache-line 隔离有效；sync 类型（Mix/AIVOnly）无关。

### 解决
每核 histogram 独占一条 32B cache line：

```cpp
// moe_token_permute_common.hpp
constexpr int32_t kMoeWsSlotStride = 8;                       // 每核 32B（8 int32）
constexpr int32_t kMoeWsTotal = kMoeNumCores * kMoeWsSlotStride;
```

- Phase1 写入 `workspaceGm + cid * kMoeWsSlotStride`，并在写后补 `pipe_barrier + dcci + dsb`（写侧可见性）；
- Phase2 按 `wsUb[c * kMoeWsSlotStride + e]` 索引读回。

---

## 问题 3：输出 32B 粒度溢写（sio / perm）

### 现象
sio 读回为大随机数（如 49209）；perm 大量行不匹配。

### 根因
`copy_ubuf_to_gm` 以 32B 为粒度写：
- sio 每核只拥有 `chunk_size=4` 个 int32（16B），32B 写会溢出到邻核区域并写入未初始化的 UB 尾部垃圾；
- perm 每个 vid 只写 `HALF_H=4` 个 half（8B），32B 写会跨行覆盖。

### 解决
改用按字节精确写的 align 变体（`lenBurst` 单位为字节）：

```cpp
// sio：精确写 16 字节
copy_ubuf_to_gm_align_b32(sioOutGm + myStart, sioUb, 0, 1, kMoeChunkSize * sizeof(int32_t), 0, 0, 0, 0);

// perm：精确写 8 字节（half）
copy_ubuf_to_gm_align_b16(permOutGm + wpUb[0] * kMoeHiddenSize + hOff, rowUb, 0, 1, kMoeHalfH * sizeof(half), 0, 0, 0, 0);
```

---

## 问题 4：MTE2↔标量↔MTE3 缺同步，直方图/偏移错乱（最隐蔽）

### 现象
修了上述问题后，`core0` 的 sio 完全正确，但 `core1` 起逐核出现小偏差（非确定性）。

### 定位
- 用 golden python（同 seed）算出各核 `indices/hist/offsets/sio`，与 device 对比。
- 反推发现：`core0` 之所以"恰好正确"，是因为它的 offset 只用 `running`（**全体求和**），对 slot 顺序/错乱不敏感；而 `core1+` 用前缀和 `cpre`（只累加 `c<cid`），对每核数据正确性敏感。
- 在 Phase1 把 `cid / idxUb[0..2]` dump 进 slot padding，确认 **`idxUb` 内容本身就是乱的**：`cid=0` 载入后读到的不是 `indices[0:4]`。

### 根因
设备侧直方图、偏移、sio 均为**标量运算**读写 UB，但同步原语 `set_flag(MTE2,V)/(V,MTE3)` 只在 MTE2 / Vector / MTE3 之间排序，**不覆盖标量管线（PIPE_S）**：

- MTE2 把 `idxUb`/`wsUb` 载入后，标量在 DMA 完成前就读 → 读到陈旧/垃圾数据；
- 标量写完 `sioUb` 后，MTE3 在标量完成前就拷出 → 拷到垃圾。

### 解决
在所有「标量 ↔ DMA」边界补 `pipe_barrier(PIPE_ALL)`：

- **Phase1**：`idxUb` 载入后（MTE2→标量）、直方图算完后（标量→MTE3 写 workspace）；
- **Phase2**：`wsUb` 载入后（MTE2→标量）、`sioUb` 算完后（标量→MTE3 拷出）。

> 经验：用 raw intrinsics 手写 kernel 时，`set_flag/wait_flag` 不覆盖标量管线；凡是「DMA 载入 → 标量读」「标量写 → DMA 拷出」都需显式 `pipe_barrier(PIPE_ALL)`（或 MTE2_S / S_MTE3 事件）。TileLang 生成代码由框架自动插入这些屏障，手工移植时极易遗漏。

---

## 验证结果

```
[ RUN      ] MoeTokenPermuteRegisterTest.case_fp16_standard
[       OK ] MoeTokenPermuteRegisterTest.case_fp16_standard
[  PASSED  ] 1 test.
```

复现命令：

```bash
source /mnt/data/ntlab/liulei/set_env_new.sh
cd code/pto-isa-main
python3 tests/script/build_st.py -r npu -v a3 -t moe_token_permute_register
python3 tests/script/run_st.py -r npu -v a3 -t moe_token_permute_register \
    -g MoeTokenPermuteRegisterTest.case_fp16_standard
```

## 备注
- 上述设备侧改动位于共享的 `moe_token_permute_body.hpp`，对 chevron 路径同为正确性增强。
- chevron 路径另受其已知限制影响：`-xcce` 单文件 mix 编译缺少 `usedCrossCoreSync`（`F_TYPE_CROSS_CORE_SYNC`）meta，SYNCALL 不生效，属独立问题，不在本次修复范围内。
