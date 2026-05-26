# moe_combine — A5 PTO MoE Combine Kernel

本项目是 `kernels/manual/a2a3/moe_combine` 的 A5 / Ascend950 翻译版本。host 主流程、fixture 生成、
CPU golden、workspace 语义和 kernel ABI 与 A3 项目保持一致，只保留 A5 必须不同的编译配置、AIV 默认值、
HCCL context 初始化和 HCCL window 头部 padding。

## A5 差异

- CANN 环境默认：`/home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh`。
- 默认 SOC：`Ascend950PR_958b`。
- kernel arch：`dav-c310-vec`。
- 架构宏：`PTO_NPU_ARCH_A5`。
- AIV 默认值：`20 AIC x 2 AIV ratio = 40`；可通过 `--aiv-blocks N` 覆盖。
- HCCL window：live peer-window payload 从 4096-byte head guard 后开始。该 guard 是本项目 layout 显式保留，
  不假设 HCCL 自动跳过。

## 编译

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_combine -B /tmp/moe_combine_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_combine_a5_build --target moe_combine -j8
```

## A5 运行

```bash
cd kernels/manual/a5/moe_combine
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```
