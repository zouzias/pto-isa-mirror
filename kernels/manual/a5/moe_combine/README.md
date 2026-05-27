# moe_combine — A5 PTO MoE Combine Kernel

This project is the A5 / Ascend950 translation of `kernels/manual/a2a3/moe_combine`.
The host flow, fixture generation, CPU golden, workspace semantics, and kernel ABI stay aligned with the A3 project.
Only A5-specific build settings, AIV defaults, HCCL context bootstrap, and HCCL window head padding differ.

## A5 Differences

- CANN environment: defaults to `/home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh`.
- Default SOC: `Ascend950PR_958b`.
- Kernel arch: `dav-c310-vec`.
- Architecture macro: `PTO_NPU_ARCH_A5`.
- Default AIV blocks: `40` from `20 AIC x 2 AIV ratio`; pass `--aiv-blocks N` to override.
- HCCL window: live peer-window payload starts after a 4096-byte head guard. The guard is owned by this project layout;
  do not assume HCCL automatically skips it.

## Build

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_combine -B /tmp/moe_combine_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_combine_a5_build --target moe_combine -j8
```

## Run On A5

```bash
cd kernels/manual/a5/moe_combine
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```
