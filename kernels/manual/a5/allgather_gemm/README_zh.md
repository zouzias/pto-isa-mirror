# AllGather GEMM A5 v3 示例

本目录以最新 `kernels/manual/a2a3/allgather_gemm` 为基线，只保留 A5 必需适配，不再额外引入一套独立的脚本和工程壳子。

## 适配原则

- 保持 A3 基线的算法主体、tile 切分、streaming 流水和 PTO ISA 调用
- `run.sh` 保持 A3 风格：生成数据、重新编译、直接运行
- `CMakeLists.txt` 保持 A3 结构，只替换 A5 必需的编译架构、头文件路径和少量链接项
- host 侧只补 A5 必需的 HCCL window guard / window size 检查
- compute kernel 只补 A5 必需的 `pipe_barrier(PIPE_ALL)` 和编译器宏名规避

## 相对 A3 的必要差异

1. Cube / Vec 架构从 `dav-c220-*` 切到 `dav-c310-*`
2. 通信相关头文件从 `tests/npu/a2a3/comm/st/testcase` 切到 `tests/npu/a5/comm/st/testcase`
3. 为兼容当前机器的工具链路径，CMake 增加了 `x86_64-linux` / `aarch64-linux` 根路径探测
4. `main.cpp` 增加 A5 HCCL window guard 和容量检查
5. `allgather_gemm_compute_kernel.cpp` 增加 A5 所需的写回屏障，并规避 `block_idx/block_num` 宏冲突
6. 保留本目录下的 `test_common.h`，用于修正结果比对计数逻辑

## 运行方式

当前 `run.sh` 已回到 A3 风格，不再拆 `build/run/all` 三种模式。

执行前需要先准备环境，例如：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
```

然后执行：

```bash
cd /home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a5/allgather_gemm_v3
bash run.sh -r npu -v Ascend910_950z -n 2

set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
set -u
cd /home/ntlab/zy/code/zhangyuan/pto-isa-zy/kernels/manual/a5/allgather_gemm_v3
rm -rf build
mkdir build
cd build
cmake -DRUN_MODE=npu -DSOC_VERSION=Ascend910_950z -DG_M=2048 -DG_K=2048 -DG_N=1024 ..
make -j16

```

可选 shape 参数与 A3 保持一致：

```bash
bash run.sh -r npu -v Ascend910_950z -n 2 --gm 2048 --gk 2048 --gn 1024
```

## 当前验证状态

- 已在当前非 A5 机器上完成 compile-only 验证
- 由于当前机器不是 A5 运行环境，功能运行仍需在 A5 机器上验证
