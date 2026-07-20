# TPUT_ASYNC Device Baseline

该目录提供与`tget_bandwidth`对称的`TPUT_ASYNC`硬件基准。源数据位于root rank的本地对称内存，
目标数据位于peer rank的远端对称内存；计时范围只包含Kernel内连续的Post+Wait，不包含Host拷贝。

## 构建

```bash
source /mnt/data/ntlab/ltr/cann/cann/set_env.sh
cmake -S . -B build -DRUN_MODE=npu -DSOC_VERSION=Ascend910B1
cmake --build build -j16
```

## 运行

```bash
export PATH=/home/mpich/bin:$PATH
export MPI_LIB_PATH=/home/mpich/lib/libmpi.so
export LD_LIBRARY_PATH=/home/mpich/lib:${ASCEND_HOME_PATH}/lib64

export TPUT_BENCH_MODE=device_baseline
export TPUT_DEVICE_BASELINE_FIRST_DEVICE_ID=5
export TPUT_DEVICE_BASELINE_BYTES=131072
export TPUT_DEVICE_BASELINE_BLOCK_DIVISOR=1
export TPUT_DEVICE_BASELINE_QUEUE_NUM=1
export TPUT_DEVICE_BASELINE_POST_COUNT=1
export TPUT_DEVICE_BASELINE_BLOCK_NUM=1
export TPUT_DEVICE_BASELINE_OUTER_WARMUP=2
export TPUT_DEVICE_BASELINE_OUTER_ITERS=20
export TPUT_DEVICE_BASELINE_INNER_WARMUP=50
export TPUT_DEVICE_BASELINE_INNER_ITERS=300

mpirun -n 2 ./build/tput_bandwidth
```

`BLOCK_DIVISOR=1`表示`block_bytes=total_bytes`，即每次Post只生成一个data SQE。测试会在每个outer
迭代刷新源数据和目标哨兵值，并在Kernel结束后校验peer rank的目标对称内存。
