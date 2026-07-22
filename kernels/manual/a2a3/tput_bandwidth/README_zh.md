# TPUT_ASYNC Device Baseline

该目录提供与`tget_bandwidth`对称的`TPUT_ASYNC`硬件基准。源数据位于root rank的本地对称内存，
目标数据位于peer rank的远端对称内存；计时范围只包含Kernel内连续的Post+Wait，不包含Host拷贝。

## 构建与运行

前置条件：CANN、MPICH和两张Ascend A2/A3设备。

```bash
source <CANN_INSTALL_PATH>/set_env.sh
bash run.sh -r npu -v a3 -n 2
```

`run.sh`会自动配置MPICH、构建Benchmark并运行默认的`device_baseline`用例。

## 可选配置

只需设置需要覆盖的参数：

```bash
export TPUT_DEVICE_BASELINE_BYTES=131072
export TPUT_DEVICE_BASELINE_BLOCK_DIVISOR=1
export TPUT_DEVICE_BASELINE_QUEUE_NUM=1
export TPUT_DEVICE_BASELINE_POST_COUNT=1
export TPUT_DEVICE_BASELINE_OUTER_ITERS=20
export TPUT_DEVICE_BASELINE_INNER_ITERS=300

bash run.sh -r npu -v a3 -n 2
```

`BLOCK_DIVISOR=1`表示`block_bytes=total_bytes`，即每次Post只生成一个data SQE。

## 计时与校验

计时范围只包含Device侧连续的`TPUT_ASYNC` Post+Wait。每个outer迭代都会刷新源数据和目标哨兵值；
Host拷贝及peer buffer校验均位于计时区间之外。
