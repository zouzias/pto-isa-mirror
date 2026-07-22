# TPUT_ASYNC Device Baseline

This benchmark mirrors `tget_bandwidth` for the TPUT direction. It posts from the root rank's local symmetric
buffer to the peer rank's remote symmetric buffer, waits for the returned event, and verifies the peer buffer
after every outer iteration.

## Build and Run

Prerequisites: CANN, MPICH, and two Ascend A2/A3 devices.

```bash
source <CANN_INSTALL_PATH>/set_env.sh
bash run.sh -r npu -v a3 -n 2
```

`run.sh` configures MPICH, builds the benchmark, and runs the default `device_baseline` case.

## Optional Configuration

Set only the parameters that need to differ from their defaults:

```bash
export TPUT_DEVICE_BASELINE_BYTES=131072
export TPUT_DEVICE_BASELINE_BLOCK_DIVISOR=1
export TPUT_DEVICE_BASELINE_QUEUE_NUM=1
export TPUT_DEVICE_BASELINE_POST_COUNT=1
export TPUT_DEVICE_BASELINE_OUTER_ITERS=20
export TPUT_DEVICE_BASELINE_INNER_ITERS=300

bash run.sh -r npu -v a3 -n 2
```

`BLOCK_DIVISOR=1` means `block_bytes=total_bytes`, so each Post contains one data SQE.

## Timing and Verification

The timed device region contains only consecutive `TPUT_ASYNC` Post+Wait operations. Every outer iteration
refreshes the source and destination sentinel values. Host copies and peer-buffer verification run outside the
timed region.
