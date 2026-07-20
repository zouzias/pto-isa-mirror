# TPUT_ASYNC Device Baseline

This benchmark mirrors `tget_bandwidth` for the TPUT direction. It posts from the root rank's local symmetric
buffer to the peer rank's remote symmetric buffer, waits for the returned event, and verifies the peer buffer
after every outer iteration.

The timed device region contains only consecutive `TPUT_ASYNC` Post+Wait operations. Host copies and verification
are outside the measured region.

See [README_zh.md](README_zh.md) for build, run, and environment-variable examples.
