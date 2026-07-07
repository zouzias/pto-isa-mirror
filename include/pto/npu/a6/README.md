PTO NPU A6 (Ascend960 / dav-920r1) Implementation Directory

This directory will contain A6-specific NPU implementations when available.
Currently, A6 shares the PTO instruction implementations with A5 where applicable.
The CCE architecture name for A6 is `dav-920r1` with variants:
- `dav-920r1-vec` for vector operations
- `dav-920r1-cube` for cube operations

The __NPU_ARCH__ macro value for A6 is 9201, which defines PTO_NPU_ARCH_A6.
