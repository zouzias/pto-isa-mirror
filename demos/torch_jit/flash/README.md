To reproduce the numerical tolerance comparison between the JIT Flash Attention kernel and the reference implementation:

```bash
export PTO_LIB_PATH=...
python fa_compile_and_run.py
```

To instead run benchmark:
```bash
export PTO_LIB_PATH=...
python fa_benchmark.py
```

Which produces a CSV file containing one row per `(sq, sk, kernel)` configuration with the following fields:

- `sq`, `sk` — query and key sequence lengths  
- `head_size` — attention head dimension (fixed at 128)  
- `kernel` — attention implementation (`gemm_ref`, `npu_fused_attention`, `jit_flash`)  
- `time_us` — average execution time in microseconds over 200 iterations  
- `tflops` — achieved throughput for the full attention forward pass  
- `flops_total` — total operation count used to compute TFLOP/s  


## Attention Performance Benchmarks

All benchmarks were run on 910B2 after:

- **50 warm-up iterations**
- **200 timed iterations (average reported)**

The JIT flash kernel parallelizes work across tiles of size 128 along the query dimension.

We define:

- jit_cores = sq / 128

Since larger `sq` launches more parallel JIT cores, raw TFLOP/s naturally increases with `sq`.  
To compare performance independent of parallelism, we normalize throughput to a fixed 24-core equivalent for the 910 B2.

The normalized throughput is computed as:
- normalized_jit_tflops = jit_tflops * 24 / jit_cores

### Kernels

| Kernel | Description |
|-------|------------|
| `gemm_ref(qk_softmax_pv)` | PyTorch attention via GEMMs |
| `npu_fused_attention` | NPU fused attention kernel |
| `jit_flash` | JIT compiled fa_kernel |

---

## Results (Batch = 1, Head Size = 128)

**Speedup vs Best Baseline = min time between GEMM and fused**

---

### Sq = 128

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|96.07|65.51|12.68|5.17×|5.35|128.48|
|2048|93.04|64.74|12.43|5.21×|10.92|262.09|
|4096|95.04|65.13|12.29|5.30×|22.10|530.45|
|8192|88.10|62.17|12.47|4.99×|43.55|1045.27|
|16384|95.64|66.41|12.62|5.26×|86.05|2065.26|

---

### Sq = 256

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|90.12|65.23|12.44|5.24×|10.91|130.96|
|2048|90.28|65.46|12.71|5.15×|21.37|256.42|
|4096|89.54|66.02|12.26|5.38×|44.31|531.72|
|8192|90.06|67.08|12.23|5.48×|88.81|1065.76|
|16384|97.27|67.84|12.17|5.58×|178.50|2141.98|

---

### Sq = 512

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|89.87|66.04|12.45|5.31×|21.81|131.00|
|2048|90.15|65.51|12.50|5.24×|43.46|260.78|
|4096|89.27|67.41|12.48|5.40×|87.03|522.18|
|8192|92.39|71.43|12.49|5.72×|173.90|1043.38|
|16384|125.44|90.39|12.46|7.26×|348.78|2092.69|

---

### Sq = 1024

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|91.11|65.56|12.49|5.25×|43.48|130.44|
|2048|89.21|64.88|12.54|5.18×|86.65|259.95|
|4096|90.96|70.50|12.49|5.64×|173.91|521.74|
|8192|91.11|87.40|12.43|7.03×|349.56|1048.68|
|16384|198.20|125.84|12.45|10.11×|697.99|2093.97|

---

### Sq = 2048

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|89.98|64.68|12.50|5.17×|86.93|130.40|
|2048|91.90|70.27|12.47|5.64×|174.20|261.30|
|4096|111.89|88.81|12.48|7.12×|348.20|522.30|
|8192|149.13|125.47|12.52|10.02×|693.99|1040.99|
|16384|339.21|176.51|12.43|14.20×|1397.89|2096.84|

---

### Sq = 3072

| Sk | GEMM µs | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized TFLOP/s |
|----|--------|---------|-------|---------|------------|--------------------|
|1024|91.56|66.79|12.46|5.36×|130.77|130.77|
|2048|93.88|90.69|12.58|7.21×|259.13|259.13|
|4096|112.89|113.19|12.47|9.06×|522.77|522.77|
|8192|203.73|153.18|12.50|12.25×|1042.69|1042.69|
|16384|491.17|221.25|12.39|17.85×|2104.82|2104.82|
