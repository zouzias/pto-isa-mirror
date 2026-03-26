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
- `kernel` — attention implementation (`npu_fused_attention`, `jit_flash`)  
- `time_us` — average execution time in microseconds over 50 iterations  
- `tflops` — achieved throughput for the full attention forward pass  
- `flops_total` — total operation count used to compute TFLOP/s  


## Attention Performance Benchmarks

All benchmarks were run on 910B2 after:

- **10 warm-up iterations**
- **50 timed iterations (average reported)**

The JIT flash kernel parallelizes work across tiles of size 128 along the query dimension.

We define:

- jit_cores = sq / 128

Since larger `sq` launches more parallel JIT cores, raw TFLOP/s naturally increases with `sq`.  
To compare performance independent of parallelism, we normalize throughput to a fixed 24-core equivalent for the 910 B2.

The normalized throughput is computed as:
- normalized_jit_tflops = jit_tflops * 24 / jit_cores

---

#### Results (Batch = 1, Head Size = 128)

**Speedup vs Fused Baseline aarch64**

---

| S0 | S1 | Fused µs | JIT µs | Speedup | JIT TFLOP/s | Normalized JIT TFLOP/s |
|----|----|---------:|-------:|--------:|-----------:|----------------------:|
|128|1024|217.88|99.88|2.18×|0.68|16.31|
|128|2048|222.70|94.57|2.35×|1.44|34.46|
|128|4096|219.19|98.03|2.24×|2.77|66.49|
|128|8192|225.68|100.46|2.25×|5.41|129.76|
|256|1024|223.60|97.94|2.28×|1.39|16.64|
|256|2048|222.50|92.92|2.39×|2.92|35.07|
|256|4096|220.14|92.52|2.38×|5.87|70.45|
|256|8192|223.02|103.86|2.15×|10.46|125.51|
|512|1024|223.10|96.47|2.31×|2.82|16.89|
|512|2048|218.00|90.74|2.40×|5.99|35.92|
|512|4096|219.30|90.30|2.43×|12.03|72.18|
|512|8192|218.54|97.41|2.24×|22.30|133.82|
|1024|1024|218.62|90.46|2.42×|6.00|18.01|
|1024|2048|217.34|91.09|2.39×|11.93|35.78|
|1024|4096|219.67|91.26|2.41×|23.81|71.42|
|1024|8192|216.55|104.93|2.06×|41.41|124.23|
|2048|1024|216.99|91.10|2.38×|11.92|17.89|
|2048|2048|219.29|90.04|2.44×|24.13|36.19|
|2048|4096|218.10|89.91|2.43×|48.33|72.49|
|2048|8192|246.70|130.56|1.89×|66.56|99.85|
