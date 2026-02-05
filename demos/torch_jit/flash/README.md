To reproduce the numerical tolerance comparison between the JIT Flash Attention kernel and the reference implementation:

```bash
export PTO_LIB_PATH=...
python fa_compile_and_run.py
```

To instead run benchmarks:
```bash
export PTO_LIB_PATH=...
python fa_benchmark.py
```
NOTE: Modify Sq and Sk accordingly in both ` fa_benchmark.py` and `fa_kernel.cpp`.

All benchmarks were conducted after 50 warm-up iterations, with performance reported as the average over 200 measured iterations to ensure steady-state execution.

### Benchmark Methodology

All benchmarks were conducted after 50 warm-up iterations to reach steady state, with performance metrics reported as the average over 200 measured iterations.

To enable fair comparison across different levels of JIT parallelism, we additionally report a normalized JIT throughput defined as:

- JIT cores = Sq / 128
- Normalized JIT TFLOP/s = (JIT TFLOP/s) * 24 / (JIT cores)

This normalization scales performance to a fixed 24-core equivalent, allowing throughput trends to be compared independent of the number of active JIT cores (where JIT cores = # of tiles = Sq / 128).

| B | Sq  | **JIT cores** | Sk   | H   | FLOPs           | **Torch NPU GEMM attention** time (µs) | **Torch NPU GEMM attention** TFLOP/s | **Fused torch_npu.npu_fused_infer_attention_score** time (µs) | **Fused torch_npu.npu_fused_infer_attention_score** TFLOP/s | **JIT compiled flash kernel** time (µs) | **JIT compiled flash kernel** TFLOP/s | **JIT speedup vs best (time)** | **Normalized JIT TFLOP/s** |
|---|-----|-------------|------|-----|-----------------|----------------------------------------|------------------------------------|-------------------------------------------------------------|------------------------------------------------------------|--------------------------------------|--------------------------------------|------------------------------|---------------------------|
| 1 | 128 | 1 | 1024 | 128 | 67,108,864      | 60.123 | 1.116197 | 68.735 | 0.976345 | 16.268 | **4.125131** | **3.70×** | 99.003 |
| 1 | 256 | 2 | 1024 | 128 | 134,217,728     | 44.078 | 3.045032 | 68.513 | 1.959014 | 16.528 | **8.120529** | **2.67×** | 97.446 |
| 1 | 512 | 4 | 1024 | 128 | 268,435,456     | 60.581 | 4.430981 | 67.892 | 3.953866 | 17.137 | **15.664269** | **3.54×** | 93.986 |
| 1 | 1024| 8 | 1024 | 128 | 536,870,912     | 65.399 | 8.209136 | 70.002 | 7.669387 | 18.915 | **28.382592** | **3.46×** | 85.148 |
| 1 | 128 | 1 | 2048 | 128 | 134,217,728     | 71.658 | 1.873045 | 74.909 | 1.791746 | 23.187 | **5.788415** | **3.09×** | 138.922 |
| 1 | 256 | 2 | 2048 | 128 | 268,435,456     | 63.335 | 4.238336 | 80.121 | 3.350376 | 23.948 | **11.208957** | **2.64×** | 134.507 |
| 1 | 512 | 4 | 2048 | 128 | 536,870,912     | 66.823 | 8.034235 | 73.713 | 7.283279 | 23.880 | **22.482220** | **2.80×** | 134.893 |
| 1 | 1024| 8 | 2048 | 128 | 1,073,741,824   | 63.706 | 16.854563 | 70.119 | 15.313093 | 26.139 | **41.078783** | **2.44×** | 123.236 |
| 1 | 128 | 1 | 4096 | 128 | 268,435,456     | 62.175 | 4.317411 | 71.704 | 3.743656 | 38.098 | **7.045865** | **1.63×** | 169.101 |
| 1 | 256 | 2 | 4096 | 128 | 536,870,912     | 58.798 | 9.130800 | 70.272 | 7.639941 | 38.538 | **13.930769** | **1.53×** | 167.169 |
| 1 | 512 | 4 | 4096 | 128 | 1,073,741,824   | 63.263 | 16.972720 | 71.013 | 15.120398 | 42.146 | **25.476537** | **1.50×** | 152.859 |
| 1 | 1024| 8 | 4096 | 128 | 2,147,483,648   | 60.047 | 35.763259 | 70.864 | 30.304467 | 40.731 | **52.723696** | **1.47×** | 158.171 |
| 1 | 128 | 1 | 8192 | 128 | 536,870,912     | 61.161 | **8.777966** | 67.072 | 8.004385 | 68.175 | 7.874848 | 0.90× | 188.996 |
| 1 | 256 | 2 | 8192 | 128 | 1,073,741,824   | 59.491 | **18.048902** | 68.459 | 15.684520 | 68.574 | 15.658216 | 0.87× | 187.899 |
| 1 | 512 | 4 | 8192 | 128 | 2,147,483,648   | 57.969 | **37.045446** | 72.393 | 29.664325 | 69.252 | 31.009744 | 0.84× | 186.058 |
| 1 | 1024| 8 | 8192 | 128 | 4,294,967,296   | 79.325 | 54.143726 | 90.021 | 47.710464 | 75.708 | **56.730994** | 1.05× | 170.193 |

