# HCA — Helper Compression + Indexer + Quantization

The "support path" that produces the inputs `sparse_attn` (CSA) consumes:
compressed KV cache, top-k compressed-cache indices, and the FP8/FP4
quantized activations along the non-rope dimensions.

## Sub-groups

| Sub-group | Reference | Role |
|-----------|-----------|------|
| [Compressor/](Compressor/) | [model.py:280](../../../../../deepseek/model.py) | Gated-pooling KV compression over `compress_ratio` consecutive tokens, with optional overlap |
| [Indexer/](Indexer/) | [model.py:381](../../../../../deepseek/model.py) | Top-k compressed-cache selector — has its **own** Compressor (with Hadamard rotation) for scoring |
| [quant/](quant/) | [kernel.py:41](../../../../../deepseek/kernel.py), [kernel.py:128](../../../../../deepseek/kernel.py) | Block-wise FP8 / FP4 activation quant (inplace fused quant+dequant variant) |

## Wiring (where each sub-group's outputs flow)

```
x  --[Compressor]-->  compressed_kv  -->  kv_cache (tail)
x  --[Indexer]----->  topk_idxs (compressed-cache positions)
kv --[quant]------->  fp8 / fp4 act in-place on the non-rope dims
```

The full Attention path then calls `sparse_attn(q, kv, attn_sink, topk_idxs,
softmax_scale)` (= the CSA family).
