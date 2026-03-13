# Grouped MoE FFN Stage-1 Operator Example

## Overview

This example adds a PTO-based grouped MoE FFN custom operator and exposes it through `torch_npu` as `torch.ops.npu.pto_moe_grouped_ffn`.

The current baseline intentionally targets the hottest part of routed MoE execution:

- tokens are already packed contiguously per expert
- two grouped expert projections (`gate` and `up`) are executed by the PTO cube kernel
- host side fuses `SiLU(gate) * up` into the returned activation tensor
- Python registers an autograd fallback so backward remains available while only forward uses the PTO kernel

It does not yet fuse the final down projection. That split keeps the first version compact while preserving the dominant grouped GEMM path for later tuning.

## Supported AI Processors

- A2/A3

## Fixed Kernel Profile

The first version is shape-specialized for a common demo profile:

- hidden size: `4096`
- intermediate size: `1920`
- input dtype: `bfloat16`
- output dtype: `float32`
- weight layout: `DN`, stored as `[num_experts, 1920, 4096]`

`x` must be packed by expert already. `group_offsets` is a prefix-sum array of length `num_experts + 1`.

## Performance Direction

The kernel is built around a worklist-driven grouped tiler:

- one launch covers all experts and all output `N` tiles
- each work item computes one `[rows<=64, 192]` output tile
- dynamic `validM` handles ragged expert token counts without padding the full batch
- the cube path keeps the `TLOAD -> TEXTRACT -> TMATMUL -> TSTORE` double-buffered structure from the existing GEMM performance example

This is the right base if the next step is pushing more fusion into-kernel, such as:

- dual-output gate/up accumulation in one launch
- in-kernel `SiLU * up`
- grouped down projection
- better work packing for small expert batches

## Build and Run

```bash
cd demos/baseline/moe_grouped_ffn
python -m venv virEnv
source virEnv/bin/activate
python3 -m pip install -r requirements.txt

export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/bin/setenv.bash
export PTO_LIB_PATH=[YOUR_PATH]/pto-isa

rm -rf build op_extension.egg-info
python3 setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

cd test
python3 test.py

# Explicit backend comparison: ACLNN grouped matmul vs fused PTO path
cd ..
python3 benchmark.py --case aligned_2k_8e --impl aclnn,fused --mode forward --json
```

## Operator Signature

```python
torch.ops.npu.pto_moe_grouped_ffn(x, gate_weight_dn, up_weight_dn, group_offsets) -> Tensor
```

- `x`: `[total_tokens, 4096]`, `bfloat16`
- `gate_weight_dn`: `[num_experts, 1920, 4096]`, `bfloat16`
- `up_weight_dn`: `[num_experts, 1920, 4096]`, `bfloat16`
- `group_offsets`: `[num_experts + 1]`, integer prefix sum
- return: `[total_tokens, 1920]`, `float32`

## Benchmark Implementations

- `aclnn`: default `npu_grouped_matmul` host path
- `custom_split`: PTO gate projection + PTO up projection + host `SiLU * up`
- `fused`: single PTO fused grouped FFN kernel, currently experimental and can stall on hardware
- `eager`: PyTorch eager reference on NPU
