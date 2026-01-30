import torch
import torch_npu
from jit_util_gemm import jit_compile_gemm

gemm = jit_compile_gemm()
dtype = torch.float16

a = torch.rand((6144,6144), device="npu", dtype=dtype)
b = torch.rand((6144,6144), device="npu", dtype=dtype)
c = torch.empty((6144,6144), device="npu", dtype=torch.float32)

gemm(c, a, b)
torch.npu.synchronize()

ref = a.float() @ b.float()
print(c)
print(ref)
torch.testing.assert_close(c, ref, rtol=2e-1, atol=2e-1)

print("GEMM OK")

