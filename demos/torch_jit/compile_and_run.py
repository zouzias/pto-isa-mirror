import torch
import torch_npu

from jit_util import jit_compile


def test_add():
    device = "npu"
    dtype = torch.float16

    shape = [20, 2048]
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    add_func = jit_compile("add_custom.cpp")
    add_func(x, y, z)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)
    print("test pass!")

if __name__ == "__main__":
    test_add()
