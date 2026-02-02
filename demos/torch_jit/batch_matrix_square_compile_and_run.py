import torch
import torch_npu

from jit_util_batch_matrix_square import jit_compile

def test_batch_matrix_square(
    block_dim: int, matrix_size: int
):

    device = "npu:0"
    dtype = torch.float16
    out_dtype = torch.float32

    x = torch.rand((block_dim, matrix_size, matrix_size), device=device, dtype=dtype)
    z = torch.empty((block_dim, matrix_size, matrix_size), device=device, dtype=out_dtype)

    batch_matrix_square_func = jit_compile(verbose=False)

    # Correctness check (single run)
    batch_matrix_square_func(z, x)
    torch.npu.synchronize()
    z_ref = torch.matmul(x, x)
    torch.npu.synchronize()
    torch.testing.assert_close(z, z_ref.to(torch.float32), rtol=0.1, atol=1e-6)
    print("Batch matrix square test pass!")


if __name__ == "__main__":
    for block_dim in range(1,21):
        for matrix_size in [128]:
            test_batch_matrix_square(i, j)