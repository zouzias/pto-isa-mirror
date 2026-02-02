import torch
import torch_npu

from jit_util_batch_matrix_square import jit_compile

def test_batch_matrix_square():
    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.float16
    out_dtype = torch.float32
    batch_matrix_square_func = jit_compile(verbose=False)
    for block_dim in range(1,21):
        for matrix_size in [16, 32, 64, 96, 128]:
        # block_dim = 17
        # matrix_size = 32
            x = torch.rand((block_dim, matrix_size, matrix_size), device=device, dtype=dtype)
            z = torch.empty((block_dim, matrix_size, matrix_size), device=device, dtype=out_dtype)

            # Correctness check (single run)
            torch.npu.synchronize()
            batch_matrix_square_func(z, x)
            torch.npu.synchronize()
            z_ref = torch.matmul(x, x)
            torch.npu.synchronize()
            torch.testing.assert_close(z, z_ref.to(torch.float32), rtol=0.05, atol=1e-6)
            print(f"Batch matrix square test pass! block_dim: {block_dim}, matrix_size: {matrix_size} x {matrix_size}")


if __name__ == "__main__":
    
    test_batch_matrix_square()