import torch
import torch_npu

from jit_util_flash import jit_compile_flash


def reference_fa(q, k, v, is_causal: bool = False):
    """
    q: [S0, H]
    k: [H, S1]
    v: [S1, H]
    returns: [S0, H]
    """

    # scores: [S0, S1]
    scores = q @ k   # q [S0,H] @ k [H,S1]

    # scale
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = scores * scale

    # causal mask if needed
    if is_causal:
        s0, s1 = scores.shape
        mask = torch.triu(torch.ones((s0, s1), dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    # softmax + output
    p = torch.softmax(scores, dim=-1)           # [S0,S1]
    out = p @ v                                # [S0,H]
    return out.to(torch.float32)


def test_flash_small():
    # case_float_H_128_S0_128_S1_1024
    s0 = 128
    s1 = 1024
    head = 128

    device = "npu"
    dtype = torch.float16
    out_dtype = torch.float32

    # -------------------------
    # Inputs
    # -------------------------
    q = torch.rand((s0, head), device=device, dtype=dtype)          # [S0, HEAD]
    k = torch.rand((head, s1), device=device, dtype=dtype)          # [HEAD, S1]
    v = torch.rand((s1, head), device=device, dtype=dtype)          # [S1, HEAD]

    # -------------------------
    # Outputs
    # -------------------------
    o_out = torch.empty((s0, head), device=device, dtype=out_dtype)                 # [S0, HEAD]
    

    flash = jit_compile_flash(verbose=False)

    # Launch: blockDim must be block_rows (S0/CUBE_S0)
    flash(q, k, v, o_out)

    torch.npu.synchronize()

    print(o_out)

    # Reference check (using k as [HEAD,S1])
    o_ref = reference_fa(q, k, v)
    print(o_ref)
    torch.testing.assert_close(o_out, o_ref, rtol=1e-2, atol=5e-2)
    print("✅ FlashAttention small test passed!")


if __name__ == "__main__":
    test_flash_small()
