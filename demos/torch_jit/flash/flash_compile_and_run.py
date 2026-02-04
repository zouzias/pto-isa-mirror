import torch
import torch_npu
from jit_util_flash import jit_compile_flash


def reference_fa(q, k_s1h, v, scale: bool = False):
    scores = q @ k_s1h.transpose(0, 1)
    if scale:
        scores = scores * (1.0 / (q.shape[-1] ** 0.5))
    p = torch.softmax(scores, dim=-1)
    return (p @ v).to(torch.float32)


def test_flash_small():
    s0, s1, head = 128, 1024, 128
    num_tiles = s1 // 128

    device = "npu:0"
    torch.npu.set_device(device)
    dtype = torch.float16
    out_dtype = torch.float32

    torch.npu.synchronize()

    # Inputs
    q = torch.rand((s0, head), device=device, dtype=dtype)
    k = torch.rand((s1, head), device=device, dtype=dtype)
    v = torch.rand((s1, head), device=device, dtype=dtype)

    # Outputs / workspaces
    o_out = torch.empty((s0, head), device=device, dtype=out_dtype)

    outDevice = torch.empty((s0, s1), device=device, dtype=torch.float32)
    xexpDevice = torch.empty((s0, s1), device=device, dtype=torch.float16)
    pOutFp32Device = torch.empty((s0, s1), device=device, dtype=torch.float32)

    out2Device = torch.empty((num_tiles, s0, head), device=device, dtype=torch.float32)
    gSumDevice = torch.empty((num_tiles, s0), device=device, dtype=torch.float32)
    expMaxDevice = torch.empty((num_tiles, s0), device=device, dtype=torch.float32)
    oPartsDevice = torch.empty((num_tiles, s0, head), device=device, dtype=torch.float32)

    flash = jit_compile_flash(verbose=False)

    # ==========================
    # Time FlashAttention kernel
    # ==========================
    start_flash = torch.npu.Event(enable_timing=True)
    end_flash = torch.npu.Event(enable_timing=True)

    torch.npu.synchronize()
    start_flash.record()

    flash(
        q,
        k,
        v,
        o_out,
        outDevice,
        xexpDevice,
        pOutFp32Device,
        out2Device,
        gSumDevice,
        expMaxDevice,
        oPartsDevice,
    )

    end_flash.record()
    torch.npu.synchronize()
    flash_ms = start_flash.elapsed_time(end_flash)

    # ==========================
    # Time reference implementation
    # ==========================
    start_ref = torch.npu.Event(enable_timing=True)
    end_ref = torch.npu.Event(enable_timing=True)

    torch.npu.synchronize()
    start_ref.record()

    o_ref = reference_fa(q, k, v, scale=True)

    end_ref.record()
    torch.npu.synchronize()
    ref_ms = start_ref.elapsed_time(end_ref)

    # ==========================
    # Results
    # ==========================
    print(f"FlashAttention kernel time : {flash_ms:.3f} ms")
    print(f"Reference FA time         : {ref_ms:.3f} ms")
    print(f"Speedup                  : {ref_ms / flash_ms:.2f}×")

    print(o_out)
    print(o_ref)

    torch.testing.assert_close(o_out, o_ref, rtol=0.1, atol=5e-2)
    print("✅ FlashAttention small test passed!")


if __name__ == "__main__":
    test_flash_small()
