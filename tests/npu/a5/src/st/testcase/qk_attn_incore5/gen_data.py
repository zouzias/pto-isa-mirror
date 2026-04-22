import numpy as np

np.random.seed(42)


def to_bf16(arr_f32):
    """Convert float32 array to BF16 (top 2 bytes of float32 representation)."""
    raw = arr_f32.astype(np.float32).view(np.uint32)
    bf16 = (raw >> 16).astype(np.uint16)
    return bf16


def bf16_to_f32(bf16_arr):
    """Reconstruct float32 from BF16 uint16 array."""
    return (bf16_arr.astype(np.uint32) << 16).view(np.float32)


# Generate random Q [16, 128] and K [64, 128] in float32
Q_f32 = np.random.uniform(-1, 1, [16, 128]).astype(np.float32)
K_f32 = np.random.uniform(-1, 1, [64, 128]).astype(np.float32)

# Convert to BF16
Q_bf16 = to_bf16(Q_f32)  # [16, 128] uint16
K_bf16 = to_bf16(K_f32)  # [64, 128] uint16

# K in DN layout = row-major K[64,128] bf16 stored as-is
K_bf16.tofile("k_gm.bin")
Q_bf16.tofile("q_gm.bin")

# Golden: compute (Q @ K^T) * attn_scale using BF16-precision inputs
Q_back = bf16_to_f32(Q_bf16).reshape(16, 128)
K_back = bf16_to_f32(K_bf16).reshape(64, 128)

attn_scale = 0.0883883461
golden = (Q_back @ K_back.T) * attn_scale  # [16, 64]

golden.astype(np.float32).tofile("golden.bin")
print(f"Generated: k_gm.bin ({K_bf16.nbytes} bytes), q_gm.bin ({Q_bf16.nbytes} bytes), golden.bin ({golden.nbytes} bytes)")
print(f"Golden shape: {golden.shape}, range: [{golden.min():.4f}, {golden.max():.4f}]")
