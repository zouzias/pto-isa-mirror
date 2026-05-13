import os
import numpy as np

np.random.seed(31)

def gen_golden_data(kT, kH, kF, kE, kTileM):
    X = np.random.randint(-10,10, size=(kT, kH)).astype(np.float16) / 3
    W_router = np.random.randint(-10,10, size=(kH, kE)).astype(np.float16) / 3
    W1 = np.random.randint(-10,10, size=(kE, kH, kF)).astype(np.float16) / 3
    W2 = np.random.randint(-10,10, size=(kE, kF, kH)).astype(np.float16) / 3

    logits = (X @ W_router).astype(np.float32)
    expert_id = np.argmax(logits, axis=1).astype(np.uint32)
    

    Z = np.zeros((kT, kH), dtype=np.float32)
    for t in range(kT):
        Y = X[t] @ W1[expert_id[t]]
        Y = np.maximum(Y, 0.).astype(np.float16)
        Z[t] = Y @ W2[expert_id[t]]
    
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    X.tofile("./input/input_X.bin")
    W_router.tofile("./input/input_W_router.bin")
    W1.tofile("./input/input_W1.bin")
    W2.tofile("./input/input_W2.bin")

    logits.tofile("./output/golden_logits.bin")
    expert_id.tofile("./output/golden_expert_id.bin")
    Z.tofile("./output/golden_Z.bin")


if __name__ == "__main__":
    kT = 256 # num tokens
    kH = 64  # hidden size
    kF = 64  # FFN intermediate size 
    kE = 16  # experts
    kTileM = 128 #expert microtile rows
    gen_golden_data(kT, kH, kF, kE, kTileM)