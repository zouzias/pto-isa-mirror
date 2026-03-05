import os
import numpy as np

ROWS = 2048
COLS = 128
SCALE = 1.0 / np.sqrt(COLS)

INPUT_DIR = "input_data"
GOLDEN_DIR = "golden_data"
IN_NAME = "in.data"
OUT_NAME = "out.data"

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_x = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_x

def gen():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    # 1. Generate Input Data
    low, high = -10, 10
    
    q = np.random.uniform(low, high, size=(ROWS, COLS)).astype(np.float16)
    k = np.random.uniform(low, high, size=(ROWS, COLS)).astype(np.float16)
    v = np.random.uniform(low, high, size=(ROWS, COLS)).astype(np.float16)

    # 2. Transpose K
    k_transposed = k.T.copy()

    # ---------------------------------------------------------
    # GOLDEN CALCULATION
    # ---------------------------------------------------------
    q_f32 = q.astype(np.float32)
    k_f32 = k.astype(np.float32)

    # Step A: Matmul 1 (Q * K.T)
    scores = np.matmul(q_f32, k_f32.T)
    scores = scores * SCALE
    
    # Step B: Softmax
    probs = softmax(scores, axis=-1)
    
    # Step C: Matmul 2 (Probs * V)
    probs_half = probs.astype(np.float16).astype(np.float32)
    v_half = v.astype(np.float32)
    out_golden = np.matmul(probs_half, v_half).astype(np.float32)
    # ---------------------------------------------------------

    # 3. Save Inputs
    all_inputs = np.concatenate((q.flatten(), k_transposed.flatten(), v.flatten()))
    
    in_path = os.path.join(INPUT_DIR, IN_NAME)
    with open(in_path, "wb") as f:
        all_inputs.tofile(f)

    # 4. Save Golden Output
    out_path = os.path.join(GOLDEN_DIR, OUT_NAME)
    with open(out_path, "wb") as f:
        out_golden.tofile(f)
        
    print(f"Saved inputs at input_data.")

if __name__ == "__main__":
    gen()