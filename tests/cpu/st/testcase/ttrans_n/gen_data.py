import os
from enum import Enum
import numpy as np

class DataFormat(Enum):
    NCHW2NC1HWC0        = 1
    NC1HWC02C1HWN1N0C0  = 2
    GNCHW2GNC1HWC0      = 3
    GNC1HWC02C1HWN1N0C0 = 4


import numpy as np

def golden_NCHW2NC1HWC0(g_info):
    """
    Generates golden data for Mode 1: NCHW -> NC1HWC0
    Maps g_info shapes to [N, H, W, C] and computes C0 from data type.
    """
    # 1. Map g_info shapes to standard NCHW
    # Assuming mapping: shape0=N, shape1=H, shape2=W, shape3=C
    n = g_info.g_shape0
    h = g_info.g_shape1
    w = g_info.g_shape2
    c = g_info.g_shape3
    
    # 2. Calculate C0 based on data type size
    # Assuming 32-byte hardware alignment as discussed
    dtype_size = np.dtype(g_info.data_type).itemsize
    c0 = 32 // dtype_size
    
    # 3. Generate random input for NCHW
    input_arr = np.random.randint(1, 5, size=(n, c, h, w)).astype(g_info.data_type)
    input_arr.tofile("./input.bin")
    
    # 4. Transform to NC1HWC0
    # The transformation function logic:
    # C is split into C1 and C0.
    c1 = (c + c0 - 1) // c0
    
    # Pad channel dimension to be multiple of C0
    padded_c = c1 * c0
    padding = padded_c - c
    if padding > 0:
        # Pad only the C dimension (axis 1)
        input_arr = np.pad(input_arr, ((0, 0), (0, padding), (0, 0), (0, 0)), mode='constant')
        
    # Reshape and Permute
    # Current: [N, Padded_C, H, W] -> Reshape: [N, C1, C0, H, W]
    # Permute to: [N, C1, H, W, C0]
    output_arr = input_arr.reshape(n, c1, c0, h, w).transpose(0, 1, 3, 4, 2)
    
    # 5. Dump to disk
    output_arr.tofile("./golden.bin")
    
    return input_arr, output_arr


def gen_golden_data(g_info):
    """
    Generates aligned runtime raw binaries for C++ unit test validation suites.
    """
    data_type = g_info.data_type
    mode = g_info.shape 
    
    # -------------------------------------------------------------
    # MODE 1: NCHW -> NC1HWC0
    # -------------------------------------------------------------
    if mode == DataFormat.NCHW2NC1HWC0.value:
        golden_NCHW2NC1HWC0(g_info)

    # -------------------------------------------------------------
    # MODE 2: NC1HWC0 -> C1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.NC1HWC02C1HWN1N0C0.value:
        pass

    # -------------------------------------------------------------
    # MODE 3: GNCHW -> GNC1HWC0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNCHW2GNC1HWC0.value:
        pass

    # -------------------------------------------------------------
    # MODE 4: GNC1HWC0 -> GC1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNC1HWC02C1HWN1N0C0.value:
        pass

    else:
        pass


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        g_shape0,
        g_shape1,
        g_shape2,
        g_shape3,
        g_shape4=1,
        g_shape5=1
    ):
        self.case_name = case_name
        self.data_type = data_type
        self.shape = shape
        self.g_shape0 = g_shape0
        self.g_shape1 = g_shape1
        self.g_shape2 = g_shape2
        self.g_shape3 = g_shape3
        self.g_shape4 = g_shape4
        self.g_shape5 = g_shape5


test_cases_registry = [
    TTRANSParams("NCHW2NC1HWC0_1", np.float32, DataFormat.NCHW2NC1HWC0.value, 5, 4, 3, 8),
    TTRANSParams("NCHW2NC1HWC0_2", np.int32, DataFormat.NCHW2NC1HWC0.value, 5, 14, 13, 16),
    TTRANSParams("NCHW2NC1HWC0_3", np.uint16, DataFormat.NCHW2NC1HWC0.value, 1, 11, 13, 16),
    TTRANSParams("NCHW2NC1HWC0_4", np.int32, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7),
    TTRANSParams("NCHW2NC1HWC0_5", np.int8, DataFormat.NCHW2NC1HWC0.value, 4, 32, 3, 7),
]

SUITE_NAME = "TTRANSConvTest"

if __name__ == "__main__":
    print(f"Beginning validation matrix deployment for {len(test_cases_registry)} test scenarios...\n")
    
    for case in test_cases_registry:
        dirname = f"{SUITE_NAME}.{case.case_name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        original_dir = os.getcwd()
        os.chdir(dirname)
        print(f"Running Transformation Mode [{case.shape}] | Configuration Identifier: {case.case_name}")
        gen_golden_data(case)
        os.chdir(original_dir)
        
    print("\nAll binary test files (input.bin, golden.bin) have been generated successfully.")