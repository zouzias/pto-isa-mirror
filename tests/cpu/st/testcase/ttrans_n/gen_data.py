from enum import Enum
import numpy as np

class DataFormat(Enum):
    NCHW2NC1HWC0        = 1
    NC1HWC02C1HWN1N0C0  = 2
    GNCHW2GNC1HWC0      = 3
    GNC1HWC02C1HWN1N0C0 = 4


def nchw_to_nc1hwc0(tensor: np.ndarray, c0: int) -> np.ndarray:
    """
    Mode 1: [N, C, H, W] -> [N, C1, H, W, C0]
    Pads channels to a multiple of C0, splits C, and blocks into hardware vectors.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (NCHW), got {tensor.ndim}D")
        
    n, c, h, w = tensor.shape
    c1 = (c + c0 - 1) // c0
    padded_c = c1 * c0
    
    # 1. Pad the C dimension if unaligned
    if padded_c > c:
        tensor = np.pad(tensor, ((0,0), (0, padded_c - c), (0,0), (0,0)), mode='constant')
        
    # 2. Move C to the fastest-moving axis: (N, H, W, C)
    tensor_nhwc = np.transpose(tensor, axes=(0, 2, 3, 1))
    
    # 3. Split C safely: (N, H, W, C1, C0)
    tensor_split = tensor_nhwc.reshape(n, h, w, c1, c0)
    
    # 4. Permute to target: (N, C1, H, W, C0)
    return np.transpose(tensor_split, axes=(0, 3, 1, 2, 4))


def nc1hwc0_to_c1hwn1n0c0(tensor: np.ndarray, n0: int) -> np.ndarray:
    """
    Mode 2: [N, C1, H, W, C0] -> [C1, H, W, N1, N0, C0]
    Pads batch size N to a multiple of N0, splits N, and re-orders for weight/backprop matrices.
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor (NC1HWC0), got {tensor.ndim}D")
        
    n, c1, h, w, c0 = tensor.shape
    n1 = (n + n0 - 1) // n0
    padded_n = n1 * n0
    
    # 1. Pad the N dimension if unaligned
    if padded_n > n:
        tensor = np.pad(tensor, ((0, padded_n - n), (0,0), (0,0), (0,0), (0,0)), mode='constant')
        
    # 2. N is already the slowest axis (axis 0). Since we split N, it won't corrupt 
    #    inner dimensions, but to be completely safe and clean, we reshape directly:
    tensor_split = tensor.reshape(n1, n0, c1, h, w, c0)
    
    # 3. Permute to target: C1(2), H(3), W(4), N1(0), N0(1), C0(5)
    return np.transpose(tensor_split, axes=(2, 3, 4, 0, 1, 5))


def gnchw_to_gnc1hwc0(tensor: np.ndarray, c0: int) -> np.ndarray:
    """
    Mode 3: [G, N, C, H, W] -> [G, N, C1, H, W, C0]
    Grouped variant of NCHW2NC1HWC0.
    """
    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor (GNCHW), got {tensor.ndim}D")
        
    g, n, c, h, w = tensor.shape
    c1 = (c + c0 - 1) // c0
    padded_c = c1 * c0
    
    # 1. Pad the C dimension
    if padded_c > c:
        tensor = np.pad(tensor, ((0,0), (0,0), (0, padded_c - c), (0,0), (0,0)), mode='constant')
        
    # 2. Move C to the fastest-moving axis: (G, N, H, W, C)
    tensor_gnhwc = np.transpose(tensor, axes=(0, 1, 3, 4, 2))
    
    # 3. Split C: (G, N, H, W, C1, C0)
    tensor_split = tensor_gnhwc.reshape(g, n, h, w, c1, c0)
    
    # 4. Permute to target: (G, N, C1, H, W, C0)
    return np.transpose(tensor_split, axes=(0, 1, 4, 2, 3, 5))


def gnc1hwc0_to_c1hwn1n0c0(tensor: np.ndarray, n0: int) -> np.ndarray:
    """
    Mode 4: [G, N, C1, H, W, C0] -> [G, C1, H, W, N1, N0, C0]
    Grouped variant of NC1HWC02C1HWN1N0C0.
    """
    if tensor.ndim != 6:
        raise ValueError(f"Expected 6D tensor (GNC1HWC0), got {tensor.ndim}D")
        
    g, n, c1, h, w, c0 = tensor.shape
    n1 = (n + n0 - 1) // n0
    padded_n = n1 * n0
    
    # 1. Pad the N dimension
    if padded_n > n:
        tensor = np.pad(tensor, ((0,0), (0, padded_n - n), (0,0), (0,0), (0,0), (0,0)), mode='constant')
        
    # 2. Move N to a localized outer position before splitting: (G, N, C1, H, W, C0)
    #    Since N is at axis 1 and G is axis 0, reshaping directly works if shapes align perfectly, 
    #    but splitting explicitly ensures memory tracking matches:
    tensor_split = tensor.reshape(g, n1, n0, c1, h, w, c0)
    
    # 3. Permute to target: G(0), C1(3), H(4), W(5), N1(1), N0(2), C0(6)
    return np.transpose(tensor_split, axes=(0, 3, 4, 5, 1, 2, 6))


def gen_golden_data(g_info):
    """
    Generates aligned runtime raw binaries for C++ unit test validation suites.
    """
    data_type = g_info.data_type
    mode = g_info.shape  # Enum integer flag mapped from test parameters
    
    # -------------------------------------------------------------
    # MODE 1: NCHW -> NC1HWC0
    # -------------------------------------------------------------
    if mode == DataFormat.NCHW2NC1HWC0.value:
        c0 = g_info.g_shape4  # Hardware alignment block width (e.g. 16 or 32)
        
        # Initialize unpadded random base matching true global footprint
        src_n = g_info.g_whole_shape0 if hasattr(g_info, 'g_whole_shape0') else 1
        src_c = g_info.g_whole_shape2
        src_h = g_info.g_whole_shape3
        src_w = g_info.g_whole_shape4
        
        input_raw = np.random.randint(1, 5, size=(src_n, src_c, src_h, src_w)).astype(data_type)
        
        # Calculate full runtime tile dimension bounds requested by execution context
        dst_n = g_info.g_shape0
        dst_c1 = g_info.g_shape1
        dst_h = g_info.g_shape2
        dst_w = g_info.g_shape3
        dst_c = dst_c1 * c0
        
        # Clip or track valid windows to avoid downstream array mismatch bounds
        valid_n = min(input_raw.shape[0], dst_n)
        valid_c = min(input_raw.shape[1], dst_c)
        valid_h = min(input_raw.shape[2], dst_h)
        valid_w = min(input_raw.shape[3], dst_w)
        
        cropped_tensor = input_raw[0:valid_n, 0:valid_c, 0:valid_h, 0:valid_w]
        
        # Full structural padding up to explicit execution grid configurations
        input_arr = np.pad(cropped_tensor, (
            (0, dst_n - valid_n),
            (0, dst_c - valid_c),
            (0, dst_h - valid_h),
            (0, dst_w - valid_w)
        ), mode='constant', constant_values=0)
        
        # Transform the padded matrix using our clean method
        output_arr = nchw_to_nc1hwc0(input_arr, c0)

    # -------------------------------------------------------------
    # MODE 2: NC1HWC0 -> C1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.NC1HWC02C1HWN1N0C0.value:
        n0 = g_info.g_shape4  # Batch block factor matching hardware loop strides
        n1 = g_info.g_shape3  # Structural outer loops parameter
        
        src_n  = g_info.g_whole_shape0
        src_c1 = g_info.g_whole_shape1
        src_h  = g_info.g_whole_shape2
        src_w  = g_info.g_whole_shape3
        src_c0 = g_info.g_whole_shape4
        
        input_raw = np.random.randint(1, 5, size=(src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
        
        # Explicitly pad the outer N dimension to fit into complete structural vector tiles
        target_n = n1 * n0
        if target_n > src_n:
            input_arr = np.pad(input_raw, ((0, target_n - src_n), (0,0), (0,0), (0,0), (0,0)), mode='constant')
        else:
            input_arr = input_raw[0:target_n, :, :, :, :]
            
        output_arr = nc1hwc0_to_c1hwn1n0c0(input_arr, n0)

    # -------------------------------------------------------------
    # MODE 3: GNCHW -> GNC1HWC0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNCHW2GNC1HWC0.value:
        c0 = g_info.g_shape4
        g  = g_info.g_shape6
        
        src_n = g_info.g_whole_shape1
        src_c = g_info.g_whole_shape2
        src_h = g_info.g_whole_shape3
        src_w = g_info.g_whole_shape4
        
        input_raw = np.random.randint(1, 5, size=(g, src_n, src_c, src_h, src_w)).astype(data_type)
        
        dst_n = g_info.g_shape0
        dst_c1 = g_info.g_shape1
        dst_h = g_info.g_shape2
        dst_w = g_info.g_shape3
        dst_c = dst_c1 * c0
        
        valid_g = g
        valid_n = min(input_raw.shape[1], dst_n)
        valid_c = min(input_raw.shape[2], dst_c)
        valid_h = min(input_raw.shape[3], dst_h)
        valid_w = min(input_raw.shape[4], dst_w)
        
        cropped = input_raw[:, 0:valid_n, 0:valid_c, 0:valid_h, 0:valid_w]
        input_arr = np.pad(cropped, (
            (0, 0),
            (0, dst_n - valid_n),
            (0, dst_c - valid_c),
            (0, dst_h - valid_h),
            (0, dst_w - valid_w)
        ), mode='constant')
        
        output_arr = gnchw_to_gnc1hwc0(input_arr, c0)

    # -------------------------------------------------------------
    # MODE 4: GNC1HWC0 -> GC1HWN1N0C0
    # -------------------------------------------------------------
    elif mode == DataFormat.GNC1HWC02C1HWN1N0C0.value:
        n0 = g_info.g_shape4
        n1 = g_info.g_shape3
        g  = g_info.g_whole_shape5
        
        src_n  = g_info.g_whole_shape0
        src_c1 = g_info.g_whole_shape1
        src_h  = g_info.g_whole_shape2
        src_w  = g_info.g_whole_shape3
        src_c0 = g_info.g_whole_shape4
        
        input_raw = np.random.randint(1, 5, size=(g, src_n, src_c1, src_h, src_w, src_c0)).astype(data_type)
        target_n = n1 * n0
        
        if target_n > src_n:
            input_arr = np.pad(input_raw, ((0,0), (0, target_n - src_n), (0,0), (0,0), (0,0), (0,0)), mode='constant')
        else:
            input_arr = input_raw[:, 0:target_n, :, :, :, :]
            
        output_arr = gnc1hwc0_to_c1hwn1n0c0(input_arr, n0)

    # -------------------------------------------------------------
    # FALLBACK MODE (Identity/Bypass Initialization)
    # -------------------------------------------------------------
    else:
        dst_h = g_info.g_shape3
        dst_w = g_info.g_shape4
        input_arr = np.random.randint(1, 5, [dst_h, dst_w]).astype(data_type)
        output_arr = np.zeros([dst_h, dst_w]).astype(data_type)

    # Dump binary blobs directly to disk sequentially flat
    input_arr.tofile("./input.bin")
    output_arr.tofile("./golden.bin")


class TTRANSParams:
    def __init__(
        self,
        case_name,
        data_type,
        shape,
        g_shape0,         # Dst/Tile N (or Src N for Mode 2/4)
        g_shape1,         # Dst/Tile C1
        g_shape2,         # Dst/Tile H
        g_shape3,         # Dst/Tile W (or N1 factor for Mode 2/4)
        g_shape4,         # C0 block width (or N0 factor for Mode 2/4)
        g_shape5,         # Unused fallback slot
        g_whole_shape0,   # Src Whole N
        g_whole_shape1,   # Src Whole C1 (or Whole N for Mode 3)
        g_whole_shape2,   # Src Whole C (or Whole C for Mode 3)
        g_whole_shape3,   # Src Whole H
        g_whole_shape4,   # Src Whole W
        g_shape6=1,       # Group Count (G) for Mode 3
        g_whole_shape5=1, # Group Count (G) for Mode 4
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
        self.g_whole_shape0 = g_whole_shape0
        self.g_whole_shape1 = g_whole_shape1
        self.g_whole_shape2 = g_whole_shape2
        self.g_whole_shape3 = g_whole_shape3
        self.g_whole_shape4 = g_whole_shape4
        self.g_shape6 = g_shape6
        self.g_whole_shape5 = g_whole_shape5


test_cases_registry = [
    # =========================================================================
    # MODE 1: NCHW2NC1HWC0 (6 Cases)
    # =========================================================================
    # Case 1.1: Standard float16 layout, perfectly aligned C (32 channels, C0=16)
    TTRANSParams(
        case_name="Mode1_FP16_Aligned",
        data_type=np.float16,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=1, g_shape1=2, g_shape2=14, g_shape3=14, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=32, g_whole_shape3=14, g_whole_shape4=14
    ),
    # Case 1.2: Fixed alignment error from earlier (uint16, unaligned C=26, C0=16, heavy padding)
    TTRANSParams(
        case_name="Mode1_UInt16_Unaligned_Fixed",
        data_type=np.uint16,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=1, g_shape1=2, g_shape2=2, g_shape3=16, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=26, g_whole_shape3=2, g_whole_shape4=16
    ),
    # Case 1.3: int8 data type with tight hardware configuration pack (C0=32, unaligned C=3)
    TTRANSParams(
        case_name="Mode1_Int8_RGB_Input",
        data_type=np.int8,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=4, g_shape1=1, g_shape2=224, g_shape3=224, g_shape4=32, g_shape5=1,
        g_whole_shape0=4, g_whole_shape1=1, g_whole_shape2=3, g_whole_shape3=224, g_whole_shape4=224
    ),
    # Case 1.4: Multi-batch execution sequence stress testing boundary dimensions
    TTRANSParams(
        case_name="Mode1_FP32_MultiBatch",
        data_type=np.float32,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=16, g_shape1=4, g_shape2=7, g_shape3=7, g_shape4=8, g_shape5=1,
        g_whole_shape0=16, g_whole_shape1=1, g_whole_shape2=32, g_whole_shape3=7, g_whole_shape4=7
    ),
    # Case 1.5: Micro boundary validation case (1x1x1x1 unit matrix block sizes)
    TTRANSParams(
        case_name="Mode1_Micro_Boundary",
        data_type=np.float16,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=1, g_shape1=1, g_shape2=1, g_shape3=1, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=1, g_whole_shape3=1, g_whole_shape4=1
    ),
    # Case 1.6: Massive deep network enterprise layer configuration block (Large Channels)
    TTRANSParams(
        case_name="Mode1_Large_Enterprise_Layer",
        data_type=np.float16,
        shape=DataFormat.NCHW2NC1HWC0.value,
        g_shape0=2, g_shape1=64, g_shape2=56, g_shape3=56, g_shape4=16, g_shape5=1,
        g_whole_shape0=2, g_whole_shape1=1, g_whole_shape2=1000, g_whole_shape3=56, g_whole_shape4=56
    ),

    # =========================================================================
    # MODE 2: NC1HWC02C1HWN1N0C0 (6 Cases)
    # =========================================================================
    # Case 2.1: Perfectly aligned matrix breakdown (N=16 split into N1=2, N0=8)
    TTRANSParams(
        case_name="Mode2_Aligned_Split",
        data_type=np.float16,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=2, g_shape4=8, g_shape5=1,
        g_whole_shape0=16, g_whole_shape1=4, g_whole_shape2=14, g_whole_shape3=14, g_whole_shape4=16
    ),
    # Case 2.2: Unaligned batch size forcing automatic destination padding (N=13 padded to 16)
    TTRANSParams(
        case_name="Mode2_Unaligned_Batch_Padding",
        data_type=np.float16,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=2, g_shape4=8, g_shape5=1,
        g_whole_shape0=13, g_whole_shape1=2, g_whole_shape2=7, g_whole_shape3=7, g_whole_shape4=16
    ),
    # Case 2.3: Single batch allocation layout (N1=1, N0=1, unpadded N=1 tracking)
    TTRANSParams(
        case_name="Mode2_Single_Batch_Static",
        data_type=np.float32,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=1, g_shape4=1, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=8, g_whole_shape2=28, g_whole_shape3=28, g_whole_shape4=8
    ),
    # Case 2.4: Int8 quantization optimized configurations (C0=32 vector tracking blocks)
    TTRANSParams(
        case_name="Mode2_Quantized_Int8_Blocks",
        data_type=np.int8,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=4, g_shape4=16, g_shape5=1,
        g_whole_shape0=64, g_whole_shape1=16, g_whole_shape2=10, g_whole_shape3=10, g_whole_shape4=32
    ),
    # Case 2.5: Micro dimension boundary pass testing minimal footprint states
    TTRANSParams(
        case_name="Mode2_Micro_Boundary",
        data_type=np.float16,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=1, g_shape4=2, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=1, g_whole_shape3=1, g_whole_shape4=16
    ),
    # Case 2.6: Massive heavy pipeline scale footprint check (High Channel count depth)
    TTRANSParams(
        case_name="Mode2_Heavy_Industrial_Scale",
        data_type=np.float16,
        shape=DataFormat.NC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=4, g_shape4=8, g_shape5=1,
        g_whole_shape0=32, g_whole_shape1=32, g_whole_shape2=32, g_whole_shape3=32, g_whole_shape4=16
    ),

    # =========================================================================
    # MODE 3: GNCHW2GNC1HWC0 (6 Cases)
    # =========================================================================
    # Case 3.1: Standard Grouped Convolution mode (G=2, perfectly aligned channels)
    TTRANSParams(
        case_name="Mode3_Grouped_Standard",
        data_type=np.float16,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=1, g_shape1=2, g_shape2=14, g_shape3=14, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=32, g_whole_shape3=14, g_whole_shape4=14,
        g_shape6=2
    ),
    # Case 3.2: High group count with unaligned channel width structures (G=8, C=11)
    TTRANSParams(
        case_name="Mode3_HighGroup_Unaligned_Channels",
        data_type=np.float16,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=2, g_shape1=1, g_shape2=28, g_shape3=28, g_shape4=16, g_shape5=1,
        g_whole_shape0=2, g_whole_shape1=2, g_whole_shape2=11, g_whole_shape3=28, g_whole_shape4=28,
        g_shape6=8
    ),
    # Case 3.3: Int8 structural vector testing tracking inside multi-group pipelines
    TTRANSParams(
        case_name="Mode3_Quantized_Int8_Groups",
        data_type=np.int8,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=1, g_shape1=2, g_shape2=40, g_shape3=40, g_shape4=32, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=45, g_whole_shape3=40, g_whole_shape4=40,
        g_shape6=4
    ),
    # Case 3.4: Deep Depthwise Convolution simulation profile (G=32, C=1 layout tracking)
    TTRANSParams(
        case_name="Mode3_Depthwise_Simulation",
        data_type=np.float16,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=1, g_shape1=1, g_shape2=112, g_shape3=112, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=1, g_whole_shape3=112, g_whole_shape4=112,
        g_shape6=32
    ),
    # Case 3.5: Minimal boundary unit execution tracking matrix blocks
    TTRANSParams(
        case_name="Mode3_Micro_Boundary",
        data_type=np.float16,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=1, g_shape1=1, g_shape2=1, g_shape3=1, g_shape4=16, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=1, g_whole_shape3=1, g_whole_shape4=1,
        g_shape6=1
    ),
    # Case 3.6: Massive layout workload verification check (High Group + Spatial density)
    TTRANSParams(
        case_name="Mode3_Massive_Spatial_Footprint",
        data_type=np.float32,
        shape=DataFormat.GNCHW2GNC1HWC0.value,
        g_shape0=2, g_shape1=4, g_shape2=128, g_shape3=128, g_shape4=8, g_shape5=1,
        g_whole_shape0=2, g_whole_shape1=2, g_whole_shape2=24, g_whole_shape3=128, g_whole_shape4=128,
        g_shape6=4
    ),

    # =========================================================================
    # MODE 4: GNC1HWC02C1HWN1N0C0 (6 Cases)
    # =========================================================================
    # Case 4.1: Aligned group layout transformation monitoring (G=2, N=16 -> N1=2, N0=8)
    TTRANSParams(
        case_name="Mode4_Grouped_Blocked_Aligned",
        data_type=np.float16,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=2, g_shape4=8, g_shape5=1,
        g_whole_shape0=16, g_whole_shape1=4, g_whole_shape2=14, g_whole_shape3=14, g_whole_shape4=16,
        g_whole_shape5=2
    ),
    # Case 4.2: Unaligned layout step with group overhead configurations (G=4, N=7 -> padded to 8)
    TTRANSParams(
        case_name="Mode4_Grouped_Unaligned_Batch",
        data_type=np.float16,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=2, g_shape4=4, g_shape5=1,
        g_whole_shape0=7, g_whole_shape1=2, g_whole_shape2=7, g_whole_shape3=7, g_whole_shape4=16,
        g_whole_shape5=4
    ),
    # Case 4.3: High group count block layout execution verification testing (G=16)
    TTRANSParams(
        case_name="Mode4_HighGroup_Weight_Transform",
        data_type=np.float16,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=1, g_shape4=4, g_shape5=1,
        g_whole_shape0=4, g_whole_shape1=2, g_whole_shape2=5, g_whole_shape3=5, g_whole_shape4=16,
        g_whole_shape5=16
    ),
    # Case 4.4: Int8 quantized multi-group matrix arrays checking sequence configurations
    TTRANSParams(
        case_name="Mode4_Quantized_Int8_GroupBlocks",
        data_type=np.int8,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=2, g_shape4=16, g_shape5=1,
        g_whole_shape0=32, g_whole_shape1=8, g_whole_shape2=10, g_whole_shape3=10, g_whole_shape4=32,
        g_whole_shape5=2
    ),
    # Case 4.5: Micro boundary baseline test execution limits
    TTRANSParams(
        case_name="Mode4_Micro_Boundary",
        data_type=np.float16,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=1, g_shape4=2, g_shape5=1,
        g_whole_shape0=1, g_whole_shape1=1, g_whole_shape2=1, g_whole_shape3=1, g_whole_shape4=16,
        g_whole_shape5=1
    ),
    # Case 4.6: Massive deep production configuration network pipeline load verification
    TTRANSParams(
        case_name="Mode4_Max_Scale_Industrial_Stress",
        data_type=np.float32,
        shape=DataFormat.GNC1HWC02C1HWN1N0C0.value,
        g_shape0=0, g_shape1=0, g_shape2=0, g_shape3=4, g_shape4=4, g_shape5=1,
        g_whole_shape0=16, g_whole_shape1=16, g_whole_shape2=20, g_whole_shape3=20, g_whole_shape4=8,
        g_whole_shape5=8
    ),
]

# Assuming 'gen_golden_data' is imported or defined in the active execution space
if __name__ == "__main__":
    print(f"Beginning validation matrix deployment for {len(test_cases_registry)} test scenarios...\n")
    
    for case in test_cases_registry:
        print(f"Running Transformation Mode [{case.shape}] | Configuration Identifier: {case.case_name}")
        
        # This will automatically compute shapes, pad safely, transform, 
        # and dump the pristine files to disk sequentially.
        gen_golden_data(case)
        
    print("\nAll binary test files (input.bin, golden.bin) have been generated successfully.")