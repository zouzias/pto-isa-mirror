#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import numpy as np
import os
from typing import Tuple
import ml_dtypes
bfloat16 = ml_dtypes.bfloat16
# 全局随机种子
np.random.seed(19)

class ConvTestParams:
    """卷积测试参数类"""
    def __init__(
        self,
        input_shape_nc1hwc0: Tuple[int, int, int, int, int],
        weight_shape: Tuple[int, int, int, int, int],  # 改为5维: (C1, H_k, W_k, N, C0)
        stride: Tuple[int, int],  # 二元组: (stride_h, stride_w)
        dilation: Tuple[int, int],  # 二元组: (dilation_h, dilation_w)
        padding: Tuple[int, int, int, int],  # 四元组: (top, bottom, left, right)
        dtype: type = np.float32  # 直接传入NumPy数据类型，如np.float16, np.float32, 或ml_dtypes.bfloat16
    ):
        # 卷积参数
        self.input_shape_nc1hwc0 = input_shape_nc1hwc0
        self.weight_shape = weight_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dtype = dtype
        
        # 根据数据类型计算C0
        N, C1, H, W, C0_input = input_shape_nc1hwc0
        
        # 获取数据类型大小
        if dtype == np.float32:
            dtype_size = 4
        elif dtype == np.int8:
            dtype_size = 1
        else:
            dtype_size = 2

        expected_c0 = 32 // dtype_size  # 32字节 / 数据类型大小
        
        if C0_input != expected_c0:
            raise ValueError(f"对于{dtype}类型，C0应为{expected_c0}，但输入为{C0_input}")

# 核心卷积计算函数
def calculate_output_shape(input_shape, weight_shape, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """计算卷积输出特征图形状，返回NHWC格式"""
    N, H, W, C_in = input_shape
    C_out, C_in_w, H_k, W_k = weight_shape
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    # 验证输入通道数匹配
    if C_in != C_in_w:
        raise ValueError("输入通道数不匹配: 输入有%d个通道，但权重有%d个输入通道" % (C_in, C_in_w))
    
    # 计算输出高度和宽度
    H_out = (H + pad_top + pad_bottom - dilation_h * (H_k - 1) - 1) // stride_h + 1
    W_out = (W + pad_left + pad_right - dilation_w * (W_k - 1) - 1) // stride_w + 1

    return (N, H_out, W_out, C_out)

def nhwc_to_nc1hwc0(input_nhwc, C0=16):
    """
    将NHWC格式的输入张量转换为NC1HWC0格式
    """
    N, H, W, C_in = input_nhwc.shape
    
    # 计算C1（向上取整）
    C1 = (C_in + C0 - 1) // C0
    
    # 如果需要，在通道维度上填充0
    if C_in % C0 != 0:
        pad_size = C1 * C0 - C_in
        input_padded = np.pad(
            input_nhwc,
            pad_width=((0, 0), (0, 0), (0, 0), (0, pad_size)),
            mode='constant',
            constant_values=0
        )
    else:
        input_padded = input_nhwc
    
    # 重塑为NC1HWC0格式
    output = input_padded.reshape(N, H, W, C1, C0).transpose(0, 3, 1, 2, 4)
    
    return output, C1

def nc1hwc0_to_nhwc(input_nc1hwc0, original_C_in):
    """
    将NC1HWC0格式的张量转换回NHWC格式
    """
    N, C1, H, W, C0 = input_nc1hwc0.shape
    
    output = input_nc1hwc0.transpose(0, 2, 3, 1, 4).reshape(N, H, W, C1 * C0)
    output = output[:, :, :, :original_C_in]
    
    return output

def img2col_nhwc(input_data, kernel_size, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    将NHWC格式的输入特征图转换为img2col矩阵
    输出形状: [C_in*H_k*W_k, N*H_out*W_out]
    使用输入数据的数据类型
    """
    N, H, W, C_in = input_data.shape
    H_k, W_k = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    # 计算输出尺寸
    H_out = (H + pad_top + pad_bottom - dilation_h * (H_k - 1) - 1) // stride_h + 1
    W_out = (W + pad_left + pad_right - dilation_w * (W_k - 1) - 1) // stride_w + 1
    
    # 对输入进行padding
    input_padded = np.pad(
        input_data,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    # 创建输出矩阵，使用输入数据的数据类型
    col_matrix = np.zeros((C_in * H_k * W_k, N * H_out * W_out), dtype=input_data.dtype)
    
    # 计算img2col
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                col_idx = n * H_out * W_out + h_out * W_out + w_out
                col = np.zeros((C_in, H_k, W_k), dtype=input_data.dtype)
                
                for hk in range(H_k):
                    for wk in range(W_k):
                        h_in = h_out * stride_h + hk * dilation_h
                        w_in = w_out * stride_w + wk * dilation_w
                        if 0 <= h_in < (H + pad_top + pad_bottom) and 0 <= w_in < (W + pad_left + pad_right):
                            col[:, hk, wk] = input_padded[n, h_in, w_in, :]
                
                col_matrix[:, col_idx] = col.flatten()
    
    return col_matrix, (N, H_out, W_out)

def kernel2matrix_new(weight):
    """
    将卷积核转换为矩阵
    输出形状: [C_out, C_in*H_k*W_k]
    使用权重的数据类型
    """
    C_out, C_in, H_k, W_k = weight.shape
    return weight.reshape(C_out, C_in * H_k * W_k)

def conv2d_matmul_nhwc_float(input_data, weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    通过矩阵乘法实现NHWC格式的浮点卷积
    使用输入和权重的数据类型进行计算，矩阵乘法时转换为float32
    返回: (输出特征图 [N, H_out, W_out, C_out], col_matrix, kernel_matrix)
    """
    H_k, W_k = weight.shape[2], weight.shape[3]
    
    # 将输入转换为img2col矩阵
    col_matrix, (N, H_out, W_out) = img2col_nhwc(input_data, (H_k, W_k), stride, dilation, padding)
    
    # 将卷积核转换为矩阵
    kernel_matrix = kernel2matrix_new(weight)
    
    # 矩阵乘法: 转换为float32进行计算
    # [C_out, C_in*H_k*W_k] x [C_in*H_k*W_k, N*H_out*W_out] = [C_out, N*H_out*W_out]
    output_flat = np.dot(
        kernel_matrix.astype(np.float32), 
        col_matrix.astype(np.float32)
    )
    
    # 调整形状: [N, H_out, W_out, C_out]
    output = output_flat.reshape(weight.shape[0], N, H_out, W_out).transpose(1, 2, 3, 0)
    
    return output, col_matrix, kernel_matrix

def conv2d_matmul_nhwc_int8(input_data, weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    通过矩阵乘法实现NHWC格式的int8卷积
    输入和权重为int8，计算时转换为int32
    返回: (输出特征图 [N, H_out, W_out, C_out], col_matrix, kernel_matrix)
    """
    H_k, W_k = weight.shape[2], weight.shape[3]
    
    # 将输入转换为img2col矩阵
    col_matrix, (N, H_out, W_out) = img2col_nhwc(input_data, (H_k, W_k), stride, dilation, padding)
    
    # 将卷积核转换为矩阵
    kernel_matrix = kernel2matrix_new(weight)
    
    # 矩阵乘法: 转换为int32进行计算
    # [C_out, C_in*H_k*W_k] x [C_in*H_k*W_k, N*H_out*W_out] = [C_out, N*H_out*W_out]
    output_flat = np.dot(
        kernel_matrix.astype(np.int32), 
        col_matrix.astype(np.int32)
    )
    
    # 调整形状: [N, H_out, W_out, C_out]
    output = output_flat.reshape(weight.shape[0], N, H_out, W_out).transpose(1, 2, 3, 0)
    
    return output, col_matrix, kernel_matrix

# 文件保存函数
def save_matrix_bin(matrix, filepath):
    """
    保存矩阵到二进制文件
    只保存数据，不包含任何头部信息
    """
    # 确保目录存在
    dirname = os.path.dirname(filepath)
    if dirname:  # 只有当目录名不为空时才创建
        os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        # 写入数据
        matrix.flatten().tofile(f)
    
    return filepath

def save_matrix_txt(matrix, filepath, max_values=1000):
    """
    保存矩阵到文本文件，便于查看
    max_values: 最多保存的元素数量，避免文件过大
    """
    # 确保目录存在
    dirname = os.path.dirname(filepath)
    if dirname:  # 只有当目录名不为空时才创建
        os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, 'w') as f:
        # 写入形状信息
        f.write(f"Shape: {matrix.shape}\n")
        f.write(f"Total elements: {matrix.size}\n")
        f.write(f"Dtype: {matrix.dtype}\n")
        f.write("=" * 50 + "\n")
        
        # 扁平化矩阵
        flat_data = matrix.flatten()
        total_elements = flat_data.size
        
        # 写入数据
        for i in range(min(total_elements, max_values)):
            f.write(f"{flat_data[i]}\n")
        
        if total_elements > max_values:
            f.write(f"\n... (只显示前{max_values}个值，共{total_elements}个)\n")


def gen_golden_data(case_name: str, params: ConvTestParams):
    
    # 解析卷积参数
    N, C1_input, H, W, C0_input = params.input_shape_nc1hwc0
    C_in = C1_input * C0_input
    
    # 解析权重形状 (C1, H_k, W_k, N, C0)
    C1_weight, H_k, W_k, N_out, C0_weight = params.weight_shape
    dtype = params.dtype
    
    # 验证输入通道数匹配
    if C1_input != C1_weight or C0_input != C0_weight:
        raise ValueError(f"输入通道分块不匹配: 输入有(C1={C1_input}, C0={C0_input})，但权重有(C1={C1_weight}, C0={C0_weight})")
    
    # 1. 生成输入张量 (NC1HWC0格式) - 使用传入的数据类型
    if dtype == np.int8:
        # 对于int8，生成[-128, 127]范围内的数据
        input_nc1hwc0 = np.random.randint(-128, 128, size=params.input_shape_nc1hwc0, dtype=np.int8)
    else:
        input_nc1hwc0 = np.random.uniform(-5, 5, size=params.input_shape_nc1hwc0).astype(dtype)
    input_nhwc_path_bin = "x1_gm.bin"
    input_nhwc_path_txt = "x1_gm.txt"
    
    # 保存为指定类型的二进制文件
    save_matrix_bin(input_nc1hwc0, input_nhwc_path_bin)
    # 保存文本文件用于查看
    save_matrix_txt(input_nc1hwc0, input_nhwc_path_txt)
    
    # 2. 生成权重张量 - 使用传入的数据类型
    if dtype == np.int8:
        # 对于int8，生成[-128, 127]范围内的数据
        weight = np.random.randint(-128, 128, size=params.weight_shape, dtype=np.int8)
    else:
        weight = np.random.uniform(-5, 5, size=params.weight_shape).astype(dtype)
    weight_path_bin = "x2_gm.bin"
    weight_path_txt = "x2_gm.txt"
    
    save_matrix_bin(weight, weight_path_bin)
    save_matrix_txt(weight, weight_path_txt)
    
    # 3. 将NC1HWC0格式的输入转换为NHWC格式用于卷积计算
    input_nhwc_temp = input_nc1hwc0.transpose(0, 2, 3, 1, 4)
    input_nhwc = input_nhwc_temp.reshape(N, H, W, C_in)
    # 4. 计算卷积
    # 将权重从 [C1, H_k, W_k, N, C0] 转换为 [N, C1*C0, H_k, W_k] 用于计算
    weight_for_calc = weight.transpose(3, 0, 4, 1, 2).reshape(N_out, C1_weight * C0_weight, H_k, W_k)
    
    # 计算输出形状
    output_shape = calculate_output_shape(
        (N, H, W, C_in), (N_out, C1_weight * C0_weight, H_k, W_k), 
        params.stride, params.dilation, 
        params.padding
    )
    N_out_calc, H_out, W_out, C_out_calc = output_shape
    # 通过矩阵乘法计算卷积
    if dtype == np.int8:
        # 对于int8，使用int8卷积函数
        output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc_int8(
            input_nhwc, weight_for_calc, 
            params.stride, params.dilation, 
            params.padding
        )
    else:
        # 对于浮点类型，使用原来的浮点卷积函数
        output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc_float(
            input_nhwc, weight_for_calc, 
            params.stride, params.dilation, 
            params.padding
        )
    
    # 5. 将输出转换为NC1HWC0格式
    output_nc1hwc0, C1_out = nhwc_to_nc1hwc0(output_nhwc, C0_input)
    output_nc1hwc0_path_bin = "golden_NC1HWC0.bin"
    output_nc1hwc0_path_txt = "golden_NC1HWC0.txt"
    
    # 对于int8输入，输出已经是int32
    # 对于浮点输入，输出是float32
    save_matrix_bin(output_nc1hwc0, output_nc1hwc0_path_bin)
    save_matrix_txt(output_nc1hwc0, output_nc1hwc0_path_txt)
    # 6. 保存img2col矩阵 (l0A.bin) - 使用原始数据类型
    l0a_path_bin = "l0A.bin"
    l0a_path_txt = "l0A.txt"
    
    save_matrix_bin(col_matrix, l0a_path_bin)
    save_matrix_txt(col_matrix, l0a_path_txt, max_values=2000)
    
    # 7. 保存kernel矩阵 (l0B.bin) - 使用原始数据类型
    l0b_path_bin = "l0B.bin"
    l0b_path_txt = "l0B.txt"
    
    save_matrix_bin(kernel_matrix, l0b_path_bin)
    save_matrix_txt(kernel_matrix, l0b_path_txt)
    
    # 8. 计算并保存二维矩阵 (M*N格式)
    if dtype == np.int8:
        # 对于int8，转换为int32进行计算
        output_2d = np.dot(
            kernel_matrix.astype(np.int32), 
            col_matrix.astype(np.int32)
        )
    else:
        # 对于浮点类型，转换为float32进行计算
        output_2d = np.dot(
            kernel_matrix.astype(np.float32), 
            col_matrix.astype(np.float32)
        )
    # 转置为 [M, N_out_ch] 格式
    output_2d_transposed = output_2d.T
    
    # 保存二维矩阵
    output_2d_path_bin = "golden.bin"
    output_2d_path_txt = "golden_ND.txt"
    save_matrix_bin(output_2d_transposed, output_2d_path_bin)
    save_matrix_txt(output_2d_transposed, output_2d_path_txt)

if __name__ == "__main__":
    
    # 定义测试用例列表
    case_name_list = [
        "TIMG2COLTest.case1_float16", 
        "TIMG2COLTest.case2_float32", 
        "TIMG2COLTest.case3_bfloat16", 
        "TIMG2COLTest.case4_float16", 
        "TIMG2COLTest.case5_stride", 
        "TIMG2COLTest.case6_bigshape", 
        "TIMG2COLTest.case7_bigshape", 
        "TIMG2COLTest.case8_bigshape", 
        "TIMG2COLTest.case9_int8",
        "TIMG2COLTest.case10_bigshape",
    ]

    # 定义测试用例参数
    case_params_list = [
        # 用例1: float16类型，C0=16
        ConvTestParams(
            input_shape_nc1hwc0=(3, 2, 4, 16, 16),  # N=1, C1=2, H=8, W=16, C0=16
            weight_shape=(2, 2, 2, 16, 16),  # C1=2, H_k=3, W_k=3, N=16, C0=16
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 0, 1, 0),  # 填充四元组
            dtype=np.float16
        ),
        # 用例2: float32类型，C0=8
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 4, 16, 8),  # N=1, C1=4, H=8, W=16, C0=8
            weight_shape=(2, 3, 3, 16, 8),  # C1=4, H_k=3, W_k=3, N=16, C0=8
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.float32
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 4, 16, 16),  # N=1, C1=2, H=8, W=16, C0=16
            weight_shape=(2, 3, 3, 16, 16),  # C1=2, H_k=3, W_k=3, N=16, C0=16
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=bfloat16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 4, 16, 16),  # 较大输入，C_in=64
            weight_shape=(4, 3, 3, 16, 16),  # C1=4, H_k=3, W_k=3, N=16, C0=16
            stride=(1, 1),  # 步长二元组
            dilation=(2, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.float16
        ),
        # 用例5: 步长=2，float32类型
        ConvTestParams(
            input_shape_nc1hwc0=(1, 4, 8, 16, 8),  # N=1, C1=4, H=8, W=16, C0=8
            weight_shape=(4, 3, 3, 16, 8),  # C1=4, H_k=3, W_k=3, N=8, C0=8
            stride=(2, 2),  # 步长为2
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.float32
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(4, 4, 8, 16, 16),  # 较大输入，C_in=64
            weight_shape=(4, 3, 3, 16, 16),  # C1=4, H_k=3, W_k=3, N=16, C0=16
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.float16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(2, 2, 16, 32, 8),  # 较大输入，C_in=64
            weight_shape=(2, 4, 4, 16, 8),  # C1=4, H_k=3, W_k=3, N=16, C0=16
            stride=(2, 2),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.float32
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(2, 4, 16, 64, 16),  # N=1, C1=2, H=8, W=16, C0=16
            weight_shape=(4, 3, 3, 16, 16),  # C1=2, H_k=3, W_k=3, N=16, C0=16
            stride=(2, 2),  # 步长二元组
            dilation=(2, 2),  # 膨胀率二元组
            padding=(1, 2, 1, 2),  # 填充四元组
            dtype=bfloat16
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 1, 8, 16, 32),  # 较大输入，C_in=64
            weight_shape=(1, 3, 3, 16, 32),  # C1=4, H_k=3, W_k=3, N=16, C0=16
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1),  # 填充四元组
            dtype=np.int8
        ),
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 32, 64, 32),  # N=1, C1=2, H=8, W=16, C0=16
            weight_shape=(2, 2, 2, 64, 32),  # C1=2, H_k=3, W_k=3, N=16, C0=16
            stride=(2, 2),  # 步长二元组
            dilation=(2, 2),  # 膨胀率二元组
            padding=(1, 1, 1, 0),  # 填充四元组
            dtype=np.int8
        ),
    ]

    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)