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
import struct
import os
import shutil
import time
from typing import Tuple, List

np.random.seed(19)

C0 = 16  # 通道分块大小

class ConvTestParams:
    def __init__(
        self,
        input_shape_nc1hwc0: Tuple[int, int, int, int, int] = (1, 2, 8, 16, 16),
        weight_shape: Tuple[int, int, int, int] = (1, 32, 3, 3),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int, int, int] = (1, 1, 1, 1)
    ):
        self.input_shape_nc1hwc0 = input_shape_nc1hwc0
        self.weight_shape = weight_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding


def calculate_output_shape(input_shape, weight_shape, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """计算卷积输出特征图形状，返回NHWC格式"""
    N, H, W, C_in = input_shape
    C_out, C_in_w, H_k, W_k = weight_shape
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding

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
    """
    N, H, W, C_in = input_data.shape
    H_k, W_k = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    pad_top, pad_bottom, pad_left, pad_right = padding
    
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
    
    # 创建输出矩阵
    col_matrix = np.zeros((C_in * H_k * W_k, N * H_out * W_out), dtype=np.float32)
    
    # 计算img2col
    for n in range(N):
        for h_out in range(H_out):
            for w_out in range(W_out):
                col_idx = n * H_out * W_out + h_out * W_out + w_out
                col = np.zeros((C_in, H_k, W_k), dtype=np.float32)
                
                for hk in range(H_k):
                    for wk in range(W_k):
                        h_in = h_out * stride_h + hk * dilation_h
                        w_in = w_out * stride_w + wk * dilation_w
                        if 0 <= h_in < (H + pad_top + pad_bottom) and 0 <= w_in < (W + pad_left + pad_right):
                            col[:, hk, wk] = input_padded[n, h_in, w_in, :]
                
                col_matrix[:, col_idx] = col.flatten()
    
    return col_matrix, (N, H_out, W_out)

def kernel2matrix(weight):
    """
    将卷积核转换为矩阵
    输出形状: [C_out, C_in*H_k*W_k]
    """
    C_out, C_in, H_k, W_k = weight.shape
    return weight.reshape(C_out, C_in * H_k * W_k)

def conv2d_matmul_nhwc(input_data, weight, stride=(1, 1), dilation=(1, 1), padding=(0, 0, 0, 0)):
    """
    通过矩阵乘法实现NHWC格式的卷积（使用np.dot）
    返回: (输出特征图 [N, H_out, W_out, C_out], col_matrix, kernel_matrix)
    """
    H_k, W_k = weight.shape[2], weight.shape[3]
    
    # 将输入转换为img2col矩阵
    col_matrix, (N, H_out, W_out) = img2col_nhwc(input_data, (H_k, W_k), stride, dilation, padding)
    
    # 将卷积核转换为矩阵
    kernel_matrix = kernel2matrix(weight)
    
    # 矩阵乘法: [C_out, C_in*H_k*W_k] x [C_in*H_k*W_k, N*H_out*W_out] = [C_out, N*H_out*W_out]
    output_flat = np.dot(kernel_matrix, col_matrix)
    
    # 调整形状: [N, H_out, W_out, C_out]
    output = output_flat.reshape(weight.shape[0], N, H_out, W_out).transpose(1, 2, 3, 0)
    
    return output, col_matrix, kernel_matrix


def save_matrix_bin(matrix, filepath):
    """
    保存矩阵到二进制文件
    """
    # 确保目录存在
    dirname = os.path.dirname(filepath)
    if dirname:  # 只有当目录名不为空时才创建
        os.makedirs(dirname, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        matrix.flatten().astype(np.float32).tofile(f)
    
    return filepath

def gen_golden_data(case_name: str, params: ConvTestParams):
    """
    生成测试数据
    """
    print(f"\n{'='*80}")
    print(f"生成测试用例: {case_name}")
    print(f"{'='*80}")
    
    # 解析卷积参数
    N, C1, H, W, C0_input = params.input_shape_nc1hwc0
    C_in = C1 * C0_input
    C_out, C_in_w, H_k, W_k = params.weight_shape
    
    print(f"输入形状 (NC1HWC0): {params.input_shape_nc1hwc0}")
    print(f"权重形状: {params.weight_shape}")
    print(f"步长: {params.stride}")
    print(f"膨胀率: {params.dilation}")
    print(f"填充: {params.padding}")
    
    # 验证输入通道数匹配
    if C_in != C_in_w:
        raise ValueError(f"输入通道数不匹配: 输入有{C_in}个通道，但权重有{C_in_w}个输入通道")
    
    # 1. 生成输入张量 (NC1HWC0格式)
    input_nc1hwc0 = np.random.uniform(-5, 5, size=params.input_shape_nc1hwc0).astype(np.float32)
    input_nhwc_path = "x1_gm.bin"
    save_matrix_bin(input_nc1hwc0, input_nhwc_path)

    # 2. 生成权重张量
    weight = np.random.uniform(-5, 5, size=params.weight_shape).astype(np.float32)
    weight_path = "x2_gm.bin"
    save_matrix_bin(weight, weight_path)
    print(f"  weight形状: {weight.shape}")
    # 3. 将NC1HWC0格式的输入转换为NHWC格式用于卷积计算
    input_nhwc_temp = input_nc1hwc0.transpose(0, 2, 3, 1, 4)
    input_nhwc = input_nhwc_temp.reshape(N, H, W, C_in)

    # 4. 计算卷积
    start_time = time.time()
    
    # 计算输出形状
    output_shape = calculate_output_shape(
        (N, H, W, C_in), params.weight_shape, 
        params.stride, params.dilation, 
        params.padding
    )
    N_out, H_out, W_out, C_out_calc = output_shape
    print(f"  输出形状 (NHWC): {output_shape}")
    
    # 通过矩阵乘法计算卷积
    output_nhwc, col_matrix, kernel_matrix = conv2d_matmul_nhwc(
        input_nhwc, weight, 
        params.stride, params.dilation, 
        params.padding
    )
    
    elapsed_time = time.time() - start_time

    # 5. 将输出转换为NC1HWC0格式
    output_nc1hwc0, C1_out = nhwc_to_nc1hwc0(output_nhwc, C0_input)
    output_nc1hwc0_path = "golden_NC1HWC0.bin"
    save_matrix_bin(output_nc1hwc0, output_nc1hwc0_path)
    print(f"  输出形状(NC1HWC0): {output_nc1hwc0.shape}")
    
    # 6. 计算并保存二维矩阵 (M*N格式)
    # 计算矩阵乘法的维度
    M = N * H_out * W_out
    N_out_ch = C_out
    K = C_in * H_k * W_k
    
    print(f"  矩阵维度: M={M}, N={N_out_ch}, K={K}")
    print(f"  FM矩阵 (img2col) 形状: {col_matrix.shape}")
    print(f"  FT矩阵 (权重) 形状: {kernel_matrix.shape}")
    
    # 计算二维矩阵乘法结果
    output_2d = np.dot(kernel_matrix, col_matrix)

    # 转置为 [M, N_out_ch] 格式
    output_2d_transposed = output_2d.T
    # 保存二维矩阵
    output_2d_path = "golden_ND.bin"
    save_matrix_bin(output_2d_transposed, output_2d_path)

    # 8. 验证数据一致性
    output_2d_reshaped = output_2d.reshape(C_out, N, H_out, W_out).transpose(1, 2, 3, 0)
    diff = np.abs(output_nhwc - output_2d_reshaped).max()

    if diff >= 1e-5:
        print("  ⚠ 数据一致性验证失败")

    return {
        "input_nc1hwc0": input_nc1hwc0,
        "weight": weight,
        "output_nhwc": output_nhwc,
        "output_nc1hwc0": output_nc1hwc0,
        "output_2d": output_2d,
        "output_2d_transposed": output_2d_transposed,
        "col_matrix": col_matrix,
        "kernel_matrix": kernel_matrix,
        "M": M,
        "N": N_out_ch,
        "K": K
    }

if __name__ == "__main__":
    # 定义测试用例列表
    case_name_list = [
        "ConvTest.case1_param", 
        "ConvTest.case2_stride2", 
        "ConvTest.case3_dilation2", 
        "ConvTest.case4_large", 
        "ConvTest.case5_asym_padding"
    ]
    
    # 定义测试用例参数
    case_params_list = [
        # 用例1: 基本参数 - 调整输出通道数为16（C0的倍数）
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 8, 16, 16),  # N=1, C1=2, H=8, W=16, C0=16, C_in=32
            weight_shape=(16, 32, 3, 3),  # C_out=16（C0的倍数）, C_in=32, 3x3卷积核
            stride=(1, 1),  # 步长二元组
            dilation=(1, 1),  # 膨胀率二元组
            padding=(1, 1, 1, 1)  # 填充四元组
        ),
        
        # 用例2: 步长=2
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 8, 16, 16),  # C_in=32
            weight_shape=(16, 32, 3, 3),  # C_out=16
            stride=(2, 2),  # 步长为2
            dilation=(1, 1),
            padding=(1, 1, 1, 1)
        ),
        
        # 用例3: 膨胀率=2
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 8, 16, 16),  # C_in=32
            weight_shape=(16, 32, 3, 3),  # C_out=16
            stride=(1, 1),
            dilation=(2, 2),  # 膨胀率为2
            padding=(2, 2, 2, 2)  # 填充增加以适应膨胀卷积
        ),
        
        # 用例4: 较大尺寸
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 16, 16, 16),  # 较大输入，C_in=32
            weight_shape=(16, 32, 3, 3),  # C_out=16
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 1, 1, 1)
        ),
        
        # 用例5: 不对称填充
        ConvTestParams(
            input_shape_nc1hwc0=(1, 2, 8, 16, 16),  # C_in=32
            weight_shape=(16, 32, 3, 3),  # C_out=16
            stride=(1, 1),
            dilation=(1, 1),
            padding=(1, 2, 3, 4)  # 不对称填充: top=1, bottom=2, left=3, right=4
        )
    ]
    
    for i, case_name in enumerate(case_name_list):
        if not os.path.exists(case_name):
            os.makedirs(case_name)
        original_dir = os.getcwd()
        os.chdir(case_name)
        gen_golden_data(case_name, case_params_list[i])
        os.chdir(original_dir)