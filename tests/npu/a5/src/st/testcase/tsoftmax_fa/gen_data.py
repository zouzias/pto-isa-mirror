"""
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
"""
#!/usr/bin/env python3
import os
import numpy as np
import math

np.random.seed(2025)

# HEAD_SIZE = 128
# SCALE = 0.8
KEEP_PROB = 1.0

def gen_golden_data(param):
    B = param.batch
    N = param.num_head
    S = param.seq
    h = param.head_size
    init = param.init
    layout_DN = param.transpose
    casual_mask = param.casual_mask

    SCALE = 1/np.sqrt(param.head_size)

    if init:
        if layout_DN:
            qk_shape = [B, N, S, h]
            input = np.random.uniform(-10, 10, qk_shape).astype(np.float32)
            input.tofile("./input.bin")
            #attention mask
            if casual_mask:
                casual_mask = np.triu((np.ones(input.shape) * float(-3.40282e+38)).astype(np.float32), 1)
                input = input + casual_mask
            drop_mask = np.ones(qk_shape).astype(np.uint8)
            if False:
                input = input * drop_mask

            local_max_unscaled = np.max(input, axis=2)
            local_max_golden = local_max_unscaled.reshape(-1).astype(np.float32)
            local_max_golden.tofile("golden_local_max.bin")
            local_max_golden.tofile("golden_global_max.bin")
            local_max_golden = local_max_golden * SCALE
            # local_max_golden.tofile("golden_local_max.bin")

            shifted_scaled = (input - local_max_unscaled) * np.float32(SCALE)
            exp_vals = np.exp(shifted_scaled).astype(np.float32)
            exp_max = np.max(exp_vals, axis=2, keepdims=True).reshape(-1).astype(np.float32)
            exp_max.tofile("golden_exp_max.bin")    #no need to check exp max in init stage, no update

            new_global_max = np.zeros(h).astype(np.float32)    #init to be 0, no need to check global max in init stage, no update
            
            local_sum = np.sum(exp_vals, axis=2, keepdims=True).astype(np.float32)
            local_sum_golden = local_sum.reshape(-1).astype(np.float32)
            local_sum_golden.tofile("golden_local_sum.bin")

            new_global_sum = local_sum.reshape(-1).astype(np.float32)
            new_global_sum.tofile("golden_global_sum.bin")
            
            x_exp_golden = exp_vals.astype(np.float16)
            x_exp_golden.tofile("golden_x_exp.bin")
            # x_exp_golden = np.transpose(exp_vals.reshape(B,N,S,4,16), [0,1,3,2,4]).astype(np.float16)  #NZ layout
            # x_exp_golden.tofile("golden_x_exp.bin")

            os.chdir(original_dir)

        else:
            qk_shape = [B, N, h, S]
            input = np.random.uniform(-10, 10, qk_shape).astype(np.float32)
            input.tofile("./input.bin")
            if casual_mask:
                casual_mask = np.triu((np.ones(input.shape) * float(-3.40282e+38)).astype(np.float32), 1)
                input = input + casual_mask
            drop_mask = np.ones(qk_shape).astype(np.uint8)
            if False:
                input = input * drop_mask

            local_max_unscaled = np.max(input, axis=3)
            local_max_golden = local_max_unscaled.reshape(-1).astype(np.float32)
            local_max_golden.tofile("golden_local_max.bin")
            local_max_golden.tofile("golden_global_max.bin")    #just to check golden, no need to check global max in init stage, no update
            local_max_golden = local_max_golden * SCALE
            # local_max_golden.tofile("golden_local_max.bin")
            # local_max_golden.tofile("golden_global_max.bin")    #just to check golden, no need to check global max in init stage, no update

            shifted_scaled = (input - local_max_unscaled[:,:,:,np.newaxis]) * np.float32(SCALE)
            exp_vals = np.exp(shifted_scaled).astype(np.float32)
            exp_max = np.max(exp_vals, axis=3, keepdims=True).reshape(-1).astype(np.float32)
            exp_max.tofile("golden_exp_max.bin")    #no need to check exp max in init stage, no update
            
            local_sum = np.sum(exp_vals, axis=3, keepdims=True).astype(np.float32)
            local_sum_golden = local_sum.reshape(-1).astype(np.float32)
            local_sum_golden.tofile("golden_local_sum.bin")

            new_global_sum = local_sum.reshape(-1).astype(np.float32)
            new_global_sum.tofile("golden_global_sum.bin")
            
            x_exp_golden = exp_vals.astype(np.float16)
            x_exp_golden.tofile("golden_x_exp.bin")

            os.chdir(original_dir)

    else:
        if layout_DN:
            qk_shape = [B, N, S, h]
            input = np.random.uniform(-10, 10, qk_shape).astype(np.float32)
            input.tofile("./input.bin")
            if casual_mask:
                casual_mask = np.triu((np.ones(input.shape) * float(-3.40282e+38)).astype(np.float32), 1)
                input = input + casual_mask
            drop_mask = np.ones(qk_shape).astype(np.uint8)
            if False:
                input = input * drop_mask

            local_max_unscaled = np.max(input, axis=2)
            local_max_unscaled.tofile("golden_local_max.bin")

            new_global_max = np.zeros(h).astype(np.float32)    #as input, should be local_max last time for whole FA cut seq situation
            new_global_max.tofile("golden_global_max_in.bin")
            local_max_golden = np.maximum(local_max_unscaled*SCALE, new_global_max).astype(np.float32)
            # local_max_golden.tofile("golden_local_max.bin")

            exp_max = new_global_max - local_max_golden
            exp_max = np.exp(exp_max)
            exp_max.tofile("golden_exp_max.bin")

            shifted_scaled = (input*SCALE - local_max_golden).astype(np.float32)
            exp_vals = np.exp(shifted_scaled).astype(np.float32)
            
            local_sum = np.sum(exp_vals, axis=2, keepdims=True).astype(np.float32)
            local_sum_golden = local_sum.reshape(-1).astype(np.float32)
            local_sum_golden.tofile("golden_local_sum.bin")

            new_global_sum = np.zeros(h).astype(np.float32)    #read last time global sum and update
            new_global_sum.tofile("golden_global_sum_in.bin")
            new_global_sum = exp_max * new_global_sum + local_sum
            new_global_sum.tofile("golden_global_sum.bin")
            
            x_exp_golden = exp_vals.astype(np.float16)
            x_exp_golden.tofile("golden_x_exp.bin")

            os.chdir(original_dir)

        else:
            qk_shape = [B, N, h, S]
            input = np.random.uniform(-10, 10, qk_shape).astype(np.float32)
            input.tofile("./input.bin")
            if casual_mask:
                casual_mask = np.triu((np.ones(input.shape) * float(-3.40282e+38)).astype(np.float32), 1)
                input = input + casual_mask
            drop_mask = np.ones(qk_shape).astype(np.uint8)
            if False:
                input = input * drop_mask

            local_max_unscaled = np.max(input, axis=3)
            local_max_unscaled.tofile("golden_local_max.bin")

            new_global_max = np.zeros(h).astype(np.float32)    #as input, should be local_max last time for whole FA cut seq situation
            new_global_max.tofile("golden_global_max_in.bin")
            local_max_golden = np.maximum(local_max_unscaled*SCALE, new_global_max).astype(np.float32)
            # local_max_golden.tofile("golden_local_max.bin")

            exp_max = new_global_max - local_max_golden
            exp_max = np.exp(exp_max)
            exp_max.tofile("golden_exp_max.bin")

            shifted_scaled = (input*SCALE - local_max_golden[:,:,:,np.newaxis]).astype(np.float32)
            exp_vals = np.exp(shifted_scaled).astype(np.float32)
            
            local_sum = np.sum(exp_vals, axis=3, keepdims=True).astype(np.float32)
            local_sum_golden = local_sum.reshape(-1).astype(np.float32)
            local_sum_golden.tofile("golden_local_sum.bin")

            new_global_sum = np.zeros(h).astype(np.float32)    #read last time global sum and update
            new_global_sum.tofile("golden_global_sum_in.bin")
            new_global_sum = exp_max[:,:,:,np.newaxis] * new_global_sum[:,np.newaxis] + local_sum
            new_global_sum.tofile("golden_global_sum.bin")
            
            x_exp_golden = exp_vals.astype(np.float16)
            x_exp_golden.tofile("golden_x_exp.bin")

            os.chdir(original_dir)


class TSoftmaxFAParams:
    def __init__(self, name, batch, num_head, seq, head_size, init=True, transpose=False, casual_mask=False):
        self.name = name
        self.batch = batch
        self.num_head = num_head
        self.seq = seq
        self.head_size = head_size
        self.init = init
        self.transpose = transpose
        self.casual_mask = casual_mask
        #TODO: support NZ golden

if __name__ == "__main__":
    case_params_list = [
        TSoftmaxFAParams("TSOFTMAXFATest.case1_B1_N1_S128_H64_DN_fusion_init", 1, 1, 128, 64, True, True),     #cce hand write DN fusion version, init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case2_B1_N1_S256_H64_DN_fusion_init", 1, 1, 256, 64, True, True), 
        TSoftmaxFAParams("TSOFTMAXFATest.case3_B1_N1_S256_H64_DN_no_fusion_init", 1, 1, 256, 64, True, True),     #PTO no fusion DN version, init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case4_B1_N1_S128_H64_DN_no_fusion_init", 1, 1, 128, 64, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case5_B1_N1_S256_H64_ND_no_fusion_init", 1, 1, 256, 64, True, False),    #PTO no fusion ND version, init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case6_B1_N1_S128_H64_ND_no_fusion_init", 1, 1, 128, 64, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case7_B1_N1_S128_H64_DN_fusion_no_init", 1, 1, 128, 64, False, True),    #cce hand write DN fusion version, no_init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case8_B1_N1_S128_H64_DN_no_fusion_no_init", 1, 1, 128, 64, False, True),    #PTO no fusion DN version, no_init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case9_B1_N1_S128_H64_ND_no_fusion_no_init", 1, 1, 128, 64, False, False),    #PTO no fusion ND version, no_init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case10_B1_N1_S256_H64_ND_no_fusion_no_init", 1, 1, 256, 64, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case11_B1_N1_S256_H64_DN_fusion_no_init", 1, 1, 256, 64, False, True), 
        TSoftmaxFAParams("TSOFTMAXFATest.case12_B1_N1_S256_H64_DN_no_fusion_no_init", 1, 1, 256, 64, False, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case13_B1_N1_S128_H64_ND_fusion_init", 1, 1, 128, 64, True, False),     #cce hand write ND fusion version, init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case14_B1_N1_S128_H64_ND_fusion_no_init", 1, 1, 128, 64, False, False),    #cce hand write ND fusion version, no init stage
        TSoftmaxFAParams("TSOFTMAXFATest.case15_B1_N1_S256_H64_ND_fusion_init", 1, 1, 256, 64, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case16_B1_N1_S256_H64_ND_fusion_no_init", 1, 1, 256, 64, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case17_B1_N1_S64_H64_DN_fusion_init", 1, 1, 64, 64, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case18_B1_N1_S64_H64_DN_fusion_no_init", 1, 1, 64, 64, False, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case19_B1_N1_S64_H64_ND_fusion_init", 1, 1, 64, 64, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case20_B1_N1_S64_H64_ND_fusion_no_init", 1, 1, 64, 64, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case21_B1_N1_S64_H128_ND_no_fusion_no_init", 1, 1, 64, 128, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case22_B1_N1_S64_H128_ND_no_fusion_init", 1, 1, 64, 128, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case23_B1_N1_S64_H128_ND_fusion_no_init", 1, 1, 64, 128, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case24_B1_N1_S64_H128_ND_fusion_init", 1, 1, 64, 128, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case25_B1_N1_S128_H128_ND_no_fusion_no_init", 1, 1, 128, 128, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case26_B1_N1_S128_H128_ND_no_fusion_init", 1, 1, 128, 128, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case27_B1_N1_S128_H128_ND_fusion_no_init", 1, 1, 128, 128, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case28_B1_N1_S128_H128_ND_fusion_init", 1, 1, 128, 128, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case29_B1_N1_S64_H128_ND_fusion_no_init", 1, 1, 64, 128, False, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case30_B1_N1_S128_H128_DN_no_fusion_init", 1, 1, 128, 128, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case31_B1_N1_S128_H128_DN_fusion_init", 1, 1, 128, 128, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case32_B1_N1_S128_H128_DN_no_fusion_no_init", 1, 1, 128, 128, False, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case33_B1_N1_S128_H128_DN_fusion_no_init", 1, 1, 128, 128, False, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case34_B1_N1_S128_H64_ND_no_fusion_init", 1, 1, 128, 64, True, False, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case35_B1_N1_S512_H32_ND_no_fusion_init", 1, 1, 512, 32, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case36_B1_N1_S1024_H16_ND_no_fusion_init", 1, 1, 1024, 16, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case37_B1_N1_S512_H32_ND_fusion_init", 1, 1, 512, 32, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case38_B1_N1_S1024_H16_ND_fusion_init", 1, 1, 1024, 16, True, False),
        TSoftmaxFAParams("TSOFTMAXFATest.case39_B1_N1_S512_H32_DN_fusion_init", 1, 1, 512, 32, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case40_B1_N1_S1024_H16_DN_fusion_init", 1, 1, 1024, 16, True, True),
        TSoftmaxFAParams("TSOFTMAXFATest.case41_B1_N1_S128_H64_DN_no_fusion_init", 1, 1, 128, 64, False, True, True),
    ]
    for case in case_params_list:
        if not os.path.exists(case.name):
            os.makedirs(case.name)
        original_dir = os.getcwd()
        os.chdir(case.name)
        gen_golden_data(case)
        os.chdir(original_dir)