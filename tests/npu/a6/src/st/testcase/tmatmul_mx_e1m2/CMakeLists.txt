# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# e1m2 MX oracle testcase — same BF16 input as tmatmul_mx_hif4 but quantized
# as float4_e1m2x2_t with per-32 e8m0 MX scales. Used to verify the L0C
# fractal layout and TExtract scale path independent of HiF4 complexity.
# --------------------------------------------------------------------------------

pto_cube_st(tmatmul_mx_e1m2)
