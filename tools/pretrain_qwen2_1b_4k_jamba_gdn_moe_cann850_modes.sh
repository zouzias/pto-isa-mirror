#!/bin/bash
set -euo pipefail

# Simple preset launcher aligned to the original native Qwen2/Jamba 1B MoE script.
# Despite the original filename containing "30b", its default configuration is 1b.
#
# Presets:
#   native_base : original native baseline path (bench_native_ms + bench_native_llm)
#   mc2         : current MC2 mainline path (MindSpeed + lzm_Mindspeed-LLm)
#   pto_mc2     : current PTO-ISA validation path on top of MC2 reorder
#
# Example:
#   LAUNCH_PRESET=pto_mc2 TRAIN_ITERS=6508 LOAD_CHECKPOINT=1 \
#   CKPT_LOAD_DIR=/home/llx/pto-isa/tmp_ckpts/qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850_load6500_sparse \
#   /home/llx/pto-isa/tools/pretrain_qwen2_1b_4k_jamba_gdn_moe_cann850_modes.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${REPO_ROOT}/tools/run_native_baseline_resume.sh"

LAUNCH_PRESET="${LAUNCH_PRESET:-pto_mc2}"

# Shared model/data/checkpoint defaults. These intentionally match the native 1B script.
export MODEL_SCALE="${MODEL_SCALE:-1b}"
export NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export TRAIN_ITERS="${TRAIN_ITERS:-6508}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
export EVAL_ITERS="${EVAL_ITERS:-16}"
export LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-0}"
export ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
export LOG_TIMERS_TO_TENSORBOARD="${LOG_TIMERS_TO_TENSORBOARD:-0}"
export LOG_THROUGHPUT="${LOG_THROUGHPUT:-1}"
export EXP_NAME="${EXP_NAME:-qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850}"
export LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-1}"
export SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
export CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-/home/llx/pto-isa/tmp_ckpts/${EXP_NAME}_load6500_sparse}"
export CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/sharedata/zimoliu/ckpts/${EXP_NAME}}"
export DATA_PATH="${DATA_PATH:-/sharedata/zimoliu/data/alpaca_zh_text_document}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/sharedata/zimoliu/models/Qwen2.5-7B-Instruct}"
export DATA_SPLIT="${DATA_SPLIT:-100,0,0}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Shared feature defaults. Individual presets override what they need.
export ENABLE_MOE_GROUPED_GEMM="${ENABLE_MOE_GROUPED_GEMM:-1}"
export ENABLE_MOE_PERMUTATION_ASYNC_COMM="${ENABLE_MOE_PERMUTATION_ASYNC_COMM:-1}"
export ENABLE_MOE_PERMUTE_FUSION="${ENABLE_MOE_PERMUTE_FUSION:-1}"
export MOE_TOKEN_DISPATCHER_TYPE="${MOE_TOKEN_DISPATCHER_TYPE:-alltoall_seq}"
export ENABLE_PTO_MOE_GROUPED_FFN="${ENABLE_PTO_MOE_GROUPED_FFN:-0}"

case "${LAUNCH_PRESET}" in
    native_base)
        export LLM_REPO="${LLM_REPO:-/home/llx/bench_native_llm}"
        export MS_REPO="${MS_REPO:-/home/llx/bench_native_ms}"
        export PRETRAIN_ENTRY="${PRETRAIN_ENTRY:-${LLM_REPO}/pretrain_jamba.py}"
        export SOURCE_EXAMPLE_SCRIPT="${SOURCE_EXAMPLE_SCRIPT:-${LLM_REPO}/examples/mcore/qwen2/pretrain_qwen2_30b_4k_jamba_gdn_moe_cann850_bck.sh}"
        export ENABLE_MOE_ALLTOALL_OVERLAP_COMM="${ENABLE_MOE_ALLTOALL_OVERLAP_COMM:-1}"
        export ENABLE_MOE_ALLTOALL_MC2="${ENABLE_MOE_ALLTOALL_MC2:-0}"
        export ENABLE_MOE_BMM_MC2="${ENABLE_MOE_BMM_MC2:-0}"
        export USE_FUSED_ROTARY_POS_EMB="${USE_FUSED_ROTARY_POS_EMB:-1}"
        ;;
    mc2)
        export LLM_REPO="${LLM_REPO:-/home/llx/lzm_Mindspeed-LLm}"
        export MS_REPO="${MS_REPO:-/home/llx/MindSpeed}"
        export PRETRAIN_ENTRY="${PRETRAIN_ENTRY:-${LLM_REPO}/pretrain_jamba.py}"
        export SOURCE_EXAMPLE_SCRIPT="${SOURCE_EXAMPLE_SCRIPT:-/home/llx/bench_native_llm/examples/mcore/qwen2/pretrain_qwen2_30b_4k_jamba_gdn_moe_cann850_bck.sh}"
        export ENABLE_MOE_ALLTOALL_OVERLAP_COMM="${ENABLE_MOE_ALLTOALL_OVERLAP_COMM:-0}"
        export ENABLE_MOE_ALLTOALL_MC2="${ENABLE_MOE_ALLTOALL_MC2:-1}"
        export ENABLE_MOE_BMM_MC2="${ENABLE_MOE_BMM_MC2:-0}"
        export USE_FUSED_ROTARY_POS_EMB="${USE_FUSED_ROTARY_POS_EMB:-1}"
        ;;
    pto_mc2)
        export LLM_REPO="${LLM_REPO:-/home/llx/lzm_Mindspeed-LLm}"
        export MS_REPO="${MS_REPO:-/home/llx/MindSpeed}"
        export PRETRAIN_ENTRY="${PRETRAIN_ENTRY:-${LLM_REPO}/pretrain_jamba.py}"
        export SOURCE_EXAMPLE_SCRIPT="${SOURCE_EXAMPLE_SCRIPT:-/home/llx/bench_native_llm/examples/mcore/qwen2/pretrain_qwen2_30b_4k_jamba_gdn_moe_cann850_bck.sh}"
        export ENABLE_MOE_ALLTOALL_OVERLAP_COMM="${ENABLE_MOE_ALLTOALL_OVERLAP_COMM:-0}"
        export ENABLE_MOE_ALLTOALL_MC2="${ENABLE_MOE_ALLTOALL_MC2:-1}"
        export ENABLE_MOE_BMM_MC2="${ENABLE_MOE_BMM_MC2:-0}"
        export ENABLE_PTO_MOE_MC2_REORDER="${ENABLE_PTO_MOE_MC2_REORDER:-1}"
        export PTO_MOE_MC2_SO_PATH="${PTO_MOE_MC2_SO_PATH:-${REPO_ROOT}/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so}"
        # Current fused rotary path is unstable for true PTO validation.
        export USE_FUSED_ROTARY_POS_EMB="${USE_FUSED_ROTARY_POS_EMB:-0}"
        ;;
    *)
        echo "Unsupported LAUNCH_PRESET=${LAUNCH_PRESET}" >&2
        echo "Supported presets: native_base, mc2, pto_mc2" >&2
        exit 1
        ;;
esac

echo "launch_preset=${LAUNCH_PRESET}"
echo "llm_repo=${LLM_REPO}"
echo "ms_repo=${MS_REPO}"
echo "pretrain_entry=${PRETRAIN_ENTRY}"
echo "exp_name=${EXP_NAME}"
echo "train_iters=${TRAIN_ITERS}"
echo "ckpt_load_dir=${CKPT_LOAD_DIR}"
echo "enable_moe_alltoall_overlap_comm=${ENABLE_MOE_ALLTOALL_OVERLAP_COMM:-0}"
echo "enable_moe_alltoall_mc2=${ENABLE_MOE_ALLTOALL_MC2:-0}"
echo "enable_pto_moe_mc2_reorder=${ENABLE_PTO_MOE_MC2_REORDER:-0}"
echo "use_fused_rotary_pos_emb=${USE_FUSED_ROTARY_POS_EMB}"

exec "${WRAPPER}"
