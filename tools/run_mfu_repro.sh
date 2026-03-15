#!/bin/bash
set -euo pipefail

MODE="${MODE:-native}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-100}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
EVAL_ITERS="${EVAL_ITERS:-0}"
MODEL_SCALE="${MODEL_SCALE:-1b}"
LOG_ROOT="${LOG_ROOT:-/home/llx/pto-isa/logs_mfu_repro}"
EXP_NAME="${EXP_NAME:-${MODE}_mfu_repro_$(date +%Y%m%d_%H%M%S)}"
DATA_SPLIT="${DATA_SPLIT:-100,0,0}"
LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-0}"
DRY_RUN="${DRY_RUN:-0}"

LLM_REPO_CLEAN="${LLM_REPO_CLEAN:-/home/llx/bench_native_llm}"
MS_REPO_CLEAN="${MS_REPO_CLEAN:-/home/llx/bench_native_ms}"
LLM_REPO_PTO="${LLM_REPO_PTO:-/home/llx/lzm_Mindspeed-LLm}"
MS_REPO_PTO="${MS_REPO_PTO:-/home/llx/MindSpeed}"

DATA_PATH="${DATA_PATH:-/sharedata/zimoliu/data/alpaca_zh_text_document}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/sharedata/zimoliu/models/Qwen2.5-7B-Instruct}"

pick_free_port() {
    python3 - <<'PY'
import subprocess

used = set()
try:
    output = subprocess.check_output(["ss", "-ltn"], text=True)
except Exception:
    output = ""

for line in output.splitlines()[1:]:
    fields = line.split()
    if len(fields) < 4:
        continue
    port = fields[3].rsplit(":", 1)[-1]
    if port.isdigit():
        used.add(int(port))

for candidate in range(35000, 45000):
    if candidate not in used:
        print(candidate)
        break
else:
    print(45001)
PY
}

case "${MODE}" in
    native)
        LLM_REPO="${LLM_REPO_CLEAN}"
        MS_REPO="${MS_REPO_CLEAN}"
        ENABLE_PTO_MOE_GROUPED_FFN=0
        ;;
    pto)
        LLM_REPO="${LLM_REPO_PTO}"
        MS_REPO="${MS_REPO_PTO}"
        ENABLE_PTO_MOE_GROUPED_FFN=1
        ;;
    *)
        echo "Unsupported MODE=${MODE}. Use native or pto." >&2
        exit 1
        ;;
esac

PTO_ISA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT="${PTO_ISA_REPO_ROOT}/demos/baseline/moe_grouped_ffn/build/libop_extension.so"
if [[ "${ENABLE_PTO_MOE_GROUPED_FFN}" == "1" ]]; then
    # Keep training-side config simple: use the fast split kernels by default.
    export PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT="${PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT:-1}"
    export PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT="${PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT:-}"
    # Prefer the freshly built .so from this repo if available, unless user overrides it.
    if [[ -z "${PTO_MOE_GROUPED_FFN_SO_PATH:-}" && -f "${PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT}" ]]; then
        export PTO_MOE_GROUPED_FFN_SO_PATH="${PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT}"
    fi
fi

if [[ -z "${MASTER_PORT}" ]]; then
    MASTER_PORT="$(pick_free_port)"
fi

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export ENABLE_PTO_MOE_GROUPED_FFN

mkdir -p "${LOG_ROOT}"
LOG_DIR="${LOG_ROOT}/${EXP_NAME}_${NODE_RANK}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
META_FILE="${LOG_DIR}/meta.txt"

case "${MODEL_SCALE}" in
    1b)
        TENSOR_MODEL_PARALLEL_SIZE=1
        EXPERT_MODEL_PARALLEL_SIZE=16
        PIPELINE_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=10
        NUM_EXPERTS=128
        TOPK=4
        HIDDEN_SIZE=4096
        FFN_HIDDEN_SIZE=8192
        MOE_FFN_HIDDEN_SIZE=1920
        NUM_ATTENTION_HEADS=32
        NUM_QUERY_GROUPS=8
        MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
        SEQ_LEN="${SEQ_LEN:-4096}"
        ROUTER_BALANCING_TYPE='aux_loss'
        ;;
    *)
        echo "Unsupported MODEL_SCALE=${MODEL_SCALE}. Only 1b is scripted." >&2
        exit 1
        ;;
esac

cat > "${META_FILE}" <<EOF
mode=${MODE}
llm_repo=${LLM_REPO}
ms_repo=${MS_REPO}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
nproc_per_node=${NPUS_PER_NODE}
nnodes=${NNODES}
node_rank=${NODE_RANK}
train_iters=${TRAIN_ITERS}
log_interval=${LOG_INTERVAL}
data_path=${DATA_PATH}
tokenizer_path=${TOKENIZER_PATH}
enable_pto_moe_grouped_ffn=${ENABLE_PTO_MOE_GROUPED_FFN}
pto_moe_grouped_ffn_use_custom_split=${PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT:-}
pto_moe_grouped_ffn_cache_dn_weight=${PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT:-}
pto_moe_grouped_ffn_so_path=${PTO_MOE_GROUPED_FFN_SO_PATH:-}
seq_len=${SEQ_LEN}
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
data_split=${DATA_SPLIT}
log_params_norm=${LOG_PARAMS_NORM}
EOF

export PYTHONPATH="${LLM_REPO}:${MS_REPO}:${PYTHONPATH:-}"

TORCHRUN_CMD=(
    torchrun
    --nproc_per_node "${NPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    "${LLM_REPO}/pretrain_jamba.py"
    --use-mcore-models
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path "${TOKENIZER_PATH}"
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings "${SEQ_LEN}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --padded-vocab-size 152576
    --rotary-base 1000000
    --lr 1.25e-6
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --position-embedding-type rope
    --normalization RMSNorm
    --use-fused-rmsnorm
    --swiglu
    --use-flash-attn
    --use-fused-rotary-pos-emb
    --use-rotary-position-embeddings
    --use-fused-swiglu
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
    --add-qkv-bias
    --initial-loss-scale 4096
    --no-load-optim
    --no-load-rng
    --seed 42
    --bf16
    --transformer-impl local
    --use-distributed-optimizer
    --ckpt-format torch
    --reuse-fp32-param
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --norm-epsilon 1e-6
    --recompute-method uniform
    --recompute-granularity full
    --recompute-num-layers 1
    --mamba-num-groups 8
    --mamba-chunk-size 128
    --mamba-state-dim 128
    --mamba-d-conv 4
    --mamba-expand 2
    --mamba-head-dim 64
    --hybrid-attention-ratio 1.0
    --hybrid-mlp-ratio 1.0
    --linear-attention-type mamba
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --num-experts "${NUM_EXPERTS}"
    --moe-router-dtype fp32
    --moe-router-score-function sigmoid
    --moe-router-topk "${TOPK}"
    --moe-router-load-balancing-type "${ROUTER_BALANCING_TYPE}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
    --moe-grouped-gemm
    --moe-permutation-async-comm
    --moe-token-dispatcher-type alltoall_seq
    --moe-alltoall-overlap-comm
    --moe-permute-fusion
    --moe-layer-freq -1
    --moe-aux-loss-coeff 0.001
    --seq-aux
    --norm-topk-prob
    --n-shared-experts 1
    --data-path "${DATA_PATH}"
    --split "${DATA_SPLIT}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-iters "${EVAL_ITERS}"
    --tensorboard-dir "${LOG_DIR}/tensorboard"
    --log-timers-to-tensorboard
    --log-throughput
    --distributed-backend nccl
)

if [[ "${LOG_PARAMS_NORM}" == "1" ]]; then
    TORCHRUN_CMD+=(--log-params-norm)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${TORCHRUN_CMD[@]}"
    printf '\n'
    exit 0
fi

"${TORCHRUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
