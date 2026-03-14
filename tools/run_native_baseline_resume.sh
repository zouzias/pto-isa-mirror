#!/bin/bash
set -euo pipefail

# Wrapper for the native 1B Jamba MoE baseline resume run.
# It mirrors the example launch configuration but handles PYTHONPATH and
# logging at the wrapper level so the run is reproducible and logs are durable.

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLM_REPO="${LLM_REPO:-/home/llx/bench_native_llm}"
MS_REPO="${MS_REPO:-/home/llx/bench_native_ms}"
PRETRAIN_ENTRY="${PRETRAIN_ENTRY:-${LLM_REPO}/pretrain_jamba.py}"
SOURCE_EXAMPLE_SCRIPT="${SOURCE_EXAMPLE_SCRIPT:-${LLM_REPO}/examples/mcore/qwen2/pretrain_qwen2_30b_4k_jamba_gdn_moe_cann850_bck.sh}"

MODEL_SCALE="${MODEL_SCALE:-1b}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-}"
NPUS_PER_NODE="${NPUS_PER_NODE:-16}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
TRAIN_ITERS="${TRAIN_ITERS:-6000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EVAL_ITERS="${EVAL_ITERS:-16}"
LOG_PARAMS_NORM="${LOG_PARAMS_NORM:-1}"
DATA_SPLIT="${DATA_SPLIT:-100,0,0}"
LOAD_CHECKPOINT="${LOAD_CHECKPOINT:-1}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}"
DRY_RUN="${DRY_RUN:-0}"

EXP_NAME="${EXP_NAME:-qwen2_1b_fp16_test_4k_jamba_gdn_moe_8npu_cann850}"
CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-/sharedata/zimoliu/ckpts/${EXP_NAME}}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/sharedata/zimoliu/ckpts/${EXP_NAME}}"
DATA_PATH="${DATA_PATH:-/sharedata/zimoliu/data/alpaca_zh_text_document}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/sharedata/zimoliu/models/Qwen2.5-7B-Instruct}"

LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs_native_resume}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${LOG_ROOT}/${EXP_NAME}_${NODE_RANK}"
LOG_DIR="${RUN_ROOT}/${RUN_ID}"
LOG_FILE="${LOG_DIR}/train.log"
META_FILE="${LOG_DIR}/meta.txt"
COMMAND_FILE="${LOG_DIR}/launch.sh"
LLM_STATUS_FILE="${LOG_DIR}/git_status_llm.txt"
MS_STATUS_FILE="${LOG_DIR}/git_status_ms.txt"

if [[ -z "${MASTER_PORT}" ]]; then
    MASTER_PORT="$(pick_free_port)"
fi

if [[ ! -d "${LLM_REPO}" ]]; then
    echo "Missing LLM_REPO: ${LLM_REPO}" >&2
    exit 1
fi

if [[ ! -d "${MS_REPO}" ]]; then
    echo "Missing MS_REPO: ${MS_REPO}" >&2
    exit 1
fi

if [[ ! -f "${PRETRAIN_ENTRY}" ]]; then
    echo "Missing pretrain entry: ${PRETRAIN_ENTRY}" >&2
    exit 1
fi

if [[ "${LOAD_CHECKPOINT}" == "1" && ! -d "${CKPT_LOAD_DIR}" ]]; then
    echo "Checkpoint load dir does not exist: ${CKPT_LOAD_DIR}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
ln -sfn "${LOG_DIR}" "${RUN_ROOT}/latest"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date -Is)] Starting native baseline resume wrapper"

LLM_HEAD="$(git -C "${LLM_REPO}" rev-parse HEAD)"
MS_HEAD="$(git -C "${MS_REPO}" rev-parse HEAD)"

git -C "${LLM_REPO}" status --short --branch > "${LLM_STATUS_FILE}"
git -C "${MS_REPO}" status --short --branch > "${MS_STATUS_FILE}"

LLM_REMOTE="$(git -C "${LLM_REPO}" remote get-url origin 2>/dev/null || true)"
MS_REMOTE="$(git -C "${MS_REPO}" remote get-url origin 2>/dev/null || true)"

LATEST_CKPT_ITERATION=""
if [[ -f "${CKPT_LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LATEST_CKPT_ITERATION="$(cat "${CKPT_LOAD_DIR}/latest_checkpointed_iteration.txt")"
fi

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

TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
mkdir -p "${TENSORBOARD_DIR}"

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export ENABLE_PTO_MOE_GROUPED_FFN="${ENABLE_PTO_MOE_GROUPED_FFN:-0}"

if [[ "${ENABLE_PTO_MOE_GROUPED_FFN}" == "1" ]]; then
    export PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT="${PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT:-1}"
    export PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT="${PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT:-1}"
    PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT="${REPO_ROOT}/demos/baseline/moe_grouped_ffn/build/lib/libop_extension.so"
    if [[ -z "${PTO_MOE_GROUPED_FFN_SO_PATH:-}" && -f "${PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT}" ]]; then
        export PTO_MOE_GROUPED_FFN_SO_PATH="${PTO_MOE_GROUPED_FFN_SO_PATH_DEFAULT}"
    fi
fi

PYTHONPATH_ENTRIES=(
    "/sharedata/zimoliu/code/zimo_mindspeed"
    "${MS_REPO}"
    "${LLM_REPO}"
)

EXISTING_PYTHONPATH="${PYTHONPATH:-}"
EXISTING_PYTHONPATH="${EXISTING_PYTHONPATH#:}"
EXISTING_PYTHONPATH="${EXISTING_PYTHONPATH%:}"

if [[ -n "${EXISTING_PYTHONPATH}" ]]; then
    PYTHONPATH_ENTRIES+=("${EXISTING_PYTHONPATH}")
fi

export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")"

TORCHRUN_CMD=(
    torchrun
    --nproc_per_node "${NPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
    "${PRETRAIN_ENTRY}"
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
    --tensorboard-dir "${TENSORBOARD_DIR}"
    --log-timers-to-tensorboard
    --log-throughput
    --distributed-backend nccl
)

if [[ "${LOG_PARAMS_NORM}" == "1" ]]; then
    TORCHRUN_CMD+=(--log-params-norm)
fi

if [[ "${SAVE_CHECKPOINT}" == "1" ]]; then
    mkdir -p "${CKPT_SAVE_DIR}"
    TORCHRUN_CMD+=(--save "${CKPT_SAVE_DIR}")
fi

if [[ "${LOAD_CHECKPOINT}" == "1" ]]; then
    TORCHRUN_CMD+=(--load "${CKPT_LOAD_DIR}")
fi

cat > "${META_FILE}" <<EOF
run_id=${RUN_ID}
started_at=$(date -Is)
repo_root=${REPO_ROOT}
source_example_script=${SOURCE_EXAMPLE_SCRIPT}
llm_repo=${LLM_REPO}
llm_head=${LLM_HEAD}
llm_remote=${LLM_REMOTE}
ms_repo=${MS_REPO}
ms_head=${MS_HEAD}
ms_remote=${MS_REMOTE}
pretrain_entry=${PRETRAIN_ENTRY}
model_scale=${MODEL_SCALE}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
nproc_per_node=${NPUS_PER_NODE}
nnodes=${NNODES}
node_rank=${NODE_RANK}
exp_name=${EXP_NAME}
train_iters=${TRAIN_ITERS}
log_interval=${LOG_INTERVAL}
save_interval=${SAVE_INTERVAL}
eval_interval=${EVAL_INTERVAL}
eval_iters=${EVAL_ITERS}
log_params_norm=${LOG_PARAMS_NORM}
seq_len=${SEQ_LEN}
micro_batch_size=${MICRO_BATCH_SIZE}
global_batch_size=${GLOBAL_BATCH_SIZE}
data_split=${DATA_SPLIT}
load_checkpoint=${LOAD_CHECKPOINT}
save_checkpoint=${SAVE_CHECKPOINT}
checkpoint_load_dir=${CKPT_LOAD_DIR}
checkpoint_save_dir=${CKPT_SAVE_DIR}
latest_checkpoint_iteration=${LATEST_CKPT_ITERATION}
data_path=${DATA_PATH}
tokenizer_path=${TOKENIZER_PATH}
log_root=${LOG_ROOT}
log_dir=${LOG_DIR}
log_file=${LOG_FILE}
tensorboard_dir=${TENSORBOARD_DIR}
enable_pto_moe_grouped_ffn=${ENABLE_PTO_MOE_GROUPED_FFN}
pto_moe_grouped_ffn_use_custom_split=${PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT:-}
pto_moe_grouped_ffn_cache_dn_weight=${PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT:-}
pto_moe_grouped_ffn_so_path=${PTO_MOE_GROUPED_FFN_SO_PATH:-}
pythonpath=${PYTHONPATH}
EOF

{
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    printf 'export HCCL_CONNECT_TIMEOUT=%q\n' "${HCCL_CONNECT_TIMEOUT}"
    printf 'export CUDA_DEVICE_MAX_CONNECTIONS=%q\n' "${CUDA_DEVICE_MAX_CONNECTIONS}"
    printf 'export ENABLE_PTO_MOE_GROUPED_FFN=%q\n' "${ENABLE_PTO_MOE_GROUPED_FFN}"
    if [[ "${ENABLE_PTO_MOE_GROUPED_FFN}" == "1" ]]; then
        printf 'export PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT=%q\n' "${PTO_MOE_GROUPED_FFN_USE_CUSTOM_SPLIT:-}"
        printf 'export PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT=%q\n' "${PTO_MOE_GROUPED_FFN_CACHE_DN_WEIGHT:-}"
        if [[ -n "${PTO_MOE_GROUPED_FFN_SO_PATH:-}" ]]; then
            printf 'export PTO_MOE_GROUPED_FFN_SO_PATH=%q\n' "${PTO_MOE_GROUPED_FFN_SO_PATH}"
        fi
    fi
    printf 'export PYTHONPATH=%q\n' "${PYTHONPATH}"
    printf '%q ' "${TORCHRUN_CMD[@]}"
    printf '\n'
} > "${COMMAND_FILE}"
chmod +x "${COMMAND_FILE}"

echo "[$(date -Is)] Log directory: ${LOG_DIR}"
echo "[$(date -Is)] Log file: ${LOG_FILE}"
echo "[$(date -Is)] Command file: ${COMMAND_FILE}"
echo "[$(date -Is)] Latest checkpoint iteration: ${LATEST_CKPT_ITERATION:-unknown}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[$(date -Is)] DRY_RUN=1, not launching training."
    exit 0
fi

echo "[$(date -Is)] Launching native baseline resume run"
"${TORCHRUN_CMD[@]}"
