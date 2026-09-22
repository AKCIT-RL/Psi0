#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on one LeRobot dataset. Runs inside the image; `ovx/run.sh train` calls it.
#
#   train.sh --dataset carry_totes \
#            --base-model gr00t_n1d7_finetune_output_totes_shelf_to_table_render/final \
#            --epochs 30
#
# The recipe is the one that trained the OVX specialists (submit_slurm.sh at f0a2e4e): same
# launcher, embodiment tag, modality config, optimiser, augmentation and gradient checkpointing.
# Anything after `--` goes to the launcher unchanged, after these defaults.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
    cat >&2 <<'EOF'
usage: train.sh --dataset NAME --base-model NAME [options] [-- launcher args]

  --dataset NAME      directory under /home/ovx/data: a validated LeRobot dataset (meta/modality.json)
  --base-model NAME   model directory under /home/ovx/checkpoints (or an absolute path) with config.json
  --epochs N          N passes over this dataset; steps derived from its frame count
  --max-steps N       exact number of steps instead (default 50000 without --epochs)
  --batch-size N      global batch size (default 16)
  --save-steps N      checkpoint interval (default max-steps/5, clamped to 200..10000)
  --run-name NAME     output directory under /home/ovx/checkpoints
                      (default gr00t_n1d7_finetune_output_<run_slug or dataset>)
  --num-gpus N        GPUs for torchrun (default 1)
  --workers N         dataloader workers (default 16)
  --no-wandb          do not log to Weights & Biases
  --config FILE       experiment YAML: FinetuneConfig fields that override the frozen recipe
                      (learning_rate, tune_llm, state_dropout_prob, ...). Applied after the
                      recipe and before anything past --, and recorded in ovx_runs.jsonl.
EOF
}

DATASET=""; BASE_MODEL=""; RUN_NAME=""; EPOCHS=""; MAX_STEPS=""; SAVE_STEPS=""; CONFIG=""
BATCH=16; NUM_GPUS=1; WORKERS=16; USE_WANDB=1; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)    DATASET="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --run-name)   RUN_NAME="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --max-steps)  MAX_STEPS="$2"; shift 2 ;;
        --batch-size) BATCH="$2"; shift 2 ;;
        --save-steps) SAVE_STEPS="$2"; shift 2 ;;
        --num-gpus)   NUM_GPUS="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --no-wandb)   USE_WANDB=0; shift ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --)           shift; EXTRA=("$@"); break ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage; die "unknown option: $1" ;;
    esac
done
[ -n "${DATASET}" ]    || { usage; die "--dataset is required"; }
[ -n "${BASE_MODEL}" ] || { usage; die "--base-model is required"; }

setup_caches
gr00t_env

# Exported, not just passed as --dataset-path: the G1 modality config (g1_locomanip_n1d7.py) reads
# <DATASET_PATH>/meta/modality.json from this variable when it is imported, and raises without it.
export DATASET_PATH="${DATA_ROOT}/${DATASET}"
BASE_PATH="$(resolve_ckpt "${BASE_MODEL}")"

# ---- preflight: each of these used to fail only after a GPU had been allocated
[ -d "${DATASET_PATH}" ] || die "dataset not found: ${DATASET_PATH}"
[ -f "${DATASET_PATH}/meta/info.json" ] || die "${DATASET_PATH}/meta/info.json missing -- not a LeRobot dataset"
[ -f "${DATASET_PATH}/meta/modality.json" ] \
    || die "${DATASET_PATH}/meta/modality.json missing -- dataset not prepared (docs/runbook_modality.md)"
[ -f "${DATASET_PATH}/PROVENANCE.json" ] \
    || log "WARNING: no PROVENANCE.json -- not produced by scripts/prepare_simple_datasets.py"
[ -f "${BASE_PATH}/config.json" ] \
    || die "${BASE_PATH}/config.json missing -- --base-model needs a model directory, not a run above it"
ensure_processor_at_root "${BASE_PATH}"
if [ "${USE_WANDB}" = 1 ] && [ -z "${WANDB_API_KEY:-}" ]; then
    die "WANDB_API_KEY is not set: put it in the env file, or pass --no-wandb"
fi
if [ -z "${HF_TOKEN:-}" ] && [ ! -d "${HF_HOME}/hub/models--nvidia--Cosmos-Reason2-2B" ]; then
    log "WARNING: no HF_TOKEN and nvidia/Cosmos-Reason2-2B is not cached -- the gated backbone will fail to download"
fi

# ---- experiment config: the fields the recipe below fixes, so config_args can say which of them
# an experiment is changing. Everything else in the YAML is simply added.
RECIPE_KEYS=(embodiment_tag save_steps save_total_limit max_steps warmup_ratio weight_decay
             learning_rate global_batch_size gradient_accumulation_steps dataloader_num_workers
             output_dir eval_strategy num_gpus color_jitter_params modality_config_path
             gradient_checkpointing base_model_path dataset_path use_wandb)
CONFIG_ARGS=()
if [ -n "${CONFIG}" ]; then
    [ -f "${CONFIG}" ] || die "--config file not found: ${CONFIG}"
    tokens="$(config_args "${CONFIG}" "${RECIPE_KEYS[@]}")" \
        || die "--config ${CONFIG}: could not be read"
    if [ -n "${tokens}" ]; then
        mapfile -t CONFIG_ARGS <<< "${tokens}"
    else
        log "WARNING: ${CONFIG} defines no fields -- running the recipe unchanged"
    fi
fi

# ---- schedule: --epochs keeps training comparable across datasets of different sizes
read -r FRAMES EPISODES < <("${GR00T_PY}" -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get("total_frames", 0), d.get("total_episodes", 0))' "${DATASET_PATH}/meta/info.json")
if [ -n "${EPOCHS}" ]; then
    [ "${FRAMES}" -gt 0 ] || die "--epochs needs total_frames in meta/info.json"
    MAX_STEPS=$("${GR00T_PY}" -c "print(max(500, int(${FRAMES} * ${EPOCHS} / ${BATCH})))")
fi
MAX_STEPS="${MAX_STEPS:-50000}"
# A fixed 10000 wrote no intermediate checkpoint at all on shorter runs.
if [ -z "${SAVE_STEPS}" ]; then
    SAVE_STEPS=$(( MAX_STEPS / 5 ))
    [ "${SAVE_STEPS}" -ge 200 ]   || SAVE_STEPS=200
    [ "${SAVE_STEPS}" -le 10000 ] || SAVE_STEPS=10000
fi

# ---- output directory: same naming as submit_slurm.sh, so an existing run is resumed
if [ -z "${RUN_NAME}" ]; then
    SLUG=""
    if [ -f "${DATASET_PATH}/PROVENANCE.json" ]; then
        SLUG=$("${GR00T_PY}" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("run_slug", ""))' \
               "${DATASET_PATH}/PROVENANCE.json" 2>/dev/null || true)
    fi
    [ -n "${SLUG}" ] || SLUG=$(printf '%s' "${DATASET}" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' \
                               | sed 's/__*/_/g; s/^_//; s/_$//')
    RUN_NAME="gr00t_n1d7_finetune_output_${SLUG}"
fi
OUT="$(resolve_ckpt "${RUN_NAME}")"
mkdir -p "${OUT}"

# ---- provenance: one line per start (a resumed run keeps the history of every attempt)
write_provenance "${OUT}/ovx_runs.jsonl" \
    "dataset=${DATASET}" "base_model=${BASE_MODEL}" "max_steps=${MAX_STEPS}" \
    "save_steps=${SAVE_STEPS}" "global_batch_size=${BATCH}" "epochs=${EPOCHS}" \
    "frames=${FRAMES}" "episodes=${EPISODES}" "num_gpus=${NUM_GPUS}" \
    "extra_args=${EXTRA[*]:-}" "slurm_job_id=${SLURM_JOB_ID:-}" \
    "psi0_code=$(repo_state "${PSI0_ROOT}")" "simple_code=$(repo_state "${SIMPLE_ROOT}")" \
    "dataset_provenance_file=${DATASET_PATH}/PROVENANCE.json" \
    "experiment_config_file=${CONFIG}"

PORT=$(free_port "$(port_base 29500)")
WANDB_FLAG=()
[ "${USE_WANDB}" = 1 ] && WANDB_FLAG=(--use-wandb)

log "image:      $(build_info)"
log "dataset:    ${DATASET_PATH} (${EPISODES} episodes, ${FRAMES} frames)"
log "base model: ${BASE_PATH}"
log "output:     ${OUT}"
log "schedule:   ${MAX_STEPS} steps, batch ${BATCH}, checkpoint every ${SAVE_STEPS}${EPOCHS:+ (${EPOCHS} epochs)}"
log "gpus:       ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}), rendezvous port ${PORT}"

set +e
"${GR00T_PY}" -m torch.distributed.run \
    --nproc_per_node "${NUM_GPUS}" \
    --master_port "${PORT}" \
    "${PSI0_ROOT}/baselines/gr00t-n1.7/launch_finetune_n1d7_inner.py" \
    --base-model-path "${BASE_PATH}" \
    --dataset-path "${DATASET_PATH}" \
    --embodiment-tag G1_LOCO_DOWNSTREAM \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit 1 \
    --max-steps "${MAX_STEPS}" \
    --warmup-ratio 0.05 \
    --weight-decay 1e-05 \
    --learning-rate 0.0001 \
    --global-batch-size "${BATCH}" \
    --gradient-accumulation-steps 2 \
    --dataloader-num-workers "${WORKERS}" \
    --output-dir "${OUT}" \
    --eval-strategy no \
    --num-gpus "${NUM_GPUS}" \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --modality-config-path "${PSI0_ROOT}/src/gr00t/gr00t/configs/modality/g1_locomanip_n1d7.py" \
    --gradient-checkpointing \
    "${WANDB_FLAG[@]}" \
    "${CONFIG_ARGS[@]}" \
    "${EXTRA[@]}"
STATUS=$?
set -e

if [ "${STATUS}" -eq 0 ]; then
    # The final model is written to the run root; make it servable as it is.
    ensure_processor_at_root "${OUT}"
    log "done: ${OUT}"
else
    log "training failed (exit ${STATUS})"
fi
exit "${STATUS}"
