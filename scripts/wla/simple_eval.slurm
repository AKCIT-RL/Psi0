#!/usr/bin/env bash
#SBATCH --job-name=simple-eval
#SBATCH --partition=b200n1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# Eval closed-loop no SIMPLE (task shelf->table, MuJoCo headless EGL) com servidor WLA no mesmo job via localhost.
#   RUN=<nome>            -> $WLA_EXP/$RUN/
#   MODE=oracle_raw|oracle_adapter|model
#   GPU=0 + sbatch --gres=none -> CPU-only (GL=osmesa)
#   SPLIT=val|train|test   (test só no relatório final congelado!)   N=<episódios>   SEED=<seed>
#   model: CKPT=<.safetensors> SOURCE=<norm source> PROFILE=wbt|psi0_tote BASE_VLM=<tokenizer dir> [EXEC_STEPS=30] [INSTR=...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
: "${RUN:?}" "${MODE:?}"; SPLIT=${SPLIT:-val}; N=${N:-1}; SEED=${SEED:-0}; PORT=${PORT:-22085}
PSI0=$PWD; SIMPLE=$PSI0/third_party/SIMPLE; SB=$RAID/containers/simple-260829.sandbox
OUT=$WLA_EXP/$RUN; mkdir -p "$OUT"
export APPTAINER_CACHEDIR=$RAID/cache/apptainer APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp; mkdir -p "$APPTAINER_TMPDIR"

{ echo "psi0: $(git rev-parse HEAD) ($(git rev-parse --abbrev-ref HEAD))"; echo "simple: $(git -C $SIMPLE rev-parse HEAD)"
  echo "unifolm-wla: $(git -C $WLA_REPO rev-parse HEAD)"; } > "$OUT/git_commit.txt"
cp "$0" "$OUT/simple_eval.slurm"
env | grep -E "^(RUN|MODE|SPLIT|N|SEED|CKPT|SOURCE|PROFILE|BASE_VLM|EXEC_STEPS|INSTR)=" > "$OUT/config.env" || true
cat > "$OUT/eval_config.yaml" <<EOF
datasets:
  - name: $SPLIT
    path: $WLA_EXP/s0_eval_data/roots/$SPLIT
    prompt: recorded
EOF
echo "$SPLIT roots @ $(cat $WLA_EXP/s0_eval_data/git_commit.txt)" > "$OUT/dataset_version.txt"

# ---- servidor (venv do WLA, host) ----
SRV=(python scripts/wla/simple_wla_server.py --port "$PORT")
case $MODE in
  oracle_raw)     SRV+=(--oracle raw --profile psi0_tote); ORACLE=$WLA_EXP/s0_eval_data/oracle/$SPLIT ;;
  oracle_adapter) SRV+=(--oracle adapter --profile psi0_tote); ORACLE=$WLA_EXP/s0_eval_data/oracle/$SPLIT ;;
  model)          module load cuda12.6/toolkit/12.6.3 2>/dev/null || true; export CUDA_HOME=/cm/shared/apps/cuda12.6/toolkit/12.6.3
                  SRV+=(--ckpt_path "${CKPT:?}" --source_name "${SOURCE:?}" --profile "${PROFILE:?}" --base_vlm "${BASE_VLM:?}" --use_bf16 --debug_dir "$OUT/server_debug")
                  [[ -n ${INSTR:-} ]] && SRV+=(--instruction "$INSTR"); ORACLE= ;;
  *) echo "MODE inválido"; exit 2 ;;
esac
( source "$WLA_REPO/.venv/bin/activate"; export PYTHONPATH="$PSI0/src:$WLA_REPO"; exec "${SRV[@]}" ) > "$OUT/server.log" 2>&1 &
SRV_PID=$!
cleanup() { kill $SRV_PID 2>/dev/null || true; }
trap cleanup EXIT
trap 'echo "SIGUSR1: saindo"; cleanup; exit 0' USR1
for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break; sleep 5; done
curl -sf "http://127.0.0.1:$PORT/health" || { echo "servidor não subiu"; tail -30 "$OUT/server.log"; exit 1; }

# ---- simulador (container; código/ submódulos do nosso checkout por cima da imagem) ----
BIND=(-B "$RAID:$RAID" -B "$SIMPLE/src:/workspace/simple/src" -B "$SIMPLE/third_party/gear_sonic:/workspace/simple/third_party/gear_sonic" -B "$SIMPLE/third_party/decoupled_wbc:/workspace/simple/third_party/decoupled_wbc")
GL=${GL:-egl}; NV=(); [[ ${GPU:-1} == 1 ]] && NV=(--nv)   # GPU=0 -> job CPU (render por software, GL=osmesa)
ENVV=(--env MUJOCO_GL=$GL --env PYOPENGL_PLATFORM=$GL --env HF_HUB_OFFLINE=1 --env OMNI_KIT_ACCEPT_EULA=Y --env SIMPLE_EVAL_EPISODE_ID= \
      --env WLA_EXEC_STEPS=${EXEC_STEPS:-30} ${ORACLE:+--env WLA_ORACLE_DIR=$ORACLE})
apptainer exec "${NV[@]}" --cleanenv "${BIND[@]}" "${ENVV[@]}" --pwd /workspace/simple "$SB" \
  /workspace/simple/.venv/bin/eval-decoupled-wbc simple/G1WholebodyLocomotionPickTotesShelfToTableTeleop-v0 wla_decoupled_wbc "$SPLIT" \
  --eval-config "$OUT/eval_config.yaml" --num-episodes "$N" --seed "$SEED" --host 127.0.0.1 --port "$PORT" \
  --sim-mode mujoco --headless --eval-dir "$OUT/eval" --success-criteria 0.9 --save-video 2>&1 | tee "$OUT/sim.log"

# ---- metrics.json a partir de eval_stats.txt ----
python3 - "$OUT" <<'PY'
import json, re, sys
out = sys.argv[1]
res = {}
for l in open(f"{out}/eval/eval_stats.txt"):
    m = re.match(r"(\S+__episode_\d+): (True|False)", l)
    if m: res[m.group(1)] = m.group(2) == "True"
json.dump({"n": len(res), "successes": sum(res.values()), "success_rate": (sum(res.values()) / len(res)) if res else None, "episodes": res},
          open(f"{out}/metrics.json", "w"), indent=2)
print(open(f"{out}/metrics.json").read())
PY
echo EVAL_DONE
