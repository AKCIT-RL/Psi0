# Treino Psi0 + baseline DP na OVX (L40S)

Procedimento validado em 2026-08 no cluster `ovx-l40` (partição `ovx01`, node
`ovx-l40s-01` com 5x NVIDIA L40S 48GB). Reproduz o treino
`psi0-wmo-totes-original-260804` com uma diferença autorizada: split 90/10 de
episódios (seed 42) para logar métricas de validação no W&B. O mesmo split é
usado no baseline Diffusion Policy.

Runs de referência produzidos com este guia (projeto `ih-akcit/psi`):

| Modelo | Script | Steps | Eval a cada | Batch |
| --- | --- | --- | --- | --- |
| Psi0 (finetune lerobot) | `scripts/train/psi0/submit_ovx_psi0.sh` | ~279.750 (50 épocas nos frames de treino) | 2.500 | 8 |
| Diffusion Policy | `scripts/train/dp/submit_ovx_dp.sh` | 42.000 | 500 | 16 |

Regras do cluster: GPU somente via Slurm; tudo (dados, caches, SIF, logs) em
`/raid/$USER`, nunca na home.

## 1. Preparar o diretório no RAID

```bash
export RAID_ROOT=/raid/$USER/psi0
mkdir -p "$RAID_ROOT"/{containers,data,psi_home/cache/checkpoints,runs,logs,secrets,cache/huggingface,cache/apptainer}
cd "$RAID_ROOT"
git clone --branch dev/marcos https://github.com/AKCIT-RL/Psi0.git repo
cd repo
git submodule update --init --recursive
mkdir -p logs
```

Se o submódulo `third_party/SIMPLE` falhar via SSH, clone por HTTPS no mesmo
commit registrado pelo repo:

```bash
git clone https://github.com/AKCIT-RL/SIMPLE.git third_party/SIMPLE
git -C third_party/SIMPLE checkout "$(git ls-tree HEAD third_party/SIMPLE --object-only)"
```

## 2. Obter o container (SIF)

A imagem correta é a tag `gr00t_devel` do Docker Hub — é a única com
`flash_attn` instalado (obrigatório: `src/psi/trainers/finetune.py` fixa
`attn_implementation="flash_attention_2"`). A tag `gr00t` (5.2GB) NÃO serve.

```bash
export APPTAINER_CACHEDIR="$RAID_ROOT/cache/apptainer"
export APPTAINER_TMPDIR="$RAID_ROOT/cache/apptainer"
apptainer pull "$RAID_ROOT/containers/industrial_humanoids_psi0-train.gr00t_devel.sif" \
  docker://rafaeljose/industrial_humanoids:psi0-train.gr00t_devel
```

Valide sem GPU (o venv utilizável é `/workspace/.venv`; o `.venv-psi` das
imagens atuais é um stub vazio — os scripts de submit já tratam isso):

```bash
apptainer exec "$RAID_ROOT/containers/industrial_humanoids_psi0-train.gr00t_devel.sif" \
  bash -lc 'source /workspace/.venv/bin/activate && python -c "import torch, flash_attn; print(torch.__version__)"'
```

Importante: o pacote `psi` do container aponta para o código montado em
`/workspace/src`, mas exige `PYTHONPATH=/workspace/src` (o finder do editable
install só mapeia `third_party`). Os scripts de submit já exportam isso.

## 3. Baixar e converter o dataset

O dataset fonte é o HF `lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop`
(teleop whole-body bruto, 43 dof). Use a variante `render/level-0`
(fotorrealista; a `raw/` é viewport MuJoCo sem textura).

```bash
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'lGabrielJJ/G1WholebodyLocomotionPickTotesShelfToTableTeleop',
    repo_type='dataset',
    allow_patterns=['render/level-0/*'],
    local_dir=f"/raid/{os.environ['USER']}/psi0/data/_hf_download",
)
EOF
```

Converta para o formato Psi0 (state 32, action 36, câmera `egocentric`) com o
conversor do submódulo SIMPLE:

```bash
python third_party/SIMPLE/scripts/postprocess_psi0_teleop_wbc.py \
  --sim-root "$RAID_ROOT/data/_hf_download/render/level-0" \
  --out-dir  "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0"
```

Gere os metadados que o fork do lerobot usado pelo Psi0 exige — os dois passos
abaixo são obrigatórios (o conversor não os produz completos):

```bash
# 1) stats_psi0.json (o treino lê meta/stats_psi0.json)
python scripts/data/calc_modality_stats.py \
  --task-dir "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0"
cp "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0/meta/stats.json" \
   "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0/meta/stats_psi0.json"

# 2) episodes_stats.jsonl com o campo "count" (sem ele o lerobot quebra com KeyError)
python docker/gen_episodes_stats.py \
  "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0"
```

Valide (exit 0; avisos esperados: `left_hand` constante 0 e `height` constante
0.74):

```bash
python scripts/validate_lerobot_modality.py \
  "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0" --expect-psi0
```

## 4. Checkpoints pré-treinados do Psi0

O finetune parte destes checkpoints, que precisam existir em:

```text
$RAID_ROOT/psi_home/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k
$RAID_ROOT/psi_home/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k
```

Copie-os de uma máquina que já os tenha (`rsync -ah --info=progress2 ...`).

## 5. Segredos (W&B / HF)

Crie fora do repositório, nunca versione e não passe tokens no `sbatch`:

```bash
cat > "$RAID_ROOT/secrets/psi0.env" <<'EOF'
PSI_HOME=/workspace/psi_home
DATA_HOME=/workspace/data
HF_HOME=/workspace/hf_cache
WANDB_ENTITY=ih-akcit
WANDB_API_KEY=COLOQUE_A_CHAVE_AQUI
HF_TOKEN=COLOQUE_O_TOKEN_AQUI
EOF
chmod 600 "$RAID_ROOT/secrets/psi0.env"
```

## 6. Submeter o treino Psi0

```bash
cd "$RAID_ROOT/repo"
sbatch scripts/train/psi0/submit_ovx_psi0.sh
```

Defaults do script (todos sobrescritíveis por variável de ambiente):

- `VAL_EPISODE_FRACTION=0.1`: split determinístico por episódio com seed 42.
  Para 50 episódios, os de validação são `[1, 7, 17, 40, 47]`.
- `VALIDATION_STEPS=2500`: eval (loss + erros L1 por canal) a cada 2.500 steps.
- `TARGET_EPOCHS=50`, `TRAIN_BATCH_SIZE=8`: `MAX_TRAINING_STEPS` é calculado a
  partir dos frames de TREINO (pós-split), preservando 50 épocas reais.
  Ex.: 44.760 frames de treino → 279.750 steps, checkpoint a cada 1/10.
- Para reproduzir o run original SEM split: `VAL_EPISODE_FRACTION=0 sbatch ...`.

O job usa 1 GPU para preservar batch efetivo 8. Em uma L40S: ~4.3 it/s,
~25–35GB de VRAM, ~18h no total.

## 7. Submeter o baseline Diffusion Policy

Hiperparâmetros idênticos a `baselines/dp/train_dp_g1_real.sh` (seed 2026,
batch 16, 42.000 steps, chunk 16, obs 32, action 36), acrescidos apenas do
split `--data.val_episode_fraction=0.1`:

```bash
cd "$RAID_ROOT/repo"
sbatch scripts/train/dp/submit_ovx_dp.sh
```

Eval a cada 500 steps (`val_num_batches=20`), checkpoint a cada 5.000.

## 8. Monitorar

```bash
squeue -u "$USER"
tail -f logs/psi0-<JOB_ID>.err   # tqdm de progresso
tail -f logs/dp-<JOB_ID>.out
```

No log do Psi0, confirme: nome da GPU impresso, `LeRobot validation episodes:
[1, 7, 17, 40, 47]`, `Max training steps` coerente com o split e a URL do run
W&B. As métricas aparecem no projeto `ih-akcit/psi`: `train/loss`, `eval/loss`
e `eval/err_l1_*` (vx, vy, vyaw, height, torso_rpy, arm_joints, hand_joints,
target_yaw).

Os jobs Slurm sobrevivem ao fechamento do terminal/VS Code. Cancelar:
`scancel <JOB_ID>`.

## 9. Retomar de checkpoint

```bash
export LAST_CKPT=$(find "$RAID_ROOT/runs/finetune" -type d -name 'ckpt_*' | sort -V | tail -1)
RESUME_FROM_CHECKPOINT="${LAST_CKPT/$RAID_ROOT\/runs/\/workspace\/.runs}" \
  sbatch scripts/train/psi0/submit_ovx_psi0.sh
```

Confira no log a linha `Resuming from:`.

## Problemas conhecidos (e correções já aplicadas nos scripts)

- `ModuleNotFoundError: psi` → falta `PYTHONPATH=/workspace/src` (já exportado
  pelos submit scripts).
- `KeyError: 'count'` no `compute_stats` do lerobot → `episodes_stats.jsonl`
  do conversor é incompleto; regenerar com `docker/gen_episodes_stats.py`
  (passo 3.2).
- SIF sem `flash_attn` (tag `gr00t`) → use a tag `gr00t_devel`.
- `.venv-psi` vazio no SIF → os submit scripts escolhem automaticamente o venv
  que importa `torch`.
