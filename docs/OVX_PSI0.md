# Treino Psi0 na OVX

Este guia executa o mesmo treino Psi0 usado em `psi0-wmo-totes-original-260804`: uma GPU, batch 8, 50 épocas, action chunk/horizon 30 e W&B. O dataset precisa estar no formato convertido do Psi0, com state 32, action 36 e câmera `egocentric`.

## 1. Entrar na OVX e escolher o node

```bash
ssh <usuario>@<host-da-ovx>
sinfo
```

Use a partição do node onde seus dados estão armazenados: `h100n2`, `h100n3` ou `b200n1`. Nunca execute GPU fora do Slurm.

## 2. Preparar o diretório no RAID

```bash
export RAID_ROOT=/raid/$USER/psi0
mkdir -p "$RAID_ROOT"/{containers,data,psi_home/cache/checkpoints,runs,logs,secrets,cache/huggingface}
cd "$RAID_ROOT"
git clone --branch dev/marcos https://github.com/AKCIT-RL/Psi0.git repo
cd repo
git submodule update --init --recursive
```

Todos os dados, caches, checkpoints, logs e imagens SIF ficam em `/raid/$USER`, nunca na home.

## 3. Obter o container Psi0

Peça ao Olives o SIF Psi0 correspondente ao GR00T usado na OVX e salve como:

```text
/raid/$USER/psi0/containers/industrial_humanoids_psi0-train.psi_devel.sif
```

Valide sem GPU:

```bash
module load apptainer
apptainer exec \
  "$RAID_ROOT/containers/industrial_humanoids_psi0-train.psi_devel.sif" \
  test -x /workspace/.venv-psi/bin/python
```

Se esse comando falhar, o SIF fornecido contém apenas o ambiente GR00T. Não submeta o treino até receber uma imagem com `/workspace/.venv-psi`.

## 4. Copiar dataset e checkpoints

Na máquina que contém o dataset convertido:

```bash
rsync -ah --info=progress2 \
  ~/hfm/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0/ \
  <usuario>@<host-da-ovx>:/raid/<usuario>/psi0/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0/

rsync -ah --info=progress2 \
  ~/hfm/cache/checkpoints/psi0/ \
  <usuario>@<host-da-ovx>:/raid/<usuario>/psi0/psi_home/cache/checkpoints/psi0/
```

Na OVX, confirme:

```bash
test -f "$RAID_ROOT/data/G1WholebodyLocomotionPickTotesShelfToTableTeleop-psi0/meta/stats_psi0.json"
test -d "$RAID_ROOT/psi_home/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
test -d "$RAID_ROOT/psi_home/cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
```

## 5. Configurar W&B e Hugging Face

Crie o arquivo fora do repositório:

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

Não versione esse arquivo e não coloque tokens no comando `sbatch`.

## 6. Submeter o treino

O diretório de logs precisa existir antes do `sbatch`:

```bash
cd "$RAID_ROOT/repo"
mkdir -p logs

RAID_ROOT="$RAID_ROOT" \
SIF_PATH="$RAID_ROOT/containers/industrial_humanoids_psi0-train.psi_devel.sif" \
sbatch scripts/train/psi0/submit_ovx_psi0.sh
```

Para usar outra partição:

```bash
sbatch --partition=h100n2 scripts/train/psi0/submit_ovx_psi0.sh
```

O job padrão usa uma GPU para preservar exatamente batch 8 e 315.437 optimizer steps. Usar duas GPUs muda o batch efetivo e reduz o número de steps.

## 7. Monitorar

```bash
squeue -u "$USER"
tail -f logs/psi0-<JOB_ID>.out
scontrol show job <JOB_ID>
```

No log, confirme:

```text
GPUs              : 1
Effective batch   : 8
Max training steps: 315437
Logging           : wandb
```

O run aparece no projeto `ih-akcit/psi`. Para cancelar:

```bash
scancel <JOB_ID>
```

## 8. Retomar de checkpoint

Os checkpoints ficam em:

```text
/raid/$USER/psi0/runs/finetune/<run>/checkpoints/
```

Localize o último checkpoint e submeta novamente:

```bash
export LAST_CKPT=$(find "$RAID_ROOT/runs/finetune" -type d -name 'ckpt_*' | sort -V | tail -1)

RESUME_FROM_CHECKPOINT="${LAST_CKPT/$RAID_ROOT\/runs/\/workspace\/.runs}" \
EXP_NAME=psi0-wmo-totes-original-ovx \
sbatch scripts/train/psi0/submit_ovx_psi0.sh
```

Confira no log a linha `Resuming from:`. O `nohup` não é usado na OVX; o Slurm mantém o job após fechar o terminal.

## 9. Atualizar a branch

```bash
cd "$RAID_ROOT/repo"
git status --short
git pull --ff-only origin dev/marcos
git submodule update --init --recursive
```

Não execute `git pull` enquanto um job estiver usando esse checkout. Para mudanças durante um treino, use outro clone ou worktree.