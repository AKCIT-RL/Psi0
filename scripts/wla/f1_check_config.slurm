#!/bin/bash
#SBATCH --job-name=f1_check_config
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
#
# F1 — validação CPU-only da config de dados configs/wla/unitree_wbt_shelf.yaml
# com o loader oficial (load_config + create_single_source_dataset).
# Não usa GPU (regra do cluster: validação leve, mas roda via Slurm).
set -euo pipefail

PSI0=/raid/user_marcospaulo/Psi0
source "$PSI0/scripts/wla/env_raid.sh"

DATA_CONFIG="$PSI0/configs/wla/unitree_wbt_shelf.yaml"
SCRATCH_CFG="$PSI0/_scratch/f1_check_unitree_wbt_shelf_noimg.yaml"
mkdir -p "$PSI0/_scratch"

cd "$WLA_REPO"
source .venv/bin/activate

# Cópia temporária da config com image_keys=[] para evitar decodificação
# de vídeo no __getitem__ (validação leve de action/state/máscaras).
.venv/bin/python - "$DATA_CONFIG" "$SCRATCH_CFG" <<'EOF'
import sys, yaml
src_path, out_path = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open(src_path))
assert len(cfg["datasets"]) == 1, "esperado exatamente 1 dataset habilitado"
cfg["datasets"][0]["image_keys"] = []
yaml.safe_dump(cfg, open(out_path, "w"))
print(f"scratch config (sem imagens): {out_path}")
EOF

.venv/bin/python - "$DATA_CONFIG" "$SCRATCH_CFG" <<'EOF'
import sys
from unifolm_wla.dataloader.multi_source_dataset.config import load_config
from unifolm_wla.dataloader.multi_source_dataset.single_source_dataset import create_single_source_dataset

data_config, scratch_cfg = sys.argv[1], sys.argv[2]

# 1) Config completa (com imagens): constrói o dataset e valida shapes de
# imagem via metadados, sem decodificar vídeo.
cfg_full = load_config(data_config)
enabled = [d for d in cfg_full.datasets if d.enabled]
assert len(enabled) == 1, f"esperado 1 source habilitado, got {len(enabled)}"
print(f"[full] data_path={enabled[0].data_path}")
print(f"[full] precollected_stats_path={enabled[0].precollected_stats_path}")
print(f"[full] image_keys={[ (ik.key, ik.role) for ik in enabled[0].image_keys ]}")
src_full = create_single_source_dataset(enabled[0], cfg_full)
print(f"[full] LEN={len(src_full)}  image_shapes={src_full._image_shapes}")

# 2) Config sem imagens: __getitem__ leve para inspecionar tensores.
cfg = load_config(scratch_cfg)
src = create_single_source_dataset(cfg.datasets[0], cfg)
print(f"[noimg] LEN={len(src)}")
s = src[0]
print(f"[noimg] action      shape={tuple(s['action'].shape)} dtype={s['action'].dtype}")
print(f"[noimg] state       shape={tuple(s['state'].shape)} dtype={s['state'].dtype}")
print(f"[noimg] action_mask shape={tuple(s['action_mask'].shape)} sum={int(s['action_mask'].sum())}")
print(f"[noimg] state_mask  shape={tuple(s['state_mask'].shape)} sum={int(s['state_mask'].sum())}")
print(f"[noimg] images={list(s['images'].keys())} image_mask={s['image_mask']}")
print(f"[noimg] task={s['task'][:80]!r}")
print(f"[noimg] action finite={bool(s['action'].isfinite().all())} state finite={bool(s['state'].isfinite().all())}")
# sanity no último frame também
s_last = src[len(src) - 1]
print(f"[noimg] last-frame ok: action shape={tuple(s_last['action'].shape)}")
print("F1_CHECK_CONFIG_OK")
EOF
