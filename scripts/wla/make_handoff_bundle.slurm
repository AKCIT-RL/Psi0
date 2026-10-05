#!/usr/bin/env bash
#SBATCH --job-name=wla-bundle
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: empacota o que NÃO se baixa de nenhum remoto, p/ rodar o eval closed-loop em outra máquina (docs/wla/HANDOFF_4090.md).
# Saída: $RAID/handoff/wla_handoff.zip (+ MANIFEST.sha256). Layout interno = $WLA_HOME da máquina de destino.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
H=$RAID/handoff; S=$H/stage/wla_handoff; ZIP=$H/wla_handoff.zip
mkdir -p "$S/checkpoints/wla/f5_psi0_tote/final_model" "$S/experiments/wla/s0_eval_data/roots" "$S/experiments/wla/s0_eval_data/oracle" \
         "$S/models/UnifoLM-WLA-1.0-Base" "$S/reference_results"
C=$WLA_CKPT/f5_psi0_tote; E=$WLA_EXP/s0_eval_data
cp "$C/final_model/model.safetensors" "$S/checkpoints/wla/f5_psi0_tote/final_model/"
cp "$C/config.yaml" "$C/dataset_statistics.json" "$C/git_commit.txt" "$S/checkpoints/wla/f5_psi0_tote/"
for s in val test; do cp -rL "$E/roots/$s" "$S/experiments/wla/s0_eval_data/roots/$s"; done      # symlinks resolvidos (vídeos/meta reais)
cp -rL "$E/oracle/val" "$S/experiments/wla/s0_eval_data/oracle/val"; cp "$E/git_commit.txt" "$S/experiments/wla/s0_eval_data/"
cp -r "$RAID/simple_data" "$S/simple_data"
cp -r "$WLA_MODELS/UnifoLM-WLA-1.0-Base/tokenizer" "$S/models/UnifoLM-WLA-1.0-Base/tokenizer"
cp "$WLA_EXP/s8_offline_val3/summary.csv" "$S/reference_results/s8_offline_val3_summary.csv" 2>/dev/null || true
cp "$WLA_EXP/s9_it1/metrics.json" "$S/reference_results/s9_it1_mujoco_metrics.json" 2>/dev/null || true
(cd "$S" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
rm -f "$ZIP"   # sem binário zip no nó: zipfile da stdlib (safetensors sem compressão)
python3 - "$H/stage" "$ZIP" <<'PY'
import os, sys, zipfile
root, out = sys.argv[1:3]
with zipfile.ZipFile(out, "w", allowZip64=True) as z:
    for d, _, fs in os.walk(os.path.join(root, "wla_handoff")):
        for f in sorted(fs):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, root), compress_type=zipfile.ZIP_STORED if f.endswith((".safetensors", ".mp4")) else zipfile.ZIP_DEFLATED)
PY
ls -lh "$ZIP"; du -sh "$S"/*; sha256sum "$ZIP" | tee "$H/wla_handoff.zip.sha256"
echo BUNDLE_DONE
