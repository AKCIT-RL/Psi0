#!/usr/bin/env bash
# Copia os vídeos (câmera esquerda) de um run para ./videos/<run>/<episodio>.mp4. Uso: collect_videos.sh <run> [<run>...]
set -euo pipefail
cd "$(dirname "$0")/../.."; 
for run in "$@"; do
  for f in $(find "$HOME/wla_data/experiments/wla/$run/eval" -name "head_stereo_left*.mp4"); do
    ep=$(basename "$(dirname "$f")"); mkdir -p "videos/$run"; cp "$f" "videos/$run/${ep}.mp4"
  done
done
ls -lh videos
