#!/usr/bin/env bash
# Copia os vídeos (câmera esquerda da cabeça) dos runs de eval para ./videos com nome plano: <run>__ep<N>__<resultado>.mp4
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=${WLA_EXP:-/raid/user_marcospaulo/experiments/wla}
mkdir -p videos
n=0
for f in "$EXP"/*/eval/*/*/*/*/head_stereo_left_*.mp4; do
  [ -e "$f" ] || continue
  run=$(echo "$f" | sed -E "s#^$EXP/([^/]+)/.*#\1#")
  ep=$(basename "$(dirname "$f")" | sed -E 's/.*__episode_//')
  res=$(basename "$f" .mp4 | sed -E 's/head_stereo_left_//')
  cp -f "$f" "videos/${run}__ep${ep}__${res}.mp4"; n=$((n+1))
done
echo "$n vídeos em $(pwd)/videos"
