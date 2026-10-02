#!/usr/bin/env bash
#SBATCH --job-name=s2-gl-check-nv
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only (sem GPU alocada, sem CUDA): só carregamento de libs GL com --nv, para achar por que o EGL falha no container.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
SB=$RAID/containers/simple-260829.sandbox
export APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp; mkdir -p "$APPTAINER_TMPDIR"
apptainer exec --nv --cleanenv --env MUJOCO_GL=egl --env LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/.singularity.d/libs "$SB" bash -c '
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "## /.singularity.d/libs (GL/EGL)"; ls -la /.singularity.d/libs 2>/dev/null | grep -iE "egl|gl|nvidia-e|nvidia-gl" | head -30
echo "## ldd libEGL.so.1 (primeiras falhas)"; ldd /.singularity.d/libs/libEGL.so.1 2>&1 | grep -iE "not found|version" | head
for l in libEGL.so.1 libGL.so.1 libGLX.so.0 libGLdispatch.so.0 libOpenGL.so.0 libEGL_nvidia.so.0; do
  /workspace/simple/.venv/bin/python -c "
import ctypes,sys
try:
    h=ctypes.CDLL(\"$l\"); print(\"CDLL $l OK\")
except OSError as e: print(\"CDLL $l FAIL\", e)"
done
/workspace/simple/.venv/bin/python -c "
import ctypes.util;print(\"find_library EGL:\", ctypes.util.find_library(\"EGL\"))
try:
    import mujoco; print(\"import mujoco OK\")
except Exception as e: print(\"import mujoco FAIL\", type(e).__name__, e)" 2>&1 | tail -3
'
echo GL_NV_DONE
