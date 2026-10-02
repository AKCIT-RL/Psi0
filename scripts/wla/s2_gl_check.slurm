#!/usr/bin/env bash
#SBATCH --job-name=s2-gl-check
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: diagnóstico do carregamento de GL/EGL no container (sem criar contexto).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
SB=$RAID/containers/simple-260829.sandbox
export APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp; mkdir -p "$APPTAINER_TMPDIR"
for SHIM in "" "$PWD/scripts/wla/pyshim"; do
  echo "=== PYTHONPATH shim='$SHIM'"
  apptainer exec --cleanenv --env MUJOCO_GL=egl ${SHIM:+--env PYTHONPATH=$SHIM -B $SHIM:$SHIM} "$SB" bash -c '
    which ldconfig gcc ld 2>&1 | head -3
    /workspace/simple/.venv/bin/python - <<PY
import ctypes.util, ctypes
print("find_library EGL/GL:", ctypes.util.find_library("EGL"), ctypes.util.find_library("GL"))
try:
    ctypes.CDLL("libEGL.so.1"); print("CDLL libEGL.so.1 OK")
except OSError as e: print("CDLL libEGL FAIL", e)
try:
    import mujoco; print("import mujoco OK", mujoco.__version__)
except Exception as e: print("import mujoco FAIL", type(e).__name__, e)
PY'
done
echo GL_CHECK_DONE
