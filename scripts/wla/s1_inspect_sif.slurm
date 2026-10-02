#!/usr/bin/env bash
#SBATCH --job-name=s1-inspect-sif
#SBATCH --partition=b200n1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x-%j.out
# CPU-only: inspeciona a imagem do SIMPLE (layout, venv, pacotes).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/wla/env_raid.sh
export APPTAINER_CACHEDIR=$RAID/cache/apptainer APPTAINER_TMPDIR=$RAID/cache/apptainer/tmp
SIF=$RAID/containers/simple-260829.sif
apptainer exec "$SIF" bash -lc '
echo "## /workspace"; ls /workspace /workspace/simple 2>&1 | head -40
echo "## third_party"; ls /workspace/simple/third_party 2>&1
echo "## python"; ls /workspace/simple/.venv/bin | grep -E "^(python|eval|pip)" | head
P=/workspace/simple/.venv/bin/python
$P -c "import sys;print(sys.version)"
$P -m pip --version 2>&1 | head -1
(uv pip list --python $P 2>/dev/null || $P -m pip list 2>/dev/null) | grep -iE "^(simple|gear|decoupled|lerobot|nvidia-curobo|curobo|isaacsim|mujoco|torch|onnx|numpy|requests|pyarrow|pin|warp|huggingface)" 
echo "## editable locations"; $P -c "
import importlib
for m in (\"simple\",\"gear_sonic\",\"decoupled_wbc\",\"curobo\",\"lerobot\",\"mujoco\"):
    try:
        x=importlib.import_module(m); print(m, getattr(x,\"__file__\",None), getattr(x,\"__version__\",\"\"))
    except Exception as e: print(m,\"ERR\",type(e).__name__,e)
" 2>&1 | tail -12
echo "## nvcc"; which nvcc; nvcc --version 2>&1 | tail -2
echo "## git info in image"; cd /workspace/simple && (git log -1 --format=%h 2>&1 | head -1); cat /workspace/simple/pyproject.toml | grep -n "lerobot" | head -3
'
echo INSPECT_DONE
