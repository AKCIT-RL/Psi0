#!/usr/bin/env bash
#
# Does --nvccli leak other people's GPUs into the container?
#
# Apptainer emits this when --nvccli is used:
#     INFO: Setting 'NVIDIA_VISIBLE_DEVICES=all' to emulate legacy GPU binding.
#
# "all" means every GPU on the node. On a shared machine that includes the ones SLURM
# allocated to other users. The device cgroup usually blocks the access anyway, but
# "usually" is not something to train 48 hours on top of — and a container that can see a
# colleague's GPU is one CUDA_VISIBLE_DEVICES mistake away from using it.
#
# This job allocates one GPU and only *lists* devices. It runs no CUDA, allocates no VRAM,
# and finishes in seconds.
#
# Usage:
#   sbatch check_gpu_visibility.sh
#   cat logs/gpucheck-<jobid>.out
#
# Verdict:
#   PASS  -> the container sees exactly the GPU SLURM granted; go ahead and train
#   FAIL  -> --nvccli leaks the whole node; fix the image instead (matching base distro)

#SBATCH --job-name=gpu-visibility
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/gpucheck-%j.out
#SBATCH --error=logs/gpucheck-%j.err

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(realpath "$(dirname "${BASH_SOURCE[0]}")")}}"
SIF_PATH="${SIF_PATH:-${PROJECT_DIR}/industrial_humanoids_psi0-train.gr00t_devel.sif}"

module load apptainer 2>/dev/null || true

rule() { printf '%s\n' "-----------------------------------------------------------------"; }
# UUIDs, not indices: indices are renumbered per container, UUIDs are not.
uuids() { grep -oE 'GPU-[0-9a-f-]+' | sort; }

echo "================================================================="
echo "GPU visibility under --nvccli"
echo "Job ${SLURM_JOB_ID:-<local>} on ${SLURM_NODELIST:-$(hostname)} — $(date)"
echo "================================================================="

[ -f "${SIF_PATH}" ] || { echo "ERROR: image not found: ${SIF_PATH}"; exit 1; }

########################
# 1. WHAT SLURM GRANTED
########################

rule
echo "1. Allocation seen from the host side of this job"
rule
echo "SLURM_JOB_GPUS       = ${SLURM_JOB_GPUS:-<unset>}"
echo "GPU_DEVICE_ORDINAL   = ${GPU_DEVICE_ORDINAL:-<unset>}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo
echo "nvidia-smi -L (the device cgroup already filters this):"
nvidia-smi -L 2>&1 | sed 's/^/    /'

HOST_UUIDS=$(nvidia-smi -L 2>/dev/null | uuids)
HOST_N=$(printf '%s\n' "${HOST_UUIDS}" | grep -c . || true)
echo
echo "-> ${HOST_N} GPU(s) granted to this job"

TOTAL_N=$(scontrol show node "${SLURM_NODELIST:-$(hostname)}" 2>/dev/null \
          | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
echo "-> ${TOTAL_N:-?} GPU(s) exist on the node in total"

########################
# 2. WHAT THE CONTAINER SEES
########################

ALLOC="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-${CUDA_VISIBLE_DEVICES:-}}}"

probe() {   # probe <label> <apptainer flags...>
    local label="$1"; shift
    rule
    echo "2.${PROBE_N}. ${label}"
    rule
    local out
    out=$(apptainer exec "$@" "${SIF_PATH}" nvidia-smi -L 2>&1)
    printf '%s\n' "${out}" | sed 's/^/    /'
    printf '%s\n' "${out}" | uuids > "/tmp/gpucheck.$$.${PROBE_N}"
    PROBE_N=$((PROBE_N + 1))
}

PROBE_N=1
probe "plain --nv (reference)"                       --nv
probe "--nv --nvccli, Apptainer's default"           --nv --nvccli
if [ -n "${ALLOC}" ]; then
    probe "--nv --nvccli + NVIDIA_VISIBLE_DEVICES=${ALLOC}" \
          --nv --nvccli --env NVIDIA_VISIBLE_DEVICES="${ALLOC}"
fi

########################
# 3. VERDICT
########################

rule
echo "3. Verdict"
rule

printf '%s\n' "${HOST_UUIDS}" > "/tmp/gpucheck.$$.host"

for i in 1 2 3; do
    f="/tmp/gpucheck.$$.${i}"
    [ -f "${f}" ] || continue
    case "${i}" in
        1) name="--nv" ;;
        2) name="--nv --nvccli (default)" ;;
        3) name="--nv --nvccli (pinned)" ;;
    esac
    n=$(grep -c . "${f}" || true)
    if diff -q "${f}" "/tmp/gpucheck.$$.host" >/dev/null 2>&1; then
        printf '  %-34s %s GPU(s)  MATCHES the allocation\n' "${name}" "${n}"
    else
        printf '  %-34s %s GPU(s)  DIFFERS from the allocation\n' "${name}" "${n}"
        comm -23 "${f}" "/tmp/gpucheck.$$.host" 2>/dev/null | sed 's/^/      extra: /'
    fi
done

echo
if diff -q "/tmp/gpucheck.$$.2" "/tmp/gpucheck.$$.host" >/dev/null 2>&1; then
    echo "  RESULT: PASS — --nvccli exposes only what SLURM granted, with no extra env."
elif [ -f "/tmp/gpucheck.$$.3" ] && diff -q "/tmp/gpucheck.$$.3" "/tmp/gpucheck.$$.host" >/dev/null 2>&1; then
    echo "  RESULT: PASS WITH PINNING — --nvccli alone leaks the node, but setting"
    echo "          NVIDIA_VISIBLE_DEVICES from the allocation fixes it. submit_slurm.sh"
    echo "          already does that when NV_FLAGS contains --nvccli."
else
    echo "  RESULT: FAIL — the container sees GPUs this job was not granted, even pinned."
    echo "          Do not train this way. Rebuild the image on a base matching the host"
    echo "          distro so plain --nv works."
fi

rm -f "/tmp/gpucheck.$$."* 2>/dev/null || true
echo
echo "Finished: $(date)"
