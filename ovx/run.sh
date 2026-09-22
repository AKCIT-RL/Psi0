#!/usr/bin/env bash
# Host-side launcher for the OVX image. The one place that decides mounts, environment and GPU
# flags -- Docker on a workstation and Apptainer on the cluster get the same container contract.
#
#   ovx/run.sh smoke [--isaac]
#   ovx/run.sh train --dataset carry_totes --base-model <model dir> --epochs 30
#   ovx/run.sh serve --model <run dir> --port 5555
#   ovx/run.sh eval  --model <run dir> --env-id simple/<Task>-v0 --data-dir <dataset>
#   ovx/run.sh shell
#
# The image carries the environment (venvs, Isaac Sim, cuRobo) under /opt; everything a run touches
# lives under /home/ovx, the workspace:
#
#   /home/ovx/Psi0          the code -- GR00T, SIMPLE and its submodules, and these scripts
#   /home/ovx/data          datasets (read-only)
#   /home/ovx/checkpoints   base models and training runs
#   /home/ovx/cache         HF, torch, triton, wandb, Isaac Sim
#   /home/ovx/evals         evaluation outputs
#
# The code comes from THIS checkout, mounted over /home/ovx/Psi0, so a code change is tried by
# rerunning, with no rebuild. Dependency, system library and cuRobo changes still need one. Each
# subcommand runs ovx/bin/<subcommand>.sh from the mount; `--help` lists its options.
#
# Configuration, all optional (host paths, relative to the current directory -- normally the repo
# root -- and mapped to the workspace directories above):
#
#   OVX_SRC       checkout to mount             -> /home/ovx/Psi0 (default: this script's repo)
#                 OVX_SRC=image uses the copy baked into the image and mounts nothing
#   OVX_SIF       .sif image or sandbox dir -> run with apptainer
#   OVX_IMAGE     docker image (default ovx-gr00t:eval), used when OVX_SIF is not set
#   OVX_ENGINE    force apptainer or docker
#   OVX_DATA      datasets (default data/simple/simple-converted)
#   OVX_CKPT      base models and training runs  (default checkpoints)
#   OVX_CACHE     caches                         (default cache)
#   OVX_EVALS     evaluation outputs             (default evals)
#   OVX_ENV_FILE  secrets: WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN (default .env when present)
#   OVX_NV_FLAGS  apptainer GPU flags (default --nv; the image matches the OVX hosts, so --nvccli
#                 should not be needed)
#   OVX_GPUS      docker --gpus value (default all)
#
# Nothing else from the host environment reaches the container: apptainer runs with --cleanenv,
# and host caches, venvs and dotfiles stay outside.

set -euo pipefail

die() { echo "ovx/run.sh: $*" >&2; exit 1; }

CMD="${1:-}"
case "${CMD}" in
    smoke|train|serve|eval|shell) shift ;;
    *) die "usage: ovx/run.sh <smoke|train|serve|eval|shell> [options]" ;;
esac

# Absolute, symlink-free host paths: binds must name the real directory.
DATA="${OVX_DATA:-data/simple/simple-converted}"
[ -d "${DATA}" ] || die "dataset directory not found: ${DATA} (set OVX_DATA)"
DATA="$(cd "${DATA}" && pwd -P)"
mkabs() { mkdir -p "$1" && (cd "$1" && pwd -P); }
CKPT="$(mkabs "${OVX_CKPT:-checkpoints}")"
CACHE="$(mkabs "${OVX_CACHE:-cache}")"
EVALS="$(mkabs "${OVX_EVALS:-evals}")"
# HOME inside the container. setup_caches creates it for every subcommand, but `shell` runs bash
# directly and never calls it: without this, `cd ~` and any dotfile write fail in that session.
mkdir -p "${CACHE}/home"

ENV_FILE=""
if [ -n "${OVX_ENV_FILE:-}" ]; then
    [ -f "${OVX_ENV_FILE}" ] || die "OVX_ENV_FILE not found: ${OVX_ENV_FILE}"
    ENV_FILE="${OVX_ENV_FILE}"
elif [ -f .env ]; then
    ENV_FILE=".env"
fi

# The code mount. SIMPLE and its submodules are installed editable against /home/ovx/Psi0, so the
# mount has to land on exactly that path -- the checkout itself can live anywhere.
SRC_APPTAINER=(); SRC_DOCKER=()
SRC="${OVX_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
if [ "${SRC}" = image ]; then
    echo "ovx/run.sh: using the code baked into the image (OVX_SRC=image)" >&2
else
    [ -d "${SRC}" ] || die "OVX_SRC is not a directory: ${SRC}"
    SRC="$(cd "${SRC}" && pwd -P)"
    # A directory that exists but is EMPTY is exactly what an uninitialised submodule looks like,
    # so check for content, not just presence.
    have() { [ -d "${SRC}/$1" ] && [ -n "$(ls -A "${SRC}/$1" 2> /dev/null)" ]; }
    need() { have "$1" || die "OVX_SRC is missing $1 -- initialise the submodules (see ovx/README.md)"; }

    for p in src/gr00t/gr00t baselines/gr00t-n1.7 ovx/bin; do need "${p}"; done

    # SIMPLE and the four nested repos its venv installs editable: that code is read at run time,
    # so a missing one fails deep inside an eval instead of here. train and serve never import
    # SIMPLE (the train image does not even ship it), so they must not require it; smoke and shell
    # only warn, since both are useful against the train image too.
    simple_parts=(third_party/SIMPLE/src)
    for p in openpi-client gear_sonic unitree_sdk2_python decoupled_wbc; do
        simple_parts+=("third_party/SIMPLE/third_party/${p}")
    done
    case "${CMD}" in
        eval)
            for p in "${simple_parts[@]}"; do need "${p}"; done ;;
        smoke|shell)
            for p in "${simple_parts[@]}"; do
                have "${p}" || echo "ovx/run.sh: warning: ${p} is missing or empty -- anything using SIMPLE will fail" >&2
            done ;;
    esac
    SRC_APPTAINER=(--bind "${SRC}:/home/ovx/Psi0")
    SRC_DOCKER=(-v "${SRC}:/home/ovx/Psi0")
fi

if [ "${CMD}" = shell ]; then
    # Arguments go to bash, so `run.sh shell -lc 'cmd'` runs one command and exits -- what
    # ovx/slurm/attach.sh and any non-interactive caller need.
    INNER=(/bin/bash "$@")
else
    INNER=("/home/ovx/Psi0/ovx/bin/${CMD}.sh" "$@")
fi

# Job identity: port selection and output naming inside the container, and the node name for the
# provenance record (--cleanenv drops everything not listed here).
PASS_VARS=(CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_ARRAY_TASK_ID SLURMD_NODENAME)

ENGINE="${OVX_ENGINE:-}"
[ -n "${ENGINE}" ] || { [ -n "${OVX_SIF:-}" ] && ENGINE=apptainer || ENGINE=docker; }

case "${ENGINE}" in
apptainer)
    # A .sif file, or a sandbox directory (apptainer build --sandbox): same runtime, no squashfs copy.
    [ -n "${OVX_SIF:-}" ] && [ -e "${OVX_SIF}" ] || die "OVX_SIF not set or not found: ${OVX_SIF:-}"
    command -v apptainer > /dev/null || module load apptainer 2> /dev/null || true
    command -v apptainer > /dev/null || die "apptainer not found"

    read -ra NV <<< "${OVX_NV_FLAGS:---nv}"
    args=(exec "${NV[@]}" --cleanenv --no-home --pwd /home/ovx/Psi0
          "${SRC_APPTAINER[@]}"
          --bind "${DATA}:/home/ovx/data:ro"
          --bind "${CKPT}:/home/ovx/checkpoints"
          --bind "${CACHE}:/home/ovx/cache"
          --bind "${EVALS}:/home/ovx/evals"
          --env HOME=/home/ovx/cache/home)
    [ -z "${ENV_FILE}" ] || args+=(--env-file "${ENV_FILE}")
    for v in "${PASS_VARS[@]}"; do
        [ -z "${!v:-}" ] || args+=(--env "${v}=${!v}")
    done
    # --nvccli exposes GPUs through NVIDIA_VISIBLE_DEVICES, which apptainer sets to "all": on a
    # shared node that means other people's GPUs too. Pin it to the allocation.
    for f in "${NV[@]}"; do
        if [ "${f}" = --nvccli ]; then
            gpus="${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-${CUDA_VISIBLE_DEVICES:-}}}"
            [ -z "${gpus}" ] || args+=(--env "NVIDIA_VISIBLE_DEVICES=${gpus}")
        fi
    done
    exec apptainer "${args[@]}" "${OVX_SIF}" "${INNER[@]}"
    ;;
docker)
    command -v docker > /dev/null || die "docker not found"
    IMAGE="${OVX_IMAGE:-ovx-gr00t:eval}"
    args=(run --rm --init --ipc=host
          --gpus "${OVX_GPUS:-all}"
          --user "$(id -u):$(id -g)"
          -e HOME=/home/ovx/cache/home -e USER=ovx -e LOGNAME=ovx
          -w /home/ovx/Psi0
          "${SRC_DOCKER[@]}"
          -v "${DATA}:/home/ovx/data:ro"
          -v "${CKPT}:/home/ovx/checkpoints"
          -v "${CACHE}:/home/ovx/cache"
          -v "${EVALS}:/home/ovx/evals")
    # -i always, so `echo 'cmd' | ovx/run.sh shell` works with no terminal; -t only with one.
    args+=(-i)
    if [ -t 0 ] && [ -t 1 ]; then args+=(-t); fi
    if [ -n "${ENV_FILE}" ]; then
        # docker --env-file takes values literally (quotes included) and rejects `export`; strip
        # both so one .env serves apptainer and docker.
        clean_env="$(mktemp)"
        trap 'rm -f "${clean_env}"' EXIT
        sed -E -e '/^[[:space:]]*(#|$)/d' -e 's/^[[:space:]]*export[[:space:]]+//' \
               -e "s/^([A-Za-z_][A-Za-z0-9_]*)=[\"'](.*)[\"'][[:space:]]*$/\1=\2/" \
               "${ENV_FILE}" > "${clean_env}"
        args+=(--env-file "${clean_env}")
    fi
    for v in "${PASS_VARS[@]}"; do
        [ -z "${!v:-}" ] || args+=(-e "${v}=${!v}")
    done
    # Not exec: the trap must remove the sanitised env file afterwards.
    docker "${args[@]}" "${IMAGE}" "${INNER[@]}"
    ;;
*)
    die "OVX_ENGINE must be apptainer or docker, not '${ENGINE}'"
    ;;
esac
