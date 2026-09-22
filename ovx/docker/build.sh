#!/usr/bin/env bash
# Build the OVX image -- after checking that the working tree holds what the image must contain.
# The Dockerfile copies files from disk, not from git: a nested submodule at the wrong commit or a
# Git LFS pointer would end up inside the image without any error.
#
#   ovx/docker/build.sh                         both targets (train, eval)
#   ovx/docker/build.sh train                   one target
#   ovx/docker/build.sh eval --sif ~/images     also convert to a .sif for apptainer
#
#   UV_NO_CACHE=1 MAX_JOBS=2 ovx/docker/build.sh eval     leaner on disk, gentler on RAM
#
# Tags: <name>:<target> and <name>:<target>-<version>, with version YYYY.MM.DD-<psi0 commit>
# (-dirty when the image inputs have uncommitted changes). Name: OVX_IMAGE_NAME, default ovx-gr00t.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${ROOT}"

NAME="${OVX_IMAGE_NAME:-ovx-gr00t}"
TARGETS=(); SIF_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        train|eval) TARGETS+=("$1"); shift ;;
        --sif)      SIF_DIR="$2"; shift 2 ;;
        -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
        *)          echo "build.sh: unknown argument: $1" >&2; exit 1 ;;
    esac
done
[ "${#TARGETS[@]}" -gt 0 ] || TARGETS=(train eval)

needs_simple=0
for t in "${TARGETS[@]}"; do [ "${t}" = eval ] && needs_simple=1; done

problems=()
[ -f ovx/docker/gr00t/uv.lock ] \
    || problems+=("ovx/docker/gr00t/uv.lock missing -- run 'uv lock' in ovx/docker/gr00t")

# `git submodule status` prefixes '-' when not initialised and '+' when checked out at a commit
# other than the one recorded -- the second is how a rolled-back decoupled_wbc slips in.
check_submodule() {   # check_submodule <repo> <path>
    local st
    st="$(git -C "$1" submodule status -- "$2" 2>/dev/null || true)"
    case "${st:0:1}" in
        "") problems+=("$1/$2: not a submodule here") ;;
        -)  problems+=("$1/$2 not initialised -- git -C $1 submodule update --init $2") ;;
        +)  problems+=("$1/$2 is not at the recorded commit -- git -C $1 submodule update $2") ;;
    esac
}
if [ "${needs_simple}" = 1 ]; then
    check_submodule . third_party/SIMPLE
    if [ -d third_party/SIMPLE/src ]; then
        for sub in openpi-client gear_sonic decoupled_wbc unitree_sdk2_python curobo; do
            check_submodule third_party/SIMPLE "third_party/${sub}"
        done
        # git lfs ls-files marks files that are still pointers with '-'.
        if git -C third_party/SIMPLE lfs ls-files 2>/dev/null | grep -q ' - '; then
            problems+=("third_party/SIMPLE has Git LFS files not downloaded -- git -C third_party/SIMPLE lfs pull")
        fi
    fi
fi
if [ "${#problems[@]}" -gt 0 ]; then
    printf 'build.sh: %s\n' "${problems[@]}" >&2
    exit 1
fi

# Must return 0 when clean: it runs inside an assignment, where a failure trips `set -e`.
dirty() {
    if [ -n "$(git -C "$1" status --porcelain -- "${@:2}")" ]; then printf -- '-dirty'; fi
}
PSI0_COMMIT="$(git rev-parse --short HEAD)$(dirty . ovx src/gr00t src/psi baselines/gr00t-n1.7)"
SIMPLE_COMMIT="none"
if [ "${needs_simple}" = 1 ]; then
    SIMPLE_COMMIT="$(git -C third_party/SIMPLE rev-parse --short HEAD)$(dirty third_party/SIMPLE .)"
fi
VERSION="${OVX_VERSION:-$(date +%Y.%m.%d)-${PSI0_COMMIT}}"

# UV_NO_CACHE=1 ovx/docker/build.sh ...  keeps uv's downloads out of the build cache (~30 GB less
# disk on the build machine, slower rebuilds). Passed only when set: uv rejects an empty value.
EXTRA_BUILD_ARGS=()
[ -z "${UV_NO_CACHE:-}" ] || EXTRA_BUILD_ARGS+=(--build-arg "UV_NO_CACHE=${UV_NO_CACHE}")
[ -z "${MAX_JOBS:-}" ]    || EXTRA_BUILD_ARGS+=(--build-arg "MAX_JOBS=${MAX_JOBS}")   # cuRobo/nvcc: cap parallel compiles on a small-RAM host

for t in "${TARGETS[@]}"; do
    echo "==> ${NAME}:${t}-${VERSION}  (psi0 ${PSI0_COMMIT}, simple ${SIMPLE_COMMIT})"
    docker buildx build \
        -f ovx/docker/Dockerfile \
        --target "${t}" \
        "${EXTRA_BUILD_ARGS[@]}" \
        --build-arg OVX_VERSION="${VERSION}" \
        --build-arg PSI0_COMMIT="${PSI0_COMMIT}" \
        --build-arg SIMPLE_COMMIT="${SIMPLE_COMMIT}" \
        -t "${NAME}:${t}" \
        -t "${NAME}:${t}-${VERSION}" \
        --load \
        .
    if [ -n "${SIF_DIR}" ]; then
        mkdir -p "${SIF_DIR}"
        sif="${SIF_DIR}/${NAME}_${t}_${VERSION}.sif"
        echo "==> ${sif}"
        apptainer build --force "${sif}" "docker-daemon://${NAME}:${t}-${VERSION}"
    fi
done
