# syntax=docker/dockerfile:1.7
#
# OVX pipeline image: GR00T N1.7 training + policy server, and SIMPLE closed-loop evaluation.
#
# The image is the ENVIRONMENT; the code is a checkout mounted over /home/ovx/Psi0 at run time
# (ovx/run.sh does it by default). The same code is also baked in, so the image runs standalone --
# a mount simply replaces it. That is what lets a research change to GR00T, to SIMPLE or to one of
# its submodules be tried with no rebuild; only dependency, system library and cuRobo changes need
# one.
#
# The path matters: SIMPLE and its submodules are installed EDITABLE into /opt/venvs/simple, which
# records absolute paths. Build path and mount path must be the same, hence /home/ovx/Psi0
# everywhere.
#
# One Dockerfile, two targets on a shared base:
#
#   train   base + /opt/venvs/gr00t                           what training jobs run
#   eval    base + /opt/venvs/simple (Isaac Sim, MuJoCo)      server and simulator side by side
#                + /opt/venvs/gr00t
#
# Every tool gets its own locked venv; they never share site-packages. The build context is the
# repo root. Build through ovx/docker/build.sh, which checks the submodules and Git LFS first:
# the image copies the working tree, so a nested submodule checked out at the wrong commit would
# silently end up inside it.
#
# Ubuntu 24.04 on purpose: it matches the OVX hosts (glibc 2.39), so the driver libraries that
# apptainer --nv injects load with plain --nv, without --nvccli.

ARG UV_VERSION=0.11.16

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ------------------------------------------------------------------------------------ base
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1

# build-essential  triton JIT-compiles its kernels and launcher stub with the system gcc
# ffmpeg           torchcodec decodes the dataset videos through the system FFmpeg libraries
# GL / X / Vulkan  what Isaac Sim and MuJoCo (EGL) load: the ELF dependencies of a working
#                  SIMPLE venv plus the loaders they dlopen at runtime. The NVIDIA side
#                  (libcuda, libEGL_nvidia, libGLX_nvidia) comes from the host driver.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git git-lfs build-essential \
        ffmpeg \
        libgl1 libglx0 libegl1 libgles2 libglu1-mesa libvulkan1 \
        libglib2.0-0t64 libgomp1 libsdl2-2.0-0 libevdev2 \
        libx11-6 libxext6 libxrandr2 libxi6 libxt6t64 libsm6 libice6 \
        libxcursor1 libxinerama1 libxkbcommon0 libxrender1 libxfixes3 libxxf86vm1 \
        libfontconfig1 libfreetype6 libnss3 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# The EGL vendor file registers the NVIDIA driver with the loader. It sits at the exact path both
# runtimes overwrite with the host's copy (docker's toolkit, apptainer --nv), so it can never
# conflict -- it only matters on a host that ships none. Same content as NVIDIA's own images.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
 && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# The Vulkan equivalent is deliberately NOT in a loader search path: docker's toolkit writes its own
# under /etc/vulkan/icd.d, and two manifests for the same driver make the loader register the GPU
# twice, which segfaults Isaac Sim at startup. Kept here as an emergency spare for a host that has
# no manifest at all; common.sh has vulkan_fallback_icd() to point the loader at it, unused by
# default.
RUN mkdir -p /opt/ovx/vulkan \
 && printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}\n' \
        > /opt/ovx/vulkan/nvidia_icd.json

# The workspace: everything a run touches (code, datasets, checkpoints, caches, results) lives
# under /home/ovx, and /opt holds only installed software. These are the mount points ovx/run.sh
# binds to. Apptainer can usually create them on the fly, but not on every configuration, and a
# missing one fails the job before any log line explains why.
RUN mkdir -p /home/ovx/Psi0 /home/ovx/data /home/ovx/checkpoints /home/ovx/cache /home/ovx/evals

COPY --from=uv /uv /uvx /usr/local/bin/

# ------------------------------------------------------------------------------------ gr00t
FROM base AS gr00t

# UV_NO_CACHE=1 (a build arg, passed by build.sh when set) keeps uv's downloads out of the build:
# each uv step uses a temporary cache instead of the persistent cache mount. Saves ~30 GB of disk
# on the build machine at the price of slower rebuilds. Unset by default; every stage running uv
# declares it.
ARG UV_NO_CACHE

# Build-time project directory: it holds only the manifest pair uv resolves from. The venv itself
# goes to /opt/venvs/gr00t (UV_PROJECT_ENVIRONMENT), like every other environment in the image.
WORKDIR /opt/ovx/gr00t
COPY ovx/docker/gr00t/pyproject.toml ovx/docker/gr00t/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/gr00t uv sync --frozen

# The model code is not installed into the venv: it goes on PYTHONPATH (set by the ovx/bin scripts,
# never globally, so the SIMPLE venv never sees these packages). Only what training and serving
# import; the server needs psi.deploy.helpers for its wire format (psi.deploy is a namespace
# package, it has no __init__.py).
COPY src/gr00t/gr00t /home/ovx/Psi0/src/gr00t/gr00t
COPY src/psi/__init__.py /home/ovx/Psi0/src/psi/__init__.py
COPY src/psi/deploy/helpers.py /home/ovx/Psi0/src/psi/deploy/helpers.py
COPY baselines/gr00t-n1.7 /home/ovx/Psi0/baselines/gr00t-n1.7

# ------------------------------------------------------------------------------------ simple-deps
FROM base AS simple-deps

ARG UV_NO_CACHE

ENV GIT_LFS_SKIP_SMUDGE=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.7.7 \
    UV_HTTP_TIMEOUT=1800

WORKDIR /home/ovx/Psi0/third_party/SIMPLE

# The locked install a working workstation uses: the lerobot group plus the full sonic group,
# minus the two teleop-only packages that cannot be built here (televuer's submodule is not
# initialised, xrobotoolkit needs a native SDK build). eval-decoupled-wbc imports neither.
# Dependencies first: Isaac Sim alone is several GB, and this layer then survives code changes.
ARG SIMPLE_SYNC="--frozen --group lerobot --group sonic --no-install-package televuer --no-install-package xrobotoolkit-sdk"
COPY third_party/SIMPLE/pyproject.toml third_party/SIMPLE/uv.lock third_party/SIMPLE/README.md ./
COPY third_party/SIMPLE/third_party ./third_party
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/simple uv sync ${SIMPLE_SYNC} --no-install-project

COPY third_party/SIMPLE/src ./src
COPY third_party/SIMPLE/configs ./configs
COPY third_party/SIMPLE/data ./data
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/venvs/simple uv sync ${SIMPLE_SYNC}

# ------------------------------------------------------------------------------------ curobo
# cuRobo is a hard import-time dependency of SIMPLE (simple/robots/mixin.py raises without it) that
# is not in SIMPLE's lock: SIMPLE installs it from its submodule, compiling five CUDA extensions,
# and when they are missing cuRobo tries to JIT-compile them at runtime with an nvcc the image does
# not have. So build a wheel here, with the CUDA 12.8 toolkit that matches the SIMPLE venv's torch
# 2.7.0+cu128, and let only the wheel reach the image.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS cuda-toolkit

FROM simple-deps AS curobo

ARG UV_NO_CACHE

COPY --from=cuda-toolkit /usr/local/cuda-12.8 /usr/local/cuda-12.8
ENV CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:${PATH}

# Every GPU the image is meant to run on: 8.0 A100, 8.6 RTX 30xx, 8.9 RTX 4090 and L40, 9.0 H100,
# 12.0 RTX 50xx. +PTX lets newer GPUs JIT the last one through the driver.
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0+PTX"
ARG MAX_JOBS=4

# This stage is thrown away, so the build tools it adds to the venv never reach the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venvs/simple/bin/python pip setuptools ninja \
        setuptools-scm==10.0.5 vcs-versioning==1.1.1 \
 && cd /home/ovx/Psi0/third_party/SIMPLE/third_party/curobo \
 && TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
    /opt/venvs/simple/bin/python -m pip wheel --no-build-isolation --no-deps -w /wheels .

# ------------------------------------------------------------------------------------ simple
FROM simple-deps AS simple

ARG UV_NO_CACHE

# System libraries the SIMPLE venv's native wheels link against beyond the base set: the ELF
# dependencies of every .so in a working workstation venv, mapped to their packages. envlogger
# needs libgmpxx at import; much of the rest (Qt platform and SQL plugins, GTK, MPI, RealSense
# USB) is optional, but finding out which one by one costs a rebuild per library. Kept out of the
# base stage so the GR00T side does not carry them, and so changing them does not invalidate the
# venv layers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgmp10 libgmpxx4ldbl \
        libacl1 libatomic1 libbrotli1 libbz2-1.0 libcrypt1 libexpat1 liblz4-1 liblzma5 \
        libuuid1 libzstd1 libudev1 libusb-1.0-0 libpcsclite1 libgpgme11t64 libgssapi-krb5-2 \
        libdrm2 libgbm1 libx11-xcb1 libxkbcommon-x11-0 \
        libxcb1 libxcb-cursor0 libxcb-glx0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-randr0 libxcb-render0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 \
        libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xkb1 \
        libwayland-client0 libwayland-cursor0 libwayland-egl1 \
        libgtk-3-0t64 libatk1.0-0t64 libcairo2 libcairo-gobject2 libgdk-pixbuf-2.0-0 \
        libharfbuzz0b libpango-1.0-0 libpangocairo-1.0-0 \
        libcups2t64 libpulse0 libspeechd2 \
        libmysqlclient21 libodbc2 libpq5 \
        libibverbs1 ibverbs-providers librdmacm1t64 libucx0 libopenmpi3t64 \
    && rm -rf /var/lib/apt/lists/*

# The cuRobo wheel plus the dependencies it needs that the synced groups do not install, pinned to
# the versions of a working workstation venv. Some of them are in SIMPLE's lock, but only for
# groups not synced here -- warp-lang 1.7.0 among them, which SIMPLE's own installer pins the same
# way. --no-deps so nothing locked moves.
#
# Then two checks, neither needing a GPU: walk cuRobo's whole requirement tree and name anything
# still missing, and load the five CUDA extensions (torch first: they link against its libraries).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=curobo,source=/wheels,target=/tmp/wheels <<'EOF'
set -eu
uv pip install --python /opt/venvs/simple/bin/python --no-deps /tmp/wheels/*.whl \
    lazy-loader==0.5 numpy-quaternion==2024.0.13 scikit-image==0.25.2 \
    setuptools-scm==10.0.5 tifffile==2025.5.10 vcs-versioning==1.1.1 \
    warp-lang==1.7.0 wheel==0.47.0 yourdfpy==0.0.60

/opt/venvs/simple/bin/python - <<'PY'
import importlib.metadata as md
import re
import sys

from packaging.requirements import Requirement

seen, missing, stack = set(), [], ["nvidia-curobo"]
while stack:
    name = re.sub(r"[-_.]+", "-", stack.pop()).lower()
    if name in seen:
        continue
    seen.add(name)
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        missing.append(name)
        continue
    for spec in dist.requires or []:
        req = Requirement(spec)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            stack.append(req.name)
if missing:
    sys.exit("cuRobo dependencies missing: " + ", ".join(sorted(missing)))
print(f"curobo requirement tree complete: {len(seen)} packages")

import torch  # noqa: E402,F401 -- first: the extensions link against its libraries
import curobo.curobolib.geom_cu  # noqa: E402,F401
import curobo.curobolib.kinematics_fused_cu  # noqa: E402,F401
import curobo.curobolib.lbfgs_step_cu  # noqa: E402,F401
import curobo.curobolib.line_search_cu  # noqa: E402,F401
import curobo.curobolib.tensor_step_cu  # noqa: E402,F401
import warp  # noqa: E402,F401
print("curobo ok")
PY
EOF

# Importing the eval CLI and the GR00T agent walks SIMPLE's whole import graph (envs, tasks, robots,
# cuRobo): a missing module fails the build here, not the first evaluation job on the cluster.
RUN OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y MUJOCO_GL=egl \
    /opt/venvs/simple/bin/python -c "import simple.cli.eval_decoupled_wbc, \
        simple.baselines.gr00t_n16_decoupled_wbc; print('simple eval imports ok')"

# Isaac Sim's pip install runs Kit in portable mode, rooted at site-packages/omni: on first start
# Kit creates cache/, data/ and logs/ there. That directory is read-only in the image (to a non-root
# docker user, and always under apptainer), so Kit cannot create them, the RTX pipeline cache fails
# to initialise and the app hangs after "app ready". Point the three into the workspace cache;
# simple_env in ovx/bin/common.sh creates the targets. The install ships none of them.
RUN cd /opt/venvs/simple/lib/python3.10/site-packages/omni \
 && ln -s /home/ovx/cache/isaac/cache cache \
 && ln -s /home/ovx/cache/isaac/data data \
 && ln -s /home/ovx/cache/isaac/logs logs

# ------------------------------------------------------------------------------------ train
FROM gr00t AS train

ARG OVX_VERSION=dev
ARG PSI0_COMMIT=unknown
ARG SIMPLE_COMMIT=unknown

COPY --chmod=0755 ovx/bin /home/ovx/Psi0/ovx/bin
COPY --chmod=0755 ovx/run.sh /home/ovx/Psi0/ovx/run.sh
RUN printf '{"version": "%s", "target": "train", "psi0_commit": "%s", "simple_commit": "%s"}\n' \
        "${OVX_VERSION}" "${PSI0_COMMIT}" "${SIMPLE_COMMIT}" > /opt/ovx/build-info.json

ENV PATH=/home/ovx/Psi0/ovx/bin:${PATH}
WORKDIR /home/ovx/Psi0
CMD ["/bin/bash"]

# ------------------------------------------------------------------------------------ eval
# Built on the simple stage with the GR00T side copied in, not the other way round: copying the
# Isaac Sim venv (~20 GB) into a new layer would store it twice on every build machine, while the
# GR00T venv is a fraction of that.
FROM simple AS eval

ARG OVX_VERSION=dev
ARG PSI0_COMMIT=unknown
ARG SIMPLE_COMMIT=unknown

# /opt/python merges: the simple stage put CPython 3.10 there, the gr00t stage 3.12. Each venv's
# bin/python points at its own interpreter by absolute path, identical in both stages. The code
# copies merge the same way, into the tree that already holds third_party/SIMPLE.
COPY --from=gr00t /opt/python /opt/python
COPY --from=gr00t /opt/venvs/gr00t /opt/venvs/gr00t
COPY --from=gr00t /home/ovx/Psi0/src /home/ovx/Psi0/src
COPY --from=gr00t /home/ovx/Psi0/baselines /home/ovx/Psi0/baselines

COPY --chmod=0755 ovx/bin /home/ovx/Psi0/ovx/bin
COPY --chmod=0755 ovx/run.sh /home/ovx/Psi0/ovx/run.sh
RUN printf '{"version": "%s", "target": "eval", "psi0_commit": "%s", "simple_commit": "%s"}\n' \
        "${OVX_VERSION}" "${PSI0_COMMIT}" "${SIMPLE_COMMIT}" > /opt/ovx/build-info.json

ENV PATH=/home/ovx/Psi0/ovx/bin:${PATH}
WORKDIR /home/ovx/Psi0
CMD ["/bin/bash"]
