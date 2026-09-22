# Shared by the ovx/bin entrypoints. Sourced, never executed.
#
# These scripts run INSIDE the image. Everything a run touches lives under /home/ovx, the
# workspace; /opt holds only installed software (the venvs, their interpreters, the image's
# metadata). ovx/run.sh decides which host directory lands on each mount point:
#
#   /home/ovx/Psi0          the code: a checkout mounted over the copy baked into the image
#   /home/ovx/data          LeRobot datasets, one directory per dataset       (read-only)
#   /home/ovx/checkpoints   base models and training runs
#   /home/ovx/cache         HF, torch, triton, wandb and Isaac Sim caches
#   /home/ovx/evals         evaluation outputs

set -euo pipefail

readonly GR00T_PY=/opt/venvs/gr00t/bin/python
readonly SIMPLE_BIN=/opt/venvs/simple/bin
readonly OVX_HOME=/home/ovx
readonly PSI0_ROOT="${OVX_HOME}/Psi0"
readonly SIMPLE_ROOT="${OVX_HOME}/Psi0/third_party/SIMPLE"
readonly DATA_ROOT="${OVX_HOME}/data"
readonly CKPT_ROOT="${OVX_HOME}/checkpoints"
readonly CACHE_ROOT="${OVX_HOME}/cache"
readonly EVAL_ROOT="${OVX_HOME}/evals"

log() { printf '[ovx %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Every cache under /home/ovx/cache. HOME too: Isaac Sim writes ~16 GB under ~/.cache/ov and friends, which
# would blow the home quota on the cluster, and the host's dotfiles have no business in here.
setup_caches() {
    [ -w "${CACHE_ROOT}" ] || die "${CACHE_ROOT} is not writable -- is the cache directory mounted?"
    export HOME="${CACHE_ROOT}/home"
    export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
    export HF_HOME="${CACHE_ROOT}/huggingface"
    export TORCH_HOME="${CACHE_ROOT}/torch"
    export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
    export WANDB_DIR="${CACHE_ROOT}/wandb"
    export WANDB_CACHE_DIR="${CACHE_ROOT}/wandb/cache"
    export WANDB_CONFIG_DIR="${CACHE_ROOT}/wandb/config"
    # getpass.getuser() (torch inductor, triton) raises when the uid has no passwd entry, as
    # under `docker run --user`. It checks these variables first.
    export USER="${USER:-ovx}" LOGNAME="${LOGNAME:-ovx}"
    mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" \
             "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
}

# Environment for the GR00T venv: the model code is not installed, it is on PYTHONPATH.
gr00t_env() {
    export PYTHONPATH="${PSI0_ROOT}/src:${PSI0_ROOT}/src/gr00t"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export NO_ALBUMENTATIONS_UPDATE=1   # otherwise every import phones home for a version check
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
}

# Environment for the SIMPLE venv. Nothing from the GR00T side may leak in. Importing isaacsim
# asks for the EULA interactively unless it is accepted here -- a batch job would just hang.
simple_env() {
    unset PYTHONPATH
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y
    # Isaac Sim's Kit writes its cache, data and logs through symlinks in the image that point here
    # (see the Dockerfile). The first start compiles the RTX shaders into this cache (~5 min);
    # later starts reuse it.
    mkdir -p "${CACHE_ROOT}/isaac/cache" "${CACHE_ROOT}/isaac/data" "${CACHE_ROOT}/isaac/logs"
}

# Emergency spare, deliberately NOT called: the Vulkan manifest comes from the container runtime
# (docker's toolkit writes /etc/vulkan/icd.d/nvidia_icd.json; apptainer --nv binds the host's into
# /usr/share/vulkan/icd.d). Only if a host ships none -- Isaac Sim then fails with no Vulkan device
# -- add a call to this in simple_env, or export the same variable before running:
#
#   export VK_ICD_FILENAMES=/opt/ovx/vulkan/nvidia_icd.json
#
# Never point the loader at two manifests for the same driver: it registers the GPU twice and Isaac
# Sim segfaults while starting up.
vulkan_fallback_icd() {
    export VK_ICD_FILENAMES=/opt/ovx/vulkan/nvidia_icd.json
    log "WARNING: using the image's spare Vulkan manifest (${VK_ICD_FILENAMES})"
}

# Which code a run actually used: the commit of a mounted checkout, marked when it has uncommitted
# changes. safe.directory because the files belong to the host user, not to the container's.
repo_state() {
    local dir="$1" commit
    commit="$(git -c safe.directory='*' -C "${dir}" rev-parse --short HEAD 2> /dev/null || true)"
    [ -n "${commit}" ] || { printf 'unknown'; return; }
    [ -z "$(git -c safe.directory='*' -C "${dir}" status --porcelain 2> /dev/null | head -1)" ] \
        || commit="${commit}-dirty"
    printf '%s' "${commit}"
}

# The nested repos the SIMPLE venv installs editable: their code is read at run time, so their
# commits describe what actually ran. decoupled_wbc is the controller that turns policy output into
# joint commands -- a different commit there is a different robot. curobo (installed as a wheel)
# and evdev (a plain path install) are baked into the venv, so their sources change nothing.
SIMPLE_EDITABLE_SUBMODULES=(openpi-client gear_sonic unitree_sdk2_python decoupled_wbc)

submodule_states() {   # submodule_states <simple root> -> "name:sha[-dirty],name:sha[-dirty],..."
    local dir="$1" out="" s
    for s in "${SIMPLE_EDITABLE_SUBMODULES[@]}"; do
        out="${out}${out:+,}${s}:$(repo_state "${dir}/third_party/${s}")"
    done
    printf '%s' "${out}"
}

# Append one JSON object describing a run to <file>: what code, what image, what arguments, where
# and when. Call it before the work starts, so a crash still leaves the record behind.
#
# Arguments are key=value pairs (the value may contain '='). Any key ending in _submodules is
# expanded from "a:sha,b:sha" into an object; the image, host and timestamp are added here so every
# record carries them in the same shape.
write_provenance() {   # write_provenance <file> <key=value>...
    local out="$1"; shift
    "${GR00T_PY}" - "${out}" "$@" <<'PY'
import datetime, json, os, pathlib, socket, sys

out = pathlib.Path(sys.argv[1])
meta = dict(arg.split("=", 1) for arg in sys.argv[2:])
# Options that were not used arrive as empty strings. Leaving them out keeps the record to what
# actually applied, and makes --data-dir vs --eval-config obvious at a glance.
meta = {k: v for k, v in meta.items() if v != ""}
for key in [k for k in meta if k.endswith("_submodules")]:
    meta[key] = dict(p.split(":", 1) for p in meta[key].split(",") if p)
# key_file=<path> embeds that JSON file under "key", or null when it is not there -- how a run
# records the provenance of what it consumed (a dataset's own PROVENANCE.json).
for key in [k for k in meta if k.endswith("_file")]:
    path = pathlib.Path(meta.pop(key))
    value = None
    if path.exists():
        text = path.read_text()
        try:
            value = json.loads(text)
        except ValueError:      # not JSON: an experiment YAML
            import yaml
            value = yaml.safe_load(text)
    meta[key[: -len("_file")]] = value
info = pathlib.Path("/opt/ovx/build-info.json")
meta["image"] = json.loads(info.read_text()) if info.exists() else None
# gethostname() is the container id under docker; SLURMD_NODENAME, passed through by ovx/run.sh,
# is the node that actually ran the job.
meta["host"] = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
meta["started_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("a") as f:
    f.write(json.dumps(meta) + "\n")
PY
}

# Turn an experiment YAML into launcher arguments, one token per line (read with mapfile). Keys are
# FinetuneConfig field names; values follow tyro's spelling, which is why this exists rather than a
# plain loop: true -> --flag, false -> --no-flag (tyro rejects "--flag false"), and a mapping ->
# --flag k v k v, the form color_jitter_params takes. Any key named in <recipe key>... is reported
# on stderr, because overriding one of those is what makes a run stop being the baseline.
config_args() {   # config_args <file.yaml> [recipe key]...
    "${GR00T_PY}" - "$@" <<'PY'
import sys, yaml

path, recipe = sys.argv[1], set(sys.argv[2:])
data = yaml.safe_load(open(path)) or {}
if not isinstance(data, dict):
    sys.exit(f"{path}: expected a mapping of FinetuneConfig fields at the top level")

overridden = sorted(k for k in data if k in recipe)
if overridden:
    print(f"[ovx] {path} overrides the frozen recipe: {', '.join(overridden)}", file=sys.stderr)

for key, value in data.items():
    flag = "--" + str(key).replace("_", "-")
    if value is None:
        continue
    if isinstance(value, bool):
        print(flag if value else "--no-" + str(key).replace("_", "-"))
    elif isinstance(value, dict):
        print(flag)
        for k, v in value.items():
            print(k)
            print(v)
    elif isinstance(value, (list, tuple)):
        print(flag)
        for v in value:
            print(v)
    else:
        print(flag)
        print(value)
PY
}

# A checkpoint argument is a directory name under /home/ovx/checkpoints, or an absolute container path.
resolve_ckpt() {
    case "$1" in
        /*) printf '%s' "$1" ;;
        *)  printf '%s' "${CKPT_ROOT}/$1" ;;
    esac
}

# Training saves the processor under <run>/processor/, but both the fine-tune setup and the policy
# server load it from the model directory itself -- the "processor not found" failure. Copy it up,
# never overwriting anything already there.
ensure_processor_at_root() {
    local dir="$1"
    if [ -d "${dir}/processor" ]; then
        if [ -w "${dir}" ]; then
            cp -rn "${dir}/processor/." "${dir}/"
        else
            log "WARNING: ${dir} is read-only; cannot copy processor/ files to its root"
        fi
    fi
}

free_port() {
    "${GR00T_PY}" - "${1:-29500}" <<'PY'
import socket, sys
start = int(sys.argv[1])
for port in range(start, start + 500):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
    print(port)
    break
else:
    raise SystemExit("no free port found")
PY
}

# Default ports derive from the SLURM job so two jobs on one node do not race for the same one.
port_base() {
    printf '%s' "$(( ${1} + ${SLURM_JOB_ID:-$$} % 10000 ))"
}

build_info() {
    cat /opt/ovx/build-info.json 2>/dev/null || printf '{"version": "unknown"}\n'
}
