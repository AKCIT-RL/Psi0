#!/usr/bin/env bash
# Environment check for the image: seconds, no dataset, no checkpoint. Run it first on every new
# machine and after every rebuild -- each check is one of the failures that used to surface hours
# into a queued job.
#
#   smoke.sh                 GPU, the GR00T venv, and (in the eval image) SIMPLE + MuJoCo EGL
#   smoke.sh --isaac         also boot Isaac Sim headless once (minutes the first time: shaders)
#   smoke.sh --video FILE    also decode a real dataset video (under /home/ovx/data, or absolute)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ISAAC=0; VIDEO=""
while [ $# -gt 0 ]; do
    case "$1" in
        --isaac)   ISAAC=1; shift ;;
        --video)   VIDEO="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0" >&2; exit 0 ;;
        *)         die "unknown option: $1" ;;
    esac
done

setup_caches
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
FAILS=0

check() {
    local name="$1"; shift
    if "$@" > "${TMP}/out" 2>&1; then
        log "PASS  ${name}  $(tail -n 1 "${TMP}/out")"
    else
        log "FAIL  ${name}"
        tail -n 25 "${TMP}/out" | sed 's/^/        /' >&2
        FAILS=$((FAILS + 1))
    fi
}

# GR00T checks run in a subshell so PYTHONPATH never reaches the SIMPLE checks.
gpy() { ( gr00t_env; "${GR00T_PY}" "$@" ); }
spy() { ( simple_env; "${SIMPLE_BIN}/python" "$@" ); }

log "image: $(build_info)"

check "driver visible" nvidia-smi -L

check "gr00t: torch on the GPU" gpy -c '
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(1024, 1024, device="cuda")
(x @ x).sum().item()
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))'

check "gr00t: flash-attn kernel" gpy -c '
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(q, q, q, causal=True)
torch.cuda.synchronize()
print("flash_attn", flash_attn.__version__)'

check "gr00t: bitsandbytes 8-bit AdamW" gpy -c '
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(4096, device="cuda"))
opt = bnb.optim.AdamW8bit([p])
p.sum().backward(); opt.step()
print("bitsandbytes", bnb.__version__)'

# triton needs a source file (it reads the kernel with inspect), gcc, and a writable cache.
cat > "${TMP}/triton_check.py" <<'PY'
import torch, triton, triton.language as tl

@triton.jit
def add_one(ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    tl.store(ptr + i, tl.load(ptr + i, mask=mask) + 1, mask=mask)

x = torch.zeros(1000, device="cuda")
add_one[(4,)](x, 1000, BLOCK=256)
assert x.sum().item() == 1000
print("triton", triton.__version__)
PY
check "gr00t: triton JIT compile" gpy "${TMP}/triton_check.py"

# Importing the decoders loads the FFmpeg-backed core; this is where the GLIBC / libnpp errors hit.
check "gr00t: torchcodec + FFmpeg" gpy -c '
import torchcodec
from torchcodec.decoders import VideoDecoder
print("torchcodec", torchcodec.__version__)'

if [ -n "${VIDEO}" ]; then
    case "${VIDEO}" in /*) ;; *) VIDEO="${DATA_ROOT}/${VIDEO}" ;; esac
    check "gr00t: decode ${VIDEO##*/}" gpy -c '
import sys
from torchcodec.decoders import VideoDecoder
d = VideoDecoder(sys.argv[1])
print("frames", d.metadata.num_frames, "first frame", tuple(d[0].shape))' "${VIDEO}"
fi

check "gr00t: transformers Qwen3-VL" gpy -c '
import transformers
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
print("transformers", transformers.__version__)'

check "gr00t: model, trainer and server code" gpy -c '
import gr00t.model.gr00t_n1d7.gr00t_n1d7
import gr00t.experiment.experiment
import gr00t.deploy.gr00t_serve_simple
import wandb
print("wandb", wandb.__version__)'

if [ -x "${SIMPLE_BIN}/python" ]; then
    check "simple: MuJoCo EGL render" env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "${SIMPLE_BIN}/python" -c '
import mujoco
m = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d)
img = r.render()
r.close()   # left to __del__, the EGL teardown at exit prints a spurious EGLError
assert img.mean() > 0, "rendered an all-black image"
print("mujoco", mujoco.__version__, img.shape)'

    # SIMPLE refuses to import without cuRobo, and cuRobo JIT-compiles missing kernels with an nvcc
    # the image does not have: the five extensions must have been built into the image.
    check "simple: cuRobo CUDA extensions" spy -c '
import torch
import curobo.curobolib.lbfgs_step_cu, curobo.curobolib.kinematics_fused_cu
import curobo.curobolib.line_search_cu, curobo.curobolib.tensor_step_cu, curobo.curobolib.geom_cu
print("curobo ok, torch", torch.__version__, "cuda", torch.cuda.is_available())'

    check "simple: eval CLI and GR00T agent import" spy -c '
import simple.cli.eval_decoupled_wbc
import simple.baselines.gr00t_n16_decoupled_wbc
import isaacsim
print("simple ok")'

    if [ "${ISAAC}" = 1 ]; then
        # Through simple_env like the real eval (Vulkan ICD choice, EULA), and through close():
        # a run that boots but never shuts down would hold its SLURM job until the time limit.
        isaac_boot() {
            ( simple_env; timeout 1200 "${SIMPLE_BIN}/python" -c '
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.kit.app
version = omni.kit.app.get_app().get_build_version()
app.close()
print("isaac sim", version, "booted and closed")' )
        }
        check "simple: Isaac Sim headless boot" isaac_boot
    fi
else
    log "SKIP  simple checks (train image)"
fi

if [ "${FAILS}" -gt 0 ]; then
    die "${FAILS} check(s) failed"
fi
log "all checks passed"
