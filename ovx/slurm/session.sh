#!/usr/bin/env bash
# Interactive session on the OVX: allocates a node with salloc and drops you into a shell inside
# the container. Open more terminals on the same allocation with ovx/slurm/attach.sh.
#
#   export OVX_SIF=/raid/$USER/images/ovx-gr00t_eval_<version>.sif
#   ovx/slurm/session.sh -p <l40 partition>        a shell inside the container
#   ovx/slurm/session.sh --host                    a shell on the node, outside the container
#   ovx/slurm/session.sh -- smoke --isaac          run a subcommand instead of a shell
#
# Anything this script does not recognise goes to salloc, so resources are overridden as usual --
# pass an option and this script drops its own default for it, rather than sending both:
#
#   ovx/slurm/session.sh --time=08:00:00 --gres=gpu:2 -p <partition>
#
# Passing your own --job-name is allowed, but attach.sh finds sessions by the default name
# (ovx-session): with another name, attach with OVX_JOBID=<jobid> ovx/slurm/attach.sh.
#
# The allocation lives exactly as long as this command: close the terminal and SLURM releases it.
# To survive a dropped connection, run this inside tmux on the login node.
#
# Defaults match eval.sbatch: 1 GPU, 16 CPUs, 64 GB, 4 h.

set -euo pipefail
die() { echo "session.sh: $*" >&2; exit 1; }

MODE=container; ARGS=(); SALLOC=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host)    MODE=host; shift ;;
        --)        shift; ARGS=("$@"); break ;;
        -h|--help) sed -n '2,22p' "$0" >&2; exit 0 ;;
        *)         SALLOC+=("$1"); shift ;;   # straight through to salloc, in order
    esac
done

command -v salloc > /dev/null || die "salloc not found -- run this on the cluster login node"
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || die "cannot reach the repo root"

if [ "${MODE}" = host ]; then
    inner=(bash -l)
else
    [ -n "${OVX_SIF:-}" ] || die "OVX_SIF is not set -- export the .sif this session should run"
    [ "${#ARGS[@]}" -gt 0 ] || ARGS=(shell)
    inner=(ovx/run.sh "${ARGS[@]}")
fi

# A default is added only when you did not pass that option, rather than trusting salloc to let
# the later flag win -- getting 4 h after asking for 8 would be silent and expensive. Your flags
# still come after the defaults, so an unusual spelling (-c8) also wins if this misses it.
# --job-name is what attach.sh looks for; keep it in step with that script.
given() {   # given <flag>... -- true if any of them is already in SALLOC
    local a f
    for a in ${SALLOC[@]+"${SALLOC[@]}"}; do
        for f in "$@"; do
            case "${a}" in "${f}"|"${f}"=*) return 0 ;; esac
        done
    done
    return 1
}
defaults=()
given --job-name      -J || defaults+=(--job-name=ovx-session)
given --nodes         -N || defaults+=(--nodes=1)
given --gres             || defaults+=(--gres=gpu:1)
given --cpus-per-task -c || defaults+=(--cpus-per-task=16)
given --mem              || defaults+=(--mem=64G)
given --time          -t || defaults+=(--time=04:00:00)

# salloc runs its command on the LOGIN node, not the allocated one -- `srun --pty` is what puts the
# shell on the compute node. This is the first step of the allocation, so it needs no --overlap;
# the extra terminals from attach.sh do.
echo "session.sh: requesting an allocation (Ctrl-C to give up while it queues)" >&2
echo "session.sh: more terminals -> ovx/slurm/attach.sh   |   end -> exit this shell" >&2
exec salloc "${defaults[@]}" "${SALLOC[@]}" srun --pty "${inner[@]}"
