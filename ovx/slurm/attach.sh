#!/usr/bin/env bash
# Open another terminal in a session allocated by ovx/slurm/session.sh. Run it once per extra
# terminal you want -- they all land on the same node, GPU and binds.
#
#   ovx/slurm/attach.sh                a shell inside the container
#   ovx/slurm/attach.sh --host         a shell on the node, outside the container (nvidia-smi, top)
#   ovx/slurm/attach.sh -- smoke       run an ovx subcommand instead of a shell
#   ovx/slurm/attach.sh -- shell -lc 'nvidia-smi'      one command, then exit
#   OVX_JOBID=12345 ovx/slurm/attach.sh                a specific session, if several run
#
# Runs from the repo root, like the sbatch wrappers, so relative OVX_DATA and friends resolve the
# same way. Needs OVX_SIF exported (not for --host).

set -euo pipefail
die() { echo "attach.sh: $*" >&2; exit 1; }

MODE=container; ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host)    MODE=host; shift ;;
        --)        shift; ARGS=("$@"); break ;;
        -h|--help) sed -n '2,12p' "$0" >&2; exit 0 ;;
        *)         die "unknown option: $1 (use -- to pass arguments to the container)" ;;
    esac
done

command -v srun > /dev/null || die "srun not found -- run this on the cluster, not in the container"
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || die "cannot reach the repo root"

# Which session to enter.
JOBID="${OVX_JOBID:-}"
if [ -z "${JOBID}" ]; then
    mapfile -t sessions < <(squeue -h -u "${USER}" -n ovx-session -t RUNNING -o '%i %N %L' 2>/dev/null)
    case "${#sessions[@]}" in
        0) die "no running job named ovx-session -- start one with ovx/slurm/session.sh, or set
    OVX_JOBID=<jobid> if your session runs under a different --job-name" ;;
        1) JOBID="${sessions[0]%% *}" ;;
        *) printf 'several sessions are running (jobid node time-left):\n' >&2
           printf '  %s\n' "${sessions[@]}" >&2
           die "pick one with OVX_JOBID=<jobid>" ;;
    esac
fi

if [ "${MODE}" = host ]; then
    inner=(bash -l)
else
    [ -n "${OVX_SIF:-}" ] || die "OVX_SIF is not set -- export the same .sif the session should run"
    [ "${#ARGS[@]}" -gt 0 ] || ARGS=(shell)
    inner=(ovx/run.sh "${ARGS[@]}")
fi

# SLURM 20.11+ requires --overlap for a step to share resources the job already holds; without it
# the second terminal waits forever for exclusive access. Older versions do not know the flag.
OVERLAP=()
srun --help 2>&1 | grep -q -- '--overlap' && OVERLAP=(--overlap)

echo "attach.sh: entering session ${JOBID}${OVERLAP:+ (--overlap)}" >&2
exec srun --jobid="${JOBID}" "${OVERLAP[@]}" --pty "${inner[@]}"
