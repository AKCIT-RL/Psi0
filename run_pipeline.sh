#!/usr/bin/env bash
#
# End-to-end driver: .zip archives -> validated datasets -> fine-tuning -> Hugging Face.
#
# Runs a fixed number of datasets at a time and keeps the SLURM queue empty of the rest.
# Each training job, when it ends, submits its own upload job and pulls in the next
# dataset itself (submit_slurm.sh, "HAND OFF THE LANE"). Nothing waits in the queue.
#
# That matters on this cluster: priority is age-based with PriorityWeightFairShare=0, so
# a wall of pending 48h GPU jobs sits in front of everything anyone submits afterwards.
# Two running jobs take two GPUs; ten pending jobs take the whole machine's future.
#
# Usage:
#   ./run_pipeline.sh --dry-run              # show the plan, submit nothing
#   ./run_pipeline.sh                        # start 2 lanes (default)
#   ./run_pipeline.sh --max-inflight 3       # 3 at a time
#   ./run_pipeline.sh --serial               # same as --max-inflight 1
#   ./run_pipeline.sh --status               # what is claimed, running, published
#   ./run_pipeline.sh --stop                 # stop advancing after the current jobs
#   ./run_pipeline.sh --reset-claims         # let failed datasets be picked up again
#   ./run_pipeline.sh --after 1402           # first lane waits for an existing job
#
# Environment overrides: HF_REPO_ID, HF_BRANCH, UPLOAD_WHAT, GRES, TRAIN_TIME.

set -euo pipefail

# Without this, `set -e` aborts silently and a caller only sees a non-zero status. This
# script is invoked from inside SLURM jobs, where a silent death means the pipeline
# quietly stops advancing and nobody finds out for hours.
trap 'echo "run_pipeline.sh: aborted at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

PROJECT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
PY="${PY:-${PROJECT_DIR}/src/gr00t/.venv-gr00t/bin/python}"
CONVERTED_DIR="${PROJECT_DIR}/data/simple/simple-converted"
STATE_DIR="${PROJECT_DIR}/.pipeline"
CLAIM_DIR="${STATE_DIR}/claims"
STOP_FILE="${STATE_DIR}/STOP"

HF_REPO_ID="${HF_REPO_ID:-lucasolives/gr00t_1.7_Psi}"
HF_BRANCH="${HF_BRANCH:-simple-converted}"
UPLOAD_WHAT="${UPLOAD_WHAT:-final+last}"
GRES="${GRES:-gpu:1}"
TRAIN_TIME="${TRAIN_TIME:-48:00:00}"

DRY_RUN=0
SKIP_PREPARE=0
MAX_INFLIGHT=2
AFTER_JOB=""
ADVANCE=0
ACTION="run"
ONLY=()
PREPARE_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY_RUN=1 ;;
        --skip-prepare)  SKIP_PREPARE=1 ;;
        --serial)        MAX_INFLIGHT=1 ;;
        --max-inflight)  shift; MAX_INFLIGHT="$1" ;;
        --all-at-once)   MAX_INFLIGHT=0 ;;
        --after)         shift; AFTER_JOB="$1" ;;
        --advance)       ADVANCE=1; SKIP_PREPARE=1 ;;
        --status)        ACTION="status" ;;
        --stop)          ACTION="stop" ;;
        --resume)        ACTION="resume" ;;
        --reset-claims)  ACTION="reset" ;;
        --only)          shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ONLY+=("$1"); shift; done; continue ;;
        --prepare-arg)   shift; PREPARE_ARGS+=("$1") ;;
        -h|--help)       sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)               echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

say() { printf '%s\n' "$*"; }
rule() { printf '%s\n' "==============================================================================="; }

mkdir -p "${CLAIM_DIR}"

########################
# DATASET BOOKKEEPING  #
########################

# Every prepared dataset, in a stable order.
all_datasets() {
    [ -d "${CONVERTED_DIR}" ] || return 0
    find "${CONVERTED_DIR}" -mindepth 2 -maxdepth 2 -name PROVENANCE.json 2>/dev/null \
        | sort | while IFS= read -r p; do basename "$(dirname "${p}")"; done
}

run_name_of() {
    local slug
    slug=$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['run_slug'])" \
           "${CONVERTED_DIR}/$1/PROVENANCE.json")
    printf 'gr00t_n1d7_finetune_output_%s' "${slug}"
}

is_published()  { [ -f "${PROJECT_DIR}/checkpoints/$(run_name_of "$1")/UPLOADED.json" ]; }
is_claimed()    { [ -d "${CLAIM_DIR}/$1" ]; }

# mkdir is atomic: two lanes finishing at the same instant cannot both take a dataset.
try_claim() { mkdir "${CLAIM_DIR}/$1" 2>/dev/null; }

selected() {
    [ ${#ONLY[@]} -eq 0 ] && return 0
    local d
    for d in "${ONLY[@]}"; do [ "$d" = "$1" ] && return 0; done
    return 1
}

# Datasets that could still be started, in order.
pending_datasets() {
    local d
    while IFS= read -r d; do
        [ -n "${d}" ] || continue
        selected "${d}" || continue
        is_claimed "${d}" && continue
        is_published "${d}" && continue
        printf '%s\n' "${d}"
    done < <(all_datasets)
}

########################
# SUBMISSION           #
########################

submit_chain() {
    local dataset="$1" dependency="${2:-}" run_name train_args train_id
    run_name=$(run_name_of "${dataset}")

    train_args=(--parsable
                --job-name "gr00t-${run_name#gr00t_n1d7_finetune_output_}"
                --gres "${GRES}"
                --time "${TRAIN_TIME}"
                --export "ALL,DATASET_NAME=${dataset},RUN_NAME=${run_name},PIPELINE_ADVANCE=1,HF_REPO_ID=${HF_REPO_ID},HF_BRANCH=${HF_BRANCH},UPLOAD_WHAT=${UPLOAD_WHAT}")
    [ -n "${dependency}" ] && train_args+=(--dependency "afterany:${dependency}")

    train_id=$(sbatch "${train_args[@]}" "${PROJECT_DIR}/submit_slurm.sh")
    printf '%s\n' "${train_id}" > "${CLAIM_DIR}/${dataset}/train_job"
    date -Is > "${CLAIM_DIR}/${dataset}/claimed_at"
    printf '%s' "${train_id}"
}

########################
# ACTIONS              #
########################

if [ "${ACTION}" = "stop" ]; then
    touch "${STOP_FILE}"
    say "Stop flag set: running jobs finish and upload, but no new dataset is started."
    say "Undo with: ./run_pipeline.sh --resume"
    exit 0
fi

if [ "${ACTION}" = "resume" ]; then
    rm -f "${STOP_FILE}"
    say "Stop flag cleared. Start lanes again with: ./run_pipeline.sh"
    exit 0
fi

if [ "${ACTION}" = "reset" ]; then
    n=0
    while IFS= read -r d; do
        [ -n "${d}" ] || continue
        selected "${d}" || continue
        is_published "${d}" && continue
        if squeue -h -u "${USER}" -o "%j" 2>/dev/null \
           | grep -qx "gr00t-$(run_name_of "${d}" | sed 's/^gr00t_n1d7_finetune_output_//')"; then
            say "  still queued, claim kept: ${d}"
            continue
        fi
        if [ -d "${CLAIM_DIR}/${d}" ]; then
            rm -rf "${CLAIM_DIR}/${d}"
            say "  claim cleared: ${d}"
            n=$((n + 1))
        fi
    done < <(all_datasets)
    say "${n} claim(s) cleared. Published and currently queued datasets were left alone."
    exit 0
fi

if [ "${ACTION}" = "status" ]; then
    rule; say "PIPELINE STATUS"; rule
    [ -f "${STOP_FILE}" ] && say "!! STOP flag is set — lanes will not advance." && say ""
    printf '%-54s %-12s %s\n' "DATASET" "STATE" "DETAIL"
    running=$(squeue -h -u "${USER}" -o "%j %i %t" 2>/dev/null || true)
    while IFS= read -r d; do
        [ -n "${d}" ] || continue
        run_name=$(run_name_of "${d}")
        slug="${run_name#gr00t_n1d7_finetune_output_}"
        if is_published "${d}"; then
            printf '%-54s %-12s %s\n' "${d}" "published" "$(run_name_of "${d}")"
        elif line=$(printf '%s\n' "${running}" | grep "^gr00t-${slug} " || true); [ -n "${line}" ]; then
            printf '%-54s %-12s %s\n' "${d}" "training" "job $(echo "${line}" | awk '{print $2, $3}')"
        elif is_claimed "${d}"; then
            printf '%-54s %-12s %s\n' "${d}" "claimed" "job $(cat "${CLAIM_DIR}/${d}/train_job" 2>/dev/null || echo '?') — finished or failed"
        else
            printf '%-54s %-12s %s\n' "${d}" "waiting" "-"
        fi
    done < <(all_datasets)
    say ""
    say "In the SLURM queue right now:"
    squeue -u "${USER}" -o "  %.7i %.28j %.2t %.10M %R" 2>/dev/null || true
    exit 0
fi

########################
# ADVANCE (from a job) #
########################

if [ "${ADVANCE}" -eq 1 ]; then
    if [ -f "${STOP_FILE}" ]; then
        say "[advance] STOP flag set — not starting another dataset."
        exit 0
    fi
    # Deliberately not `pending_datasets | head -1`: head closes the pipe after the first
    # line, the producer dies of SIGPIPE (141), pipefail propagates that to the
    # assignment, and set -e kills this script with no message at all. Read one line from
    # a process substitution instead.
    next=""
    while IFS= read -r candidate; do next="${candidate}"; break; done < <(pending_datasets)

    if [ -z "${next}" ]; then
        say "[advance] nothing left to start."
        exit 0
    fi
    if ! try_claim "${next}"; then
        say "[advance] ${next} was claimed by another lane; stopping here."
        exit 0
    fi
    train_id=$(submit_chain "${next}")
    say "[advance] started ${next} as job ${train_id}"
    exit 0
fi

########################
# 1. PREPARE           #
########################

if [ "${SKIP_PREPARE}" -eq 0 ]; then
    rule; say "STAGE 1 — preparing datasets"; rule
    PREP_CMD=("${PY}" "${PROJECT_DIR}/scripts/prepare_simple_datasets.py")
    [ ${#ONLY[@]} -gt 0 ] && PREP_CMD+=(--only "${ONLY[@]}")
    [ ${#PREPARE_ARGS[@]} -gt 0 ] && PREP_CMD+=("${PREPARE_ARGS[@]}")

    if [ "${DRY_RUN}" -eq 1 ]; then
        say "would run: ${PREP_CMD[*]}"
        say ""
        "${PY}" "${PROJECT_DIR}/scripts/prepare_simple_datasets.py" --list
    else
        # A dataset that cannot be converted must not stop the ones that can. Stage 3
        # only starts datasets that have a PROVENANCE.json, so a partial preparation is
        # safe to continue from; if nothing is ready at all, stage 2 stops anyway.
        PREP_STATUS=0
        "${PREP_CMD[@]}" || PREP_STATUS=$?
        if [ "${PREP_STATUS}" -ne 0 ]; then
            say ""
            say "!! Some datasets failed to prepare (see the summary above)."
            say "!! Continuing with the ones that did — their raw trees were left in place"
            say "!! for investigation, and nothing was deleted for them."
        fi
    fi
else
    say "STAGE 1 — skipped"
fi

########################
# 2. ENUMERATE         #
########################

rule; say "STAGE 2 — datasets ready to train"; rule

mapfile -t READY < <(all_datasets)
mapfile -t PENDING < <(pending_datasets)

if [ ${#READY[@]} -eq 0 ]; then
    say "No prepared dataset found under ${CONVERTED_DIR}."
    say "(Datasets prepared before this pipeline existed have no PROVENANCE.json and are"
    say " skipped on purpose — their provenance and validation were never recorded.)"
    exit 1
fi
say "${#READY[@]} prepared, ${#PENDING[@]} not yet started."

if [ -f "${STOP_FILE}" ]; then
    say ""
    say "!! STOP flag is set — clear it with ./run_pipeline.sh --resume before starting."
    exit 1
fi

########################
# 3. START THE LANES   #
########################

rule
if [ "${MAX_INFLIGHT}" -eq 0 ]; then
    say "STAGE 3 — starting every remaining dataset at once"
else
    say "STAGE 3 — starting up to ${MAX_INFLIGHT} lane(s); each pulls the next when it ends"
fi
rule

if [ ${#PENDING[@]} -eq 0 ]; then
    say "Nothing to start: everything is claimed or published (./run_pipeline.sh --status)."
    exit 0
fi

printf '%-54s %-46s %s\n' "DATASET" "RUN" "TRAIN JOB"

started=0
for dataset in "${PENDING[@]}"; do
    if [ "${MAX_INFLIGHT}" -ne 0 ] && [ "${started}" -ge "${MAX_INFLIGHT}" ]; then
        break
    fi
    run_name=$(run_name_of "${dataset}")

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf '%-54s %-46s %s\n' "${dataset}" "${run_name}" "(dry)"
        started=$((started + 1))
        continue
    fi

    if ! try_claim "${dataset}"; then
        printf '%-54s %-46s %s\n' "${dataset}" "${run_name}" "already claimed — skipped"
        continue
    fi
    train_id=$(submit_chain "${dataset}" "${AFTER_JOB}")
    AFTER_JOB=""   # only the first lane waits for the pre-existing job
    printf '%-54s %-46s %s\n' "${dataset}" "${run_name}" "${train_id}"
    started=$((started + 1))
done

say ""
if [ "${DRY_RUN}" -eq 1 ]; then
    say "Dry run — nothing was submitted."
    remaining=$(( ${#PENDING[@]} - started ))
    [ "${remaining}" -gt 0 ] && say "${remaining} dataset(s) would stay out of the queue until a lane frees up."
else
    say "Started ${started} lane(s). The rest stay OUT of the SLURM queue until a lane ends."
    say ""
    say "  ./run_pipeline.sh --status     what is running, claimed, published"
    say "  ./run_pipeline.sh --stop       finish the current jobs, then stand down"
    say "  squeue -u \$USER"
    say ""
    say "Published models land at: https://huggingface.co/${HF_REPO_ID}/tree/${HF_BRANCH}"
fi
