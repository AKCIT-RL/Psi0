#!/usr/bin/env python3
"""Publish finished fine-tuning runs to a Hugging Face branch, verify, then reclaim disk.

For each run directory (checkpoints/gr00t_n1d7_finetune_output_<slug>) this uploads

    <repo>/<run_name>/final/              the inference model (top-level files)
    <repo>/<run_name>/checkpoint-<N>/     the last checkpoint, for resuming training

to the branch given by --branch, then checks every uploaded file back against the
remote tree -- name and byte size -- and only then deletes the local copy.

WHY THE VERIFY STEP EXISTS
--------------------------
upload_folder can return without raising while a file is missing or truncated, and
"delete after upload" turns that into permanent data loss. So deletion is gated on
reading the repo tree back and matching it against what is on disk. One mismatched
byte and nothing is deleted.

Every deletion is recorded in UPLOADED.json inside the run directory, with the remote
path and commit sha, so nothing disappears without a trace.

The intermediate checkpoints (everything but the last) are NOT uploaded and NOT deleted
by default -- deleting an unpublished checkpoint is unrecoverable. Pass
--purge-unuploaded when you have decided you do not want them.

Usage:
    python scripts/upload_checkpoints_hf.py --all --dry-run
    python scripts/upload_checkpoints_hf.py --all
    python scripts/upload_checkpoints_hf.py --run gr00t_n1d7_finetune_output_handover
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time

# The Xet backend fails with "MerkleDB Shard error" on this filesystem; the classic
# LFS path works. Set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, CommitInfo  # noqa: E402
from huggingface_hub.hf_api import RepoFile  # noqa: E402
from huggingface_hub.utils import HfHubHTTPError  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
ENV_FILE = PROJECT_DIR / ".env"

DEFAULT_REPO = "lucasolives/gr00t_1.7_Psi"
DEFAULT_BRANCH = "simple-converted"

# Run directories that are not fine-tuning outputs of ours.
EXCLUDE_RUNS = {"GR00T-N1.7-3B", "upload_tmp"}

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def log(msg: str, indent: int = 0) -> None:
    print(("  " * indent) + msg, flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def load_env_token() -> str | None:
    """HF_TOKEN from the environment, falling back to .env. Never taken from argv:
    command lines are visible in `ps` and echoed into the SLURM logs."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.strip().startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# --------------------------------------------------------------------------------------
# Discovering what to upload
# --------------------------------------------------------------------------------------

def local_files(folder: Path, ignore_checkpoints: bool = False) -> dict[str, int]:
    """Relative path -> byte size for every file under `folder`."""
    out: dict[str, int] = {}
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(folder)
        if ignore_checkpoints and CHECKPOINT_RE.match(rel.parts[0]):
            continue
        if rel.name in ("UPLOADED.json", ".DS_Store"):
            continue
        out[rel.as_posix()] = path.stat().st_size
    return out


def checkpoint_dirs(run: Path) -> list[tuple[int, Path]]:
    found = []
    for path in run.iterdir():
        match = CHECKPOINT_RE.match(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path))
    return sorted(found)


def run_is_complete(run: Path) -> tuple[bool, str]:
    """A finished run has the merged model at the top level and a checkpoint that
    reached max_steps."""
    if not (run / "config.json").exists() or not (run / "model.safetensors.index.json").exists():
        return False, "no top-level model (config.json / model.safetensors.index.json missing)"
    ckpts = checkpoint_dirs(run)
    if not ckpts:
        return False, "no checkpoint-* directory"
    state = ckpts[-1][1] / "trainer_state.json"
    if state.exists():
        data = json.loads(state.read_text())
        step, target = data.get("global_step"), data.get("max_steps")
        if step is not None and target is not None and step < target:
            return False, f"training stopped early ({step}/{target} steps)"
    return True, "complete"


# --------------------------------------------------------------------------------------
# Upload and verification
# --------------------------------------------------------------------------------------

def upload_with_retry(api: HfApi, folder: Path, repo_id: str, branch: str,
                      path_in_repo: str, ignore: list[str] | None,
                      attempts: int = 3) -> CommitInfo:
    for attempt in range(1, attempts + 1):
        try:
            return api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                path_in_repo=path_in_repo,
                ignore_patterns=ignore,
                commit_message=f"add {path_in_repo}",
            )
        except HfHubHTTPError as exc:
            if "storage limit reached" in str(exc).lower():
                raise  # retrying will not help
            if attempt == attempts:
                raise
            wait = 30 * attempt
            log(f"upload attempt {attempt} failed ({exc}); retrying in {wait}s", 3)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def verify_remote(api: HfApi, repo_id: str, branch: str, path_in_repo: str,
                  expected: dict[str, int]) -> list[str]:
    """Compare the remote tree against `expected`. Returns a list of problems."""
    remote: dict[str, int] = {}
    for entry in api.list_repo_tree(repo_id=repo_id, repo_type="model", revision=branch,
                                    path_in_repo=path_in_repo, recursive=True):
        if isinstance(entry, RepoFile):
            rel = entry.path[len(path_in_repo):].lstrip("/")
            # For LFS files the real size lives on the lfs blob, not the pointer.
            size = entry.lfs.size if entry.lfs is not None else entry.size
            remote[rel] = size

    problems = []
    for rel, size in sorted(expected.items()):
        if rel not in remote:
            problems.append(f"missing on the hub: {rel}")
        elif remote[rel] != size:
            problems.append(f"size mismatch {rel}: local {size} != remote {remote[rel]}")
    return problems


def publish_part(api: HfApi, run: Path, run_name: str, repo_id: str, branch: str,
                 part_name: str, folder: Path, ignore: list[str] | None,
                 args) -> dict:
    """Upload one part (the final model, or one checkpoint), verify it, delete it."""
    expected = local_files(folder, ignore_checkpoints=(ignore is not None))
    total = sum(expected.values())
    path_in_repo = f"{run_name}/{part_name}"
    log(f"{part_name}: {len(expected)} files, {human(total)} -> {repo_id}/{path_in_repo}@{branch}", 2)

    if args.dry_run:
        return {"part": part_name, "status": "dry-run", "files": len(expected),
                "bytes": total, "path_in_repo": path_in_repo}

    started = time.time()
    commit = upload_with_retry(api, folder, repo_id, branch, path_in_repo, ignore)
    log(f"uploaded in {(time.time() - started) / 60:.1f} min "
        f"(commit {str(commit.oid)[:8]})", 3)

    problems = verify_remote(api, repo_id, branch, path_in_repo, expected)
    if problems:
        log(f"VERIFICATION FAILED -- nothing deleted:", 3)
        for problem in problems[:10]:
            log(problem, 4)
        return {"part": part_name, "status": "verify-failed", "problems": problems,
                "path_in_repo": path_in_repo}
    log(f"verified: all {len(expected)} files present with matching sizes", 3)

    record = {
        "part": part_name,
        "status": "uploaded",
        "path_in_repo": path_in_repo,
        "revision": branch,
        "commit": str(commit.oid),
        "commit_url": str(commit.commit_url),
        "files": len(expected),
        "bytes": total,
        "uploaded_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "deleted_local": False,
    }

    if args.no_delete:
        log("local copy kept (--no-delete)", 3)
        return record

    # Delete exactly what was verified, nothing else.
    for rel in expected:
        (folder / rel).unlink(missing_ok=True)
    for path in sorted(folder.rglob("*"), key=lambda p: -len(p.parts)):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    if folder != run and folder.exists() and not any(folder.iterdir()):
        folder.rmdir()
    record["deleted_local"] = True
    log(f"local copy removed ({human(total)} reclaimed)", 3)
    return record


def process_run(api: HfApi, run: Path, repo_id: str, branch: str, args) -> dict:
    run_name = run.name
    log("")
    log(f"=== {run_name}")

    complete, why = run_is_complete(run)
    if not complete:
        if not args.allow_incomplete:
            log(f"not a finished run: {why} -- skipping (--allow-incomplete to override)", 1)
            return {"run": run_name, "status": "incomplete", "reason": why}
        log(f"WARNING: {why} -- proceeding because of --allow-incomplete", 1)

    ckpts = checkpoint_dirs(run)
    last_step, last_dir = ckpts[-1]
    older = [d for step, d in ckpts[:-1]]

    parts: list[dict] = []
    if args.what in ("final+last", "final"):
        parts.append(publish_part(api, run, run_name, repo_id, branch, "final", run,
                                  ignore=["checkpoint-*", "checkpoint-*/**"], args=args))
    if args.what in ("final+last", "last"):
        parts.append(publish_part(api, run, run_name, repo_id, branch,
                                  last_dir.name, last_dir, ignore=None, args=args))

    if older:
        kept = human(sum(sum(local_files(d).values()) for d in older))
        if args.purge_unuploaded and not args.dry_run:
            for d in older:
                shutil.rmtree(d)
            log(f"purged {len(older)} intermediate checkpoint(s), {kept} reclaimed "
                f"(never uploaded)", 1)
        else:
            log(f"{len(older)} intermediate checkpoint(s) left on disk ({kept}): "
                f"{', '.join(d.name for d in older)}", 1)
            log("they were not uploaded; pass --purge-unuploaded to delete them", 2)

    status = "ok" if all(p["status"] in ("uploaded", "dry-run") for p in parts) else "failed"

    if not args.dry_run:
        manifest = {
            "run": run_name,
            "repo_id": repo_id,
            "revision": branch,
            "last_step": last_step,
            "parts": parts,
            "intermediate_checkpoints_kept": [d.name for d in older]
                                             if not args.purge_unuploaded else [],
            "host": socket.gethostname(),
            "written_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        run.mkdir(parents=True, exist_ok=True)
        (run / "UPLOADED.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return {"run": run_name, "status": status, "parts": parts}


# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", metavar="DIR",
                        help="run directory name(s) under checkpoints/")
    parser.add_argument("--all", action="store_true",
                        help="every finished run under checkpoints/")
    parser.add_argument("--checkpoints-dir", default=str(CHECKPOINTS_DIR))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--what", choices=["final+last", "final", "last"], default="final+last",
                        help="final = inference model; last = newest checkpoint (resumable)")
    parser.add_argument("--no-delete", action="store_true",
                        help="upload and verify but keep the local copy")
    parser.add_argument("--purge-unuploaded", action="store_true",
                        help="also delete the intermediate checkpoints, which are never uploaded")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="upload a run that did not reach max_steps")
    parser.add_argument("--private", action="store_true",
                        help="create the repo private if it does not exist yet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = load_env_token()
    if not token:
        log("ERROR: no HF_TOKEN in the environment or .env")
        return 1

    api = HfApi(token=token)
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    log(f"authenticated as {who.get('name')} (token role: {role})")
    if role == "read" and not args.dry_run:
        log("ERROR: this token is read-only; uploads would fail. Put a write token in .env.")
        return 1

    root = Path(args.checkpoints_dir)
    if args.run:
        runs = [root / name for name in args.run]
    elif args.all:
        runs = sorted(p for p in root.iterdir()
                      if p.is_dir() and not p.name.startswith(".")
                      and p.name not in EXCLUDE_RUNS)
    else:
        parser.error("pass --run <dir> or --all")

    missing = [r for r in runs if not r.is_dir()]
    if missing:
        log("ERROR: not a directory: " + ", ".join(str(m) for m in missing))
        return 1

    if not args.dry_run:
        api.create_repo(repo_id=args.repo_id, repo_type="model",
                        private=args.private, exist_ok=True)
        api.create_branch(repo_id=args.repo_id, repo_type="model",
                          branch=args.branch, exist_ok=True)
        log(f"branch ready: {args.repo_id}@{args.branch}")

    results = [process_run(api, run, args.repo_id, args.branch, args) for run in runs]

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for row in results:
        detail = row.get("reason", "")
        print(f"  {row['status']:<12} {row['run']}  {detail}")
        for part in row.get("parts", []):
            note = "deleted locally" if part.get("deleted_local") else part["status"]
            print(f"      {part['part']:<18} {note:<16} {part.get('path_in_repo', '')}")
    print(f"\nhttps://huggingface.co/{args.repo_id}/tree/{args.branch}")

    return 0 if all(r["status"] in ("ok",) for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
