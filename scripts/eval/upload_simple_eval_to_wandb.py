#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    episode: int
    left_video: Path
    right_video: Path
    success: bool
    policy_steps: int
    stabilization_steps: int
    max_reward: float
    final_reward: float
    partial_success: bool
    partial_reward_steps: int
    final_chunk_action_index: int | None
    terminated: bool
    truncated: bool
    end_reason: str

    def metrics(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "success": self.success,
            "policy_steps": self.policy_steps,
            "stabilization_steps": self.stabilization_steps,
            "max_reward": self.max_reward,
            "final_reward": self.final_reward,
            "partial_success": self.partial_success,
            "partial_reward_steps": self.partial_reward_steps,
            "final_chunk_action_index": self.final_chunk_action_index,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "end_reason": self.end_reason,
        }


def _episode_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Invalid episode name: {path.name}") from exc


def parse_eval_stats(path: Path) -> dict[int, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval stats: {path}")
    stats: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("episode_"):
            continue
        name, value = line.split(":", maxsplit=1)
        episode = int(name.removeprefix("episode_"))
        normalized = value.strip()
        if normalized not in {"True", "False"}:
            raise ValueError(f"Invalid success value in {path}: {line}")
        stats[episode] = normalized == "True"
    return stats


def parse_telemetry(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing telemetry: {path}")
    start: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if event.get("event") == "episode_start":
                start = event
            elif event.get("event") == "step":
                steps.append(event)
    if start is None or not steps:
        raise ValueError(f"Incomplete telemetry: {path}")

    rewards = [float(step["reward"]) for step in steps]
    final = steps[-1]
    terminated = bool(final.get("terminated"))
    truncated = bool(final.get("truncated"))
    if terminated:
        end_reason = "terminated"
    elif truncated:
        end_reason = "timeout"
    else:
        end_reason = "stopped"
    return {
        "policy_steps": len(steps),
        "stabilization_steps": int(start.get("stabilization_steps", 0)),
        "max_reward": max(rewards),
        "final_reward": rewards[-1],
        "partial_success": any(reward >= 0.5 for reward in rewards),
        "partial_reward_steps": sum(reward == 0.5 for reward in rewards),
        "final_chunk_action_index": final.get("chunk_action_index"),
        "terminated": terminated,
        "truncated": truncated,
        "end_reason": end_reason,
    }


def _find_episode_videos(result_dir: Path, episode: int) -> tuple[Path, Path]:
    matches = [
        path
        for path in result_dir.rglob(f"episode_{episode}")
        if path.is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one video directory for episode {episode} in {result_dir}, "
            f"found {len(matches)}"
        )
    episode_dir = matches[0]
    left = sorted(episode_dir.glob("head_stereo_left_*.mp4"))
    right = sorted(episode_dir.glob("head_stereo_right_*.mp4"))
    if len(left) != 1 or len(right) != 1:
        raise ValueError(
            f"Episode {episode} must have one left and one right MP4 in {episode_dir}"
        )
    return left[0], right[0]


def collect_run(result_dir: Path, expected_episodes: int) -> list[EpisodeRecord]:
    stats = parse_eval_stats(result_dir / "eval_stats.txt")
    expected = set(range(expected_episodes))
    if set(stats) != expected:
        raise ValueError(
            f"Expected episodes {sorted(expected)} in {result_dir}, got {sorted(stats)}"
        )

    records = []
    for episode in sorted(stats):
        telemetry = parse_telemetry(result_dir / "telemetry" / f"episode_{episode}.jsonl")
        left_video, right_video = _find_episode_videos(result_dir, episode)
        records.append(
            EpisodeRecord(
                episode=episode,
                left_video=left_video,
                right_video=right_video,
                success=stats[episode],
                **telemetry,
            )
        )
    return records


def summarize(records: list[EpisodeRecord]) -> dict[str, float | int]:
    count = len(records)
    return {
        "episodes": count,
        "successes": sum(record.success for record in records),
        "success_rate": sum(record.success for record in records) / count,
        "partial_successes": sum(record.partial_success for record in records),
        "partial_success_rate": sum(record.partial_success for record in records) / count,
        "timeout_rate": sum(record.truncated for record in records) / count,
        "mean_max_reward": sum(record.max_reward for record in records) / count,
        "mean_policy_steps": sum(record.policy_steps for record in records) / count,
        "mean_stabilization_steps": sum(record.stabilization_steps for record in records) / count,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema in {path}")
    if not manifest.get("runs"):
        raise ValueError(f"Manifest has no runs: {path}")
    return manifest


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _run_config(manifest: dict[str, Any], run_spec: dict[str, Any]) -> dict[str, Any]:
    shared = {
        key: value
        for key, value in manifest.items()
        if key not in {"schema_version", "entity", "project", "group", "runs"}
    }
    return {**shared, **run_spec["config"]}


def upload_run(
    wandb: Any,
    manifest: dict[str, Any],
    run_key: str,
    run_spec: dict[str, Any],
    result_dir: Path,
    records: list[EpisodeRecord],
    *,
    force: bool,
    upload_artifact: bool,
) -> None:
    config = _run_config(manifest, run_spec)
    run = wandb.init(
        entity=manifest.get("entity") or os.getenv("WANDB_ENTITY"),
        project=manifest["project"],
        group=manifest["group"],
        job_type="evaluation",
        id=run_spec["id"],
        resume="allow",
        name=run_spec["name"],
        tags=run_spec.get("tags", []),
        config=config,
    )
    try:
        if run.summary.get("upload_complete") and not force:
            print(f"{run_key}: already uploaded; use --force to replace the table")
            return

        columns = [
            "episode",
            "left_video",
            "right_video",
            "success",
            "max_reward",
            "final_reward",
            "partial_success",
            "partial_reward_steps",
            "policy_steps",
            "stabilization_steps",
            "final_chunk_action_index",
            "terminated",
            "truncated",
            "end_reason",
        ]
        table = wandb.Table(columns=columns)
        for record in records:
            metrics = record.metrics()
            table.add_data(
                record.episode,
                wandb.Video(str(record.left_video), format="mp4"),
                wandb.Video(str(record.right_video), format="mp4"),
                metrics["success"],
                metrics["max_reward"],
                metrics["final_reward"],
                metrics["partial_success"],
                metrics["partial_reward_steps"],
                metrics["policy_steps"],
                metrics["stabilization_steps"],
                metrics["final_chunk_action_index"],
                metrics["terminated"],
                metrics["truncated"],
                metrics["end_reason"],
            )
        run.log({"episodes": table})

        if upload_artifact:
            artifact = wandb.Artifact(
                name=f"{run_spec['id']}-telemetry",
                type="simple-eval",
                metadata=config,
            )
            artifact.add_file(str(result_dir / "eval_stats.txt"), name="eval_stats.txt")
            artifact.add_dir(str(result_dir / "telemetry"), name="telemetry")
            for log_name in ("eval_latest.log", "eval_worker_0.log"):
                log_path = result_dir / log_name
                if log_path.is_file():
                    artifact.add_file(str(log_path), name=log_name)
            run.log_artifact(artifact)

        run.summary.update(summarize(records))
        run.summary["upload_complete"] = True
    finally:
        run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and upload SIMPLE evaluation videos to Weights & Biases."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-artifact", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        selected = args.runs or list(manifest["runs"])
        unknown = sorted(set(selected) - set(manifest["runs"]))
        if unknown:
            raise ValueError(f"Unknown runs: {', '.join(unknown)}")

        prepared = []
        for run_key in selected:
            run_spec = manifest["runs"][run_key]
            result_dir = _resolve_path(run_spec["result_dir"], args.repo_root.resolve())
            records = collect_run(result_dir, int(manifest["expected_episodes"]))
            prepared.append((run_key, run_spec, result_dir, records))
            print(json.dumps({
                "run": run_key,
                "wandb_id": run_spec["id"],
                "result_dir": str(result_dir),
                **summarize(records),
                "video_bytes": sum(
                    record.left_video.stat().st_size + record.right_video.stat().st_size
                    for record in records
                ),
            }, sort_keys=True))

        if args.dry_run:
            print(f"Dry run passed for {len(prepared)} W&B runs; nothing was uploaded.")
            return 0

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "wandb is not installed in this Python environment; run with the project environment"
            ) from exc

        for run_key, run_spec, result_dir, records in prepared:
            upload_run(
                wandb,
                manifest,
                run_key,
                run_spec,
                result_dir,
                records,
                force=args.force,
                upload_artifact=not args.skip_artifact,
            )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())