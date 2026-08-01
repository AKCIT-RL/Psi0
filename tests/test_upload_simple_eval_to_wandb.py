import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval import upload_simple_eval_to_wandb as uploader


def _write_episode(root: Path, episode: int, *, right_video: bool = True) -> None:
    telemetry = root / "telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event": "episode_start",
            "episode": f"episode_{episode}",
            "stabilization_steps": 300,
        },
        {
            "event": "step",
            "reward": 0.0,
            "chunk_action_index": 0,
            "terminated": False,
            "truncated": False,
        },
        {
            "event": "step",
            "reward": 0.5 if episode == 0 else 0.0,
            "chunk_action_index": 1,
            "terminated": False,
            "truncated": True,
        },
    ]
    (telemetry / f"episode_{episode}.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    video_dir = root / "policy" / "task" / "level-0" / f"episode_{episode}"
    video_dir.mkdir(parents=True)
    (video_dir / "head_stereo_left_failed.mp4").write_bytes(b"left")
    if right_video:
        (video_dir / "head_stereo_right_failed.mp4").write_bytes(b"right")


class UploadSimpleEvalTest(unittest.TestCase):
    def test_collect_run_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "eval_stats.txt").write_text(
                "run: task - policy\nepisode_0: False\nepisode_1: True\n"
            )
            _write_episode(root, 0)
            _write_episode(root, 1)

            records = uploader.collect_run(root, expected_episodes=2)
            summary = uploader.summarize(records)

            self.assertEqual([record.episode for record in records], [0, 1])
            self.assertTrue(records[0].partial_success)
            self.assertEqual(records[0].end_reason, "timeout")
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(summary["partial_success_rate"], 0.5)
            self.assertEqual(summary["timeout_rate"], 1.0)

    def test_collect_run_rejects_missing_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "eval_stats.txt").write_text("episode_0: False\n")
            _write_episode(root, 0, right_video=False)

            with self.assertRaisesRegex(ValueError, "one left and one right MP4"):
                uploader.collect_run(root, expected_episodes=1)


if __name__ == "__main__":
    unittest.main()