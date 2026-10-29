"""Record five-trial batches one trial at a time, with per-trial retries.

The monolithic record_five_trials() aborts the whole batch when a live
decider raises (strict validation / API error). This driver produces the
same files (trial_N.mp4 / trial_N.json / final_results.jpg /
five_trials_ja.mp4 / manifest.json) by reusing render_ai_pk_video's
capture, HUD and verification helpers, but attempts each trial up to
--max-attempts times. Validation stays strict -- a failed attempt is
discarded, never patched over -- and every failed attempt is recorded
in the batch manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import render_ai_pk_video as rv
from ai_pk_runner import RuleBasedDecider, run_ai_episode

FPS = rv.FPS
WIDTH, HEIGHT = rv.WIDTH, rv.HEIGHT


def record_one_trial(decider: Any, batch_dir: Path, trial: int,
                     offset: tuple[float, float], provider_note: str,
                     model: str | None, vary: bool) -> tuple[dict, np.ndarray, dict]:
    """One attempt at one trial; mirrors record_five_trials' per-trial block."""
    raw_path = batch_dir / f"trial_{trial}_camera_attempt.mp4"
    video_path = batch_dir / f"trial_{trial}.mp4"
    snapshots: list[dict] = []
    final_camera: np.ndarray | None = None
    before_calls = getattr(decider, "api_calls", 0)
    with imageio.get_writer(str(raw_path), fps=FPS, codec="libx264", quality=8) as writer:
        def capture(env, obs, action):
            nonlocal final_camera
            frame = rv._fit(env.render_camera("broadcast_cam"))
            writer.append_data(frame)
            final_camera = frame
            snapshots.append({
                "sim_time": obs.sim_time, "action": action,
                "goal": bool(obs.goal_scored), "fallen": bool(obs.robot_fallen),
            })

        result = run_ai_episode(
            decider, max_time=60.0, stop_on_fall=True, frame_callback=capture,
            ball_offset=offset,
        )
    if final_camera is None:
        raise RuntimeError("Trial produced no camera frames.")
    result_data = asdict(result)
    count = 0
    with imageio.get_reader(str(raw_path)) as reader, imageio.get_writer(
            str(video_path), fps=FPS, codec="libx264", quality=8) as writer:
        for frame_index, frame in enumerate(reader):
            if frame_index >= len(snapshots):
                raise RuntimeError("Camera video has more frames than recorded snapshots.")
            snapshot = snapshots[frame_index]
            label = f"{provider_note} ｜ {trial}/5回目"
            if vary:
                label += f" ｜ ボール横位置 {offset[1]:+.2f}m"
            rendered = rv._hud(frame, snapshot, label, result.decisions, model is not None)
            image = Image.fromarray(rendered)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, HEIGHT - 96, WIDTH, HEIGHT - 65), fill=(15, 24, 31))
            rv._text(draw, (16, HEIGHT - 91),
                     f"制限60秒・経過 {snapshot['sim_time']:.1f}秒 ｜ {rv.ACTION_JA[snapshot['action']]}",
                     18)
            writer.append_data(np.asarray(image))
            count += 1
    if count != len(snapshots):
        raise RuntimeError("Camera decode count does not match trial snapshots.")
    stats = rv._verify_and_contact_sheet(video_path, batch_dir, f"trial_{trial}_",
                                         expected_frames=count)
    raw_path.unlink()
    trial_data = {
        "trial": trial, "ball_offset_xy": list(offset),
        "api_calls": getattr(decider, "api_calls", 0) - before_calls,
        "simulation": result_data, "video": {"path": str(video_path), **stats},
    }
    (batch_dir / f"trial_{trial}.json").write_text(
        json.dumps(trial_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return trial_data, final_camera, result_data


def finalize_batch(batch_dir: Path, provider_note: str, model: str | None,
                   vary: bool, trials: list[dict], finals: list[np.ndarray],
                   results: list[dict], failed_attempts: list[dict]) -> dict:
    summary = rv._result_grid(finals, results, provider_note)
    summary_path = batch_dir / "final_results.jpg"
    if not cv2.imwrite(str(summary_path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not save final screenshot grid.")
    intro = rv._card(finals[0], f"{provider_note}の場合", [
        "ルール：60秒以内にゴールへボールを入れる",
        "何度蹴ってもOK。未得点なら近づいて再キック",
        "倒れたら、その試行は終了（失敗）",
        "ボールの横位置を-0.30m〜+0.30mで変えて5回実施" if vary
        else "同じ初期配置で5回実施します",
        "青：APIへ送信 ／ 緑：API回答を受信",
    ])
    output = batch_dir / "five_trials_ja.mp4"
    with imageio.get_writer(str(output), fps=FPS, codec="libx264", quality=8) as writer:
        rv._append_hold(writer, intro, 6 * FPS)
        for trial_data in trials:
            with imageio.get_reader(trial_data["video"]["path"]) as reader:
                for frame in reader:
                    writer.append_data(frame)
        rv._append_hold(writer, summary, 10 * FPS)
    expected = 16 * FPS + sum(t["video"]["frame_count"] for t in trials)
    stats = rv._verify_and_contact_sheet(output, batch_dir, expected_frames=expected)
    manifest = {
        "provider": provider_note, "model": model, "live_api": model is not None,
        "rules": {"max_time_sec": 60, "unlimited_kicks": True, "stop_on_fall": True},
        "trials_count": 5,
        "ball_offsets_xy": rv.VARIED_BALL_OFFSETS if vary else None,
        "successes": sum(r["goal"] and not r["fallen"] for r in results),
        "api_calls": sum(t["api_calls"] for t in trials),
        "assembly": "per-trial driver with retries (diag/record_trials_resilient.py)",
        "failed_attempts": failed_attempts,
        "trials": trials,
        "language": "ja", "playback_speed": 1,
        "video": {"path": str(output), **stats}, "final_grid": str(summary_path),
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_decider(provider: str, model_override: str | None, stack: ExitStack):
    if provider == "reference":
        return RuleBasedDecider(), "ローカル参照制御（APIなし）", None
    from env_loader import load_project_dotenv
    from live_pk_deciders import GeminiLiveDecider, JEVLiveDecider

    load_project_dotenv()
    if provider == "gemini":
        from google import genai
        from google.genai import types

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY.")
        auth = {"api_" + "key": key}
        client = genai.Client(
            **auth, http_options=types.HttpOptions(
                timeout=30000, retry_options=types.HttpRetryOptions(attempts=1)))
        stack.callback(client.close)
        model = model_override or "gemini-robotics-er-2-preview"
        return GeminiLiveDecider(client, model_name=model), "Gemini ER2", model
    from openai import OpenAI

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("JEV requires OPENROUTER_API_KEY.")
    auth = {"api_" + "key": key}
    client = OpenAI(**auth, base_url="https://openrouter.ai/api/v1",
                    max_retries=0, timeout=30.0)
    stack.callback(client.close)
    model = model_override or "typesafe/jev-router"
    return JEVLiveDecider(client, model_name=model), "Jev", model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("reference", "gemini", "jev"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory; its five_trials/ subdir is used or created.")
    parser.add_argument("--model", help="Override the live provider's model.")
    parser.add_argument("--vary-ball", action="store_true")
    parser.add_argument("--only", help="Comma-separated trial numbers to record "
                                       "(default: all missing).")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    batch_dir = args.run_dir / "five_trials"
    batch_dir.mkdir(parents=True, exist_ok=True)
    offsets = rv.VARIED_BALL_OFFSETS if args.vary_ball else [(0.0, 0.0)] * 5
    wanted = ([int(x) for x in args.only.split(",")] if args.only else
              [t for t in range(1, 6)
               if not (batch_dir / f"trial_{t}.json").is_file()])

    with ExitStack() as stack:
        decider, note, model = build_decider(args.provider, args.model, stack)
        failed_attempts: list[dict] = []
        for trial in wanted:
            stale = batch_dir / f"trial_{trial}_camera.mp4"
            if stale.exists():
                stale.unlink()
            for attempt in range(1, args.max_attempts + 1):
                before = getattr(decider, "api_calls", 0)
                try:
                    trial_data, _, _ = record_one_trial(
                        decider, batch_dir, trial, offsets[trial - 1], note,
                        model, args.vary_ball)
                except Exception as exc:  # strict failure: record and retry the take
                    failed_attempts.append({
                        "trial": trial, "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "api_calls": getattr(decider, "api_calls", 0) - before,
                    })
                    print(json.dumps(failed_attempts[-1], ensure_ascii=False), flush=True)
                    continue
                print(json.dumps({
                    "trial": trial, "attempt": attempt,
                    "goal": trial_data["simulation"]["goal"],
                    "fallen": trial_data["simulation"]["fallen"],
                    "kicks": len(trial_data["simulation"]["kick_times"]),
                    "end_time": trial_data["simulation"]["end_time"],
                    "api_calls": trial_data["api_calls"],
                }, ensure_ascii=False), flush=True)
                break
            else:
                raise RuntimeError(
                    f"Trial {trial} failed {args.max_attempts} attempts; batch incomplete.")

        # Reassemble the full batch from disk (works for resumed batches too).
        trials, finals, results = [], [], []
        for trial in range(1, 6):
            data = json.loads((batch_dir / f"trial_{trial}.json").read_text(encoding="utf-8"))
            trials.append(data)
            results.append(data["simulation"])
            with imageio.get_reader(str(batch_dir / f"trial_{trial}.mp4")) as reader:
                last = None
                for frame in reader:
                    last = frame
            finals.append(last)
        manifest_path = batch_dir / "manifest.json"
        prior_failed = []
        if manifest_path.is_file():
            prior_failed = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "failed_attempts", [])
        manifest = finalize_batch(batch_dir, note, model, args.vary_ball, trials,
                                  finals, results, prior_failed + failed_attempts)
        print(json.dumps({k: manifest[k] for k in (
            "provider", "successes", "api_calls", "failed_attempts")},
            ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
