"""Record a closed-loop PK controller and verify a labeled 60-second cut."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai_pk_runner import RuleBasedDecider, run_ai_episode


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ai_pk_closed_loop_videos"
FPS = 30
VARIED_BALL_OFFSETS = [(0.0, -0.30), (0.0, -0.15), (0.0, 0.0),
                       (0.0, 0.15), (0.0, 0.30)]
WIDTH = 640
HEIGHT = 480
TARGET_FRAMES = 60 * FPS
ACTION_JA = {
    "GOTO_STANCE": "ボールへ再接近・位置合わせ",
    "ALIGN": "体の向きを整える", "KICK": "キック", "HOLD": "待機",
}
FONT_PATHS = (
    Path(r"C:\Windows\Fonts\meiryo.ttc"),
    Path(r"C:\Windows\Fonts\YuGothM.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise RuntimeError("Japanese font unavailable; install Meiryo or Noto Sans CJK.")


def _text(draw: ImageDraw.ImageDraw, xy: tuple, text: str, size: int = 18,
          fill: tuple = (235, 245, 250), max_width: int = WIDTH - 32) -> None:
    while size > 10 and draw.textlength(text, font=_font(size)) > max_width:
        size -= 1
    if draw.textlength(text, font=_font(size)) > max_width:
        raise ValueError(f"Video text does not fit the available width: {text}")
    draw.text(xy, text, font=_font(size), fill=fill)


def _fit(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)


def _api_state(sim_time: float, decisions: list[dict]) -> tuple[str, dict | None, int]:
    for index, decision in reversed(list(enumerate(decisions, start=1))):
        if sim_time + 1e-9 >= decision["sent_at"]:
            if sim_time + 1e-9 < decision["received_at"]:
                return "waiting", decision, index
            return "received", decision, index
    return "idle", None, 0


def _hud(frame: np.ndarray, snapshot: dict, provider_label: str,
         decisions: list[dict], live: bool) -> np.ndarray:
    image = Image.fromarray(_fit(frame))
    draw = ImageDraw.Draw(image)
    t = snapshot["sim_time"]
    phase, decision, number = _api_state(t, decisions) if live else ("idle", None, 0)
    draw.rectangle((0, 0, WIDTH, 64), fill=(15, 24, 31))
    draw.rectangle((0, HEIGHT - 96, WIDTH, HEIGHT), fill=(15, 24, 31))
    _text(draw, (16, 5), f"Unitree G1 ｜ {provider_label}", 22)
    _text(draw, (16, 35), "数値の状態を入力 ｜ RL歩行＋共通キック動作", 16,
          (255, 211, 110))
    status = "ゴール！" if snapshot["goal"] else "転倒" if snapshot["fallen"] else "プレー中"
    _text(draw, (16, HEIGHT - 91), f"{t:05.2f}秒 ｜ {ACTION_JA[snapshot['action']]} ｜ {status}", 18)
    event_color = None
    if decision is not None:
        elapsed = t - decision["sent_at"]
        wait = decision["received_at"] - decision["sent_at"]
        if phase == "waiting":
            event_color = (70, 190, 255)
            label = f"APIへ送信 #{number} ・ 回答待ち"
            _text(draw, (16, HEIGHT - 63), label, 18, event_color)
            _text(draw, (16, HEIGHT - 35), "回答待ちの間も、前の指示で動き続けます", 15)
            if elapsed < .65:
                draw.rectangle((8, 70, 632, 115), fill=event_color)
                _text(draw, (24, 77), f"↑ APIへ観測・指示を送信 #{number}", 24, (10, 25, 35))
        else:
            event_color = (95, 235, 165)
            applied = decision["applied_at"] is not None
            _text(draw, (16, HEIGHT - 63), f"API回答を受信 #{number}", 18, event_color)
            _text(draw, (16, HEIGHT - 35),
                  f"指示：{ACTION_JA[decision['action']]}" if applied else "試行終了のため指示は未適用", 16)
            if t - decision["received_at"] < .85:
                draw.rectangle((8, 70, 632, 115), fill=event_color)
                _text(draw, (24, 77), f"↓ API回答を受信 #{number} ・ {'動作に反映' if applied else '未適用'}",
                      23, (10, 25, 35))
        event_age = elapsed if phase == "waiting" else t - decision["received_at"]
        if event_age < .85:
            inset = 4 + int(5 * abs(np.sin(event_age * 12)))
            draw.rectangle((inset, inset, WIDTH - inset, HEIGHT - inset),
                           outline=event_color, width=4)
    else:
        _text(draw, (16, HEIGHT - 54), "ローカル参照制御（API呼び出しなし）" if not live
              else "観測の送信前", 17)
    return np.asarray(image).copy()


def _card(background: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    frame = _fit(background).copy()
    tint = np.zeros_like(frame)
    tint[:] = (15, 24, 31)
    frame = cv2.addWeighted(frame, 0.04, tint, 0.96, 0)
    cv2.rectangle(frame, (36, 48), (WIDTH - 36, HEIGHT - 48), (15, 24, 31), -1)
    cv2.rectangle(frame, (36, 48), (44, HEIGHT - 48), (70, 205, 165), -1)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    _text(draw, (64, 76), title, 26, max_width=WIDTH - 110)
    y = 145
    for line in lines:
        _text(draw, (64, y), line, 19, (210, 224, 229), WIDTH - 110)
        y += 47
    return np.asarray(image).copy()


def _decision_card(background: np.ndarray, index: int, total: int, decision: dict,
                   provider_note: str = "ローカル参照制御（APIなし）") -> np.ndarray:
    applied = decision["applied_at"]
    applied_note = f"{applied:.2f}秒" if applied is not None else "未適用（試行終了）"
    lines = [
        f"送信 {decision['sent_at']:.2f}秒 → 受信相当 {decision['received_at']:.2f}秒",
        f"回答：{ACTION_JA[decision['action']]} ｜ 適用：{applied_note}",
        f"API待ち時間：{decision['reported_latency_s']:.2f}秒",
        f"立ち位置まで {decision['dist_to_stance_m']:.3f}m ／ 向きのずれ {decision['yaw_err_deg']:+.1f}度",
        "受信相当時刻＝送信時刻＋実測API待ち時間",
    ]
    card = Image.fromarray(_card(background, f"API通信と判断 #{index} ／ 全{total}回", lines))
    draw = ImageDraw.Draw(card)
    draw.rectangle((64, 120, 225, 130), fill=(70, 190, 255))
    draw.rectangle((235, 120, 396, 130), fill=(95, 235, 165))
    return np.asarray(card).copy()


def _append_hold(writer: Any, frame: np.ndarray, count: int) -> None:
    for _ in range(count):
        writer.append_data(frame)


def _compose_video(run_dir: Path, frames: list[np.ndarray], result: Any,
                   prefix: str = "", provider_note: str = "Local reference; no API calls",
                   live: bool = False) -> Path:
    if not frames:
        raise RuntimeError("The simulation produced no camera frames.")

    source_video = run_dir / f"{prefix}episode.mp4"
    with imageio.get_writer(str(source_video), fps=FPS, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)

    decisions = result.decisions
    intro_frames = 4 * FPS
    minimum_result_frames = 9 * FPS
    available = TARGET_FRAMES - intro_frames - len(frames) - minimum_result_frames
    if available < 0:
        raise RuntimeError("The episode is too long to compose the requested 60-second cut.")
    shown_indices = sorted(set(np.linspace(0, len(decisions) - 1,
                                          min(3, len(decisions)), dtype=int))) if decisions else []
    shown_decisions = [(index + 1, decisions[index]) for index in shown_indices]
    decision_frames = min(2 * FPS, available // max(1, len(shown_decisions)))
    reserved = intro_frames + len(frames) + len(shown_decisions) * decision_frames + minimum_result_frames
    replay_budget = TARGET_FRAMES - reserved
    replay_factor = max(0, min(3, replay_budget // len(frames)))

    video_path = run_dir / f"{prefix}ai_pk_closed_loop_60s.mp4"
    title = _card(frames[0], "実APIで動かすロボットサッカー" if live else "再接近・再キックの検証", [
        f"判断担当：{provider_note}",
        "摩擦を増やし、未得点なら再接近して再キック",
        "青＝観測を送信・回答待ち ／ 緑＝回答を受信" if live else "ルールによる参照制御（APIなし）",
        "API待ち時間中も、前の指示で物理を進めます",
        "入力は数値のみ。画像認識の実験ではありません",
    ])
    live_frames = [_fit(frame) for frame in frames]
    replay_label = (f"同じ試行のリプレイ ｜ {replay_factor}倍スロー"
                    if replay_factor > 1 else "同じ試行のリプレイ（追加試行ではありません）")
    replay_frames = []
    for frame in live_frames:
        replay = frame.copy()
        cv2.rectangle(replay, (0, 0), (WIDTH, 38), (15, 24, 31), -1)
        replay_image = Image.fromarray(replay)
        _text(ImageDraw.Draw(replay_image), (16, 6), replay_label, 20, (255, 211, 110))
        replay_frames.append(np.asarray(replay_image).copy())

    with imageio.get_writer(str(video_path), fps=FPS, codec="libx264", quality=8) as writer:
        _append_hold(writer, title, intro_frames)
        for frame in live_frames:
            writer.append_data(frame)
        for frame in replay_frames:
            for _ in range(replay_factor):
                writer.append_data(frame)
        for index, decision in shown_decisions:
            background = live_frames[min(len(live_frames) - 1, round(decision["sent_at"] * FPS))]
            _append_hold(writer, _decision_card(background, index, len(decisions), decision,
                                               provider_note), decision_frames)

        used_frames = intro_frames + len(live_frames) + len(replay_frames) * replay_factor + len(shown_decisions) * decision_frames
        outcome = "ゴール成功" if result.goal else "未得点"
        end_note = {"goal": "ゴール", "timeout": "時間切れ", "fall_before_kick": "キック前に転倒",
                    "fall_after_kick": "キック後に転倒", "ball_live_timeout": "転倒後の観測終了"}[result.end_reason]
        mean_wait = np.mean([d["reported_latency_s"] for d in decisions]) if decisions else 0
        summary = _card(live_frames[-1], "今回の実測結果（1試行）", [
            f"{outcome} ／ 転倒：{'あり' if result.fallen else 'なし'} ／ {end_note}",
            f"キック {len(result.kick_times)}回 ／ 経過 {result.end_time:.2f}秒",
            f"判断 {len(decisions)}回 ／ 平均API待ち時間 {mean_wait:.2f}秒",
            f"ボール最大x：{result.max_ball_x:.3f}m ／ ゴールライン：11.25m",
            "成功・失敗を含め、そのまま記録しています",
        ])
        _append_hold(writer, summary, 4 * FPS)
        discussion = _card(live_frames[-1], "考察：今回わかったことと限界", [
            "実APIの判断で、共通の歩行・キック機能を操作",
            "回答を待つ間に状態が変わり、観測が古くなる",
            "位置・角度・強さはAIが直接指定していません",
            "ルール制御でも得点可能。AIの優位性は未検証",
            "各1試行のため、モデルの優劣は結論できません",
        ])
        _append_hold(writer, discussion, TARGET_FRAMES - used_frames - 4 * FPS)

    return video_path


def _verify_and_contact_sheet(video_path: Path, run_dir: Path, prefix: str = "",
                              expected_frames: int = TARGET_FRAMES,
                              event_indices: list[int] | None = None) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_indices = set(np.linspace(0, expected_frames - 1, 8, dtype=int))
    if event_indices:
        sample_indices = set(event_indices[:6] + [0, expected_frames - 1])
    samples: list[np.ndarray] = []
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (HEIGHT, WIDTH):
            capture.release()
            raise RuntimeError(f"Unexpected frame size at frame {decoded}: {frame.shape}.")
        if decoded in sample_indices:
            samples.append(frame)
        decoded += 1
    capture.release()
    while samples and len(samples) < 8:
        samples.append(samples[-1])

    duration = decoded / fps if fps else 0.0
    if decoded != expected_frames or abs(fps - FPS) > 0.01 or abs(duration - expected_frames / FPS) > 0.02:
        raise RuntimeError(
            f"Video validation failed: decoded={decoded}, reported={reported_frames}, "
            f"fps={fps:.3f}, duration={duration:.3f}s."
        )
    if width != WIDTH or height != HEIGHT or not samples or any(float(np.std(frame)) < 4.0 for frame in samples):
        raise RuntimeError("Video framing or representative-frame content validation failed.")

    sheet = np.zeros((HEIGHT * 2, WIDTH * 4, 3), dtype=np.uint8)
    for index, frame in enumerate(samples):
        row, column = divmod(index, 4)
        sheet[row * HEIGHT:(row + 1) * HEIGHT, column * WIDTH:(column + 1) * WIDTH] = frame
    if not cv2.imwrite(str(run_dir / f"{prefix}preview_contact_sheet.jpg"), sheet):
        raise RuntimeError("Could not write the representative-frame contact sheet.")
    return {
        "fps": fps,
        "frame_count": decoded,
        "duration_sec": duration,
        "width": width,
        "height": height,
        "all_frames_decoded": True,
        "representative_frames_nonblank": True,
    }


def record_run(decider: Any, run_dir: Path, run_id: str, prefix: str,
               provider_note: str, live: bool, model: str | None = None) -> dict:
    output_names = (
        "episode.mp4", "ai_pk_closed_loop_60s.mp4", "preview_contact_sheet.jpg",
        "episode_preview_contact_sheet.jpg", "manifest.json",
    )
    for name in output_names:
        path = run_dir / f"{prefix}{name}"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    frames: list[np.ndarray] = []
    snapshots: list[dict] = []

    def capture_frame(env, obs, action: str) -> None:
        frames.append(_fit(env.render_camera("broadcast_cam")))
        snapshots.append({
            "sim_time": obs.sim_time, "action": action,
            "goal": bool(obs.goal_scored), "fallen": bool(obs.robot_fallen),
        })

    result = run_ai_episode(
        decider, ball_lateral_speed=-0.11, frame_callback=capture_frame
    )
    for index, snapshot in enumerate(snapshots):
        frames[index] = _hud(frames[index], snapshot, provider_note, result.decisions, live)
    event_indices = []
    for decision in result.decisions[:3]:
        for event_time in (decision["sent_at"], decision["received_at"]):
            if event_time <= snapshots[-1]["sim_time"]:
                event_indices.append(min(len(frames) - 1, round(event_time * FPS) + 1))
    video_path = _compose_video(run_dir, frames, result, prefix, provider_note, live)
    video_stats = _verify_and_contact_sheet(video_path, run_dir, prefix)
    source_path = run_dir / f"{prefix}episode.mp4"
    source_stats = _verify_and_contact_sheet(source_path, run_dir, f"{prefix}episode_",
                                            expected_frames=len(frames), event_indices=event_indices)
    manifest = {
        "run_id": run_id,
        "controller": f"{type(decider).__name__} + RLActionExecutor",
        "provider": provider_note,
        "live_api": live,
        "model": model,
        "api_calls": getattr(decider, "api_calls", 0),
        "latency_semantics": "Physics advances under previous action during measured inference",
        "language": "ja",
        "api_event_visuals": {
            "send": "blue flash, waiting timer and progress bar",
            "receive": "green flash and applied/not-applied label",
            "time_basis": "Simulation-mapped request time + measured API latency; UTC timestamps also logged",
            "late_responses": "Responses beyond episode end appear in analysis cards, not as in-play events",
        },
        "simulation": {
            "goal": result.goal,
            "fallen": result.fallen,
            "fall_time_sec": result.fall_time,
            "end_reason": result.end_reason,
            "end_time_sec": result.end_time,
            "kick_time_sec": result.kick_time,
            "kick_times_sec": result.kick_times,
            "kick_count": len(result.kick_times),
            "peak_ball_vx_mps": result.peak_ball_vx,
            "max_ball_x_m": result.max_ball_x,
            "final_ball": result.final_ball,
            "decisions": result.decisions,
        },
        "video": {"path": str(video_path), **video_stats},
        "source_video": {"path": str(source_path), **source_stats},
    }
    (run_dir / f"{prefix}manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def _result_grid(finals: list[np.ndarray], results: list[dict],
                 provider_note: str) -> np.ndarray:
    if len(finals) != len(results) or not finals:
        raise ValueError("Final screenshots and trial results must match.")
    image = Image.new("RGB", (WIDTH, HEIGHT), (15, 24, 31))
    draw = ImageDraw.Draw(image)
    successes = sum(result["goal"] and not result["fallen"] for result in results)
    _text(draw, (16, 8), f"{provider_note} ｜ {len(results)}回中 {successes}回成功", 25)
    for index, (frame, result) in enumerate(zip(finals, results)):
        row, col = divmod(index, 3)
        x, y = 8 + col * 210, 62 + row * 204
        image.paste(Image.fromarray(frame).resize((204, 153)), (x, y))
        success = result["goal"] and not result["fallen"]
        label = "成功" if success else "転倒で終了" if result["fallen"] else "時間切れ"
        _text(draw, (x + 3, y + 157), f"{index + 1}回目：{label}", 17,
              (95, 235, 165) if success else (255, 211, 110), 200)
        _text(draw, (x + 3, y + 180), f"キック {len(result['kick_times'])}回", 14, max_width=200)
    return np.asarray(image).copy()


def record_five_trials(decider: Any, run_dir: Path, prefix: str,
                       provider_note: str, model: str | None,
                       ball_offsets: list | None = None) -> dict:
    """Record five real-time trials on disk, then concatenate without replay."""
    batch_dir = run_dir / f"{prefix}five_trials"
    batch_dir.mkdir(parents=False, exist_ok=False)
    results: list[dict] = []
    finals: list[np.ndarray] = []
    trials: list[dict] = []
    for trial in range(1, 6):
        raw_path = batch_dir / f"trial_{trial}_camera.mp4"
        video_path = batch_dir / f"trial_{trial}.mp4"
        snapshots: list[dict] = []
        final_camera: np.ndarray | None = None
        before_calls = getattr(decider, "api_calls", 0)
        offset = ball_offsets[trial - 1] if ball_offsets else (0.0, 0.0)
        with imageio.get_writer(str(raw_path), fps=FPS, codec="libx264", quality=8) as writer:
            def capture(env, obs, action):
                nonlocal final_camera
                frame = _fit(env.render_camera("broadcast_cam"))
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
        # Capture the terminal state rather than the preceding scheduled video frame.
        result_data = asdict(result)
        results.append(result_data)
        finals.append(final_camera)
        count = 0
        with imageio.get_reader(str(raw_path)) as reader, imageio.get_writer(
                str(video_path), fps=FPS, codec="libx264", quality=8) as writer:
            for frame_index, frame in enumerate(reader):
                if frame_index >= len(snapshots):
                    raise RuntimeError("Camera video has more frames than recorded snapshots.")
                snapshot = snapshots[frame_index]
                label = f"{provider_note} ｜ {trial}/5回目"
                if ball_offsets:
                    label += f" ｜ ボール横位置 {offset[1]:+.2f}m"
                rendered = _hud(frame, snapshot, label, result.decisions, model is not None)
                image = Image.fromarray(rendered)
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, HEIGHT - 96, WIDTH, HEIGHT - 65), fill=(15, 24, 31))
                _text(draw, (16, HEIGHT - 91),
                      f"制限60秒・経過 {snapshot['sim_time']:.1f}秒 ｜ {ACTION_JA[snapshot['action']]}",
                      18)
                writer.append_data(np.asarray(image))
                count += 1
        if count != len(snapshots):
            raise RuntimeError("Camera decode count does not match trial snapshots.")
        stats = _verify_and_contact_sheet(video_path, batch_dir, f"trial_{trial}_",
                                          expected_frames=count)
        raw_path.unlink()
        trial_data = {
            "trial": trial, "ball_offset_xy": list(offset),
            "api_calls": getattr(decider, "api_calls", 0) - before_calls,
            "simulation": result_data, "video": {"path": str(video_path), **stats},
        }
        trials.append(trial_data)
        (batch_dir / f"trial_{trial}.json").write_text(
            json.dumps(trial_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "trial": trial, "goal": result.goal, "fallen": result.fallen,
            "kicks": len(result.kick_times), "end_time": result.end_time,
            "api_calls": trial_data["api_calls"],
        }), flush=True)

    summary = _result_grid(finals, results, provider_note)
    summary_path = batch_dir / "final_results.jpg"
    if not cv2.imwrite(str(summary_path), cv2.cvtColor(summary, cv2.COLOR_RGB2BGR)):
        raise RuntimeError("Could not save final screenshot grid.")
    intro = _card(finals[0], f"{provider_note}の場合", [
        "ルール：60秒以内にゴールへボールを入れる",
        "何度蹴ってもOK。未得点なら近づいて再キック",
        "倒れたら、その試行は終了（失敗）",
        "ボールの横位置を-0.30m〜+0.30mで変えて5回実施" if ball_offsets
        else "同じ初期配置で5回実施します",
        "青：APIへ送信 ／ 緑：API回答を受信",
    ])
    output = batch_dir / "five_trials_ja.mp4"
    with imageio.get_writer(str(output), fps=FPS, codec="libx264", quality=8) as writer:
        _append_hold(writer, intro, 6 * FPS)
        for trial_data in trials:
            with imageio.get_reader(trial_data["video"]["path"]) as reader:
                for frame in reader:
                    writer.append_data(frame)
        _append_hold(writer, summary, 10 * FPS)
    expected = 16 * FPS + sum(t["video"]["frame_count"] for t in trials)
    stats = _verify_and_contact_sheet(output, batch_dir, expected_frames=expected)
    manifest = {
        "provider": provider_note, "model": model, "live_api": model is not None,
        "rules": {"max_time_sec": 60, "unlimited_kicks": True, "stop_on_fall": True},
        "trials_count": 5, "ball_offsets_xy": ball_offsets,
        "successes": sum(r["goal"] and not r["fallen"] for r in results),
        "api_calls": sum(t["api_calls"] for t in trials), "trials": trials,
        "language": "ja", "playback_speed": 1,
        "video": {"path": str(output), **stats}, "final_grid": str(summary_path),
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "provider", "successes", "api_calls", "video", "final_grid",
    )}, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("reference", "gemini", "jev"),
                        default="reference")
    parser.add_argument("--output-dir", type=Path,
                        help="Existing folder for uniquely named videos; nothing is overwritten.")
    parser.add_argument("--model", help="Override the selected live provider's model.")
    parser.add_argument("--trials", type=int, choices=(1, 5), default=5,
                        help="Five 60-second trials by default; 1 retains the legacy recording mode.")
    parser.add_argument("--vary-ball", action="store_true",
                        help="Shift the ball's lateral start position per trial "
                             "(-0.30/-0.15/0/+0.15/+0.30 m).")
    args = parser.parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.output_dir is None:
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        prefix = ""
    else:
        run_dir = args.output_dir.resolve()
        if not run_dir.is_dir():
            raise NotADirectoryError(f"Output folder does not exist: {run_dir}")
        prefix = f"{args.provider}_{run_id}_"

    with ExitStack() as stack:
        if args.provider == "reference":
            decider = RuleBasedDecider()
            note = "ローカル参照制御（APIなし）"
            model = None
        else:
            from env_loader import load_project_dotenv
            from live_pk_deciders import GeminiLiveDecider, JEVLiveDecider

            load_project_dotenv()
            if args.provider == "gemini":
                from google import genai
                from google.genai import types

                key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
                if not key:
                    raise RuntimeError("Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY.")
                client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=30000, retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
                stack.callback(client.close)
                model = args.model or "gemini-robotics-er-2-preview"
                decider = GeminiLiveDecider(client, model_name=model)
                note = "Gemini ER2"
            else:
                from openai import OpenAI

                key = os.environ.get("OPENROUTER_API_KEY")
                if not key:
                    raise RuntimeError("JEV requires OPENROUTER_API_KEY.")
                client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1",
                                max_retries=0, timeout=30.0)
                stack.callback(client.close)
                model = args.model or "typesafe/jev-router"
                decider = JEVLiveDecider(client, model_name=model)
                note = "Jev"
        if args.trials == 5:
            record_five_trials(decider, run_dir, prefix, note, model,
                               ball_offsets=VARIED_BALL_OFFSETS if args.vary_ball else None)
        else:
            record_run(decider, run_dir, run_id, prefix, note, args.provider != "reference", model)


if __name__ == "__main__":
    main()