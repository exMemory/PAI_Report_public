"""Compose five 60-second trial videos into one 2x3 grid segment (60 s).

Layout (1920x960, cells 640x480):
    [take1] [take2] [take3]
    [take4] [take5] [TITLE]
Trials that end early freeze on their last frame with a result banner,
so every cell runs the full 60 seconds in parallel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from render_ai_pk_video import _font as font

FPS = 30
SECONDS = 60
FRAMES = FPS * SECONDS
CELL_W, CELL_H = 640, 480
BG = (15, 24, 31)


def result_text(sim: dict) -> tuple[str, tuple[int, int, int]]:
    kicks = len(sim.get("kick_times") or [])
    if sim.get("goal") and not sim.get("fallen"):
        return (f"結果：ゴール（{sim.get('end_time', 0):.1f}秒・キック{kicks}回）",
                (95, 235, 165))
    if sim.get("fallen"):
        return (f"結果：転倒で終了（{sim.get('end_time', 0):.1f}秒・キック{kicks}回）",
                (255, 150, 130))
    return (f"結果：時間切れ（キック{kicks}回）", (255, 211, 110))


def frozen_frame(last: np.ndarray, sim: dict) -> np.ndarray:
    image = Image.fromarray(last.copy())
    draw = ImageDraw.Draw(image)
    text, color = result_text(sim)
    draw.rectangle((0, CELL_H - 64, CELL_W, CELL_H), fill=BG)
    draw.text((16, CELL_H - 48), text, font=font(26), fill=color)
    return np.asarray(image)


def title_cell(title: str, trials: list[dict]) -> np.ndarray:
    image = Image.new("RGB", (CELL_W, CELL_H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((24, 20), title, font=font(34), fill=(235, 245, 250))
    draw.text((24, 74), "11m PK ｜ 制限60秒 ｜ 転倒で終了", font=font(19),
              fill=(170, 190, 205))
    successes = 0
    y = 122
    for entry in trials:
        sim = entry["simulation"]
        dy = (entry.get("ball_offset_xy") or [0.0, 0.0])[1]
        ok = bool(sim.get("goal")) and not sim.get("fallen")
        successes += ok
        label = "ゴール" if ok else ("転倒" if sim.get("fallen") else "時間切れ")
        color = (95, 235, 165) if ok else (255, 211, 110)
        draw.text((24, y),
                  f"テイク{entry['trial']}（横 {dy:+.2f}m）：{label}"
                  f"・キック{len(sim.get('kick_times') or [])}回",
                  font=font(20), fill=color)
        y += 34
    draw.text((24, y + 10), f"成功 {successes} / {len(trials)}", font=font(26),
              fill=(235, 245, 250))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True,
                        help="five_trials directory with trial_N.mp4 / trial_N.json")
    parser.add_argument("--title", required=True, help="Pattern name for the title cell")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    trials = []
    readers = []
    try:
        for index in range(1, 6):
            data = json.loads(
                (args.batch_dir / f"trial_{index}.json").read_text(encoding="utf-8"))
            trials.append(data)
            readers.append(imageio.get_reader(str(args.batch_dir / f"trial_{index}.mp4")))
        iterators = [iter(reader) for reader in readers]
        lasts: list[np.ndarray | None] = [None] * 5
        frozen: list[np.ndarray | None] = [None] * 5
        title = title_cell(args.title, trials)
        positions = [(0, 0), (CELL_W, 0), (2 * CELL_W, 0), (0, CELL_H), (CELL_W, CELL_H)]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(args.output), fps=FPS, codec="libx264",
                                quality=8) as writer:
            for frame_index in range(FRAMES):
                canvas = np.zeros((2 * CELL_H, 3 * CELL_W, 3), dtype=np.uint8)
                canvas[CELL_H:, 2 * CELL_W:] = title
                for cell in range(5):
                    if frozen[cell] is None:
                        try:
                            frame = next(iterators[cell])
                            lasts[cell] = frame
                        except StopIteration:
                            if lasts[cell] is None:
                                raise RuntimeError(f"trial {cell + 1} produced no frames")
                            frozen[cell] = frozen_frame(
                                lasts[cell], trials[cell]["simulation"])
                    current = frozen[cell] if frozen[cell] is not None else lasts[cell]
                    x, y = positions[cell]
                    canvas[y:y + CELL_H, x:x + CELL_W] = current
                writer.append_data(canvas)
                if (frame_index + 1) % (FPS * 10) == 0:
                    print(f"frames {frame_index + 1}/{FRAMES}", flush=True)
    finally:
        for reader in readers:
            reader.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
