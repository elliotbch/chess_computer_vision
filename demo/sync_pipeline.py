"""Copy what the demo needs into demo/ so that it builds on its own (a Hugging Face Space only
receives the demo/ folder): pipeline modules from src/ (demo/tests checks they are identical),
the two annotated games and the public-domain games of data/famous_games/, training/metrics.json,
and the example photos (JPEG, quality 92, full resolution) from data/sample_frames/.

    python demo/sync_pipeline.py
"""

import glob
import os
import shutil

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["robust_calibration.py", "occupancy.py", "replay.py", "logic.py", "chessrules.py", "piece_inference.py"]
EXAMPLE_FRAMES = ["frame05", "frame06", "frame07", "frame08", "frame09", "frame20"]
GAMES = ["archive/step1_annotation/step1_Lea/gameLR_fix.json", "archive/step1_annotation/step1_Lea/gameLR_mov.json"]


def main():
    dst = os.path.join(ROOT, "demo", "pipeline")
    os.makedirs(dst, exist_ok=True)
    for m in MODULES:
        shutil.copy2(os.path.join(ROOT, "src", m), os.path.join(dst, m))
    games = os.path.join(ROOT, "demo", "static", "examples", "games")
    os.makedirs(games, exist_ok=True)
    for g in GAMES + sorted(glob.glob(os.path.join(ROOT, "data", "famous_games", "*.pgn"))):
        shutil.copy2(os.path.join(ROOT, g), os.path.join(games, os.path.basename(g)))
    shutil.copy2(os.path.join(ROOT, "training", "metrics.json"), os.path.join(ROOT, "demo", "eval_metrics.json"))
    ex = os.path.join(ROOT, "demo", "static", "examples")
    for name in EXAMPLE_FRAMES:
        img = cv2.imread(os.path.join(ROOT, "data", "sample_frames", f"{name}.png"))
        cv2.imwrite(os.path.join(ex, f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("copied", MODULES, [os.path.basename(g) for g in GAMES], "metrics.json and", EXAMPLE_FRAMES)


if __name__ == "__main__":
    main()
