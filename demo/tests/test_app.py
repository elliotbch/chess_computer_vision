import filecmp
import io
import json
import os
import sys

import cv2
import numpy as np
from fastapi.testclient import TestClient

DEMO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(DEMO)
sys.path.insert(0, DEMO)
import app as demo  # noqa: E402

client = TestClient(demo.app)


def test_pipeline_copies_are_in_sync():
    for m in ["robust_calibration.py", "occupancy.py", "replay.py", "logic.py", "chessrules.py", "piece_inference.py"]:
        assert filecmp.cmp(os.path.join(ROOT, "src", m), os.path.join(DEMO, "pipeline", m), shallow=False), \
            f"demo/pipeline/{m} differs from src/{m}: run python demo/sync_pipeline.py"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["occupancy_detector"] in ("logistic regression", "rule")


def test_page_and_examples():
    assert client.get("/").status_code == 200
    d = client.get("/api/examples").json()
    assert 3 <= len(d["examples"]) <= 6
    for ex in d["examples"]:
        assert client.get(ex["url"]).status_code == 200


def test_each_example_is_analysed():
    gt = json.load(open(os.path.join(ROOT, "data", "ground_truth", "sample_frames_gt.json")))
    for ex in client.get("/api/examples").json()["examples"]:
        r = client.post("/api/analyze", data={"example": ex["id"]})
        assert r.status_code == 200, ex["id"]
        body = r.json()
        assert body["calibrated"], ex["id"]
        assert body["input_image"].startswith("data:image/jpeg;base64,")
        assert len(body["occupancy"]) == 8 and all(len(row) == 8 for row in body["occupancy"])
        # corners within 3 % of the annotated ones (examples are JPEG copies of the annotated PNGs)
        g = np.array(gt["frames"][ex["id"]]["corners"])
        pred = np.array([body["corners"][k] for k in ["a1", "a8", "h8", "h1"]])
        side = np.mean([np.linalg.norm(g[i] - g[(i + 1) % 4]) for i in range(4)])
        assert (np.linalg.norm(pred - g, axis=1) < 0.03 * side).all(), ex["id"]
        assert body["timing_ms"]["total"] < 3000


def test_upload_png_and_errors():
    img = cv2.imread(os.path.join(ROOT, "data", "sample_frames", "frame01.png"))
    ok, buf = cv2.imencode(".png", img)
    r = client.post("/api/analyze", files={"file": ("board.png", io.BytesIO(buf.tobytes()), "image/png")})
    assert r.status_code == 200 and r.json()["calibrated"]

    blank = np.full((400, 600, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", blank)
    r = client.post("/api/analyze", files={"file": ("blank.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")})
    assert r.status_code == 200 and r.json()["calibrated"] is False and "message" in r.json()

    assert client.post("/api/analyze", files={"file": ("x.jpg", io.BytesIO(b"not an image"), "image/jpeg")}).status_code == 415
    assert client.post("/api/analyze", data={}).status_code == 400
    assert client.post("/api/analyze", data={"example": "nope"}).status_code == 404
    tiny = cv2.imencode(".png", np.zeros((50, 50, 3), np.uint8))[1].tobytes()
    assert client.post("/api/analyze", files={"file": ("t.png", io.BytesIO(tiny), "image/png")}).status_code == 400


def test_upload_too_large(monkeypatch):
    monkeypatch.setattr(demo, "MAX_UPLOAD_BYTES", 100)
    r = client.post("/api/analyze", files={"file": ("big.jpg", io.BytesIO(b"x" * 1000), "image/jpeg")})
    assert r.status_code == 413


def test_replay_examples_never_wrong():
    for g in client.get("/api/examples").json()["games"]:
        r = client.post("/api/replay", json={"game": g["id"]})
        assert r.status_code == 200, g["id"]
        d = r.json()
        steps = d["steps"]
        assert len(steps) == d["total_plies"] + 1
        assert all(s["certain_wrong"] == 0 for s in steps), g["id"]         # a certainty is never wrong
        for s in steps:
            for m in s["markers"]:
                assert m["truth"] in m["cand"]                               # the true type is never excluded
        assert steps[-1]["certain"] > 0


def test_replay_mid_game_start_and_inputs():
    d = client.post("/api/replay", json={"game": "opera_1858", "start_ply": 16}).json()
    assert d["start_ply"] == 16 and len(d["steps"]) == d["total_plies"] - 16 + 1
    assert d["steps"][0]["certain"] < d["steps"][0]["present"]
    # PGN with castling, FEN + coordinate moves with a promotion
    r = client.post("/api/replay", json={"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6"})
    assert r.status_code == 200
    last = r.json()["steps"][-1]["markers"]
    assert {m["sq"]: m["certain"] for m in last}["g1"] == "K" and {m["sq"]: m["certain"] for m in last}["f1"] == "R"
    r = client.post("/api/replay", json={"fen": "8/P6k/8/8/8/8/8/K7 w - - 0 1", "moves": "a7a8q h7g6"})
    assert r.status_code == 200
    assert all(m["truth"] in m["cand"] for m in r.json()["steps"][-1]["markers"])


def test_replay_errors():
    assert client.post("/api/replay", json={"game": "nope"}).status_code == 404
    assert client.post("/api/replay", json={"pgn": "1. e4 e5 2. Ke3"}).status_code == 400
    assert client.post("/api/replay", json={"pgn": "bonjour"}).status_code == 400
    assert client.post("/api/replay", json={"fen": "8/8/8/8/8/8/8/K7 w - - 0 1", "moves": "a1a2"}).status_code == 400
    assert client.post("/api/replay", json={"game": "opera_1858", "start_ply": 999}).status_code == 400
    assert client.post("/api/replay", json={"game": "opera_1858", "start_ply": "x"}).status_code == 400
    assert client.post("/api/replay", json={"pgn": "e4 " * 50001}).status_code == 413
    assert "error" in client.post("/api/replay", json={"game": "nope"}).json()


def test_metrics():
    m = client.get("/api/metrics").json()
    assert "test" in m["evaluation"]["results"]
    pi = m["piece_identification"]
    assert pi["lichess_test_all_starts"]["new"]["certain_wrong_total"] == 0
