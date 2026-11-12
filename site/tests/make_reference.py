"""Reference answers of the FastAPI demo (demo/app.py), for the parity check of the static site.

Start the demo, then (standard library only):

    cd demo && MODEL_PATH=../models/occupancy_logreg.json PRIOR_PATH=../models/piece_prior.json \
        uvicorn app:app --port 8142
    python site/tests/make_reference.py --url http://127.0.0.1:8142

Writes site/tests/reference.json, compared by site/tests/parity.html (in the browser, Pyodide) with
the answers of the site's Web Worker: every example game observed from move 1, 9 and 25
(start_ply = 2 * (N - 1), the slider's "coup N"), two imported games (PGN, FEN + moves), and every
example photo. Timings and JPEG images are left out (they are not results).
"""

import argparse
import json
import os
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
STARTS = [1, 9, 25]
IMPORTS = [
    {"id": "import_pgn", "label": "PGN importé (roque)", "req": {"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6"}},
    {"id": "import_fen", "label": "FEN + coups (promotion)", "req": {"fen": "8/P6k/8/8/8/8/8/K7 w - - 0 1", "moves": "a7a8q h7g6"}},
]
ANALYZE_KEYS = ["calibrated", "method", "stickers", "image_size", "corners", "occupancy", "board", "occupied",
                "white", "black", "detector", "message"]


def post(url, body, ctype):
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def strip_replay(d):
    d = dict(d)
    d.pop("timing_ms", None)
    d["steps"] = [{k: v for k, v in s.items() if k != "ms"} for s in d.get("steps", [])]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8142")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    with urllib.request.urlopen(base + "/health") as r:
        health = json.loads(r.read())
    with urllib.request.urlopen(base + "/api/examples") as r:
        ex = json.loads(r.read())
    out = {"server": base, "health": health, "replays": [], "photos": []}
    for g in ex["games"]:
        for n in STARTS:
            req = {"game": g["id"], "start_ply": 2 * (n - 1)}
            code, d = post(base + "/api/replay", json.dumps(req).encode(), "application/json")
            out["replays"].append({"id": f"{g['id']}@{n}", "label": f"{g['label']}, coup {n}", "req": req,
                                   "status": code, "body": strip_replay(d) if code == 200 else d})
    for imp in IMPORTS:
        code, d = post(base + "/api/replay", json.dumps(imp["req"]).encode(), "application/json")
        out["replays"].append({"id": imp["id"], "label": imp["label"], "req": imp["req"], "status": code,
                               "body": strip_replay(d) if code == 200 else d})
    for e in ex["examples"]:
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"example\"\r\n\r\n{e['id']}\r\n--{boundary}--\r\n").encode()
        code, d = post(base + "/api/analyze", body, f"multipart/form-data; boundary={boundary}")
        out["photos"].append({"id": e["id"], "label": e["label"], "file": e["file"], "status": code,
                              "body": {k: d[k] for k in ANALYZE_KEYS if k in d}, "timing_ms": d.get("timing_ms")})
    path = os.path.join(HERE, "reference.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    ok = sum(r["status"] == 200 for r in out["replays"])
    print(f"{path}: {len(out['replays'])} replays ({ok} played, others start beyond the game), {len(out['photos'])} photos")


if __name__ == "__main__":
    main()
