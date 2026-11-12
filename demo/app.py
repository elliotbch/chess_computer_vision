"""
Chess board vision demo (ULiège ELEN-0016 project, 2024; pipeline v2 of November 2026).

- GET  /                       page
- GET  /health                 status, occupancy detector and piece prior in use
- GET  /api/examples           example photos and example games
- POST /api/analyze            photo (upload) or example id -> corners, top view, occupied squares
- POST /api/replay             game (example id, PGN, or FEN + moves) observed from a given ply:
                               move-by-move identification of the unknown pieces (identical markers)
- GET  /api/metrics            evaluation metrics (copy of training/metrics.json)

The occupancy classifier (logistic regression, 4 coefficients) is read from MODEL_PATH, or from
the Hugging Face Hub (HF_MODEL_REPO), or from ../models/ when run from the repository. Without it
the demo uses the rule-based v2 detector (adaptive threshold, no training), and /health says so.
The piece prior (piece_prior.json, used only for "most probable" types, never for certainties)
is read from PRIOR_PATH, HF_MODEL_REPO or ../models/; without it the replay shows certainties and
remaining candidates only.
"""

import base64
import json
import os
import re
import sys
import time
import urllib.request

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pipeline"))

import chessrules  # noqa: E402
import occupancy  # noqa: E402
import piece_inference  # noqa: E402
import replay  # noqa: E402
import robust_calibration  # noqa: E402

STATIC = os.path.join(HERE, "static")
EXAMPLES_DIR = os.path.join(STATIC, "examples")
GAMES_DIR = os.path.join(EXAMPLES_DIR, "games")
METRICS_PATH = os.path.join(HERE, "eval_metrics.json")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_SIDE = 2400
MODEL_FILE = "occupancy_logreg.json"
PRIOR_FILE = "piece_prior.json"
MAX_TEXT = 100_000
MAX_PLIES = 400

EXAMPLES = [
    {"id": "frame06", "file": "frame06.jpg", "label": "Position de départ", "note": "image de test"},
    {"id": "frame08", "file": "frame08.jpg", "label": "Petite image, milieu de partie", "note": "image de test"},
    {"id": "frame09", "file": "frame09.jpg", "label": "Coin caché par une main", "note": "image de développement"},
    {"id": "frame05", "file": "frame05.jpg", "label": "Plateau tourné, fin de partie", "note": "image de développement"},
    {"id": "frame07", "file": "frame07.jpg", "label": "Main au-dessus du plateau", "note": "image de développement"},
    {"id": "frame20", "file": "frame20.jpg", "label": "Pièces en relief (capsules et vis)", "note": "image de test, autre type de marqueur"},
]
EXAMPLES_BY_ID = {e["id"]: e for e in EXAMPLES}
GAMES = {
    "gameLR_fix": {"file": "gameLR_fix.json", "label": "Partie annotée 1 (caméra fixe)", "note": "partie filmée du projet, états annotés à la main"},
    "gameLR_mov": {"file": "gameLR_mov.json", "label": "Partie annotée 2 (caméra mobile)", "note": "partie filmée du projet, états annotés à la main"},
    "opera_1858": {"file": "opera_1858.pgn", "label": "Partie de l'Opéra (Morphy, 1858)", "note": "domaine public"},
    "immortelle_1851": {"file": "immortelle_1851.pgn", "label": "L'Immortelle (Anderssen – Kieseritzky, 1851)", "note": "domaine public"},
    "toujours_vert_1852": {"file": "toujours_vert_1852.pgn", "label": "La Toujours Verte (Anderssen – Dufresne, 1852)", "note": "domaine public"},
}


def load_model():
    """Returns (model or None, description)."""
    path = os.environ.get("MODEL_PATH")
    if path:
        if os.path.exists(path):
            return occupancy.LogReg.load(path), f"régression logistique chargée depuis MODEL_PATH ({os.path.basename(path)})"
        return None, f"MODEL_PATH={path} introuvable"
    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        try:
            url = f"https://huggingface.co/{repo}/resolve/main/{MODEL_FILE}"
            with urllib.request.urlopen(url, timeout=10) as r:
                return occupancy.LogReg(json.loads(r.read().decode())), f"régression logistique chargée depuis {repo}"
        except Exception as exc:  # pragma: no cover - network dependent
            return None, f"impossible de télécharger {MODEL_FILE} depuis {repo} : {exc.__class__.__name__}"
    local = os.path.join(os.path.dirname(HERE), "models", MODEL_FILE)
    if os.path.exists(local):
        return occupancy.LogReg.load(local), "régression logistique chargée depuis models/"
    return None, "aucun poids de classifieur trouvé : détecteur à seuil adaptatif utilisé"


def load_prior():
    """Returns (Prior or None, description)."""
    path = os.environ.get("PRIOR_PATH")
    if path:
        if os.path.exists(path):
            return piece_inference.Prior.load(path), f"a priori chargé depuis PRIOR_PATH ({os.path.basename(path)})"
        return None, f"PRIOR_PATH={path} introuvable"
    repo = os.environ.get("HF_MODEL_REPO")
    if repo:
        try:
            url = f"https://huggingface.co/{repo}/resolve/main/{PRIOR_FILE}"
            with urllib.request.urlopen(url, timeout=10) as r:
                return piece_inference.Prior(json.loads(r.read().decode())), f"a priori chargé depuis {repo}"
        except Exception as exc:  # pragma: no cover - network dependent
            return None, f"impossible de télécharger {PRIOR_FILE} depuis {repo} : {exc.__class__.__name__}"
    local = os.path.join(os.path.dirname(HERE), "models", PRIOR_FILE)
    if os.path.exists(local):
        return piece_inference.Prior.load(local), "a priori chargé depuis models/"
    return None, "aucun a priori trouvé : seules les certitudes et les candidats restants sont affichés"


PRIOR, PRIOR_SOURCE = load_prior()

MODEL, MODEL_SOURCE = (None, "détecteur à règles demandé (OCCUPANCY_DETECTOR=rule)") if os.environ.get("OCCUPANCY_DETECTOR") == "rule" \
    else load_model()

app = FastAPI(title="Vision d'échiquier", docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ helpers

def method_fr(method):
    """French wording of the calibration method reported by robust_calibration (pipeline unchanged)."""
    if not method:
        return method
    m = re.fullmatch(r"full 7x7 \((\w+)\)", method)
    if m:
        return f"grille 7×7 complète ({ {'inpaint': 'marqueurs effacés', 'gray': 'niveaux de gris'}.get(m[1], m[1]) })"
    m = re.fullmatch(r"partial (\d+)x(\d+) at offset \((\d+), (\d+)\)", method)
    if m:
        return f"grille partielle {m[1]}×{m[2]}, décalage ({m[3]}, {m[4]})"
    if method == "not found":
        return "introuvable"
    if method.startswith("error: "):
        return "erreur : " + method[len("error: "):]
    return method


def jpeg_data_url(img, quality=85, max_w=1100):
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def draw_input(img, H, corners):
    vis = img.copy()
    t = max(2, int(round(img.shape[1] / 600)))
    for i in range(9):
        for a, b in (((i, 0), (i, 8)), ((0, i), (8, i))):
            p = cv2.perspectiveTransform(np.float32([a, b]).reshape(-1, 1, 2), H).reshape(-1, 2)
            cv2.line(vis, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (60, 220, 60), t, cv2.LINE_AA)
    names = ["a1", "a8", "h8", "h1"]
    colors = [(230, 120, 30), (200, 60, 230), (40, 40, 40), (250, 250, 250)]
    for (x, y), name, col in zip(corners, names, colors):
        c = (int(x), int(y))
        cv2.circle(vis, c, 4 * t, col, -1, cv2.LINE_AA)
        cv2.circle(vis, c, 4 * t, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, name, (c[0] + 5 * t, c[1] - 5 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(vis, name, (c[0] + 5 * t, c[1] - 5 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (255, 255, 255), t, cv2.LINE_AA)
    return vis


def draw_top(top, grid, square=100):
    """Top view re-oriented like a chess diagram (rank 8 at the top, file a on the left)."""
    # pipeline top view: rows = files a..h, columns = ranks 1..8 -> transpose + flip
    diag = cv2.flip(cv2.transpose(top), 0)
    vis = diag.copy()
    for k in range(9):
        cv2.line(vis, (0, k * square), (8 * square, k * square), (60, 220, 60), 2)
        cv2.line(vis, (k * square, 0), (k * square, 8 * square), (60, 220, 60), 2)
    for f in range(8):
        for r in range(8):
            v = grid[f][r]
            if v == ".":
                continue
            x, y = f * square, (7 - r) * square
            col = (255, 255, 255) if v == "W" else (20, 20, 20)
            ring = (20, 20, 20) if v == "W" else (255, 255, 255)
            cv2.rectangle(vis, (x + 5, y + 5), (x + square - 5, y + square - 5), (30, 140, 255), 4)
            cv2.circle(vis, (x + square - 18, y + 18), 11, col, -1, cv2.LINE_AA)
            cv2.circle(vis, (x + square - 18, y + 18), 11, ring, 2, cv2.LINE_AA)
    return vis


def analyze_image(img):
    t0 = time.time()
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    res = robust_calibration.calibrate(img)
    t_cal = time.time() - t0
    out = {"calibrated": bool(res["ok"]), "method": method_fr(res["info"].get("method")),
           "stickers": res["info"].get("stickers"), "image_size": [int(img.shape[1]), int(img.shape[0])]}
    if not res["ok"]:
        out["message"] = ("Plateau introuvable : les 7×7 coins intérieurs (ou une grande partie d'entre eux) et au moins une "
                          "pastille d'orientation (rose ou bleue) doivent être visibles.")
        out["timing_ms"] = {"calibration": round(1000 * t_cal), "total": round(1000 * (time.time() - t0))}
        out["input_image"] = jpeg_data_url(img)
        return out
    corners = res["corners"]
    top = robust_calibration.top_view(img, corners, square=100)
    t1 = time.time()
    grid, score = occupancy.detect(top, 100, MODEL)
    t_occ = time.time() - t1
    occupied = sum(c != "." for row in grid for c in row)
    out.update({
        "corners": {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(["a1", "a8", "h8", "h1"], corners)},
        "occupancy": grid,
        "board": [[grid[f][r] for f in range(8)] for r in range(7, -1, -1)],   # rank 8 first, file a first
        "occupied": int(occupied),
        "white": int(sum(c == "W" for row in grid for c in row)),
        "black": int(sum(c == "B" for row in grid for c in row)),
        "detector": "régression logistique" if MODEL is not None else "seuil adaptatif (sans entraînement)",
        "input_image": jpeg_data_url(draw_input(img, res["H"], corners)),
        "top_view": jpeg_data_url(draw_top(top, grid), max_w=800),
        "timing_ms": {"calibration": round(1000 * t_cal), "occupancy": round(1000 * t_occ),
                      "total": round(1000 * (time.time() - t0))},
    })
    return out


def game_positions(req):
    """(positions, moves, label) from {"game": id} or {"pgn": text} or {"fen": ..., "moves": ...}."""
    if not isinstance(req, dict):
        raise HTTPException(400, "Corps JSON attendu.")
    try:
        if req.get("game"):
            g = GAMES.get(req["game"])
            if g is None:
                raise HTTPException(404, f"Partie inconnue '{req['game']}'.")
            path = os.path.join(GAMES_DIR, g["file"])
            if g["file"].endswith(".json"):
                with open(path) as f:
                    states = [st["gs"] for st in json.load(f)["game_states"]]
                positions, moves = chessrules.play_states(states)
            else:
                with open(path) as f:
                    toks, fen = chessrules.parse_pgn_moves(f.read())
                positions, moves = chessrules.play(toks, fen)
            return positions, moves, g["label"]
        text = req.get("pgn") or ""
        fen = (req.get("fen") or "").strip() or None
        mv_text = req.get("moves") or ""
        if len(text) > MAX_TEXT or len(mv_text) > MAX_TEXT or (fen and len(fen) > 200):
            raise HTTPException(413, "Texte trop long : 100 000 caractères au plus.")
        if text.strip():
            toks, pgn_fen = chessrules.parse_pgn_moves(text)
            fen = pgn_fen or fen
            label = "PGN importé"
        else:
            toks, _ = chessrules.parse_pgn_moves(mv_text)
            label = "Position FEN + coups" if fen else "Coups importés"
        if not toks:
            raise HTTPException(400, "Aucun coup trouvé : collez un PGN, ou une FEN suivie de coups (Nf3, e2e4…).")
        if len(toks) > MAX_PLIES:
            raise HTTPException(413, f"Partie trop longue : {MAX_PLIES} demi-coups au plus.")
        positions, moves = chessrules.play(toks, fen)
        return positions, moves, label
    except chessrules.ChessError as exc:
        raise HTTPException(400, f"Partie illisible : {exc}.")


def occupancy_of(pos):
    return [0 if p is None else (1 if p.isupper() else -1) for p in pos.board]


def describe_event(ev):
    n = piece_inference.sqname
    if ev["kind"] == "castle":
        return f"roque : {n(ev['king'][0])} → {n(ev['king'][1])} et {n(ev['rook'][0])} → {n(ev['rook'][1])}"
    if ev["kind"] == "en_passant":
        return f"{n(ev['from'])} × {n(ev['to'])} (prise en passant, {n(ev['taken'])} retiré)"
    if ev["kind"] == "move":
        txt = f"{n(ev['from'])} {'×' if ev['capture'] else '→'} {n(ev['to'])}"
        if ev.get("promotion_possible"):
            txt += " (promotion possible)"
        return txt
    return "changement illisible : tout est remis à zéro"


def build_replay(positions, moves, start_ply, label):
    if not 0 <= start_ply < len(positions):
        raise HTTPException(400, f"Début d'observation hors de la partie (0 à {len(positions) - 1}).")
    t0 = time.time()
    seq = positions[start_ply:]
    eng = piece_inference.PieceInference(occupancy_of(seq[0]), PRIOR)
    old = replay.replay([replay.observation_from_state(np.array(chessrules.to_array(p))) for p in seq], fixes=True)
    steps = []
    for k, pos in enumerate(seq):
        t1 = time.perf_counter()
        ev = eng.observe(occupancy_of(pos)) if k else {"kind": "start"}
        marks = eng.markers()
        ms = 1000 * (time.perf_counter() - t1)
        out_m = []
        for m in marks:
            truth = pos.board[m["sq"]].upper()
            d = {"sq": m["square"], "c": "w" if m["color"] > 0 else "b", "id": m["label"], "cand": m["candidates"],
                 "certain": m["certain"], "since": m["identified_ply"], "truth": truth}
            if PRIOR is not None:
                d["mp"] = m["most_probable"]
                d["p"] = m["probs"].get(m["most_probable"])
            out_m.append(d)
        ob = old[k]["board"]
        old_ident = old_wrong = 0
        for sq, p in enumerate(pos.board):
            if p is None:
                continue
            v = abs(int(ob[7 - sq // 8][sq % 8]))
            if 1 <= v <= 6:
                old_ident += 1
                old_wrong += replay.PIECE_LETTERS[v] != p.upper()
        steps.append({
            "ply": start_ply + k,
            "move": describe_event(ev) if k else "début de l'observation",
            "new": [{"sq": m["sq"], "type": m["certain"]} for m in out_m if m["since"] == k and m["certain"]] if k else
                   [{"sq": m["sq"], "type": m["certain"]} for m in out_m if m["certain"]],
            "markers": out_m,
            "present": len(out_m),
            "certain": sum(1 for m in out_m if m["certain"]),
            "certain_wrong": sum(1 for m in out_m if m["certain"] and m["certain"] != m["truth"]),
            "mp_correct": sum(1 for m in out_m if m.get("mp", m["certain"]) == m["truth"]) if PRIOR is not None else None,
            "old_identified": old_ident, "old_wrong": old_wrong,
            "ms": round(ms, 2),
        })
    p0 = positions[0]
    return {"label": label, "start_ply": start_ply, "total_plies": len(positions) - 1,
            "first_turn": p0.turn, "prior": PRIOR is not None, "resets": eng.resets,
            "steps": steps, "timing_ms": round(1000 * (time.time() - t0))}


# ------------------------------------------------------------------ routes

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "occupancy_detector": "logistic regression" if MODEL is not None else "rule",
            "model_source": MODEL_SOURCE, "piece_prior": PRIOR is not None, "prior_source": PRIOR_SOURCE,
            "examples": len(EXAMPLES), "games": list(GAMES)}


@app.get("/api/examples")
def examples():
    return {"examples": [dict(e, url=f"static/examples/{e['file']}") for e in EXAMPLES],
            "games": [{"id": k, "label": v["label"], "note": v["note"]} for k, v in GAMES.items()]}


@app.get("/static/examples/{name}")
def example_file(name: str):
    path = os.path.join(EXAMPLES_DIR, os.path.basename(name))
    if not name.endswith(".jpg") or not os.path.exists(path):
        raise HTTPException(404, "Fichier introuvable.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), example: str | None = Form(None)):
    if example:
        ex = EXAMPLES_BY_ID.get(example)
        if ex is None:
            raise HTTPException(404, f"Exemple inconnu '{example}'.")
        img = cv2.imread(os.path.join(EXAMPLES_DIR, ex["file"]))
    elif file is not None:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image trop volumineuse : la démo accepte au plus 12 Mo.")
        if not raw:
            raise HTTPException(400, "Fichier vide.")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(415, "Impossible de décoder l'image (utilisez JPEG ou PNG).")
        if min(img.shape[:2]) < 200:
            raise HTTPException(400, "Image trop petite : au moins 200 px de chaque côté.")
    else:
        raise HTTPException(400, "Envoyez un fichier image ou un identifiant d'exemple.")
    return analyze_image(img)


@app.post("/api/replay")
def replay_game(req: dict = Body(...)):
    positions, moves, label = game_positions(req)
    try:
        start = int(req.get("start_ply", 0))
    except (TypeError, ValueError):
        raise HTTPException(400, "start_ply doit être un entier.")
    return build_replay(positions, moves, start, label)


@app.get("/api/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(404, "Fichier de métriques introuvable.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def _warm_up():
    """The first OpenCV call is slow (~3 s, lazy initialisation): pay it at start-up, not on the first click."""
    img = cv2.imread(os.path.join(EXAMPLES_DIR, "frame08.jpg"))
    if img is not None:
        analyze_image(img)


_warm_up()
