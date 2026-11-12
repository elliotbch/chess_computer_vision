"""
Données et simulation communes à l'entraînement de l'a priori et à l'évaluation de
l'identification des pièces inconnues.

- Parties lichess (base publique, licence CC0) : `data/lichess/lichess_db_standard_rated_2013-01.pgn`.
  Les parties valides (lisibles, sans position de départ FEN) sont prises dans l'ordre du fichier ;
  la partie n° i va au TEST si i % 3 == 0, sinon à l'ENTRAÎNEMENT. Les deux ensembles sont donc
  disjoints par construction (vérifié aussi par l'identifiant lichess).
- Un « extrait » : une partie observée à partir du coup N (demi-coup 2(N-1), trait aux Blancs),
  pendant au plus H demi-coups. Le système ne voit que l'occupation des cases (camp seulement).
"""

import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chessrules as cr  # noqa: E402
import piece_inference as pi  # noqa: E402

PGN = os.path.join(ROOT, "data", "lichess", "lichess_db_standard_rated_2013-01.pgn")
ANNOTATED = ["archive/step1_annotation/step1_Lea/gameLR_fix.json", "archive/step1_annotation/step1_Lea/gameLR_mov.json"]
STARTS = [1, 9, 15, 25, 40]          # coups (numérotation usuelle) où commence l'observation
HORIZON = 60                         # demi-coups observés au plus
T = "PNBRQK"


def iter_pgn(path=PGN):
    """Textes des parties, dans l'ordre du fichier (lecture en flux)."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
    if buf:
        yield "".join(buf)


def game_id(text):
    m = re.search(r'\[Site "([^"]+)"\]', text)
    return m.group(1) if m else None


def split_games(n_test, n_train):
    """(test, train) : listes de (id, texte PGN). Parties valides dans l'ordre du fichier, i % 3."""
    test, train = [], []
    i = 0
    for text in iter_pgn():
        if '[FEN "' in text or "1." not in text:
            continue
        toks, _ = cr.parse_pgn_moves(text)
        if len(toks) < 2:
            continue
        (test if i % 3 == 0 else train).append((game_id(text), text))
        i += 1
        if len(test) >= n_test and len(train) >= n_train:
            break
    test, train = test[:n_test], train[:n_train]
    assert not ({g for g, _ in test} & {g for g, _ in train}), "parties communes entre test et entraînement"
    return test, train


def positions_from_pgn(text):
    toks, fen = cr.parse_pgn_moves(text)
    return cr.play(toks, fen)


def positions_from_annotated(path):
    """Positions (chessrules) d'une partie annotée du dépôt : coup légal retrouvé entre deux états."""
    with open(os.path.join(ROOT, path) if not os.path.isabs(path) else path) as f:
        game = json.load(f)
    return cr.play_states([s["gs"] for s in game["game_states"]])


def occupancy(pos):
    return [0 if p is None else (1 if p.isupper() else -1) for p in pos.board]


class MarkerTracker:
    """Suit l'identité réelle des marqueurs (vérité terrain) le long des coups joués."""

    def __init__(self, pos):
        self.at = {}
        self.info = []                     # (couleur, type au début de l'observation)
        for sq, p in enumerate(pos.board):
            if p is not None:
                self.at[sq] = len(self.info)
                self.info.append((1 if p.isupper() else -1, p.upper()))

    def push(self, pos, move):
        fr, to, _ = move
        p = pos.board[fr]
        if pos.board[to] is not None:
            del self.at[to]
        if p.upper() == "P" and to == pos.ep and pos.board[to] is None:
            del self.at[to - 8 if p.isupper() else to + 8]
        if p.upper() == "K" and abs(to - fr) == 2:
            rf, rt = (fr + 3, fr + 1) if to > fr else (fr - 4, fr - 1)
            self.at[rt] = self.at.pop(rf)
        self.at[to] = self.at.pop(fr)


# ---------------------------------------------------------------------------- simulation

def _empty_stats():
    H = HORIZON
    return {"present": np.zeros(H + 1, np.int64), "certain_correct": np.zeros(H + 1, np.int64),
            "certain_wrong": np.zeros(H + 1, np.int64), "mp_correct": np.zeros(H + 1, np.int64),
            "uncertain": np.zeros(H + 1, np.int64), "mp_correct_uncertain": np.zeros(H + 1, np.int64),
            "extracts_at": np.zeros(H + 1, np.int64), "markers": [], "times_ms": [], "resets": 0, "extracts": 0}


def run_new(positions, moves, start, prior=None, horizon=HORIZON, stats=None, collect=None):
    """Nouveau moteur sur un extrait. Remplit `stats` ; `collect` (liste) reçoit (indices, type vrai)
    des marqueurs non identifiés pour entraîner l'a priori."""
    stats = stats if stats is not None else _empty_stats()
    seq = positions[start:start + horizon + 1]
    mv = moves[start:start + horizon]
    tracker = MarkerTracker(seq[0])
    eng = pi.PieceInference(occupancy(seq[0]), prior)
    first = {}            # marqueur réel -> demi-coup d'identification certaine
    wrong = set()
    for k in range(len(seq)):
        if k > 0:
            t0 = time.perf_counter()
            eng.observe(occupancy(seq[k]))
            marks = eng.markers()
            stats["times_ms"].append(1000 * (time.perf_counter() - t0))
            tracker.push(seq[k - 1], mv[k - 1])
        else:
            marks = eng.markers()
        pos = seq[k]
        stats["extracts_at"][k] += 1
        for m in marks:
            true = pos.board[m["sq"]].upper()
            mid = tracker.at[m["sq"]]
            stats["present"][k] += 1
            if m["certain"] is not None:
                if m["certain"] == true:
                    stats["certain_correct"][k] += 1
                else:
                    stats["certain_wrong"][k] += 1
                    wrong.add(mid)
                first.setdefault(mid, k)
            else:
                stats["uncertain"][k] += 1
                if m["most_probable"] == true:
                    stats["mp_correct_uncertain"][k] += 1
            if m["most_probable"] == true:
                stats["mp_correct"][k] += 1
        if collect is not None and k > 0:
            alive = [v for v in eng.vars if v.alive]
            cnt = {1: 0, -1: 0}
            for v in alive:
                cnt[v.color] += 1
            for v in alive:
                if v.dom & (v.dom - 1):
                    collect.append((pi.Prior.features(v, cnt[v.color], cnt[-v.color]), T.index(pos.board[v.sq].upper())))
    n_obs = len(seq) - 1
    for mid, (color, typ) in enumerate(tracker.info):
        stats["markers"].append((typ, first.get(mid, -1), n_obs, mid in wrong))
    stats["resets"] += eng.resets
    stats["extracts"] += 1
    return stats


def run_old(positions, moves, start, horizon=HORIZON, stats=None):
    """Ancienne logique du dépôt (`logic.py` + correctifs de `replay.py`) sur le même extrait.
    Elle n'a pas de probabilités : « le plus probable » = son type déduit (ou rien)."""
    import replay
    stats = stats if stats is not None else _empty_stats()
    seq = positions[start:start + horizon + 1]
    mv = moves[start:start + horizon]
    arrays = [np.array(cr.to_array(p)) for p in seq]
    t0 = time.perf_counter()
    steps = replay.replay([replay.observation_from_state(a) for a in arrays], fixes=True)
    dt = 1000 * (time.perf_counter() - t0)
    if len(seq) > 1:
        stats["times_ms"].extend([dt / (len(seq) - 1)] * (len(seq) - 1))
    tracker = MarkerTracker(seq[0])
    first, wrong = {}, set()
    for k, (step, pos) in enumerate(zip(steps, seq)):
        if k > 0:
            tracker.push(seq[k - 1], mv[k - 1])
        stats["extracts_at"][k] += 1
        b = step["board"]
        for sq, p in enumerate(pos.board):
            if p is None:
                continue
            true = p.upper()
            v = abs(int(b[7 - sq // 8][sq % 8]))
            claim = replay.PIECE_LETTERS.get(v) if 1 <= v <= 6 else None
            mid = tracker.at[sq]
            stats["present"][k] += 1
            if claim is not None:
                if claim == true:
                    stats["certain_correct"][k] += 1
                    stats["mp_correct"][k] += 1
                else:
                    stats["certain_wrong"][k] += 1
                    wrong.add(mid)
                first.setdefault(mid, k)
            else:
                stats["uncertain"][k] += 1
    n_obs = len(seq) - 1
    for mid, (color, typ) in enumerate(tracker.info):
        stats["markers"].append((typ, first.get(mid, -1), n_obs, mid in wrong))
    stats["extracts"] += 1
    return stats


def merge(a, b):
    for k, v in b.items():
        if isinstance(v, np.ndarray) or isinstance(v, list):
            a[k] = a[k] + v if isinstance(v, list) else a[k] + v
        else:
            a[k] += v
    return a


def summarize(stats, curve_points=(0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60)):
    pres = stats["present"]
    out = {"extracts": int(stats["extracts"]), "resets_after_contradiction": int(stats["resets"])}
    curve = []
    for k in range(len(pres)):
        if pres[k] == 0:
            continue
        curve.append({"plies_observed": k, "extracts": int(stats["extracts_at"][k]),
                      "certain_share": round(float(stats["certain_correct"][k] + stats["certain_wrong"][k]) / pres[k], 4),
                      "certain_wrong": int(stats["certain_wrong"][k]),
                      "most_probable_accuracy": round(float(stats["mp_correct"][k]) / pres[k], 4)})
    out["curve"] = curve
    out["curve_summary"] = {str(c["plies_observed"]): c["certain_share"] for c in curve if c["plies_observed"] in curve_points}
    out["certain_wrong_total"] = int(stats["certain_wrong"].sum())
    out["certain_total"] = int((stats["certain_correct"] + stats["certain_wrong"]).sum())
    unc = int(stats["uncertain"].sum())
    out["most_probable_accuracy_on_uncertain"] = round(float(stats["mp_correct_uncertain"].sum()) / unc, 4) if unc else None
    out["most_probable_accuracy_all"] = round(float(stats["mp_correct"].sum()) / pres.sum(), 4) if pres.sum() else None
    by_type = {}
    for t in T:
        rows = [(f, n) for typ, f, n, _ in stats["markers"] if typ == t]
        if not rows:
            continue
        # médiane avec censure : non identifié = +inf (fin d'extrait, prise ou horizon)
        vals = sorted(f if f >= 0 else float("inf") for f, _ in rows)
        med = vals[(len(vals) - 1) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        idf = [f for f, _ in rows if f >= 0]
        by_type[t] = {"markers": len(rows), "identified_share": round(len(idf) / len(rows), 4),
                      "median_plies_to_identification": None if med == float("inf") else med,
                      "median_plies_among_identified": float(np.median(idf)) if idf else None,
                      "identified_at_start_share": round(sum(1 for f in idf if f == 0) / len(rows), 4)}
    out["by_type"] = by_type
    tm = np.array(stats["times_ms"]) if stats["times_ms"] else np.zeros(1)
    out["ms_per_ply"] = {"mean": round(float(tm.mean()), 3), "median": round(float(np.median(tm)), 3),
                         "p95": round(float(np.percentile(tm, 95)), 3), "max": round(float(tm.max()), 2)}
    return out
