"""
Entraîne l'a priori P(type | indices) qui départage les types encore possibles d'un marqueur
(« le plus probable »). Il n'intervient jamais dans les identifications certaines.

Données : parties lichess d'ENTRAÎNEMENT uniquement (voir `inference_data.split_games`, disjointes
des parties de test). Chaque partie est observée à partir des coups 1, 9, 15, 25, 40 et d'un coup
tiré au hasard ; à chaque demi-coup, chaque marqueur non identifié donne un exemple (indices du
moteur, type réel). Validation : 20 % des parties d'entraînement (par partie) pour choisir C.

    python training/train_prior.py --train-games 4000 --seed 0 --out models/piece_prior.json
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_data as D  # noqa: E402
import piece_inference as pi  # noqa: E402


def collect_game(args):
    gid, text, seed, keep = args
    rng = random.Random(f"{seed}-{gid}")
    try:
        positions, moves = D.positions_from_pgn(text)
    except D.cr.ChessError:
        return gid, []
    starts = [2 * (n - 1) for n in D.STARTS] + [rng.randrange(0, len(positions))]
    rows = []
    for s in starts:
        if s >= len(positions) - 1:
            continue
        col = []
        D.run_new(positions, moves, s, prior=None, collect=col)
        rows += [r for r in col if rng.random() < keep]
    return gid, rows


def to_matrix(rows):
    cols = np.array([idx for idx, _ in rows], np.int64)
    y = np.array([t for _, t in rows], np.int64)
    n, k = cols.shape
    X = sparse.csr_matrix((np.ones(n * k, np.float32), cols.ravel(), np.arange(0, n * k + 1, k)),
                          shape=(n, pi.Prior.N_FEATURES))
    return X, y


def masked_accuracy(clf, X, y, doms):
    s = clf.decision_function(X)
    mask = np.array([[d >> t & 1 for t in range(6)] for d in doms], bool)
    s = np.where(mask, s, -np.inf)
    return float((s.argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-games", type=int, default=4000)
    ap.add_argument("--test-games", type=int, default=2000, help="seulement pour reconstituer le même découpage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.03, help="part des exemples gardés (sous-échantillonnage)")
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(D.ROOT, "models", "piece_prior.json"))
    ap.add_argument("--metrics", default=os.path.join(D.ROOT, "training", "metrics.json"))
    a = ap.parse_args()
    t0 = time.time()
    _, train = D.split_games(a.test_games, a.train_games)
    with Pool(a.workers) as pool:
        res = pool.map(collect_game, [(g, t, a.seed, a.keep) for g, t in train], chunksize=20)
    rng = random.Random(a.seed)
    ids = [g for g, _ in train]
    rng.shuffle(ids)
    val_ids = set(ids[: len(ids) // 5])
    tr = [r for g, rows in res if g not in val_ids for r in rows]
    va = [r for g, rows in res if g in val_ids for r in rows]
    Xtr, ytr = to_matrix(tr)
    Xva, yva = to_matrix(va)
    dva = [idx[0] for idx, _ in va]
    t_data = time.time() - t0
    print(f"exemples : entraînement {len(tr)}, validation {len(va)} ({t_data:.0f} s)", flush=True)
    scores = {}
    for C in a.Cs:
        clf = LogisticRegression(C=C, max_iter=300, random_state=a.seed)
        clf.fit(Xtr, ytr)
        scores[C] = masked_accuracy(clf, Xva, yva, dva)
        print(f"C={C}: exactitude validation (types restants) {scores[C]:.4f}", flush=True)
    best = max(scores, key=scores.get)
    Xall, yall = to_matrix(tr + va)
    clf = LogisticRegression(C=best, max_iter=300, random_state=a.seed).fit(Xall, yall)
    assert list(clf.classes_) == list(range(6))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    meta = {"features": "domaine, case vue de son camp, déplacements observés, marqueurs par camp, domaine x a bougé",
            "C": best, "train_games": len(train), "examples": len(tr) + len(va), "seed": a.seed}
    with open(a.out, "w") as f:
        json.dump({"coef": np.round(clf.coef_, 5).tolist(), "intercept": np.round(clf.intercept_, 5).tolist(),
                   "meta": meta}, f)
    dt = time.time() - t0
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "machine": f"{platform.machine()} {platform.system()} (Apple M2 Pro)",
              "seed": a.seed, "duration_s": round(dt, 1), "train_games": len(train), "validation_games": len(val_ids),
              "examples_train": len(tr), "examples_validation": len(va),
              "validation_accuracy_by_C": {str(k): round(v, 4) for k, v in scores.items()}, "chosen_C": best,
              "model": "models/" + os.path.basename(a.out), "command": "python training/train_prior.py " + " ".join(sys.argv[1:])}
    m = json.load(open(a.metrics)) if os.path.exists(a.metrics) else {}
    m["training_piece_prior"] = report
    with open(a.metrics, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
