"""
Évaluation de l'identification des pièces inconnues : nouveau moteur (`src/piece_inference.py`,
avec ou sans a priori) contre l'ancienne logique du dépôt (`src/logic.py` + correctifs de
`src/replay.py`), sur EXACTEMENT les mêmes extraits.

Protocole (fixé avant la mesure, voir training/README.md) :
- (a) les 2 parties annotées du dépôt ; (b) les parties lichess de TEST (i % 3 == 0 dans l'ordre
  du fichier, disjointes de celles de l'a priori) ;
- observation à partir des coups 1, 9, 15, 25 et 40 (si la partie va jusque-là), pendant au plus
  60 demi-coups ; le système ne reçoit que l'occupation (camp de chaque case) ;
- métriques : part des marqueurs présents identifiés avec certitude après N demi-coups observés,
  médiane du nombre de demi-coups avant identification par type (marqueurs présents au début ;
  non identifié = infini), identifications certaines fausses, exactitude du « plus probable »,
  temps par demi-coup.

    python training/evaluate_inference.py --test-games 2000 --workers 6
"""

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference_data as D  # noqa: E402
import piece_inference as pi  # noqa: E402

PRIOR = None


def _init(prior_path):
    global PRIOR
    PRIOR = pi.Prior.load(prior_path) if prior_path and os.path.exists(prior_path) else None


def eval_game(args):
    gid, text = args
    try:
        positions, moves = D.positions_from_pgn(text)
    except D.cr.ChessError:
        return None
    out = {}
    for n in D.STARTS:
        s = 2 * (n - 1)
        if s >= len(positions) - 1:
            continue
        out[n] = {"new": D.run_new(positions, moves, s, prior=PRIOR), "old": D.run_old(positions, moves, s)}
    return out


def evaluate_annotated(prior):
    res = {}
    for path in D.ANNOTATED:
        positions, moves = D.positions_from_annotated(path)
        name = os.path.splitext(os.path.basename(path))[0]
        res[name] = {"plies": len(moves), "by_start": {}}
        for n in D.STARTS:
            s = 2 * (n - 1)
            if s >= len(positions) - 1:
                continue
            new = D.summarize(D.run_new(positions, moves, s, prior=prior))
            old = D.summarize(D.run_old(positions, moves, s))
            res[name]["by_start"][str(n)] = {"new": new, "old": old}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-games", type=int, default=2000)
    ap.add_argument("--train-games", type=int, default=4000, help="seulement pour reconstituer le même découpage")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prior", default=os.path.join(D.ROOT, "models", "piece_prior.json"))
    ap.add_argument("--metrics", default=os.path.join(D.ROOT, "training", "metrics.json"))
    a = ap.parse_args()
    t0 = time.time()
    _init(a.prior)
    annotated = evaluate_annotated(PRIOR)
    test, _ = D.split_games(a.test_games, a.train_games)
    agg = {n: {"new": D._empty_stats(), "old": D._empty_stats()} for n in D.STARTS}
    with Pool(a.workers, initializer=_init, initargs=(a.prior,)) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_game, test, chunksize=10)):
            if r is None:
                continue
            for n, d in r.items():
                for k in ("new", "old"):
                    D.merge(agg[n][k], d[k])
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(test)} parties ({time.time() - t0:.0f} s)", flush=True)
    lichess = {str(n): {"new": D.summarize(agg[n]["new"]), "old": D.summarize(agg[n]["old"])} for n in D.STARTS}
    allnew, allold = D._empty_stats(), D._empty_stats()
    for n in D.STARTS:
        D.merge(allnew, agg[n]["new"])
        D.merge(allold, agg[n]["old"])
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "machine": f"{platform.machine()} {platform.system()} (Apple M2 Pro)",
        "duration_s": round(time.time() - t0, 1), "seed": "aucune graine nécessaire (protocole déterministe)",
        "protocol": {
            "observation_starts_at_move": D.STARTS, "horizon_plies": D.HORIZON,
            "input": "occupation des cases seulement (camp), ni types, ni droits de roque, ni historique",
            "lichess_file": "data/lichess/lichess_db_standard_rated_2013-01.pgn (CC0)",
            "lichess_split": "parties valides dans l'ordre du fichier ; test si i % 3 == 0, entraînement de l'a priori sinon",
            "test_games": len(test),
            "certain_share": "marqueurs identifiés avec certitude / marqueurs présents, après N demi-coups observés (extraits encore en cours)",
            "median": "demi-coups observés avant identification certaine, marqueurs présents au début de l'observation, par type au début ; non identifié = infini (null si la médiane n'est pas atteinte)",
            "old_logic": "src/logic.py + correctifs de src/replay.py (roque, prise en passant, prises), sans l'astuce des cases de départ",
            "prior": "models/piece_prior.json" if PRIOR is not None else None,
        },
        "annotated_games": annotated,
        "lichess_test": lichess,
        "lichess_test_all_starts": {"new": D.summarize(allnew), "old": D.summarize(allold)},
    }
    m = json.load(open(a.metrics)) if os.path.exists(a.metrics) else {}
    m["piece_identification"] = report
    with open(a.metrics, "w") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    for n in D.STARTS:
        L = lichess[str(n)]
        print(f"coup {n:>2} : extraits {L['new']['extracts']:>5}  certitude après 0/10/20/40 demi-coups "
              f"nouveau {[L['new']['curve_summary'].get(k) for k in ('0', '10', '20', '40')]} "
              f"ancien {[L['old']['curve_summary'].get(k) for k in ('0', '10', '20', '40')]}  "
              f"erreurs {L['new']['certain_wrong_total']} / {L['old']['certain_wrong_total']}")
    print("durée", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
