"""
Évaluation chiffrée du pipeline sur la vérité terrain de `data/ground_truth/sample_frames_gt.json`.

Compare :
- v1 : le pipeline d'origine (src/calibration.py + src/detection.py, inchangés dans leur logique) ;
- v2 : src/robust_calibration.py + src/occupancy.py (règle adaptative, puis régression
       logistique si ses poids existent dans models/occupancy_logreg.json).

Métriques (par split dev / test, l'image dupliquée frame10 est exclue) :
- calibration réussie : les 4 coins externes ordonnés (a1, a8, h8, h1) sont chacun à moins de
  3 % du côté du plateau des coins annotés (une orientation fausse compte comme un échec) ;
- erreur moyenne des coins (px et % du côté) sur les calibrations réussies ;
- exactitude d'occupation par case (vide / occupée), cases cachées exclues :
    * « bout en bout » : avec les coins trouvés par le pipeline ; une calibration ratée compte
      toutes les cases de l'image comme fausses ;
    * « coins annotés » : avec les coins de la vérité terrain, pour isoler le détecteur ;
- exactitude 3 classes (vide / blanc / noir), v2 seulement, coins annotés ;
- précision et rappel de la classe « occupée » (coins annotés).

Usage :
    python training/evaluate.py --out training/metrics.json
"""

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import calibration as v1_calib  # noqa: E402
import detection as v1_det  # noqa: E402
import occupancy  # noqa: E402
import replay  # noqa: E402
import robust_calibration as v2_calib  # noqa: E402

GT_PATH = os.path.join(ROOT, "data", "ground_truth", "sample_frames_gt.json")
FRAMES_DIR = os.path.join(ROOT, "data", "sample_frames")
DEFAULT_MODEL = os.path.join(ROOT, "models", "occupancy_logreg.json")
CORNER_TOL = 0.03


def load_gt():
    with open(GT_PATH) as f:
        return json.load(f)


def board_side(corners):
    c = np.asarray(corners, np.float64)
    return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def corner_error(pred, gt):
    if pred is None:
        return None
    d = np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(gt, np.float64), axis=1)
    return d


# ------------------------------------------------------------------ v1 (code d'origine)

def v1_corners(img):
    with contextlib.redirect_stdout(io.StringIO()):
        ok, _, _, corners = v1_calib.corners_calibration(img.copy())
    if not ok:
        return None
    return np.asarray(corners, np.float32).reshape(4, 2)


def v1_occupancy(img, corners):
    """Même boucle que src/demo.py (detect_piece_color sur la vue de dessus 800x800)."""
    _, top = v1_det.find_chessboard_corners(img.copy(), np.asarray(corners, np.float32))
    work = top.copy()
    occ = np.zeros((8, 8), bool)
    for r in range(8):
        for c in range(8):
            roi = work[r * 100:(r + 1) * 100, c * 100:(c + 1) * 100]
            occ[r, c] = v1_det.detect_piece_color(roi, 100 * 100, (r % 2) == (c % 2))
    return occ


# ------------------------------------------------------------------ v2

def v2_corners(img):
    res = v2_calib.calibrate(img)
    return res["corners"] if res["ok"] else None


def v2_predict(img, corners, model):
    top = v2_calib.top_view(img, corners, square=100)
    grid, _ = occupancy.detect(top, 100, model)
    return grid


# ------------------------------------------------------------------ métriques

def gt_arrays(occ_rows):
    lab = np.array([list(r) for r in occ_rows])
    known = lab != "?"
    return lab, known


def summarize(rows):
    """rows : liste de dicts par image. Renvoie les métriques agrégées."""
    n = len(rows)
    ok = [r for r in rows if r["calib_ok"]]
    out = {
        "images": n,
        "calibration_success": len(ok),
        "calibration_success_rate": round(len(ok) / n, 4) if n else None,
        "corner_error_px_mean": round(float(np.mean([r["corner_err_px"] for r in ok])), 2) if ok else None,
        "corner_error_pct_side_mean": round(float(np.mean([r["corner_err_pct"] for r in ok])), 3) if ok else None,
    }
    known = sum(r["known"] for r in rows)
    out["squares_evaluated"] = known
    out["occupancy_accuracy_end_to_end"] = round(sum(r["correct_e2e"] for r in rows) / known, 4)
    out["occupancy_accuracy_gt_corners"] = round(sum(r["correct_gt"] for r in rows) / known, 4)
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    out["occupied_precision_gt_corners"] = round(tp / (tp + fp), 4) if tp + fp else None
    out["occupied_recall_gt_corners"] = round(tp / (tp + fn), 4) if tp + fn else None
    if all("correct3_gt" in r for r in rows):
        out["three_class_accuracy_gt_corners"] = round(sum(r["correct3_gt"] for r in rows) / known, 4)
    return out


def evaluate(method, frames, gt, model=None):
    rows = []
    for name in frames:
        g = gt["frames"][name]
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{name}.png"))
        lab, known = gt_arrays(g["occupancy"])
        occ_gt = np.isin(lab, ["W", "B"])
        side = board_side(g["corners"])
        t0 = time.time()
        corners = v1_corners(img) if method == "v1" else v2_corners(img)
        err = corner_error(corners, g["corners"])
        calib_ok = err is not None and bool((err < CORNER_TOL * side).all())
        row = {"frame": name, "calib_ok": calib_ok, "known": int(known.sum())}
        if err is not None:
            row["corner_err_px"] = float(err.mean())
            row["corner_err_pct"] = float(100 * err.mean() / side)

        def occ_of(c):
            if method == "v1":
                return v1_occupancy(img, c), None
            grid = v2_predict(img, c, model)
            arr = np.array([list(r) for r in grid])
            return arr != ".", arr

        # bout en bout
        if calib_ok:
            pred_e2e, _ = occ_of(corners)
            row["correct_e2e"] = int(((pred_e2e == occ_gt) & known).sum())
        else:
            row["correct_e2e"] = 0
        row["seconds"] = round(time.time() - t0, 3)
        # coins annotés
        pred_gt, arr = occ_of(np.asarray(g["corners"], np.float32))
        row["correct_gt"] = int(((pred_gt == occ_gt) & known).sum())
        row["tp"] = int((pred_gt & occ_gt & known).sum())
        row["fp"] = int((pred_gt & ~occ_gt & known).sum())
        row["fn"] = int((~pred_gt & occ_gt & known).sum())
        if arr is not None:
            row["correct3_gt"] = int(((arr == lab) & known).sum())
        rows.append(row)
    return summarize(rows), rows


def synthetic_views(img, corners, n, rng):
    """Vues synthétiques d'une image : homographie aléatoire (rotation ±25°, perspective),
    échelle 0,5-1, gamma, flou, compression JPEG. Renvoie [(image, coins transformés)]."""
    h, w = img.shape[:2]
    views = []
    while len(views) < n:
        ang = np.deg2rad(rng.uniform(-25, 25))
        scale = rng.uniform(0.5, 1.0)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), np.rad2deg(ang), 1.0)
        R = np.vstack([R, [0, 0, 1]])
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-0.12, 0.12, (4, 2)).astype(np.float32) * np.float32([w, h])
        P = cv2.getPerspectiveTransform(src, dst)
        S = np.diag([scale, scale, 1.0])
        M = S @ P @ R
        out_w, out_h = int(w * scale), int(h * scale)
        c = cv2.perspectiveTransform(np.float32(corners).reshape(-1, 1, 2), M).reshape(4, 2)
        if (c < 5).any() or (c[:, 0] > out_w - 5).any() or (c[:, 1] > out_h - 5).any():
            continue
        # pas de symétrie miroir : l'orientation a1 -> a8 -> h8 -> h1 doit être conservée
        v = cv2.warpPerspective(img, M, (out_w, out_h), borderMode=cv2.BORDER_REFLECT)
        gamma = rng.uniform(0.7, 1.4)
        v = np.clip(255 * (v / 255.0) ** gamma, 0, 255).astype(np.uint8)
        k = int(rng.choice([1, 3, 5]))
        if k > 1:
            v = cv2.GaussianBlur(v, (k, k), 0)
        q = int(rng.integers(50, 95))
        v = cv2.imdecode(cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        views.append((v, c))
    return views


def evaluate_synthetic(frames, gt, per_frame=10, seed=0):
    """Taux de calibration réussie sur des vues synthétiques des images de test."""
    rng = np.random.default_rng(seed)
    res = {"v1_original": 0, "v2": 0, "views": 0}
    for name in frames:
        g = gt["frames"][name]
        img = cv2.imread(os.path.join(FRAMES_DIR, f"{name}.png"))
        for v, c in synthetic_views(img, g["corners"], per_frame, rng):
            side = board_side(c)
            res["views"] += 1
            for key, fn in (("v1_original", v1_corners), ("v2", v2_corners)):
                err = corner_error(fn(v), c)
                res[key] += int(err is not None and bool((err < CORNER_TOL * side).all()))
    for key in ("v1_original", "v2"):
        res[f"{key}_success_rate"] = round(res[key] / res["views"], 4)
    res["seed"] = seed
    res["generator"] = ("rotation ±25°, déplacement des coins de l'image ±12 % (perspective), échelle 0,5-1, "
                        "gamma 0,7-1,4, flou 1/3/5 px, JPEG qualité 50-95 ; vues où le plateau sort de l'image rejetées")
    return res


GAMES = ["archive/step1_annotation/step1_Lea/gameLR_fix.json",
         "archive/step1_annotation/step1_Lea/gameLR_mov.json"]


def evaluate_logic():
    """Logique de jeu seule : états annotés -> observations (camp seulement) -> types déduits."""
    out = {}
    for fixes, key in [(False, "logic_original"), (True, "logic_with_replay_fixes")]:
        agg = {"moves": 0, "flagged_moves": 0, "piece_steps": 0, "identified": 0, "correct": 0, "wrong": 0,
               "final_pieces": 0, "final_identified_correct": 0, "final_correct_with_start_inference": 0}
        for path in GAMES:
            with open(os.path.join(ROOT, path)) as f:
                game = json.load(f)
            truth = [np.array(s["gs"]) for s in game["game_states"]]
            steps = replay.replay([replay.observation_from_state(t) for t in truth], fixes=fixes)
            agg["moves"] += len(steps) - 1
            agg["flagged_moves"] += sum(s["flag"] for s in steps[1:])
            for s, t in zip(steps, truth):
                ident, corr, tot = replay.identification_stats(s["board"], t)
                agg["piece_steps"] += tot
                agg["identified"] += ident
                agg["correct"] += corr
                agg["wrong"] += ident - corr
            ident, corr, tot = replay.identification_stats(steps[-1]["board"], truth[-1])
            agg["final_pieces"] += tot
            agg["final_identified_correct"] += corr
            board = replay.with_start_inference(steps[-1]["board"], steps[-1]["never_moved"])
            agg["final_correct_with_start_inference"] += replay.identification_stats(board, truth[-1])[1]
        agg["identified_rate"] = round(agg["identified"] / agg["piece_steps"], 4)
        agg["precision_of_identified"] = round(agg["correct"] / agg["identified"], 4) if agg["identified"] else None
        out[key] = agg
    out["games"] = GAMES
    out["protocol"] = ("états annotés à la main de deux parties (step 1), réduits au camp de chaque pièce "
                       "(±7) comme le produirait la vision ; à chaque coup on compare le type déduit au type "
                       "annoté. 'piece_steps' = pièces présentes, sommées sur tous les états.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "training", "metrics.json"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="poids de la régression logistique (optionnel)")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--synthetic", type=int, default=10, help="vues synthétiques par image de test")
    args = ap.parse_args()

    gt = load_gt()
    model = occupancy.LogReg.load(args.model) if os.path.exists(args.model) else None
    t0 = time.time()
    results = {}
    for split in args.splits:
        frames = gt["split"][split]
        res = {}
        res["v1_original"], per_v1 = evaluate("v1", frames, gt)
        res["v2_rule"], per_rule = evaluate("v2", frames, gt, None)
        res["per_frame"] = {"v1_original": per_v1, "v2_rule": per_rule}
        if model is not None:
            res["v2_logreg"], per_lr = evaluate("v2", frames, gt, model)
            res["per_frame"]["v2_logreg"] = per_lr
        results[split] = res
        print(f"== {split} ({len(frames)} images)")
        for k, v in res.items():
            if k != "per_frame":
                print(f"  {k:12s} {json.dumps(v)}")

    if "test" in args.splits:
        results["test_synthetic_views"] = evaluate_synthetic(gt["split"]["test"], gt, per_frame=args.synthetic)
        print(f"== vues synthétiques (test) : {json.dumps(results['test_synthetic_views'])}")
    results["game_logic"] = evaluate_logic()
    print("== logique de jeu")
    for k in ("logic_original", "logic_with_replay_fixes"):
        print(f"  {k}: {json.dumps(results['game_logic'][k])}")

    previous = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            previous = json.load(f)
    previous["evaluation"] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.machine()} / {platform.system()} {platform.release()} / Python {platform.python_version()}",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "duration_seconds": round(time.time() - t0, 1),
        "ground_truth": os.path.relpath(GT_PATH, ROOT),
        "protocol": {
            "corner_tolerance": f"{int(CORNER_TOL * 100)} % du côté du plateau, pour chacun des 4 coins ordonnés",
            "excluded": "frame10 (doublon exact de frame07) ; cases marquées '?' (cachées par une main)",
            "end_to_end": "une calibration ratée compte toutes les cases de l'image comme fausses",
            "logreg_model": os.path.relpath(args.model, ROOT) if model is not None else None,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(previous, f, indent=1, ensure_ascii=False)
    print(f"Métriques écrites dans {args.out}")


if __name__ == "__main__":
    main()
