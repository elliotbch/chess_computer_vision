"""
Entraîne le petit classifieur « case vide / occupée » (régression logistique sur 4 mesures
par case, voir src/occupancy.py) à partir des images de DÉVELOPPEMENT uniquement.

- Données : cases des images du split « dev » de data/ground_truth/sample_frames_gt.json,
  vues de dessus calculées avec les coins annotés ; cases cachées ('?') ignorées.
- Validation : validation croisée « une image laissée de côté » sur le dev, pour choisir C.
- Les images « test » ne sont jamais lues ici (elles servent à training/evaluate.py).
- Sortie : models/occupancy_logreg.json (moyenne, échelle, coefficients, seuil), lu par
  src/occupancy.LogReg et par la démo ; résumé ajouté à training/metrics.json.

Usage :
    python training/train_occupancy.py --seed 0
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import occupancy  # noqa: E402
import robust_calibration  # noqa: E402

GT_PATH = os.path.join(ROOT, "data", "ground_truth", "sample_frames_gt.json")


def load_split(gt, split):
    X, y, groups = [], [], []
    for name in gt["split"][split]:
        g = gt["frames"][name]
        img = cv2.imread(os.path.join(ROOT, "data", "sample_frames", f"{name}.png"))
        top = robust_calibration.top_view(img, np.asarray(g["corners"], np.float32), square=100)
        feats = occupancy.square_features(top, 100)
        lab = np.array([c for row in g["occupancy"] for c in row])
        keep = lab != "?"
        X.append(feats[keep])
        y.append(np.isin(lab[keep], ["W", "B"]).astype(int))
        groups += [name] * int(keep.sum())
    return np.vstack(X), np.concatenate(y), np.array(groups)


def fit(X, y, C, seed):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, random_state=seed, max_iter=1000).fit(sc.transform(X), y)
    return sc, clf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    ap.add_argument("--out-model", default=os.path.join(ROOT, "models", "occupancy_logreg.json"))
    ap.add_argument("--metrics", default=os.path.join(ROOT, "training", "metrics.json"))
    args = ap.parse_args()
    np.random.seed(args.seed)
    t0 = time.time()

    with open(GT_PATH) as f:
        gt = json.load(f)
    X, y, groups = load_split(gt, "dev")
    print(f"dev : {len(y)} cases ({y.sum()} occupées) sur {len(set(groups))} images")

    # Validation croisée une image laissée de côté
    cv = {}
    for C in args.Cs:
        correct = 0
        for g in sorted(set(groups)):
            tr, va = groups != g, groups == g
            sc, clf = fit(X[tr], y[tr], C, args.seed)
            correct += int((clf.predict(sc.transform(X[va])) == y[va]).sum())
        cv[C] = correct / len(y)
        print(f"C={C:<6} exactitude LOIO = {cv[C]:.4f}")
    best_C = max(args.Cs, key=lambda c: (cv[c], -c))   # à égalité, le plus régularisé
    sc, clf = fit(X, y, best_C, args.seed)
    train_acc = float((clf.predict(sc.transform(X)) == y).mean())

    params = {
        "features": occupancy.FEATURES,
        "mean": sc.mean_.tolist(),
        "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "threshold": 0.5,
        "trained_on": gt["split"]["dev"],
        "C": best_C,
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    with open(args.out_model, "w") as f:
        json.dump(params, f, indent=1)

    metrics = {}
    if os.path.exists(args.metrics):
        with open(args.metrics) as f:
            metrics = json.load(f)
    metrics["training_occupancy_logreg"] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "machine": f"{platform.machine()} / {platform.system()} {platform.release()} / Python {platform.python_version()}",
        "duration_seconds": round(time.time() - t0, 1),
        "train_images": gt["split"]["dev"],
        "train_squares": int(len(y)),
        "features": occupancy.FEATURES,
        "leave_one_image_out_accuracy": {str(k): round(v, 4) for k, v in cv.items()},
        "selected_C": best_C,
        "train_accuracy": round(train_acc, 4),
        "model_file": os.path.relpath(args.out_model, ROOT),
    }
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)
    print(f"C retenu = {best_C}, exactitude entraînement = {train_acc:.4f}; modèle : {args.out_model}")


if __name__ == "__main__":
    main()
