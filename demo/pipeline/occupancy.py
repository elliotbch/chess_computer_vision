"""
Détection d'occupation des cases, version 2 (novembre 2026).

Travaille sur la vue de dessus produite par `robust_calibration.top_view`
(a1 en haut à gauche, lignes = colonnes a..h, colonnes = rangées 1..8,
cases de `square` pixels). Pour chaque case, quelques mesures simples :

- `sat_frac`   : part des pixels du centre de la case dont la saturation dépasse un
                 seuil adapté à l'image (médiane + 4 MAD de la saturation du plateau) ;
- `lab_diff`   : écart de couleur (Lab) entre le centre de la case et sa bordure,
                 que les marqueurs posés au milieu de la case laissent visible ;
- `edge_frac`  : densité de contours (Canny) au centre de la case ;
- `l_std`      : écart-type de la luminance au centre.

Deux décideurs utilisent ces mesures :

- `rule` : seuil adaptatif, aucun apprentissage (seuil d'Otsu par image sur `lab_diff`) ;
- `logreg` : régression logistique entraînée sur les cases des images de développement
             (`training/train_occupancy.py`), poids lus dans un petit JSON.

Couleur du camp : parmi les pixels saturés de la case, une teinte jaune-orangé
(post-it jaune, capsule dorée) donne le camp noir, sinon (post-it vert, pièce
métallique non saturée) le camp blanc. Même convention que la vérité terrain
(`data/ground_truth/sample_frames_gt.json`).
"""

import json

import cv2
import numpy as np

FEATURES = ["sat_frac", "lab_diff", "edge_frac", "l_std"]


def square_features(top, square=100):
    """Matrice (64, 4) des mesures, dans l'ordre ligne par ligne (a1, a2, ..., h8)."""
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    s = hsv[..., 1].astype(np.float32)
    med = np.median(s)
    mad = np.median(np.abs(s - med)) + 1e-6
    s_thr = max(60.0, med + 4 * 1.4826 * mad)

    c0, c1 = int(square * 0.18), int(square * 0.82)      # centre de la case
    b = max(2, int(square * 0.07))                       # épaisseur de la bordure
    feats = np.zeros((64, len(FEATURES)), np.float32)
    for r in range(8):
        for c in range(8):
            y, x = r * square, c * square
            cell_lab = lab[y:y + square, x:x + square]
            center = cell_lab[c0:c1, c0:c1].reshape(-1, 3)
            ring = np.concatenate([cell_lab[1:b + 1, :].reshape(-1, 3), cell_lab[-b - 1:-1, :].reshape(-1, 3),
                                   cell_lab[:, 1:b + 1].reshape(-1, 3), cell_lab[:, -b - 1:-1].reshape(-1, 3)])
            feats[r * 8 + c] = [
                float((s[y + c0:y + c1, x + c0:x + c1] > s_thr).mean()),
                float(np.linalg.norm(np.median(center, axis=0) - np.median(ring, axis=0))),
                float((edges[y + c0:y + c1, x + c0:x + c1] > 0).mean()),
                float(center[:, 0].std()),
            ]
    return feats


def side_of_square(top, r, c, square=100):
    """'B' (marqueur jaune/doré) ou 'W' (vert ou non saturé) pour une case occupée."""
    y, x = r * square, c * square
    c0, c1 = int(square * 0.18), int(square * 0.82)
    hsv = cv2.cvtColor(top[y + c0:y + c1, x + c0:x + c1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[(hsv[:, 1] > 80) & (hsv[:, 2] > 70)]
    if len(sat) < 0.05 * len(hsv):
        return "W"
    hue = np.median(sat[:, 0])
    return "B" if 12 <= hue <= 32 else "W"


def _otsu(values):
    v = np.asarray(values, np.float64)
    order = np.sort(v)
    best, thr = -1, order[0]
    for i in range(1, len(order)):
        a, b = order[:i], order[i:]
        w = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if w > best:
            best, thr = w, (order[i - 1] + order[i]) / 2
    return thr


def predict_rule(feats):
    """Seuil adaptatif sans apprentissage sur l'écart de couleur centre / bordure.

    Seuil d'Otsu calculé sur les 64 cases de l'image, borné à [15, 35] (unités Lab)
    pour qu'un plateau presque vide ou presque plein ne soit pas coupé en deux
    groupes arbitraires. Bornes choisies sur les images de développement uniquement.
    """
    score = feats[:, 1]
    thr = float(np.clip(_otsu(score), 15.0, 35.0))
    return score > thr, score


class LogReg:
    """Régression logistique minimale (inférence NumPy, poids en JSON)."""

    def __init__(self, params):
        self.mean = np.array(params["mean"], np.float64)
        self.scale = np.array(params["scale"], np.float64)
        self.coef = np.array(params["coef"], np.float64)
        self.intercept = float(params["intercept"])
        self.threshold = float(params.get("threshold", 0.5))
        self.features = params.get("features", FEATURES)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def proba(self, feats):
        z = ((feats - self.mean) / self.scale) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))

    def predict(self, feats):
        p = self.proba(feats)
        return p > self.threshold, p


def detect(top, square=100, model=None):
    """Grille 8x8 de caractères 'W', 'B' ou '.', et scores par case."""
    feats = square_features(top, square)
    if model is not None:
        occ, score = model.predict(feats)
    else:
        occ, score = predict_rule(feats)
    grid = [["." for _ in range(8)] for _ in range(8)]
    for i in np.flatnonzero(occ):
        r, c = divmod(int(i), 8)
        grid[r][c] = side_of_square(top, r, c, square)
    return ["".join(row) for row in grid], score.reshape(8, 8)
