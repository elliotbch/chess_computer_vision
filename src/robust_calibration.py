"""
Calibration robuste du plateau (version 2, novembre 2026).

Ajoutée après coup pour mesurer et corriger les échecs de `calibration.py`
(6 images sur 12 calibrées dans `data/sample_frames/`). Même idée de départ
(coins internes 7x7 du damier, stickers rose et bleu pour l'orientation),
avec quatre changements :

1. Les marqueurs colorés posés sur les cases (post-it jaunes et verts) sont
   effacés par inpainting avant la recherche des coins : ils cassent le motif
   du damier que cherche `findChessboardCornersSB`.
2. Recherche multi-échelle (l'image est ramenée à ~1200 px de large, y compris
   en agrandissant les petites images).
3. Si une main cache une partie du plateau, on accepte un sous-damier
   (7x6, 6x7, 7x5, ...) puis on teste toutes les positions possibles de ce
   sous-damier dans le plateau 8x8 : on garde celle qui maximise l'alternance
   clair/foncé à l'intérieur et la minimise sur la bordure (« hypothèse puis
   vérification »). L'homographie est ensuite raffinée sur les 49 coins
   internes projetés (cornerSubPix + RANSAC).
4. Les stickers sont cherchés près des quatre coins du plateau trouvé (seuils
   d'aire relatifs à la taille d'une case, plus de coordonnées en pixels codées
   en dur), et un seul sticker suffit : le sens de parcours a1 -> a8 -> h8 -> h1
   est toujours le même dans l'image (une photo n'inverse pas le plateau).

Convention de sortie identique à `calibration.py` : coins externes dans l'ordre
a1, a8, h8, h1 (a1 près du sticker bleu, a8 près du sticker rose).
"""

import cv2
import numpy as np

WORK_WIDTH = 1200          # largeur de travail pour la recherche des coins
SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_NORMALIZE_IMAGE
PARTIAL_PATTERNS = [(7, 6), (6, 7), (7, 5), (5, 7), (6, 6), (7, 4), (4, 7), (6, 5), (5, 6), (5, 5)]

# Ordre a1, a8, h8, h1 en coordonnées « plateau » (unité = une case, coins externes en 0 et 8),
# exprimé dans le repère du damier détecté ; l'orientation réelle est choisie ensuite.
BOARD_QUAD = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Prétraitement
# ----------------------------------------------------------------------------------------------

def marker_mask(bgr):
    """Masque des pixels saturés jaunes/verts (post-it), dilaté."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((s > 70) & (v > 90) & (h >= 18) & (h <= 95)).astype(np.uint8) * 255
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def erase_markers(bgr):
    """Image en niveaux de gris où les marqueurs colorés sont remplacés par inpainting."""
    out = cv2.inpaint(bgr, marker_mask(bgr), 5, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


# ----------------------------------------------------------------------------------------------
# Recherche du damier
# ----------------------------------------------------------------------------------------------

def _find_pattern(gray, size):
    ok, c = cv2.findChessboardCornersSB(gray, size, SB_FLAGS)
    if ok:
        return c.reshape(size[1], size[0], 2)
    return None


def _checker_score(gray, H, px=24):
    """Score d'une hypothèse de plateau (homographie plateau -> image).

    Vue de dessus de -1 à 9 cases ; alternance moyenne entre cases voisines à l'intérieur
    du 8x8 moins alternance sur l'anneau extérieur (bordure du plateau, uniforme).
    """
    T = np.array([[px, 0, px], [0, px, px], [0, 0, 1]], dtype=np.float64)   # -1 -> 0
    M = T @ np.linalg.inv(H)
    top = cv2.warpPerspective(gray, M, (10 * px, 10 * px), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    q = px // 4
    cells = np.array([[np.median(top[i * px + q:(i + 1) * px - q, j * px + q:(j + 1) * px - q])
                       for j in range(10)] for i in range(10)], dtype=np.float64)
    inner = cells[1:9, 1:9]
    alt_in = (np.abs(np.diff(inner, axis=0)).mean() + np.abs(np.diff(inner, axis=1)).mean()) / 2
    ring = np.concatenate([cells[0, :], cells[1:, 9], cells[9, -2::-1], cells[-2:0:-1, 0]])
    alt_ring = np.abs(np.diff(ring)).mean()
    # Le signe du damier doit aussi être cohérent (alternance régulière, pas seulement du contraste)
    sign = np.indices((8, 8)).sum(axis=0) % 2 * 2 - 1
    regular = abs(((inner - inner.mean()) * sign).mean())
    return alt_in - alt_ring + regular


def _homography_from_pattern(pattern, ox, oy):
    """Homographie plateau -> image pour un sous-damier placé à l'offset (ox, oy)."""
    rows, cols = pattern.shape[:2]
    src = np.array([[ox + 1 + j, oy + 1 + i] for i in range(rows) for j in range(cols)], np.float32)
    H, _ = cv2.findHomography(src, pattern.reshape(-1, 2).astype(np.float32))
    return H


def _refine(gray, H):
    """Raffine l'homographie sur les 49 coins internes projetés (cornerSubPix + RANSAC)."""
    pts = np.array([[i, j] for j in range(1, 8) for i in range(1, 8)], np.float32)
    proj = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    # taille approximative d'une case dans l'image
    sq = np.median(np.linalg.norm(np.diff(proj.reshape(7, 7, 2), axis=1), axis=2))
    win = int(max(3, min(15, sq * 0.2)))
    h, w = gray.shape[:2]
    inside = ((proj[:, 0] > win + 1) & (proj[:, 0] < w - win - 2) &
              (proj[:, 1] > win + 1) & (proj[:, 1] < h - win - 2))
    if inside.sum() < 12:            # plateau en grande partie hors de l'image
        return H, int(inside.sum())
    pts, proj = pts[inside], proj[inside]
    ref = cv2.cornerSubPix(gray, proj.reshape(-1, 1, 2).copy(), (win, win), (-1, -1),
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(-1, 2)
    keep = np.linalg.norm(ref - proj, axis=1) < 0.15 * sq
    if keep.sum() < 12:
        return H, int(keep.sum())
    H2, inl = cv2.findHomography(pts[keep], ref[keep], cv2.RANSAC, max(1.0, 0.05 * sq))
    if H2 is None:
        return H, int(keep.sum())
    return H2, int(inl.sum())


def find_board(bgr):
    """Trouve le plateau. Renvoie (H plateau->image en pixels d'origine, infos) ou (None, infos)."""
    scale = WORK_WIDTH / bgr.shape[1]
    small = cv2.resize(bgr, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    candidates = [("inpaint", erase_markers(small)), ("gray", cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))]
    S = np.diag([scale, scale, 1.0])
    for name, gray in candidates:
        pat = _find_pattern(gray, (7, 7))
        if pat is not None:
            H = _homography_from_pattern(pat, 0, 0)
            H, n = _refine(gray, H)
            return np.linalg.inv(S) @ H, {"method": f"full 7x7 ({name})", "inliers": n}
    gray = candidates[0][1]
    for size in PARTIAL_PATTERNS:
        pat = _find_pattern(gray, size)
        if pat is None:
            continue
        best = None
        for ox in range(0, 8 - size[0]):
            for oy in range(0, 8 - size[1]):
                H = _homography_from_pattern(pat, ox, oy)
                s = _checker_score(gray, H)
                if best is None or s > best[0]:
                    best = (s, H, ox, oy)
        H, n = _refine(gray, best[1])
        return np.linalg.inv(S) @ H, {"method": f"partial {size[0]}x{size[1]} at offset {best[2:]}",
                                      "inliers": n}
    return None, {"method": "not found"}


# ----------------------------------------------------------------------------------------------
# Orientation par les stickers
# ----------------------------------------------------------------------------------------------

def _blobs(bgr, color, min_area, max_area):
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    if color == "rose":
        m = cv2.inRange(hsv, (140, 80, 90), (179, 255, 255)) | cv2.inRange(hsv, (0, 110, 120), (4, 255, 255))
    else:
        m = cv2.inRange(hsv, (90, 100, 80), (130, 255, 255))
    k = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    out = []
    for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a = cv2.contourArea(c)
        if min_area < a < max_area:
            M = cv2.moments(c)
            out.append(np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]]))
    return out


def orient(bgr, H):
    """Choisit l'ordre a1, a8, h8, h1 des coins externes à partir des stickers.

    Renvoie (coins 4x2 ordonnés, H plateau->image réorientée, infos) ; coins = None si
    aucun sticker n'est trouvé près du plateau.
    """
    quad = cv2.perspectiveTransform(BOARD_QUAD.reshape(-1, 1, 2), H).reshape(4, 2)
    side = np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)]) / 8  # taille d'une case
    blue = _blobs(bgr, "blue", 0.03 * side ** 2, 1.5 * side ** 2)
    rose = _blobs(bgr, "rose", 0.03 * side ** 2, 1.5 * side ** 2)

    def dist(blobs, p):
        return min([np.linalg.norm(b - p) for b in blobs], default=np.inf) / side

    # Le sens a1 -> a8 -> h8 -> h1 dans l'image est fixé (voir docstring) : signe de l'aire.
    def signed_area(q):
        return 0.5 * sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))

    best = None
    for k in range(4):
        for direction in (1, -1):
            order = [(k + direction * i) % 4 for i in range(4)]
            q = quad[order]
            if signed_area(q) < 0:      # sens imposé : a1 -> a8 -> h8 -> h1 tourne dans le sens horaire à l écran
                continue
            db, dr = dist(blue, q[0]), dist(rose, q[1])
            cost = min(db, 2.5) + min(dr, 2.5)
            if best is None or cost < best[0]:
                best = (cost, order, db, dr)
    cost, order, db, dr = best
    found = {"blue": bool(db < 2.5), "rose": bool(dr < 2.5)}
    if not (found["blue"] or found["rose"]):
        return None, None, {"stickers": found}
    # H réorientée : coordonnées plateau canoniques (a1 = (0,0), a8 = (8,0), h8 = (8,8), h1 = (0,8))
    Hc, _ = cv2.findHomography(BOARD_QUAD, quad[order])
    return quad[order], Hc, {"stickers": found}


def calibrate(bgr):
    """Calibration complète. Renvoie un dict : ok, corners (a1, a8, h8, h1), H (plateau -> image), infos."""
    try:
        H, info = find_board(bgr)
    except cv2.error as exc:         # image dégénérée : échec propre plutôt qu'une exception
        return {"ok": False, "corners": None, "H": None, "info": {"method": f"error: {exc.err}"}}
    if H is None:
        return {"ok": False, "corners": None, "H": None, "info": info}
    corners, Hc, oinfo = orient(bgr, H)
    info.update(oinfo)
    if corners is None:
        return {"ok": False, "corners": None, "H": None, "info": info}
    return {"ok": True, "corners": corners, "H": Hc, "info": info}


def top_view(bgr, corners, square=100, margin=0):
    """Vue de dessus : a1 en haut à gauche, a8 en haut à droite (même convention que detection.py)."""
    n = 8 * square
    dst = np.array([[margin, margin], [margin + n, margin], [margin + n, margin + n], [margin, margin + n]],
                   np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), dst)
    return cv2.warpPerspective(bgr, M, (n + 2 * margin, n + 2 * margin))
