"""
Démo sur une image fixe : calibration + détection d'occupation des cases.

Exécute les mêmes fonctions que le pipeline vidéo (calibration.py, detection.py)
sur une seule image et enregistre une visualisation :
  - à gauche, l'image d'origine avec le contour du plateau, les coins et les stickers ;
  - à droite, la vue de dessus (homographie) avec, encadrées, les cases détectées comme occupées.

La couleur des pièces n'est pas déduite ici : dans le pipeline, elle n'est attribuée
qu'à l'initialisation (position de départ), puis suivie via les mouvements (logic.py).

Usage :
    python src/demo.py data/sample_frames/frame03.png -o demo_frame03.png
"""

import argparse
import os

import cv2
import numpy as np

from calibration import corners_calibration
from detection import detect_piece_color, find_chessboard_corners

SQUARE = 100  # taille d'une case dans la vue de dessus (pixels)


def run(image_path, output_path):
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Impossible de lire l'image : {image_path}")
    original = frame.copy()

    # 1. Calibration : stickers rose/bleu, coins du plateau, homographie
    is_calibrate, _, _, corners = corners_calibration(frame)
    if not is_calibrate:
        raise SystemExit("Calibration impossible sur cette image (plateau ou stickers non trouvés).")

    # 2. Vue de dessus 800x800 du plateau
    _, top_view = find_chessboard_corners(original.copy(), corners)
    work = top_view.copy()      # detect_piece_color dessine ses contours sur cette copie
    overlay = top_view.copy()   # visualisation propre

    # 3. Détection d'occupation case par case (segmentation HSV + aire des contours)
    game_state = np.zeros((8, 8), dtype=int)
    for row in range(8):
        for col in range(8):
            x1, y1 = col * SQUARE, row * SQUARE
            roi = work[y1:y1 + SQUARE, x1:x1 + SQUARE]
            dark_square = (row % 2) == (col % 2)
            if detect_piece_color(roi, SQUARE * SQUARE, dark_square):
                game_state[row, col] = 1
                cv2.rectangle(overlay, (x1 + 4, y1 + 4), (x1 + SQUARE - 4, y1 + SQUARE - 4), (255, 160, 0), 5)

    # 4. Composition côte à côte (même hauteur)
    h = overlay.shape[0]
    scale = h / frame.shape[0]
    left = cv2.resize(frame, (int(frame.shape[1] * scale), h))
    canvas = np.hstack([left, np.full((h, 16, 3), 255, np.uint8), overlay])
    cv2.imwrite(output_path, canvas)

    print(f"Cases occupées détectées : {int((game_state != 0).sum())}")
    print(np.rot90(game_state, k=1))  # même orientation que process.py
    print(f"Visualisation enregistrée : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="chemin d'une image du plateau")
    parser.add_argument("-o", "--output", default=None, help="image de sortie (défaut : demo_<nom>.png)")
    args = parser.parse_args()
    out = args.output or f"demo_{os.path.splitext(os.path.basename(args.image))[0]}.png"
    run(args.image, out)
