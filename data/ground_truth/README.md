# Vérité terrain des images d'exemple

`sample_frames_gt.json` annote les 12 images de `data/sample_frames/` (novembre 2026).

- **Coins** : les 4 coins externes de la zone 8×8 (ordre a1, a8, h8, h1), proposés par `src/robust_calibration.py` puis vérifiés à l'œil sur 48 agrandissements pleine résolution (grille projetée superposée aux bords des cases). Aucun n'a dû être corrigé. Le pipeline d'origine, quand il réussit, tombe à 0,2-0,3 px de ces coins : l'annotation ne favorise donc pas un pipeline plutôt que l'autre à la tolérance utilisée (3 % du côté), mais l'erreur de coin en pixels de la v2 (≈ 0,04 px) n'a pas de sens et n'est pas reprise dans le README.
- **Occupation** : chaque case annotée à la main sur la vue de dessus redressée : `W` (post-it vert ou vis métallique, camp blanc), `B` (post-it jaune ou capsule dorée, camp noir), `.` vide, `?` cachée par une main (exclue des métriques, 16 cases au total). Pour les pièces en relief de `frame20`, la case retenue est celle de la base de la pièce.
- **Doublon** : `frame10` est identique octet pour octet à `frame07` ; elle est exclue.
- **Découpage** : `dev` (frame00, 03, 05, 07, 09) sert au réglage des seuils et à l'entraînement ; `test` (frame01, 02, 04, 06, 08, 20) n'a servi qu'à la mesure finale. Certaines images des deux groupes viennent de la même partie et d'angles voisins : le test est « non vu » mais pas indépendant.

Annotateur : agent Claude pour Elliot Bouchy ; à relire par Elliot.
