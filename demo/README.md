---
title: Vision d'échiquier
emoji: ♟️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Identifier des pièces inconnues à partir de leurs coups
---

# Vision d'échiquier : pièces inconnues (projet de vision par ordinateur, ULiège)

Démo du dépôt [chess_computer_vision](https://github.com/elliotbch/chess_computer_vision). Les pièces sont remplacées par des marqueurs identiques (un modèle par camp) : seul le camp de chaque case est visible.

- **Identifier les pièces inconnues coup par coup** : parties d'exemple (2 parties annotées du projet, 3 parties célèbres du domaine public) ou votre PGN / FEN + coups. Curseur « commencer l'observation au coup N » : le moteur ne connaît ni la position de départ, ni les types, ni l'historique. Sur l'échiquier, symbole de la pièce quand elle est identifiée avec certitude, sinon les types encore possibles (le plus probable en gras) ; courbe « pièces identifiées / demi-coups observés », comparée à l'ancienne logique du dépôt ; journal des identifications.
- **Analyser une photo** (6 exemples ou votre JPEG/PNG, 12 Mo au maximum) : coins et grille détectés, vue de dessus, cases occupées et camp de chaque marqueur.
- **Métriques** : lues dans `eval_metrics.json` (copie de `training/metrics.json`).

## Lancer en local

```bash
python demo/sync_pipeline.py          # depuis la racine du dépôt, après toute modification de src/
cd demo && pip install -r requirements.txt
MODEL_PATH=../models/occupancy_logreg.json PRIOR_PATH=../models/piece_prior.json uvicorn app:app --port 7860
```

Docker : `docker build -t chess-vision demo/ && docker run --rm -p 7860:7860 chess-vision` (poids téléchargés depuis `HF_MODEL_REPO`, par défaut `elliotb/chess-occupancy` : `occupancy_logreg.json` et `piece_prior.json`). Sans le premier, le détecteur à seuil adaptatif est utilisé ; sans le second, le rejeu montre seulement les certitudes et les types restants. `/health` dit ce qui est chargé.

Tests : `pip install pytest httpx && pytest demo/tests`.

## API

| Méthode | Chemin | |
|---|---|---|
| GET | `/health` | détecteur d'occupation et a priori chargés, et leur provenance |
| GET | `/api/examples` | photos et parties d'exemple |
| POST | `/api/analyze` | champ de formulaire `example=<id>` ou `file=<image>` → coins, grille d'occupation, images (JPEG en base64), temps de calcul |
| POST | `/api/replay` | JSON `{"game": id}` ou `{"pgn": "..."}` ou `{"fen": "...", "moves": "..."}`, plus `start_ply` → pour chaque demi-coup observé : marqueurs, types possibles, certitudes, type le plus probable, comparaison avec l'ancienne logique, temps de calcul |
| GET | `/api/metrics` | métriques d'évaluation |
