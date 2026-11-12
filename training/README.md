# Entraînement et évaluation

Deux petits modèles, aucun réseau de neurones :

- **occupation des cases** : régression logistique à 4 coefficients (vide / occupée, `src/occupancy.py`) ;
- **a priori des pièces inconnues** : régression logistique multinomiale (6 types × 295 indicateurs binaires) qui classe les types encore possibles d'un marqueur ambigu (« le plus probable »). Elle n'intervient **jamais** dans les identifications certaines, qui sont des déductions logiques (`src/piece_inference.py`).

## Identification des pièces inconnues : protocole (fixé avant la mesure)

- **Entrée du système** : l'occupation des cases après chaque demi-coup (marqueur blanc, marqueur noir, vide). Ni types, ni position initiale, ni droits de roque, ni historique ; la case de départ d'une pièce n'est jamais utilisée.
- **Jeux** : (a) les 2 parties annotées du dépôt (`archive/step1_annotation/step1_Lea/gameLR_*.json`, 67 et 70 demi-coups) ; (b) la base lichess de janvier 2013 (`data/lichess/lichess_db_standard_rated_2013-01.pgn`, parties publiques, licence CC0). Les parties valides sont prises dans l'ordre du fichier : la n° i va au **test** si i % 3 == 0, sinon à l'**entraînement** de l'a priori. 2 000 parties de test, 4 000 d'entraînement, disjointes (vérifié par l'identifiant lichess).
- **Extraits** : chaque partie est observée à partir des coups 1, 9, 15, 25 et 40 (si elle va jusque-là), pendant au plus 60 demi-coups.
- **Métriques** : part des marqueurs présents identifiés avec certitude après N demi-coups observés (courbe, extraits encore en cours) ; médiane du nombre de demi-coups avant identification certaine, par type (marqueurs présents au début ; non identifié = infini, donc médiane « non atteinte » si moins de la moitié des marqueurs du type sont identifiés) ; identifications certaines fausses ; exactitude du « plus probable » ; temps par demi-coup.
- **Référence** : l'ancienne logique du dépôt (`src/logic.py` + correctifs de `src/replay.py`) sur exactement les mêmes extraits, sans l'astuce « pièce jamais déplacée = type de sa case de départ ». Elle n'a pas de notion de certitude : chaque type qu'elle affiche est compté comme une identification (juste ou fausse).
- **A priori** : C choisi sur 20 % des parties d'entraînement (validation par partie), jamais sur le test.

Résultats sur les 2 000 parties lichess de test (section `piece_identification` de `training/metrics.json`) : part des marqueurs présents **identifiés avec certitude** après N demi-coups observés.

| Début de l'observation, méthode | Identifiés après 0 → 10 → 20 → 40 → 60 demi-coups (certitudes fausses, en marqueurs × demi-coups) |
|---|---|
| coup 1 (2 000 extraits), nouveau moteur | 50,0 % → 56,4 % → 65,5 % → 85,3 % → 95,7 % (fausses : 0 sur 1 888 442) |
| coup 1 (2 000 extraits), ancienne logique | 0,0 % → 7,2 % → 15,2 % → 27,8 % → 45,0 % (fausses : 42 sur 472 122) |
| coup 9 (1 914 extraits), nouveau moteur | 0,1 % → 10,4 % → 19,1 % → 33,2 % → 46,3 % (fausses : 0 sur 368 732) |
| coup 9 (1 914 extraits), ancienne logique | 0,0 % → 9,6 % → 15,7 % → 27,2 % → 37,8 % (fausses : 74 sur 306 129) |
| coup 15 (1 795 extraits), nouveau moteur | 0,0 % → 7,0 % → 10,7 % → 17,8 % → 33,7 % (fausses : 0 sur 140 070) |
| coup 15 (1 795 extraits), ancienne logique | 0,0 % → 6,9 % → 11,8 % → 20,1 % → 24,7 % (fausses : 88 sur 147 250) |
| coup 25 (1 386 extraits), nouveau moteur | 0,0 % → 4,8 % → 8,2 % → 21,6 % → 50,9 % (fausses : 0 sur 57 062) |
| coup 25 (1 386 extraits), ancienne logique | 0,0 % → 4,9 % → 8,3 % → 11,3 % → 17,4 % (fausses : 186 sur 40 919) |
| coup 40 (584 extraits), nouveau moteur | 0,1 % → 5,9 % → 17,7 % → 48,0 % → 63,7 % (fausses : 0 sur 22 064) |
| coup 40 (584 extraits), ancienne logique | 0,0 % → 3,6 % → 5,7 % → 11,2 % → 17,0 % (fausses : 119 sur 6 978) |

Pour l'ancienne logique, « identifiés » compte tous les types qu'elle affiche, faux compris : elle suppose les effectifs de départ (8 pions, 2 fous, 1 dame…), ce qui la fait parfois conclure plus tôt, mais à tort dès qu'il y a eu des prises ou des promotions avant l'observation.

Nombre médian de demi-coups observés avant identification certaine, par type (tous départs confondus ; « non atteinte » : moins de la moitié des marqueurs de ce type identifiés dans l'extrait) :

| Type (marqueurs), méthode | Identifiés dans l'extrait ; médiane sur tous ; médiane parmi les identifiés |
|---|---|
| pion (103 707), nouveau | 37,6 % ; non atteinte ; 0 |
| pion (103 707), ancienne | 7,4 % ; non atteinte ; 43 |
| cavalier (22 059), nouveau | 83,3 % ; 11 ; 8 |
| cavalier (22 059), ancienne | 83,2 % ; 11 ; 8 |
| fou (22 554), nouveau | 23,3 % ; non atteinte ; 30 |
| fou (22 554), ancienne | 16,6 % ; non atteinte ; 30 |
| tour (28 013), nouveau | 36,1 % ; non atteinte ; 24 |
| tour (28 013), ancienne | 35,3 % ; non atteinte ; 22 |
| dame (12 707), nouveau | 56,0 % ; 44 ; 23 |
| dame (12 707), ancienne | 50,1 % ; 60 ; 24 |
| roi (15 358), nouveau | 56,1 % ; 45 ; 20 |
| roi (15 358), ancienne | 41,6 % ; non atteinte ; 17 |

- **Erreurs** : 0 identification certaine fausse sur 2 476 370 (nouveau moteur) ; 509 types faux sur 973 398 affichés (ancienne logique). Remises à zéro après observation incohérente : 0.
- **Le plus probable** (a priori) : exact pour 89,2 % des marqueurs non identifiés, 93,1 % de tous les marqueurs (certitudes comprises). Au tout début d'une observation au coup 1, l'a priori reconnaît les pièces à leur case actuelle : c'est une statistique, pas une certitude, et elle ne vaut plus après des échanges.
- **Temps par demi-coup** (propagation + a priori, Python pur, un cœur ; mesuré pendant qu'un autre entraînement occupait la machine, donc pessimiste) : moyenne 3,492 ms, médiane 1,059 ms, 95e centile 15,394 ms, maximum 317,28 ms ; ancienne logique : moyenne 0,148 ms.

Les 2 parties annotées du dépôt (part identifiée avec certitude après 20 demi-coups et à la fin de l'extrait ; certitudes fausses) :

| Partie, début, méthode | Identifiés après 20 demi-coups ; à la fin de l'extrait ; certitudes fausses |
|---|---|
| gameLR_fix, coup 1, nouveau | 74,2 % ; 85,7 % ; 0 |
| gameLR_fix, coup 1, ancienne | 22,6 % ; 14,3 % ; 0 |
| gameLR_fix, coup 9, nouveau | 32,0 % ; 85,7 % ; 0 |
| gameLR_fix, coup 9, ancienne | 12,0 % ; 7,1 % ; 0 |
| gameLR_fix, coup 15, nouveau | 4,8 % ; 7,1 % ; 0 |
| gameLR_fix, coup 15, ancienne | 4,8 % ; 7,1 % ; 0 |
| gameLR_fix, coup 25, nouveau | – ; 7,1 % ; 0 |
| gameLR_fix, coup 25, ancienne | – ; 7,1 % ; 0 |
| gameLR_mov, coup 1, nouveau | 78,1 % ; 100,0 % ; 0 |
| gameLR_mov, coup 1, ancienne | 25,0 % ; 83,3 % ; 0 |
| gameLR_mov, coup 9, nouveau | 42,9 % ; 66,7 % ; 0 |
| gameLR_mov, coup 9, ancienne | 17,9 % ; 38,9 % ; 0 |
| gameLR_mov, coup 15, nouveau | 9,1 % ; 16,7 % ; 0 |
| gameLR_mov, coup 15, ancienne | 9,1 % ; 27,8 % ; 0 |
| gameLR_mov, coup 25, nouveau | 11,1 % ; 11,1 % ; 0 |
| gameLR_mov, coup 25, ancienne | 11,1 % ; 11,1 % ; 0 |


## Pour reproduire

1. **Machine** : Apple M2 Pro, 16 Go (macOS). N'importe quel CPU convient, aucun GPU n'est utilisé.
2. **Préparation** :
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt scikit-learn scipy
   mkdir -p data/lichess && curl -L https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst | zstd -d > data/lichess/lichess_db_standard_rated_2013-01.pgn   # 89 Mo, CC0
   ```
   Les photos et vérités terrain de la vision sont dans le dépôt (`data/sample_frames/`, `data/ground_truth/`).
3. **Commandes** :
   ```bash
   # vision
   python training/train_occupancy.py --seed 0 --Cs 0.01 0.1 1 10   # images dev seulement, C choisi en validation croisée une-image-sortie
   python training/evaluate.py --out training/metrics.json --synthetic 10   # v1 vs v2 sur dev et test, vues synthétiques
   # pièces inconnues
   python training/train_prior.py --train-games 4000 --test-games 2000 --seed 0 --keep 0.03 --workers 6
   python training/evaluate_inference.py --test-games 2000 --train-games 4000 --workers 6
   python demo/sync_pipeline.py                                          # recopie modules, métriques et exemples dans demo/
   ```
   `--test-games` / `--train-games` doivent être identiques dans les deux commandes (même découpage). `--keep 0.03` garde 3 % des exemples (marqueurs non identifiés, un par demi-coup), pour limiter la mémoire.
4. **Suivi** : les scripts affichent leur avancement dans le terminal (`train_prior.py` : nombre d'exemples puis exactitude de validation pour chaque C ; `evaluate_inference.py` : parties traitées toutes les 200).
5. **Durée observée (M2 Pro, 6 processus)** : a priori 227 s (dont 213 s de simulation pour extraire les exemples) ; évaluation 215 s pour 2 000 parties × 5 points de départ, nouveau moteur et ancienne logique (machine partagée avec un autre entraînement ; 57 s quand elle était libre) ; vision : entraînement 10 s, évaluation 63 s. Mémoire < 2 Go.
6. **Métriques** : `training/metrics.json`, sections `training_occupancy_logreg` et `evaluation` (vision), `training_piece_prior` et `piece_identification` (pièces inconnues), avec date, graine, machine et durée. Poids : `models/occupancy_logreg.json` et `models/piece_prior.json` (ignorés par git).

## Publication des poids (non exécutée)

```bash
hf upload elliotb/chess-occupancy models/occupancy_logreg.json occupancy_logreg.json
hf upload elliotb/chess-occupancy models/piece_prior.json piece_prior.json
```

La démo lit `MODEL_PATH` et `PRIOR_PATH`, sinon `HF_MODEL_REPO` (défaut du Dockerfile : `elliotb/chess-occupancy`), sinon `models/`. Sans poids d'occupation, elle utilise le seuil adaptatif v2 ; sans a priori, le rejeu montre seulement les certitudes et les types restants ; `/health` indique ce qui est chargé.
