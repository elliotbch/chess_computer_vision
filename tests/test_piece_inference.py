"""Tests du moteur d'identification des pièces inconnues (src/piece_inference.py)."""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import chessrules as cr  # noqa: E402
import piece_inference as pi  # noqa: E402


def occ(pos):
    return [0 if p is None else (1 if p.isupper() else -1) for p in pos.board]


def check_sound(positions, start):
    eng = pi.PieceInference(occ(positions[start]))
    for pos in positions[start + 1:]:
        eng.observe(occ(pos))
        for v in eng.vars:
            if v.alive:
                assert v.dom >> "PNBRQK".index(pos.board[v.sq].upper()) & 1, (pos.fen(), pi.sqname(v.sq))
    assert eng.resets == 0
    return eng


def random_game(seed, plies=120):
    rng = random.Random(seed)
    pos = cr.Position.from_fen(cr.START_FEN)
    positions = [pos]
    for _ in range(plies):
        legal = pos.legal_moves()
        if not legal:
            break
        pos = pos.push(rng.choice(legal))
        positions.append(pos)
    return positions


def test_random_games_never_exclude_true_type():
    # parties aléatoires : promotions, sous-promotions, prises en passant et roques fréquents
    for seed in range(12):
        positions = random_game(seed)
        for start in (0, 7, 30):
            if start < len(positions) - 1:
                check_sound(positions, start)


def test_annotated_games():
    import json
    for name in ("gameLR_fix", "gameLR_mov"):
        with open(os.path.join(ROOT, "archive", "step1_annotation", "step1_Lea", f"{name}.json")) as f:
            states = [s["gs"] for s in json.load(f)["game_states"]]
        positions, _ = cr.play_states(states)
        for start in (0, 16, 28):
            check_sound(positions, start)


def test_full_board_pawns_and_knight():
    positions, _ = cr.play("e4 e5 Nf3".split())
    eng = pi.PieceInference(occ(positions[0]))
    m = {x["square"]: x["certain"] for x in eng.markers()}
    assert all(m[f + "2"] == "P" and m[f + "7"] == "P" for f in "abcdefgh")   # 32 marqueurs : aucune promotion
    assert m["e1"] is None                                                   # jamais la case de départ
    for p in positions[1:]:
        eng.observe(occ(p))
    assert {x["square"]: x["certain"] for x in eng.markers()}["f3"] == "N"


def test_castling_mid_game_without_history():
    fen = "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"
    positions, _ = cr.play(["O-O-O"], fen)
    eng = pi.PieceInference(occ(positions[0]))
    eng.observe(occ(positions[1]))
    m = {x["square"]: x["certain"] for x in eng.markers()}
    assert m["c1"] == "K" and m["d1"] == "R"


def test_en_passant_identifies_both_pawns():
    positions, _ = cr.play("e4 a6 e5 d5 exd6".split())
    eng = pi.PieceInference(occ(positions[2]))          # observation au 2e coup des Blancs
    for p in positions[3:]:
        ev = eng.observe(occ(p))
    assert ev["kind"] == "en_passant"
    assert {x["square"]: x["certain"] for x in eng.markers()}["d6"] == "P"


def test_unreadable_change_resets_without_error():
    positions, _ = cr.play("e4 e5 Nf3 Nc6".split())
    eng = pi.PieceInference(occ(positions[0]))
    ev = eng.observe(occ(positions[2]))                 # deux demi-coups d'un coup : illisible
    assert ev["kind"] == "unreadable" and eng.resets == 1
    for x in eng.markers():
        assert x["certain"] is None or x["certain"] == positions[2].board[x["sq"]].upper()


def test_prior_probabilities_are_normalised():
    path = os.path.join(ROOT, "models", "piece_prior.json")
    if not os.path.exists(path):
        return
    positions, _ = cr.play("d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7".split())
    eng = pi.PieceInference(occ(positions[4]), pi.Prior.load(path))
    for p in positions[5:]:
        eng.observe(occ(p))
    for m in eng.markers():
        assert abs(sum(m["probs"].values()) - 1) < 1e-3     # probabilités arrondies à 4 décimales
        assert m["most_probable"] in m["candidates"]
