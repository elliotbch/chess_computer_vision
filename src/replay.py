"""
Rejoue une partie coup par coup avec la logique de jeu de `logic.py`.

Entrée : une suite d'états « observés » du plateau, tels que les produit la vision :
8x8, +7 pour un marqueur blanc, -7 pour un marqueur noir, 0 pour une case vide
(ligne 0 = rangée 8, colonne 0 = colonne a, convention de `logic.py` et des JSON
d'annotation de `archive/step1_annotation`). Le type des pièces n'est jamais donné :
la logique le déduit des déplacements.

`logic.pieces_recognition` attend l'état précédent (avec les types déjà déduits) et
cet état auquel on a appliqué le coup. Le pipeline vidéo d'origine (`process.py`)
lui passait directement la nouvelle détection, où tous les types valent ±7 : dès
qu'une pièce était reconnue, toutes les cases connues apparaissaient « modifiées ».
`apply_observation` fait la conversion manquante.

Codes des pièces : 1 pion, 2 tour, 3 cavalier, 4 fou, 5 roi, 6 dame ; 7 = inconnu.
"""

import contextlib
import io

import numpy as np

import logic

PIECE_LETTERS = {1: "P", 2: "R", 3: "N", 4: "B", 5: "K", 6: "Q"}


def initial_pieces_number():
    base = {1: 8, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1}
    out = {}
    for side in (1, -1):
        for v, n in base.items():
            out[side * v] = tuple([n] + [""] * n)
    return out


def observation_from_state(gs):
    """État annoté (types connus) -> observation (camp seulement)."""
    gs = np.asarray(gs, dtype=int)
    return np.sign(gs) * 7


def classify_change(prev, obs):
    """Analyse le changement observé entre l'état déduit `prev` et l'observation `obs`.

    Renvoie un dict {kind: 'none'|'move'|'castle'|'en_passant'|'unreadable', ...}.
    """
    prev = np.asarray(prev, dtype=int)
    obs = np.asarray(obs, dtype=int)
    origins = [tuple(int(v) for v in p) for p in np.argwhere((prev != 0) & (obs == 0))]
    dests = [tuple(int(v) for v in p) for p in np.argwhere((obs != 0) & (np.sign(obs) != np.sign(prev)))]
    if not origins and not dests:
        return {"kind": "none"}
    if len(origins) == 1 and len(dests) == 1:
        o, d = origins[0], dests[0]
        if np.sign(prev[o]) == np.sign(obs[d]):
            return {"kind": "move", "from": o, "to": d, "side": int(np.sign(prev[o]))}
    if len(origins) == 2 and len(dests) == 2 and len({r for r, _ in origins + dests}) == 1:
        row = origins[0][0]
        cols_o = sorted(c for _, c in origins)
        cols_d = sorted(c for _, c in dests)
        if 4 in cols_o:
            rook_o = [c for c in cols_o if c != 4][0]
            king_d, rook_d = (6, 5) if rook_o == 7 else (2, 3)
            if sorted([king_d, rook_d]) == cols_d:
                return {"kind": "castle", "row": row, "rook_from": rook_o, "king_to": king_d, "rook_to": rook_d,
                        "side": int(np.sign(prev[row, 4]))}
    if len(origins) == 2 and len(dests) == 1 and prev[dests[0]] == 0:
        d = dests[0]
        movers = [o for o in origins if np.sign(prev[o]) == np.sign(obs[d])]
        taken = [o for o in origins if np.sign(prev[o]) == -np.sign(obs[d])]
        if len(movers) == 1 and len(taken) == 1 and taken[0][0] == movers[0][0] and taken[0][1] == d[1]:
            return {"kind": "en_passant", "from": movers[0], "to": d, "taken": taken[0],
                    "side": int(np.sign(obs[d]))}
    return {"kind": "unreadable"}


def _apply(prev, ch):
    new = np.asarray(prev, dtype=int).copy()
    if ch["kind"] in ("move", "en_passant"):
        new[ch["to"]] = new[ch["from"]]
        new[ch["from"]] = 0
        if ch["kind"] == "en_passant":
            new[ch["taken"]] = 0
    elif ch["kind"] == "castle":
        r = ch["row"]
        new[r, ch["king_to"]] = new[r, 4]
        new[r, ch["rook_to"]] = new[r, ch["rook_from"]]
        new[r, 4] = 0
        new[r, ch["rook_from"]] = 0
    return new


def square_name(rc):
    r, c = rc
    return "abcdefgh"[c] + str(8 - r)


def describe_move(prev, new, ch):
    """Texte du coup, avec la lettre de la pièce quand son type est connu après le coup."""
    if ch["kind"] == "castle":
        side = "White" if ch["side"] > 0 else "Black"
        return f"{side} castles {'O-O' if ch['rook_from'] == 7 else 'O-O-O'}"
    o, d = ch["from"], ch["to"]
    letter = PIECE_LETTERS.get(abs(int(new[d])), "?")
    sep = "x" if (prev[d] != 0 or ch["kind"] == "en_passant") else "-"
    suffix = " e.p." if ch["kind"] == "en_passant" else ""
    return f"{letter}{square_name(o)}{sep}{square_name(d)}{suffix}"


def _remove_from_dict(game_dict, square):
    """La pièce prise quitte le plateau : sans cela, deux entrées du dictionnaire partagent la
    même case et `logic.get_piece_name` peut renvoyer la pièce capturée au lieu de la bonne.
    L'entrée est supprimée (une position hors plateau comme (-1, -1) serait relue par
    `logic.context_update` comme l'indice NumPy de la case h1)."""
    for name in [n for n, (_, pos, _, _) in game_dict.items() if pos == square]:
        del game_dict[name]


def replay(observations, fixes=True, quiet=True):
    """Rejoue une suite d'observations avec `logic.pieces_recognition`.

    fixes=False : la logique d'origine seule (le roque y échoue toujours, la prise en passant
    n'est pas gérée, les pièces prises restent dans le dictionnaire).
    fixes=True  : roque et prise en passant traités ici (types connus par la règle : roi et tour,
    pion), pièce prise retirée du dictionnaire, resynchronisation sur l'observation si un
    changement est illisible (les pièces concernées redeviennent inconnues).

    Renvoie une liste d'étapes {board, move, kind, flag, never_moved} ; never_moved est le masque
    des pièces encore sur leur case de départ (dont le type découle de la position initiale).
    """
    obs0 = np.asarray(observations[0], dtype=int)
    state = obs0.copy()
    game_dict = {}
    k = 1
    for r in range(8):
        for c in range(8):
            if state[r, c] != 0:
                s = int(np.sign(state[r, c]))
                game_dict[f"piece{k}"] = (s * 7, (r, c), (r, c), s)
                k += 1
    pieces_number = initial_pieces_number()
    moved = np.zeros((8, 8), bool)
    steps = [{"board": state.copy(), "move": "start", "kind": "start", "flag": False, "never_moved": ~moved & (state != 0)}]
    sink = io.StringIO()
    for obs in observations[1:]:
        obs = np.asarray(obs, dtype=int)
        ch = classify_change(state, obs)
        before = state.copy()
        flag = False
        if ch["kind"] == "none":
            steps.append({"board": state.copy(), "move": "no change", "kind": "none", "flag": False,
                          "never_moved": ~moved & (state != 0)})
            continue
        if ch["kind"] == "unreadable" or (not fixes and ch["kind"] == "en_passant"):
            flag = True
            if fixes:
                # resynchronisation : on garde les types là où le camp observé est inchangé
                keep = np.sign(state) == np.sign(obs)
                state = np.where(keep, state, obs)
                moved |= ~keep
                for name, (_, pos, _, _) in list(game_dict.items()):
                    if not keep[pos]:
                        del game_dict[name]
                for r, c in np.argwhere(~keep & (obs != 0)):
                    s = int(np.sign(obs[r, c]))
                    game_dict[f"piece{k}"] = (s * 7, (int(r), int(c)), (int(r), int(c)), s)
                    k += 1
            steps.append({"board": state.copy(), "move": "unreadable change", "kind": "unreadable", "flag": True,
                          "never_moved": ~moved & (state != 0)})
            continue
        new = _apply(state, ch)
        if fixes and ch["kind"] == "castle":
            r, side = ch["row"], ch["side"]
            for sq_from, sq_to, val in [((r, 4), (r, ch["king_to"]), 5), ((r, ch["rook_from"]), (r, ch["rook_to"]), 2)]:
                name = logic.get_piece_name(sq_from, game_dict)
                if name is not None:
                    game_dict[name] = (side * val, sq_to, sq_from, side)
                new[sq_to] = side * val
            with contextlib.redirect_stdout(sink):
                state = np.asarray(logic.context_update(new, game_dict, pieces_number), dtype=int)
        else:
            prev_for_logic = state.copy()
            if fixes and ch["kind"] == "en_passant":
                _remove_from_dict(game_dict, ch["taken"])
                prev_for_logic[ch["taken"]] = 0
            if fixes and ch["kind"] == "move" and state[ch["to"]] != 0:
                _remove_from_dict(game_dict, ch["to"])
            with contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext():
                try:
                    updated, flag = logic.pieces_recognition(prev_for_logic, new.copy(), game_dict,
                                                             pieces_number, ch["side"])
                except Exception:       # la logique d'origine lève parfois sur des cas non prévus
                    updated, flag = new, True
            state = np.asarray(updated, dtype=int)
            if fixes and ch["kind"] == "en_passant" and abs(state[ch["to"]]) == 7:
                state[ch["to"]] = ch["side"] * 1          # seule une prise de pion se fait en passant
        state = np.where(np.abs(state) > 7, np.sign(state) * 7, state)
        for sq in ([ch["from"], ch["to"]] if ch["kind"] != "castle"
                   else [(ch["row"], 4), (ch["row"], ch["rook_from"]), (ch["row"], ch["king_to"]),
                         (ch["row"], ch["rook_to"])]):
            moved[sq] = True
        if ch["kind"] == "en_passant":
            moved[ch["taken"]] = True
        steps.append({"board": state.copy(), "move": describe_move(before, state, ch), "kind": ch["kind"],
                      "flag": bool(flag), "never_moved": ~moved & (state != 0)})
    return steps


STANDARD_START = np.array([
    [-2, -3, -4, -6, -5, -4, -3, -2],
    [-1] * 8, [0] * 8, [0] * 8, [0] * 8, [0] * 8, [1] * 8,
    [2, 3, 4, 6, 5, 4, 3, 2],
])


def with_start_inference(board, never_moved):
    """Complète les pièces inconnues qui n'ont jamais bougé par le type de leur case de départ."""
    board = np.asarray(board, dtype=int).copy()
    m = never_moved & (np.abs(board) == 7) & (STANDARD_START != 0) & (np.sign(STANDARD_START) == np.sign(board))
    board[m] = STANDARD_START[m]
    return board


def identification_stats(board, truth):
    """Sur les cases occupées de la vérité : (identifiées, correctes, total)."""
    board = np.asarray(board, dtype=int)
    truth = np.asarray(truth, dtype=int)
    occ = truth != 0
    known = occ & (np.abs(board) >= 1) & (np.abs(board) <= 6)
    correct = known & (board == truth)
    return int(known.sum()), int(correct.sum()), int(occ.sum())


def to_fen_board(board):
    """Partie « placement » d'une FEN ; '?' pour une pièce de type inconnu (non standard)."""
    rows = []
    for r in range(8):
        row, empty = "", 0
        for c in range(8):
            v = int(board[r][c])
            if v == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            letter = PIECE_LETTERS.get(abs(v), "?")
            row += letter if v > 0 else letter.lower()
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows)
