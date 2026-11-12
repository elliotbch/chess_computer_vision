"""
Minimal chess rules: FEN, legal moves, SAN/PGN parsing (no dependency).

Only used to turn a PGN or a FEN + moves into the sequence of true positions that the
camera would see (and to check the result); the inference itself (`piece_inference.py`)
never reads piece types from here. Written for this project instead of importing
python-chess (GPL-3.0).

Squares are 0..63 with sq = rank * 8 + file (a1 = 0, h8 = 63). Pieces are letters,
upper case for White ('PNBRQK'), lower case for Black, None for an empty square.
"""

import re

FILES = "abcdefgh"
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BISHOP_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]


class ChessError(ValueError):
    pass


def sq_name(sq):
    return FILES[sq % 8] + str(sq // 8 + 1)


def parse_sq(name):
    return (int(name[1]) - 1) * 8 + FILES.index(name[0])


def _white(p):
    return p is not None and p.isupper()


class Position:
    def __init__(self, board, turn="w", castling="KQkq", ep=None):
        self.board = list(board)
        self.turn = turn
        self.castling = castling if castling != "-" else ""
        self.ep = ep

    # ---------------------------------------------------------------- FEN
    @classmethod
    def from_fen(cls, fen):
        parts = fen.strip().split()
        if not parts:
            raise ChessError("FEN vide")
        rows = parts[0].split("/")
        if len(rows) != 8:
            raise ChessError("FEN : il faut 8 rangées")
        board = [None] * 64
        for i, row in enumerate(rows):
            rank, f = 7 - i, 0
            for ch in row:
                if ch.isdigit():
                    f += int(ch)
                elif ch.lower() in "pnbrqk":
                    if f > 7:
                        raise ChessError("FEN : rangée trop longue")
                    board[rank * 8 + f] = ch
                    f += 1
                else:
                    raise ChessError(f"FEN : caractère inattendu « {ch} »")
            if f != 8:
                raise ChessError("FEN : rangée de longueur incorrecte")
        turn = parts[1] if len(parts) > 1 else "w"
        if turn not in ("w", "b"):
            raise ChessError("FEN : trait invalide")
        castling = parts[2] if len(parts) > 2 else "-"
        ep = parse_sq(parts[3]) if len(parts) > 3 and parts[3] != "-" else None
        for color in "Kk":
            if board.count(color) != 1:
                raise ChessError("FEN : il faut exactement un roi par camp")
        return cls(board, turn, castling, ep)

    def fen(self):
        rows = []
        for rank in range(7, -1, -1):
            row, e = "", 0
            for f in range(8):
                p = self.board[rank * 8 + f]
                if p is None:
                    e += 1
                else:
                    row += (str(e) if e else "") + p
                    e = 0
            rows.append(row + (str(e) if e else ""))
        return f"{'/'.join(rows)} {self.turn} {self.castling or '-'} {sq_name(self.ep) if self.ep is not None else '-'} 0 1"

    def copy(self):
        return Position(self.board, self.turn, self.castling, self.ep)

    # ---------------------------------------------------------------- attacks
    def attacked(self, sq, by_white):
        b = self.board
        r, f = divmod(sq, 8)
        pawn = "P" if by_white else "p"
        pr = r - 1 if by_white else r + 1
        if 0 <= pr < 8:
            for df in (-1, 1):
                if 0 <= f + df < 8 and b[pr * 8 + f + df] == pawn:
                    return True
        for dr, df in KNIGHT:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                p = b[rr * 8 + ff]
                if p is not None and p.upper() == "N" and _white(p) == by_white:
                    return True
        for dr, df in KING:
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                p = b[rr * 8 + ff]
                if p is not None and p.upper() == "K" and _white(p) == by_white:
                    return True
        for dirs, kinds in ((ROOK_DIRS, "RQ"), (BISHOP_DIRS, "BQ")):
            for dr, df in dirs:
                rr, ff = r + dr, f + df
                while 0 <= rr < 8 and 0 <= ff < 8:
                    p = b[rr * 8 + ff]
                    if p is not None:
                        if p.upper() in kinds and _white(p) == by_white:
                            return True
                        break
                    rr, ff = rr + dr, ff + df
        return False

    def in_check(self, white):
        return self.attacked(self.board.index("K" if white else "k"), not white)

    # ---------------------------------------------------------------- moves
    def pseudo_moves(self):
        """(from, to, promotion letter or None)."""
        b, white = self.board, self.turn == "w"
        out = []
        for sq, p in enumerate(b):
            if p is None or _white(p) != white:
                continue
            r, f = divmod(sq, 8)
            kind = p.upper()
            if kind == "P":
                d = 1 if white else -1
                last = 7 if white else 0
                to = sq + 8 * d
                if 0 <= to < 64 and b[to] is None:
                    out += self._pawn(sq, to, last)
                    start = 1 if white else 6
                    if r == start and b[to + 8 * d] is None:
                        out.append((sq, to + 8 * d, None))
                for df in (-1, 1):
                    if 0 <= f + df < 8 and 0 <= r + d < 8:
                        t = (r + d) * 8 + f + df
                        if (b[t] is not None and _white(b[t]) != white) or t == self.ep:
                            out += self._pawn(sq, t, last)
            elif kind in "NK":
                for dr, df in (KNIGHT if kind == "N" else KING):
                    rr, ff = r + dr, f + df
                    if 0 <= rr < 8 and 0 <= ff < 8:
                        t = rr * 8 + ff
                        if b[t] is None or _white(b[t]) != white:
                            out.append((sq, t, None))
                if kind == "K":
                    out += self._castles(sq, white)
            else:
                dirs = {"R": ROOK_DIRS, "B": BISHOP_DIRS, "Q": ROOK_DIRS + BISHOP_DIRS}[kind]
                for dr, df in dirs:
                    rr, ff = r + dr, f + df
                    while 0 <= rr < 8 and 0 <= ff < 8:
                        t = rr * 8 + ff
                        if b[t] is None:
                            out.append((sq, t, None))
                        else:
                            if _white(b[t]) != white:
                                out.append((sq, t, None))
                            break
                        rr, ff = rr + dr, ff + df
        return out

    @staticmethod
    def _pawn(sq, to, last):
        if to // 8 == last:
            return [(sq, to, x) for x in "QRBN"]
        return [(sq, to, None)]

    def _castles(self, sq, white):
        b, out = self.board, []
        home = 4 if white else 60
        if sq != home:
            return out
        rights = self.castling
        for flag, rook_sq, empty, safe in (("K", home + 3, [home + 1, home + 2], [home, home + 1, home + 2]),
                                          ("Q", home - 4, [home - 1, home - 2, home - 3], [home, home - 1, home - 2])):
            flag = flag if white else flag.lower()
            if flag not in rights or b[rook_sq] != ("R" if white else "r"):
                continue
            if any(b[s] is not None for s in empty):
                continue
            if any(self.attacked(s, not white) for s in safe):
                continue
            out.append((sq, safe[-1], None))
        return out

    def push(self, move):
        """New position after a move (assumed pseudo-legal)."""
        fr, to, promo = move
        pos = self.copy()
        b = pos.board
        p = b[fr]
        white = _white(p)
        kind = p.upper()
        if kind == "P" and to == self.ep and b[to] is None:
            b[to - 8 if white else to + 8] = None
        if kind == "K" and abs(to - fr) == 2:
            if to > fr:
                b[fr + 1], b[fr + 3] = b[fr + 3], None
            else:
                b[fr - 1], b[fr - 4] = b[fr - 4], None
        b[to] = p if promo is None else (promo if white else promo.lower())
        b[fr] = None
        pos.ep = (fr + to) // 2 if kind == "P" and abs(to - fr) == 16 else None
        rights = pos.castling
        for s, flags in ((4, "KQ"), (60, "kq"), (0, "Q"), (7, "K"), (56, "q"), (63, "k")):
            if fr == s or to == s:
                rights = "".join(c for c in rights if c not in flags)
        pos.castling = rights
        pos.turn = "b" if self.turn == "w" else "w"
        return pos

    def legal_moves(self):
        white = self.turn == "w"
        return [m for m in self.pseudo_moves() if not self.push(m).in_check(white)]

    # ---------------------------------------------------------------- checks
    def validate(self):
        """Raises ChessError if the position cannot occur in a game."""
        for color, pawn, king in (("blanc", "P", "K"), ("noir", "p", "k")):
            if self.board.count(king) != 1:
                raise ChessError(f"il faut exactement un roi {color}")
            mine = [p for p in self.board if p is not None and p.isupper() == (pawn == "P")]
            if len(mine) > 16:
                raise ChessError(f"plus de 16 pièces pour le camp {color}")
            n_p = mine.count(pawn)
            extra = sum(max(0, mine.count(x if pawn == "P" else x.lower()) - n) for x, n in (("Q", 1), ("R", 2), ("N", 2)))
            for light in (0, 1):
                nb = sum(1 for sq, p in enumerate(self.board) if p == ("B" if pawn == "P" else "b") and (sq // 8 + sq % 8) % 2 == light)
                extra += max(0, nb - 1)
            if n_p + extra > 8:
                raise ChessError(f"camp {color} : pions + pièces promues en surplus > 8")
        if any(p is not None and p.upper() == "P" and sq // 8 in (0, 7) for sq, p in enumerate(self.board)):
            raise ChessError("pion sur la 1re ou la 8e rangée")
        if self.in_check(self.turn != "w"):
            raise ChessError("le camp qui n'a pas le trait est en échec")

    # ---------------------------------------------------------------- SAN
    def parse_move(self, token):
        """SAN (Nf3, exd5, O-O, e8=Q) or coordinates (g1f3, e7e8q)."""
        m = re.fullmatch(r"([a-h][1-8])-?([a-h][1-8])([qrbnQRBN])?", token.strip())
        if m:
            fr, to = parse_sq(m.group(1)), parse_sq(m.group(2))
            promo = m.group(3).upper() if m.group(3) else None
            for mv in self.legal_moves():
                if mv[0] == fr and mv[1] == to and (mv[2] == promo or (mv[2] == "Q" and promo is None)):
                    return mv
            raise ChessError(f"coup illégal : {token}")
        return self.parse_san(token)

    def parse_san(self, san):
        s = re.sub(r"[+#!?]+$", "", san.strip())
        s = s.replace("0", "O")
        legal = self.legal_moves()
        if s in ("O-O", "O-O-O"):
            for m in legal:
                p = self.board[m[0]]
                if p.upper() == "K" and abs(m[1] - m[0]) == 2 and ((m[1] > m[0]) == (s == "O-O")):
                    return m
            raise ChessError(f"roque illégal : {san}")
        mt = re.fullmatch(r"([NBRQK])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([NBRQ]))?", s)
        if not mt:
            raise ChessError(f"coup illisible : {san}")
        kind, ffile, frank, dest, promo = mt.groups()
        kind = kind or "P"
        to = parse_sq(dest)
        cands = [m for m in legal if m[1] == to and self.board[m[0]].upper() == kind
                 and (ffile is None or FILES[m[0] % 8] == ffile)
                 and (frank is None or str(m[0] // 8 + 1) == frank)
                 and (m[2] == promo or (m[2] is None and promo is None))]
        if len(cands) != 1:
            raise ChessError(f"coup {'illégal' if not cands else 'ambigu'} : {san}")
        return cands[0]


def parse_pgn_moves(text):
    """List of SAN tokens and the starting FEN if a [FEN "..."] tag is present."""
    fen = None
    m = re.search(r'\[FEN\s+"([^"]+)"\]', text)
    if m:
        fen = m.group(1)
    body = re.sub(r"\[[^\]]*\]", " ", text)
    body = re.sub(r"\{[^}]*\}", " ", body)
    body = re.sub(r";[^\n]*", " ", body)
    while re.search(r"\([^()]*\)", body):
        body = re.sub(r"\([^()]*\)", " ", body)
    body = re.sub(r"\$\d+", " ", body)
    tokens = []
    for tok in body.split():
        tok = re.sub(r"^\d+\.(\.\.)?", "", tok)
        if not tok or tok in ("1-0", "0-1", "1/2-1/2", "*") or re.fullmatch(r"\.+", tok):
            continue
        tokens.append(tok)
    return tokens, fen


def play(moves, fen=None, max_moves=None):
    """Positions [p0, p1, ...] and moves [(from, to, promo), ...] for a list of SAN or coordinate moves."""
    pos = Position.from_fen(fen or START_FEN)
    if fen:
        pos.validate()
    positions, played = [pos], []
    for i, san in enumerate(moves):
        if max_moves is not None and i >= max_moves:
            break
        try:
            m = pos.parse_move(san)
        except ChessError as e:
            raise ChessError(f"demi-coup {i + 1} ({san}) : {e}") from None
        pos = pos.push(m)
        positions.append(pos)
        played.append(m)
    return positions, played


CODES = {"P": 1, "R": 2, "N": 3, "B": 4, "K": 5, "Q": 6}


def to_array(pos):
    """8x8 list, row 0 = rank 8, column 0 = file a, codes of logic.py (+ White, - Black)."""
    out = [[0] * 8 for _ in range(8)]
    for sq, p in enumerate(pos.board):
        if p is not None:
            out[7 - sq // 8][sq % 8] = CODES[p.upper()] * (1 if p.isupper() else -1)
    return out


def from_array(arr):
    """Inverse of `to_array` (8x8, row 0 = rank 8, codes of logic.py)."""
    letters = {v: k for k, v in CODES.items()}
    board = [None] * 64
    for r in range(8):
        for c in range(8):
            v = int(arr[r][c])
            if v:
                board[(7 - r) * 8 + c] = letters[abs(v)] if v > 0 else letters[abs(v)].lower()
    return board


def play_states(states, castling="KQkq"):
    """Positions and moves of a game given as successive annotated states (8x8 arrays, codes of
    logic.py), e.g. the JSON files of archive/step1_annotation: the legal move between two
    states is found; identical successive states are skipped."""
    pos = Position(from_array(states[0]), "w", castling)
    positions, played = [pos], []
    for i, st in enumerate(states[1:]):
        b = from_array(st)
        if b == pos.board:
            continue
        found = [m for m in pos.legal_moves() if pos.push(m).board == b]
        if not found:
            raise ChessError(f"état {i + 1} : aucun coup légal ne mène à cet état")
        pos = pos.push(found[0])
        positions.append(pos)
        played.append(found[0])
    return positions, played
