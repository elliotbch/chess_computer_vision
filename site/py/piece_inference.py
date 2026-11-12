"""
Identification incrémentale des pièces inconnues (marqueurs identiques par camp) à partir des
seuls déplacements observés.

Entrée : une suite d'occupations du plateau (pour chaque case : marqueur blanc, marqueur noir ou
vide), telles que les produit la vision. Le système ne connaît ni les types, ni la position
initiale, ni les droits de roque, ni l'historique : l'observation peut commencer à n'importe quel
coup. Il ne se sert jamais de la case de départ d'une pièce.

Chaque marqueur (et, après une promotion possible, chaque « segment de vie » d'un marqueur) est une
variable d'un problème de satisfaction de contraintes (CSP) dont le domaine est l'ensemble des
types encore possibles. Les contraintes, toutes valides dans n'importe quelle position légale :

- géométrie de chaque coup observé (cavalier, roi, glissement orthogonal ou diagonal sans saut,
  pion : avance d'une case, double pas depuis sa 2e rangée, prise en diagonale ; prise en passant ;
  roque = roi + tour) et promotion (un pion qui atteint la dernière rangée change de type) ;
- jamais de pion sur la 1re ou la 8e rangée ;
- dénombrement par camp : exactement un roi, au plus 16 pièces, pions + pièces en surplus
  (dames au-delà de 1, tours et cavaliers au-delà de 2, fous au-delà de 1 PAR COULEUR DE CASE)
  <= 8 ; si les 32 marqueurs sont encore là, aucune prise n'a eu lieu, donc aucune promotion ;
- légalité : après son coup, le camp qui vient de jouer n'est pas en échec (rois non adjacents
  compris) ; avant un roque, le roi n'est pas en échec et ne traverse pas de case attaquée ; une
  pièce prise n'est jamais le roi.

Propagation : cohérence d'arc sur les contraintes binaires (« si y est le roi, x n'est pas d'un
type qui l'attaquerait »), cohérence d'arc généralisée exacte sur la contrainte de dénombrement
(couplage biparti marqueurs / emplacements {R, D, T, T, C, C, F clair, F foncé}). Chaque retrait
de type est prouvé : une pièce dont le domaine est réduit à un seul type est identifiée AVEC
CERTITUDE. La propagation n'est pas complète (certains types restent possibles alors qu'aucune
partie ne les admet), ce qui peut retarder une identification, jamais la fausser.

Optionnellement, un a priori appris (`Prior`) donne une probabilité par type restant, pour
proposer « le plus probable » ; il n'intervient jamais dans les certitudes.

Cases : 0..63, sq = rangée * 8 + colonne (a1 = 0, h8 = 63). Camps : +1 blanc, -1 noir.
"""

import json
import math
import time

P, N, B, R, Q, K = range(6)
TYPES = "PNBRQK"
NAMES_FR = ["pion", "cavalier", "fou", "tour", "dame", "roi"]
ALL = 0b111111
BIT = [1 << t for t in range(6)]
QRBN = BIT[Q] | BIT[R] | BIT[B] | BIT[N]
NON_KING = ALL & ~BIT[K]

# emplacements « gratuits » du dénombrement : roi, dame, 2 tours, 2 cavaliers, fou clair, fou foncé
SLOT_K, SLOT_Q, SLOT_R1, SLOT_R2, SLOT_N1, SLOT_N2, SLOT_BL, SLOT_BD = range(8)
SLOT_TYPE = [K, Q, R, R, N, N, B, B]


def _sq(r, f):
    return r * 8 + f


def _build_tables():
    knight, king = [0] * 64, [0] * 64
    between = [[0] * 64 for _ in range(64)]
    line = [[0] * 64 for _ in range(64)]          # 0 non alignées, 1 orthogonal, 2 diagonal
    for a in range(64):
        ar, af = divmod(a, 8)
        for dr, df in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
            if 0 <= ar + dr < 8 and 0 <= af + df < 8:
                knight[a] |= 1 << _sq(ar + dr, af + df)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if (dr or df) and 0 <= ar + dr < 8 and 0 <= af + df < 8:
                    king[a] |= 1 << _sq(ar + dr, af + df)
        for dr, df, kind in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1), (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2)):
            r, f, path = ar + dr, af + df, 0
            while 0 <= r < 8 and 0 <= f < 8:
                b = _sq(r, f)
                line[a][b] = kind
                between[a][b] = path
                path |= 1 << b
                r, f = r + dr, f + df
    return knight, king, between, line


KNIGHT, KING, BETWEEN, LINE = _build_tables()


def light(sq):
    """Case claire ? (a1 est foncée)."""
    return (sq // 8 + sq % 8) % 2 == 1


def attack_types(xsq, ysq, xcolor, occ_bits):
    """Masque des types qu'une pièce du camp `xcolor` en `xsq` devrait avoir pour attaquer `ysq`
    (l'occupation suffit : les pièces qui bloquent sont visibles, quel que soit leur type)."""
    m = 0
    yb = 1 << ysq
    if KNIGHT[xsq] & yb:
        m |= BIT[N]
    if KING[xsq] & yb:
        m |= BIT[K]
        xr, xf = divmod(xsq, 8)
        yr, yf = divmod(ysq, 8)
        if yr - xr == xcolor and abs(yf - xf) == 1:
            m |= BIT[P]
    ln = LINE[xsq][ysq]
    if ln and not (BETWEEN[xsq][ysq] & occ_bits):
        m |= (BIT[R] | BIT[Q]) if ln == 1 else (BIT[B] | BIT[Q])
    return m


def move_types(color, frm, to, occ_bits, capture):
    """Masque des types capables du déplacement frm -> to (coup ordinaire, hors roque et prise en passant)."""
    m = 0
    tb = 1 << to
    if KNIGHT[frm] & tb:
        m |= BIT[N]
    if KING[frm] & tb:
        m |= BIT[K]
    ln = LINE[frm][to]
    if ln and not (BETWEEN[frm][to] & occ_bits):
        m |= (BIT[R] | BIT[Q]) if ln == 1 else (BIT[B] | BIT[Q])
    fr, ff = divmod(frm, 8)
    tr, tf = divmod(to, 8)
    dr = (tr - fr) * color
    if capture:
        if dr == 1 and abs(tf - ff) == 1:
            m |= BIT[P]
    elif tf == ff:
        if dr == 1:
            m |= BIT[P]
        elif dr == 2 and fr == (1 if color > 0 else 6) and not (BETWEEN[frm][to] & occ_bits):
            m |= BIT[P]
    return m


def last_rank(color):
    return 7 if color > 0 else 0


def mask_types(m):
    return [t for t in range(6) if m >> t & 1]


class Contradiction(Exception):
    """Observation incompatible avec les règles (erreur de vision, coup manqué)."""


class Var:
    __slots__ = ("id", "color", "dom", "sq", "alive", "born", "parent", "identified_ply", "moves", "label")

    def __init__(self, vid, color, dom, sq, born, parent=None, label=None):
        self.id, self.color, self.dom, self.sq = vid, color, dom, sq
        self.alive, self.born, self.parent = True, born, parent
        self.identified_ply = born if dom & (dom - 1) == 0 else None
        self.moves = 0
        self.label = label


class PieceInference:
    """Moteur incrémental. Usage :

        eng = PieceInference(occupation_initiale)       # 64 valeurs dans {+1, -1, 0}
        info = eng.observe(occupation_suivante)          # après chaque coup observé
        eng.markers()                                    # état de chaque marqueur
    """

    def __init__(self, occupancy, prior=None):
        self.prior = prior
        self.ply = 0
        self.resets = 0
        self.events = []
        self._start(list(occupancy))

    # ------------------------------------------------------------------ état
    def _start(self, occ):
        if len(occ) != 64:
            raise ValueError("l'occupation doit compter 64 cases")
        self.occ = [int((v > 0) - (v < 0)) for v in occ]
        self.vars = []
        self.at = {}
        self.clauses = []            # (y, x, T) : y roi  =>  type(x) hors de T
        self.links = []              # (w, z) : z = w après une promotion possible
        self.snapshots = []          # dénombrements passés : [ids des marqueurs d'un camp, surplus permis, cache]
        self.first_move_seen = False
        self.next_label = getattr(self, "next_label", 1)
        for color in (1, -1):
            if sum(1 for v in self.occ if v == color) > 16:
                raise Contradiction("plus de 16 marqueurs pour un camp")
        for sq, c in enumerate(self.occ):
            if c:
                dom = ALL & ~BIT[P] if sq // 8 in (0, 7) else ALL
                self._new_var(c, dom, sq, label=self._label())
        self._propagate()

    def _label(self):
        lab = self.next_label
        self.next_label += 1
        return lab

    def _new_var(self, color, dom, sq, parent=None, label=None):
        v = Var(len(self.vars), color, dom, sq, self.ply, parent, label)
        self.vars.append(v)
        self.at[sq] = v.id
        return v

    def occ_bits(self, occ=None):
        occ = self.occ if occ is None else occ
        b = 0
        for sq, c in enumerate(occ):
            if c:
                b |= 1 << sq
        return b

    # ------------------------------------------------------------------ observation
    def classify(self, new):
        """Nature du changement entre l'occupation courante et `new`."""
        prev = self.occ
        vac = [s for s in range(64) if prev[s] and not new[s]]
        dst = [s for s in range(64) if new[s] and new[s] != prev[s]]
        if not vac and not dst:
            return {"kind": "none"}
        if len(vac) == 1 and len(dst) == 1 and prev[vac[0]] == new[dst[0]]:
            return {"kind": "move", "from": vac[0], "to": dst[0], "color": new[dst[0]], "capture": bool(prev[dst[0]])}
        if len(vac) == 2 and len(dst) == 2:
            c = prev[vac[0]]
            rank = last_rank(-c)
            if all(prev[s] == c for s in vac) and all(new[s] == c and not prev[s] for s in dst) \
                    and all(s // 8 == rank for s in vac + dst):
                kf = rank * 8 + 4
                files_v = sorted(s % 8 for s in vac)
                files_d = sorted(s % 8 for s in dst)
                if files_v == [4, 7] and files_d == [5, 6]:
                    return {"kind": "castle", "color": c, "king": (kf, kf + 2), "rook": (kf + 3, kf + 1)}
                if files_v == [0, 4] and files_d == [2, 3]:
                    return {"kind": "castle", "color": c, "king": (kf, kf - 2), "rook": (kf - 4, kf - 1)}
        if len(vac) == 2 and len(dst) == 1 and not prev[dst[0]]:
            d = dst[0]
            c = new[d]
            movers = [s for s in vac if prev[s] == c]
            taken = [s for s in vac if prev[s] == -c]
            if len(movers) == 1 and len(taken) == 1:
                m, t = movers[0], taken[0]
                mr, mf = divmod(m, 8)
                dr, df = divmod(d, 8)
                if mr == (4 if c > 0 else 3) and dr - mr == c and abs(df - mf) == 1 and t == mr * 8 + df:
                    return {"kind": "en_passant", "from": m, "to": d, "taken": t, "color": c}
        return {"kind": "unreadable"}

    def observe(self, new_occ):
        """Intègre une nouvelle occupation. Renvoie {kind, ..., newly_identified, ms}."""
        t0 = time.perf_counter()
        new = [int((v > 0) - (v < 0)) for v in new_occ]
        if len(new) != 64:
            raise ValueError("l'occupation doit compter 64 cases")
        ch = self.classify(new)
        before = {v.id for v in self.vars if v.alive and v.identified_ply is not None}
        if ch["kind"] == "none":
            ch["newly_identified"] = []
            ch["ms"] = 1000 * (time.perf_counter() - t0)
            return ch
        self.ply += 1
        if ch["kind"] == "unreadable":
            self._reset(new, "changement illisible")
        else:
            try:
                self._apply(ch, new)
                self._propagate()
            except Contradiction as exc:
                self._reset(new, str(exc))
                ch["contradiction"] = str(exc)
        ch["newly_identified"] = [v.id for v in self.vars
                                  if v.alive and v.identified_ply is not None and v.id not in before]
        ch["ms"] = 1000 * (time.perf_counter() - t0)
        self.events.append(ch)
        return ch

    def _reset(self, new, why):
        """Observation incohérente : on repart de zéro (on perd l'acquis mais rien de faux)."""
        self.resets += 1
        labels = {sq: self.vars[vid].label for sq, vid in self.at.items()}
        self._start(new)
        for sq, vid in self.at.items():     # garde les numéros des marqueurs restés en place (affichage)
            if sq in labels:
                self.vars[vid].label = labels[sq]

    def _restrict(self, v, mask):
        nd = v.dom & mask
        if nd == v.dom:
            return False
        if nd == 0:
            raise Contradiction(f"aucun type possible pour le marqueur de {sqname(v.sq)}")
        v.dom = nd
        if nd & (nd - 1) == 0 and v.identified_ply is None:
            v.identified_ply = self.ply
        self._dirty = True
        return True

    def _apply(self, ch, new):
        occ_before = self.occ
        bits_before = self.occ_bits(occ_before)
        c = ch["color"]
        if ch["kind"] == "en_passant" or ch.get("capture") or (
                ch["kind"] == "move" and ch["to"] // 8 == last_rank(c) and self.vars[self.at[ch["from"]]].dom & BIT[P]):
            # l'ensemble des marqueurs va changer : le dénombrement de l'instant présent reste vrai
            # pour ces variables (le type d'une variable ne change pas), on le garde
            total = sum(1 for v in self.vars if v.alive)
            for color in (1, -1):
                ids = tuple(v.id for v in self.vars if v.alive and v.color == color)
                if not any(sn[0] == ids for sn in self.snapshots):
                    self.snapshots.append([ids, total < 32, None])
        if not self.first_move_seen:
            self.first_move_seen = True
            self._not_in_check(-c, occ_before)          # le camp qui n'a pas le trait n'est pas en échec
        if ch["kind"] == "castle":
            kv = self.vars[self.at[ch["king"][0]]]
            rv = self.vars[self.at[ch["rook"][0]]]
            self._restrict(kv, BIT[K])
            self._restrict(rv, BIT[R])
            # roi pas en échec, ne traverse pas de case attaquée (l'arrivée est vérifiée après le coup)
            for s in (ch["king"][0], (ch["king"][0] + ch["king"][1]) // 2):
                for sq, vid in self.at.items():
                    if occ_before[sq] == -c:
                        self._restrict(self.vars[vid], ~attack_types(sq, s, -c, bits_before))
            for (a, b), v in ((ch["king"], kv), (ch["rook"], rv)):
                del self.at[a]
            for (a, b), v in ((ch["king"], kv), (ch["rook"], rv)):
                v.sq = b
                v.moves += 1
                self.at[b] = v.id
        else:
            frm, to = ch["from"], ch["to"]
            mv = self.vars[self.at[frm]]
            if ch["kind"] == "en_passant":
                tv = self.vars[self.at.pop(ch["taken"])]
                self._restrict(mv, BIT[P])
                self._restrict(tv, BIT[P])
                tv.alive = False
            else:
                self._restrict(mv, move_types(c, frm, to, bits_before, ch["capture"]))
                if ch["capture"]:
                    tv = self.vars[self.at[to]]
                    self._restrict(tv, NON_KING)            # on ne prend jamais le roi
                    tv.alive = False
            del self.at[frm]
            mv.moves += 1
            if to // 8 == last_rank(c) and mv.dom & BIT[P]:
                # promotion possible : nouveau segment de vie pour ce marqueur
                mv.alive = False
                mv.sq = to
                nd = (mv.dom & ~BIT[P]) | QRBN
                z = Var(len(self.vars), c, nd, to, self.ply, parent=mv.id, label=mv.label)
                z.moves = mv.moves
                self.vars.append(z)
                self.at[to] = z.id
                self.links.append((mv.id, z.id))
                ch["promotion_possible"] = True
            else:
                mv.sq = to
                self.at[to] = mv.id
        self.occ = new
        # cohérence : les marqueurs suivis doivent correspondre à l'occupation observée
        for sq, vid in self.at.items():
            if new[sq] != self.vars[vid].color:
                raise Contradiction("suivi des marqueurs incohérent")
        self._not_in_check(c, new)

    def _not_in_check(self, color, occ):
        """Le camp `color` n'est pas en échec dans l'occupation `occ` (cases -> vars courantes)."""
        bits = self.occ_bits(occ)
        kings = [self.vars[vid] for sq, vid in self.at.items() if occ[sq] == color and self.vars[vid].dom & BIT[K]]
        opp = [(sq, self.vars[vid]) for sq, vid in self.at.items() if occ[sq] == -color]
        for y in kings:
            for sq, x in opp:
                T = attack_types(sq, y.sq, -color, bits)
                if T & x.dom:
                    self.clauses.append((y.id, x.id, T))

    # ------------------------------------------------------------------ propagation
    def _propagate(self):
        self._dirty = True
        rounds = 0
        while self._dirty:
            self._dirty = False
            rounds += 1
            keep = []
            for y_id, x_id, T in self.clauses:
                y, x = self.vars[y_id], self.vars[x_id]
                if not (y.dom & BIT[K]) or not (x.dom & T):
                    continue                     # satisfaite pour toujours (les domaines ne font que décroître)
                if y.dom == BIT[K]:
                    self._restrict(x, ~T)
                    continue
                if not (x.dom & ~T):
                    self._restrict(y, ~BIT[K])
                    continue
                keep.append((y_id, x_id, T))
            self.clauses = keep
            for w_id, z_id in self.links:
                w, z = self.vars[w_id], self.vars[z_id]
                allowed_z = (w.dom & ~BIT[P]) | (QRBN if w.dom & BIT[P] else 0)
                self._restrict(z, allowed_z)
                allowed_w = (z.dom & ~BIT[P]) | (BIT[P] if z.dom & QRBN else 0)
                self._restrict(w, allowed_w)
            total = sum(1 for v in self.vars if v.alive)
            for color in (1, -1):
                self._count_gac([v for v in self.vars if v.alive and v.color == color], excess=total < 32)
            for sn in self.snapshots:
                vs = [self.vars[i] for i in sn[0]]
                key = tuple(v.dom for v in vs)
                if key != sn[2]:
                    self._count_gac(vs, excess=sn[1])
                    sn[2] = tuple(v.dom for v in vs)

    def _count_gac(self, vs, excess):
        n = len(vs)
        if n > 16:
            raise Contradiction("plus de 16 marqueurs pour un camp")
        if n == 0:
            raise Contradiction("aucun marqueur pour un camp (il faut un roi)")
        free_types = NON_KING if excess else BIT[P]
        doms = [v.dom for v in vs]
        sol = _feasible(doms, [light(v.sq) for v in vs], free_types)
        if sol is None:
            raise Contradiction("dénombrement impossible (roi, pions, pièces en surplus)")
        supported = [0] * n
        _mark(sol, doms, free_types, supported)
        lights = [light(v.sq) for v in vs]
        for i in range(n):
            for t in range(6):
                if doms[i] >> t & 1 and not supported[i] >> t & 1:
                    d2 = list(doms)
                    d2[i] = BIT[t]
                    s2 = _feasible(d2, lights, free_types)
                    if s2 is None:
                        continue
                    _mark(s2, d2, free_types, supported)
        for v, s in zip(vs, supported):
            self._restrict(v, s)

    # ------------------------------------------------------------------ lecture
    def markers(self):
        """Liste des marqueurs présents avec types possibles, certitude et probabilités."""
        alive = [v for v in self.vars if v.alive]
        probs = self.prior.probabilities(self, alive) if self.prior is not None else None
        best = most_probable(alive, probs) if probs is not None else None
        out = []
        for i, v in enumerate(alive):
            cand = mask_types(v.dom)
            d = {"id": v.id, "label": v.label, "square": sqname(v.sq), "sq": v.sq, "color": v.color,
                 "candidates": [TYPES[t] for t in cand],
                 "certain": TYPES[cand[0]] if len(cand) == 1 else None,
                 "identified_ply": v.identified_ply}
            if probs is not None:
                p = probs[i]
                d["probs"] = {TYPES[t]: round(float(p[t]), 4) for t in cand}
                d["most_probable"] = TYPES[best[i]]
            else:
                d["most_probable"] = d["certain"]
            out.append(d)
        return out


def _slots_of(dom, is_light):
    s = []
    if dom & BIT[K]:
        s.append(SLOT_K)
    if dom & BIT[Q]:
        s.append(SLOT_Q)
    if dom & BIT[R]:
        s += [SLOT_R1, SLOT_R2]
    if dom & BIT[N]:
        s += [SLOT_N1, SLOT_N2]
    if dom & BIT[B]:
        s.append(SLOT_BL if is_light else SLOT_BD)
    return s


def _feasible(doms, lights, free_types):
    """Couplage maximal marqueurs -> emplacements couvrant le roi et tous les marqueurs qui ne
    peuvent pas rester hors emplacement ; faisable si (marqueurs hors emplacement) <= 8.
    Renvoie (var_slot) ou None."""
    n = len(doms)
    slots = [_slots_of(d, l) for d, l in zip(doms, lights)]
    owner = [-1] * 8
    vslot = [-1] * n

    def try_var(v, seen):
        for s in slots[v]:
            if not seen[s]:
                seen[s] = True
                if owner[s] == -1 or try_var(owner[s], seen):
                    owner[s] = v
                    vslot[v] = s
                    return True
        return False

    def try_slot(s, seen):
        for v in range(n):
            if not seen[v] and s in slots[v]:
                seen[v] = True
                if vslot[v] == -1 or try_slot(vslot[v], seen):
                    vslot[v] = s
                    owner[s] = v
                    return True
        return False

    mandatory = [i for i in range(n) if not doms[i] & free_types]
    for i in mandatory:
        if not try_var(i, [False] * 8):
            return None
    if owner[SLOT_K] == -1 and not try_slot(SLOT_K, [False] * n):
        return None
    for i in range(n):
        if vslot[i] == -1 and slots[i]:
            try_var(i, [False] * 8)
    unmatched = sum(1 for s in vslot if s == -1)
    if unmatched > 8:
        return None
    return vslot


def _mark(vslot, doms, free_types, supported):
    for i, s in enumerate(vslot):
        if s == -1:
            supported[i] |= doms[i] & free_types
        else:
            supported[i] |= BIT[SLOT_TYPE[s]]


def sqname(sq):
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


# ---------------------------------------------------------------------------- a priori appris

class Prior:
    """Régression logistique multinomiale (poids JSON, inférence NumPy-free) donnant P(type | indices).

    Indices d'un marqueur : son domaine courant (64 valeurs possibles), sa rangée vue de son camp,
    sa colonne, le nombre de ses déplacements observés, le nombre de marqueurs de chaque camp.
    Les probabilités sont renormalisées sur les types encore possibles ; `most_probable` attribue
    le roi au candidat le plus probable de chaque camp. Ne sert qu'à « le plus probable »."""

    def __init__(self, weights):
        self.w = weights["coef"]          # 6 x F
        self.b = weights["intercept"]     # 6
        self.meta = weights.get("meta", {})

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    @staticmethod
    def features(v, n_own, n_opp):
        """Indices actifs (one-hot) d'un marqueur ; même code à l'entraînement et à l'inférence."""
        rel_rank = v.sq // 8 if v.color > 0 else 7 - v.sq // 8
        f = v.sq % 8
        mv = min(v.moves, 4)
        return [v.dom,                                  # 0..63
                64 + rel_rank * 8 + f,                  # 64..127 case vue de son camp
                128 + mv,                               # 128..132
                133 + min(n_own, 16),                   # 133..149
                150 + min(n_opp, 16),                   # 150..166
                167 + (v.dom & 63 if mv == 0 else 64 + (v.dom & 63))]   # 167..294 domaine x a bougé
    N_FEATURES = 295

    def scores(self, idx):
        return [self.b[t] + sum(self.w[t][i] for i in idx) for t in range(6)]

    def probabilities(self, eng, alive):
        counts = {1: 0, -1: 0}
        for v in alive:
            counts[v.color] += 1
        out = []
        for v in alive:
            s = self.scores(self.features(v, counts[v.color], counts[-v.color]))
            cand = mask_types(v.dom)
            mx = max(s[t] for t in cand)
            e = [math.exp(s[t] - mx) if v.dom >> t & 1 else 0.0 for t in range(6)]
            z = sum(e)
            out.append([x / z for x in e])
        return out


def most_probable(alive, probs):
    """Type le plus probable de chaque marqueur, avec exactement un roi par camp (le candidat
    dont la probabilité d'être roi est la plus forte)."""
    out = [None] * len(alive)
    for color in (1, -1):
        idx = [i for i, v in enumerate(alive) if v.color == color]
        kc = [i for i in idx if alive[i].dom & BIT[K]]
        king = max(kc, key=lambda i: probs[i][K]) if kc else None
        for i in idx:
            cand = mask_types(alive[i].dom)
            if i == king:
                out[i] = K
            else:
                nk = [t for t in cand if t != K] or cand
                out[i] = max(nk, key=lambda t: probs[i][t])
    return out


# ---------------------------------------------------------------------------- utilitaires

def occupancy_from_board8(board):
    """Grille 8x8 (ligne 0 = rangée 8, colonne 0 = colonne a ; >0 blanc, <0 noir) -> 64 valeurs."""
    occ = [0] * 64
    for r in range(8):
        for c in range(8):
            v = board[r][c]
            occ[(7 - r) * 8 + c] = (v > 0) - (v < 0)
    return occ


def run(occupancies, prior=None):
    """Rejoue une suite d'occupations ; renvoie le moteur et les événements."""
    eng = PieceInference(occupancies[0], prior)
    for occ in occupancies[1:]:
        eng.observe(occ)
    return eng
