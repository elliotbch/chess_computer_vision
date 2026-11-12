/* Comparaison réponse du site (Pyodide) / réponse de la démo FastAPI (reference.json).
 * Utilisé par parity.html (navigateur) et parity_node.mjs (Node).
 *
 * Tolérances (documentées dans le README) :
 *  - rejeu : tout doit être identique (types possibles, certitudes, demi-coup d'identification,
 *    type le plus probable, compteurs, ancienne logique, textes), sauf la probabilité « p » du type
 *    le plus probable : |écart| <= 1e-9 (exp/log de deux bibliothèques mathématiques différentes) ;
 *  - photo : calibration, méthode, pastilles, taille, grille d'occupation 8×8 et compteurs
 *    identiques ; coins à 1 px près (OpenCV 4.10 de Pyodide contre 4.11 de pip, raffinement
 *    sub-pixel) ; l'écart maximal mesuré est affiché.
 */
(function (root) {
  "use strict";
  const P_TOL = 1e-9, CORNER_TOL = 1.0;

  function diffs(a, b, path, out, ctx) {
    if (typeof a === "number" && typeof b === "number") {
      if (path.endsWith(".p")) { const d = Math.abs(a - b); ctx.maxP = Math.max(ctx.maxP, d); if (d > P_TOL) out.push(`${path}: ${a} ≠ ${b}`); }
      else if (a !== b) out.push(`${path}: ${a} ≠ ${b}`);
      return;
    }
    if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
      if (a !== b) out.push(`${path}: ${JSON.stringify(a)} ≠ ${JSON.stringify(b)}`);
      return;
    }
    if (Array.isArray(a) !== Array.isArray(b)) { out.push(`${path}: type`); return; }
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) diffs(a[k], b[k], `${path}.${k}`, out, ctx);
  }

  function stripReplay(d) {
    const c = Object.assign({}, d); delete c.timing_ms;
    if (c.steps) c.steps = c.steps.map((s) => { const t = Object.assign({}, s); delete t.ms; return t; });
    return c;
  }

  // ref : entrée de reference.json ; got : {status, body} du site
  function compareReplay(ref, got) {
    const out = [], ctx = { maxP: 0 };
    if (ref.status !== got.status) out.push(`statut ${got.status} ≠ ${ref.status}`);
    let steps = 0, same = 0;
    if (ref.status === 200 && got.status === 200) {
      const a = stripReplay(got.body), b = ref.body;
      const { steps: sa, ...ha } = a, { steps: sb, ...hb } = b;
      diffs(ha, hb, "", out, ctx);
      steps = sb.length;
      if (sa.length !== sb.length) out.push(`${sa.length} demi-coups ≠ ${sb.length}`);
      for (let i = 0; i < Math.min(sa.length, sb.length); i++) {
        const o = []; diffs(sa[i], sb[i], `steps[${i}]`, o, ctx);
        if (!o.length) same++; out.push(...o);
      }
    } else diffs(got.body, ref.body, "", out, ctx);
    return { ok: out.length === 0, steps, same, maxP: ctx.maxP, diffs: out.slice(0, 20) };
  }

  function comparePhoto(ref, got) {
    const out = [], ctx = { maxP: 0 };
    if (ref.status !== got.status) out.push(`statut ${got.status} ≠ ${ref.status}`);
    const a = got.body || {}, b = ref.body;
    let maxCorner = 0;
    for (const k of Object.keys(b)) {
      if (k === "corners") {
        for (const n of Object.keys(b.corners)) {
          const p = (a.corners || {})[n] || [NaN, NaN], q = b.corners[n];
          const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
          maxCorner = Math.max(maxCorner, d);
          if (!(d <= CORNER_TOL)) out.push(`coin ${n}: ${p} ≠ ${q}`);
        }
      } else diffs(a[k], b[k], k, out, ctx);
    }
    let cells = 0, sameCells = 0;
    if (b.occupancy && a.occupancy) b.occupancy.forEach((row, f) => [...row].forEach((v, r) => { cells++; sameCells += (a.occupancy[f] || [])[r] === v; }));
    return { ok: out.length === 0, cells, sameCells, maxCorner, diffs: out.slice(0, 20) };
  }

  const api = { compareReplay, comparePhoto, P_TOL, CORNER_TOL };
  if (typeof module === "object" && module.exports) module.exports = api; else root.ParityCompare = api;
})(typeof self !== "undefined" ? self : this);
