// Contrôle de fidélité sans navigateur : même code Python (site/py/), même Pyodide 0.27.7, sous Node.
// La vérification de référence reste parity.html (dans le navigateur, via le Web Worker du site).
//
//   npm install --prefix /tmp/pyo pyodide@0.27.7
//   NODE_PATH=/tmp/pyo/node_modules node site/tests/parity_node.mjs
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const HERE = dirname(fileURLToPath(import.meta.url));
const SITE = dirname(HERE);
const { loadPyodide } = require("pyodide");
const { compareReplay, comparePhoto } = require(join(HERE, "compare.js"));

const t0 = performance.now();
const py = await loadPyodide();
await py.loadPackage(["numpy", "opencv-python"], { messageCallback: () => {} });
const APP = "/home/pyodide/app";
py.FS.mkdirTree(APP + "/games");
for (const f of JSON.parse(readFileSync(join(SITE, "py/manifest.json"), "utf8"))) py.FS.writeFile(`${APP}/${f}`, readFileSync(join(SITE, "py", f), "utf8"));
const catalog = JSON.parse(readFileSync(join(SITE, "assets/catalog.json"), "utf8"));
for (const g of catalog.games) py.FS.writeFile(`${APP}/games/${g.file}`, readFileSync(join(SITE, "assets/examples/games", g.file), "utf8"));
py.runPython(`import sys; sys.path.insert(0, "${APP}")`);
const web = py.pyimport("web_demo");
web.init(readFileSync(join(SITE, "assets/piece_prior.json"), "utf8"));
console.log(web.load_vision(readFileSync(join(SITE, "assets/occupancy_logreg.json"), "utf8")));
console.log(`chargement ${(performance.now() - t0).toFixed(0)} ms`);

const ref = JSON.parse(readFileSync(join(HERE, "reference.json"), "utf8"));
let bad = 0;
for (const r of ref.replays) {
  const d = JSON.parse(web.replay_game(JSON.stringify(r.req)));
  const got = d.error ? { status: d.status, body: { error: d.error } } : { status: 200, body: d };
  const c = compareReplay(r, got);
  bad += !c.ok;
  console.log(`${c.ok ? "OK " : "DIFF"} ${r.id.padEnd(24)} ${c.same}/${c.steps} demi-coups identiques, max|Δp| ${c.maxP.toExponential(1)}`, c.ok ? "" : c.diffs.slice(0, 5));
}
for (const p of ref.photos) {
  // comme la page : les pixels exacts de la démo API (PNG sans perte), voir sync_site.py
  const bytes = new Uint8Array(readFileSync(join(SITE, "assets/examples", `${p.id}.png`)));
  const t = performance.now();
  const d = JSON.parse(web.analyze_bytes(bytes, p.id));
  const got = d.error ? { status: d.status, body: { error: d.error } } : { status: 200, body: d };
  const c = comparePhoto(p, got);
  bad += !c.ok;
  console.log(`${c.ok ? "OK " : "DIFF"} ${p.id.padEnd(24)} ${c.sameCells}/${c.cells} cases, max coin ${c.maxCorner.toFixed(2)} px, ${(performance.now() - t).toFixed(0)} ms`, c.ok ? "" : c.diffs.slice(0, 5));
}
console.log(bad ? `${bad} écart(s)` : "fidélité : tout est identique");
process.exit(bad ? 1 : 0);
