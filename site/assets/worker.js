/* Vision d'échiquier : Web Worker qui exécute le code Python du dépôt avec Pyodide.
 *
 * Le code Python (src/ copié tel quel dans site/py/, plus les fonctions de demo/app.py extraites
 * par site/sync_site.py) tourne ici, jamais dans le fil de la page. Deux étapes de chargement :
 *  - "init"   : Pyodide + NumPy (rejeu des parties, pur Python + NumPy) et l'a priori piece_prior.json ;
 *  - "vision" : OpenCV (paquet opencv-python de Pyodide) et occupancy_logreg.json, seulement à la
 *               première analyse de photo.
 * Les fichiers du CDN sont gardés dans le Cache API : les visites suivantes ne retéléchargent rien.
 *
 * Messages reçus : {type:"init", noCache?} | {type:"vision"} | {type:"replay", id, req}
 *                  {type:"analyze", id, example?, bytes:ArrayBuffer}
 * Messages émis  : {type:"progress", phase, loaded, total} | {type:"ready", phase, ms, bytes, transferred, info}
 * noCache (page de test seulement) : ignore les caches pour mesurer un chargement à froid.
 *                  {type:"result", id, data, wall_ms} | {type:"error", id?, phase?, message}
 */
"use strict";
const PYODIDE_VERSION = "0.27.7";
const CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const CACHE_NAME = `pyodide-${PYODIDE_VERSION}`;
// Taille décompressée de ce qui est téléchargé à chaque étape (barre de progression seulement) :
// pyodide.asm.wasm + python_stdlib.zip + pyodide.asm.js + pyodide.js + pyodide-lock.json + roue
// NumPy 2.0.2 ; roue opencv-python 4.10.0.84.
const EXPECTED = { init: 16.92e6, vision: 11.44e6 };
const APP = "/home/pyodide/app";
const SITE = new URL("../", self.location.href);

let phase = "init", noCache = false;
const loaded = { init: 0, vision: 0 };
const report = () => postMessage({ type: "progress", phase, loaded: loaded[phase], total: Math.max(EXPECTED[phase], loaded[phase]) });

// fetch() : Cache API pour les fichiers du CDN, et comptage des octets pour la barre de progression.
const netFetch = self.fetch.bind(self);
self.fetch = async (input, init) => {
  const url = input instanceof Request ? input.url : String(input);
  if (!url.startsWith(CDN)) return netFetch(input, init);
  let cache = null, resp = null;
  if (!noCache) try { cache = await caches.open(CACHE_NAME); resp = await cache.match(url); } catch { cache = null; }
  if (!resp) {
    resp = await netFetch(url, noCache ? { cache: "no-store" } : undefined);
    if (resp.ok && cache) cache.put(url, resp.clone()).catch(() => { /* quota dépassé : pas de cache */ });
  }
  if (!resp.body) return resp;
  const reader = resp.body.getReader();
  const ph = phase;
  const counted = new ReadableStream({
    async pull(ctrl) {
      const { done, value } = await reader.read();
      if (done) { ctrl.close(); return; }
      loaded[ph] += value.byteLength; report(); ctrl.enqueue(value);
    },
  });
  // Seul Content-Type est recopié : les autres en-têtes du CDN (Content-Length de la version
  // compressée, gardé par le Cache API) ne décrivent plus ce corps décompressé.
  const headers = { "Content-Type": resp.headers.get("Content-Type") || "application/octet-stream" };
  return new Response(counted, { status: resp.status, statusText: resp.statusText, headers });
};

// Fichiers du site (code Python, parties, poids) : trois essais, une coupure réseau passagère
// ne doit pas faire échouer le chargement.
const siteText = async (path, tries = 3) => {
  for (let k = 1; ; k++) {
    try {
      const r = await netFetch(new URL(path, SITE), { cache: "no-cache" });
      if (!r.ok) throw new Error(`${path} : erreur ${r.status}`);
      return await r.text();
    } catch (err) {
      if (k >= tries) throw err;
      await new Promise((ok) => setTimeout(ok, 300 * k));
    }
  }
};

// pyodide.js et pyodide.asm.js sont chargés par importScripts(), qui ne passe ni par fetch() ni
// par le Cache API : on les télécharge avec fetch() (donc mis en cache et comptés), puis
// importScripts() reçoit une URL blob: à la place de l'URL du CDN.
const blobFor = {};
const cdnImport = self.importScripts.bind(self);
self.importScripts = (...urls) => cdnImport(...urls.map((u) => blobFor[u] || u));
async function preloadScript(name) {
  const r = await self.fetch(CDN + name);
  if (!r.ok) throw new Error(`${name} : erreur ${r.status}`);
  const text = await r.text();
  blobFor[CDN + name] = URL.createObjectURL(new Blob([text], { type: "text/javascript" }));
}

let pyodide = null, web = null, initPromise = null, visionPromise = null;

// octets réellement transférés depuis le CDN (0 si servis par un cache), pour les mesures
const transferred = () => performance.getEntriesByType("resource").filter((r) => r.name.startsWith(CDN))
  .reduce((a, r) => a + (r.transferSize || 0), 0);

async function init() {
  const t0 = performance.now();
  phase = "init"; report();
  await Promise.all([preloadScript("pyodide.js"), preloadScript("pyodide.asm.js")]);
  importScripts(CDN + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: CDN });
  await pyodide.loadPackage(["numpy"]);
  const [files, catalog, prior] = await Promise.all([
    siteText("py/manifest.json").then(JSON.parse), siteText("assets/catalog.json").then(JSON.parse),
    siteText("assets/piece_prior.json").catch(() => null),
  ]);
  pyodide.FS.mkdirTree(APP + "/games");
  await Promise.all(files.map(async (f) => pyodide.FS.writeFile(`${APP}/${f}`, await siteText("py/" + f))));
  await Promise.all(catalog.games.map(async (g) => pyodide.FS.writeFile(`${APP}/games/${g.file}`, await siteText("assets/examples/games/" + g.file))));
  pyodide.runPython(`import sys; sys.path.insert(0, "${APP}")`);
  web = pyodide.pyimport("web_demo");
  const info = JSON.parse(web.init(prior));
  info.pyodide = PYODIDE_VERSION;
  info.numpy = pyodide.runPython("import numpy; numpy.__version__");
  postMessage({ type: "ready", phase: "init", ms: performance.now() - t0, bytes: loaded.init, transferred: transferred(), info });
}

async function vision() {
  initPromise = initPromise || init();
  await initPromise;
  const t0 = performance.now();
  phase = "vision"; report();
  await pyodide.loadPackage(["opencv-python"]);
  const model = await siteText("assets/occupancy_logreg.json").catch(() => null);
  const info = JSON.parse(web.load_vision(model));
  // premier appel OpenCV plus lent (initialisation paresseuse) : payé ici, comme le _warm_up() de la démo API
  const warm = new Uint8Array(await (await netFetch(new URL("assets/examples/frame08.jpg", SITE))).arrayBuffer());
  web.analyze_bytes(warm, "frame08");
  postMessage({ type: "ready", phase: "vision", ms: performance.now() - t0, bytes: loaded.vision, transferred: transferred(), info });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      noCache = !!m.noCache;
      initPromise = initPromise || init();
      await initPromise;
    } else if (m.type === "vision") {
      visionPromise = visionPromise || vision();
      await visionPromise;
    } else if (m.type === "replay") {
      initPromise = initPromise || init();
      await initPromise;
      const t0 = performance.now();
      const data = JSON.parse(web.replay_game(JSON.stringify(m.req)));
      postMessage({ type: "result", id: m.id, data, wall_ms: performance.now() - t0 });
    } else if (m.type === "analyze") {
      visionPromise = visionPromise || vision();
      await visionPromise;
      const t0 = performance.now();
      const data = JSON.parse(web.analyze_bytes(new Uint8Array(m.bytes), m.example || null));
      postMessage({ type: "result", id: m.id, data, wall_ms: performance.now() - t0 });
    }
  } catch (err) {
    const message = String((err && err.message) || err);
    if (m.type === "init" || m.type === "vision") {
      if (m.type === "init") initPromise = null; else visionPromise = null;
      postMessage({ type: "error", phase: m.type, fatal: true, message });
    } else {
      postMessage({ type: "error", id: m.id, message });
    }
  }
};
