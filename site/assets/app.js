/* Vision d'échiquier, version statique : même page que demo/static/index.html, mais les appels à
 * l'API FastAPI (/api/replay, /api/analyze) sont remplacés par des messages au Web Worker
 * assets/worker.js, qui exécute le même code Python avec Pyodide, dans le navigateur. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const mo = (b) => (b / 1e6).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "\u00a0Mo";
  const secs = (v) => (v / 1000).toLocaleString("fr-FR", { maximumFractionDigits: 1 }) + "\u00a0s";

  // ---------------- Web Worker (Pyodide)
  const hasWasm = typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function" && typeof Worker === "function";
  let worker = null, nextId = 1;
  const pending = new Map();
  const loads = { init: null, vision: null };       // {resolve, reject, promise}
  const timings = {};                               // mesures exposées à la page de test (window.chessTimings)
  window.chessTimings = timings;

  function deferred() { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { resolve, reject, promise }; }

  function onProgress(m) {
    const el = m.phase === "init" ? "py" : "cv";
    $(el + "Load").hidden = false;
    $(el + "LoadBar").max = m.total; $(el + "LoadBar").value = m.loaded;
    $(el + "LoadText").textContent = (m.phase === "init" ? "Chargement de Python dans le navigateur (Pyodide et NumPy) : " : "Chargement du module de vision (OpenCV) : ") +
      `${mo(m.loaded)} sur ${mo(m.total)}`;
  }

  function startWorker() {
    worker = new Worker("assets/worker.js");
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") onProgress(m);
      else if (m.type === "ready") {
        timings[m.phase] = { ms: m.ms, bytes: m.bytes, info: m.info };
        const el = m.phase === "init" ? "py" : "cv";
        $(el + "Load").hidden = false; $(el + "LoadBar").hidden = true;
        $(el + "LoadText").textContent = m.phase === "init"
          ? `Python ${m.info.pyodide ? "(Pyodide " + m.info.pyodide + ", NumPy " + m.info.numpy + ")" : ""} prêt en ${secs(m.ms)}${m.bytes ? ", " + mo(m.bytes) + " chargés (mis en cache pour les visites suivantes)" : ""}.`
          : `OpenCV ${m.info.opencv} prêt en ${secs(m.ms)}${m.bytes ? ", " + mo(m.bytes) + " chargés" : ""}.`;
        loads[m.phase].resolve(m);
      } else if (m.type === "result") {
        const p = pending.get(m.id); pending.delete(m.id);
        if (p) p.resolve(m);
      } else if (m.type === "error") {
        if (m.phase) {
          const el = m.phase === "init" ? "py" : "cv";
          $(el + "LoadText").textContent = `Échec du chargement : ${m.message}`;
          const d = loads[m.phase]; loads[m.phase] = null; if (d) d.reject(new Error(m.message));
        } else {
          const p = pending.get(m.id); pending.delete(m.id);
          if (p) p.reject(new Error(m.message));
        }
      }
    };
    worker.onerror = (e) => { const err = new Error(e.message || "erreur du Web Worker"); for (const d of Object.values(loads)) if (d) d.reject(err); };
  }

  function ensure(phase) {
    if (!hasWasm) return Promise.reject(new Error("WebAssembly n'est pas pris en charge par ce navigateur"));
    if (!worker) startWorker();
    if (!loads[phase]) { loads[phase] = deferred(); worker.postMessage({ type: phase }); }
    return loads[phase].promise;
  }

  // même contrat que fetch("api/…") de la démo API : corps JSON, ou {error} avec un statut
  async function call(msg, phase, transfer) {
    await ensure(phase);
    const id = nextId++;
    const p = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
    worker.postMessage(Object.assign({ id }, msg), transfer || []);
    const r = await p;
    if (r.data && r.data.error) throw new Error(r.data.error);
    r.data.wall_ms = r.wall_ms;
    return r.data;
  }
  window.chessCall = call;          // utilisé par tests/parity.html
  const ms = (v) => `${Number(v).toLocaleString("fr-FR")}\u00a0ms`;
  const GLYPH = { 1: "♟", 2: "♜", 3: "♞", 4: "♝", 5: "♚", 6: "♛" };
  let exButtons = [], seq = null, seqTimer = 0;

  function setStatus(msg, err = false) { $("status").textContent = msg; $("status").classList.toggle("error", err); }

  function renderBoard(el, cellFn) {
    el.innerHTML = "";
    for (let r = 0; r < 8; r++) {
      const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = 8 - r; el.appendChild(lab);
      for (let c = 0; c < 8; c++) {
        const d = document.createElement("div");
        d.className = "sq " + ((r + c) % 2 === 0 ? "l" : "d");
        cellFn(d, r, c);
        el.appendChild(d);
      }
    }
    el.appendChild(document.createElement("div"));
    "abcdefgh".split("").forEach((f) => { const l = document.createElement("div"); l.className = "lab"; l.textContent = f; el.appendChild(l); });
  }

  function pieceInto(d, v) {
    if (!v) return;
    const side = v > 0 ? "w" : "b";
    const a = Math.abs(v);
    if (a >= 1 && a <= 6) {
      const s = document.createElement("span"); s.className = "pc-" + side; s.textContent = GLYPH[a];
      s.setAttribute("aria-label", ["", "pion", "tour", "cavalier", "fou", "roi", "dame"][a] + (a === 2 || a === 6 ? (side === "w" ? " blanche" : " noire") : (side === "w" ? " blanc" : " noir")));
      d.appendChild(s);
    } else {
      const s = document.createElement("span"); s.className = "unk " + side; s.textContent = "?";
      s.setAttribute("aria-label", "pièce " + (side === "w" ? "blanche" : "noire") + ", type inconnu"); d.appendChild(s);
    }
  }

  // ---------------- photo analysis
  function showAnalysis(res, label) {
    $("result").hidden = false;
    $("inputImg").src = res.input_image;
    if (!res.calibrated) {
      $("topImg").removeAttribute("src");
      $("facts").innerHTML = `<span>Calibration <b>échouée</b> (${ms(res.timing_ms.total)})</span>`;
      $("photoBoard").innerHTML = "";
      setStatus(`${label} : ${res.message}`, true);
      return;
    }
    $("topImg").src = res.top_view;
    const st = res.stickers || {};
    $("facts").innerHTML =
      `<span>Cases occupées : <b>${res.occupied}</b> (blanches <b>${res.white}</b>, noires <b>${res.black}</b>)</span>` +
      `<span>Recherche du plateau : <b>${res.method}</b></span>` +
      `<span>Pastilles : rose <b>${st.rose ? "trouvée" : "introuvable"}</b>, bleue <b>${st.blue ? "trouvée" : "introuvable"}</b></span>` +
      `<span>Détecteur : <b>${res.detector}</b></span>` +
      `<span>Temps : <b>${ms(res.timing_ms.total)}</b> (calibration ${ms(res.timing_ms.calibration)})</span>`;
    renderBoard($("photoBoard"), (d, r, c) => { const v = res.board[r][c]; if (v === "W") pieceInto(d, 7); else if (v === "B") pieceInto(d, -7); });
    setStatus(`${label} : plateau trouvé et analysé.`);
  }

  // body : {example, url} (photo d'exemple) ou {file} (photo importée)
  async function analyse(body, label) {
    setStatus(loads.vision ? `Analyse de « ${label} »…` : `Chargement d'OpenCV, puis analyse de « ${label} »…`);
    try {
      const src = body.file ? body.file : await fetch(body.url).then((r) => { if (!r.ok) throw new Error(`photo introuvable (${r.status})`); return r.blob(); });
      const bytes = await src.arrayBuffer();
      const j = await call({ type: "analyze", example: body.example || null, bytes }, "vision", [bytes]);
      showAnalysis(j, label);
    } catch (e) { setStatus(`Impossible d'analyser « ${label} » : ${e.message}`, true); }
  }

  $("file").addEventListener("change", (e) => {
    const f = e.target.files[0]; if (!f) return;
    if (f.size > 12 * 1024 * 1024) { setStatus("Image trop volumineuse : la démo accepte au plus 12 Mo.", true); return; }
    exButtons.forEach((b) => b.setAttribute("aria-pressed", "false"));
    analyse({ file: f }, f.name);
  });

  // ---------------- replay : pièces inconnues
  const FR = { P: "P", N: "C", B: "F", R: "T", Q: "D", K: "R" };
  const FRNAME = { P: "pion", N: "cavalier", B: "fou", R: "tour", Q: "dame", K: "roi" };
  const GLYPHL = { P: "♟", N: "♞", B: "♝", R: "♜", Q: "♛", K: "♚" };
  let req = null, total = 0, firstTurn = "w";
  const sqIdx = (name) => [8 - Number(name[1]), "abcdefgh".indexOf(name[0])];
  const pct0 = (a, b) => b ? Math.round(100 * a / b) + " %" : "–";

  function moveLabel(ply) {
    const k = ply + (firstTurn === "b" ? 1 : 0);
    return `coup ${1 + Math.floor(k / 2)}, trait aux ${k % 2 === 0 ? "Blancs" : "Noirs"}`;
  }

  function setReplayStatus(msg, err = false) { $("replayStatus").textContent = msg; $("replayStatus").classList.toggle("error", err); }

  function markerInto(d, m, k) {
    const disc = document.createElement("span");
    disc.className = "mk " + m.c + (m.certain ? " certain" : "");
    const side = m.c === "w" ? "blanc" : "noir";
    if (m.certain) {
      const g = document.createElement("span"); g.className = "glyph"; g.textContent = GLYPHL[m.certain]; disc.appendChild(g);
      disc.setAttribute("aria-label", `marqueur ${side} n°${m.id} : ${FRNAME[m.certain]}, identifié avec certitude au demi-coup ${m.since}`);
      disc.title = `Marqueur ${side} n°${m.id} : ${FRNAME[m.certain]} (certain depuis le demi-coup ${m.since})`;
      if (m.since === k && k > 0) d.classList.add("fresh");
    } else {
      if (m.mp) {
        const b = document.createElement("span"); b.className = "mp"; b.innerHTML = `${FR[m.mp]}<small>?</small>`; disc.appendChild(b);
      }
      const c = document.createElement("span"); c.className = "cands";
      c.textContent = m.cand.map((t) => FR[t]).join("");
      disc.appendChild(c);
      const prob = m.mp ? ` ; le plus probable : ${FRNAME[m.mp]} (${Math.round(100 * m.p)} %)` : "";
      disc.setAttribute("aria-label", `marqueur ${side} n°${m.id}, types possibles : ${m.cand.map((t) => FRNAME[t]).join(", ")}${prob}`);
      disc.title = `Marqueur ${side} n°${m.id}\nPossibles : ${m.cand.map((t) => FRNAME[t]).join(", ")}${prob}`;
    }
    d.appendChild(disc);
    if ($("showTruth").checked) {
      const t = document.createElement("span"); t.className = "truth"; t.textContent = FR[m.truth]; t.setAttribute("aria-hidden", "true"); d.appendChild(t);
    }
    if (m.certain && m.certain !== m.truth) d.classList.add("wrong");
  }

  function drawChart(k) {
    const st = seq.steps, n = st.length, W = 560, H = 220, L = 38, R = 10, T = 10, B = 40;
    const x = (i) => L + (n > 1 ? i / (n - 1) : 0) * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
    const line = (f) => st.map((s, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(f(s)).toFixed(1)}`).join("");
    let g = "";
    for (const v of [0, .25, .5, .75, 1]) g += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--border)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v * 100} %</text>`;
    const step = Math.max(1, Math.ceil((n - 1) / 8 / 5) * 5);
    for (let i = 0; i < n; i += step) g += `<text x="${x(i)}" y="${H - 12}" text-anchor="middle" font-size="11" fill="var(--muted)">${i}</text>`.replace(`y="${H - 12}"`, `y="${H - B + 16}"`);
    g += `<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle" font-size="11" fill="var(--muted)">demi-coups observés</text>`;
    if (seq.prior) g += `<path d="${line((s) => s.mp_correct / s.present)}" fill="none" stroke="var(--ok)" stroke-width="2" stroke-dasharray="2 3"/>`;
    g += `<path d="${line((s) => s.old_identified / s.present)}" fill="none" stroke="var(--muted)" stroke-width="2" stroke-dasharray="6 4"/>`;
    g += `<path d="${line((s) => s.certain / s.present)}" fill="none" stroke="var(--accent)" stroke-width="3"/>`;
    g += `<line x1="${x(k)}" x2="${x(k)}" y1="${T}" y2="${H - B}" stroke="var(--warn)" stroke-width="1.5"/>`;
    $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Part des marqueurs identifiés avec certitude en fonction du nombre de demi-coups observés">${g}</svg>`;
    $("chartLegend").innerHTML = `<span><i style="background:var(--accent)"></i>identifiées avec certitude</span>` +
      (seq.prior ? `<span><i style="background:var(--ok)"></i>type le plus probable correct</span>` : "") +
      `<span><i style="background:var(--muted)"></i>ancienne logique du dépôt (types déduits, faux compris)</span>`;
  }

  function showStep(k) {
    const s = seq.steps[k];
    $("stepRange").value = k;
    $("prev").disabled = k === 0; $("next").disabled = k === seq.steps.length - 1;
    $("moveText").textContent = k === 0 ? `Début de l'observation (${moveLabel(seq.start_ply)}) : seul le camp de chaque marqueur est connu.`
      : `Demi-coup observé ${k} (${moveLabel(s.ply - 1)}) : ${s.move}` + (s.new.length ? ` · identifié : ${s.new.map((x) => `${x.sq} = ${FRNAME[x.type]}`).join(", ")}` : "");
    const by = {};
    s.markers.forEach((m) => { const [r, c] = sqIdx(m.sq); by[r * 8 + c] = m; });
    const prevS = k > 0 ? seq.steps[k - 1] : null;
    const prevOcc = {};
    if (prevS) prevS.markers.forEach((m) => { prevOcc[m.sq] = true; });
    renderBoard($("dedBoard"), (d, r, c) => {
      const m = by[r * 8 + c];
      const name = "abcdefgh"[c] + (8 - r);
      if (prevS && prevOcc[name] && !m) d.classList.add("from");
      if (m) markerInto(d, m, k);
    });
    const unc = s.present - s.certain;
    $("seqFacts").innerHTML =
      `<span>Marqueurs sur le plateau</span><b>${s.present}</b>` +
      `<span>Identifiés avec certitude</span><b>${s.certain} (${pct0(s.certain, s.present)})</b>` +
      `<span>dont faux</span><b>${s.certain_wrong}</b>` +
      (seq.prior ? `<span>Le plus probable correct (tous)</span><b>${s.mp_correct} / ${s.present}</b>` : `<span>Le plus probable</span><b>a priori indisponible</b>`) +
      `<span>Ancienne logique : types déduits</span><b>${s.old_identified} (dont ${s.old_wrong} faux)</b>` +
      `<span>Temps de calcul de ce demi-coup</span><b>${String(s.ms).replace(".", ",")} ms</b>`;
    drawChart(k);
    const log = $("identLog");
    log.querySelectorAll("li").forEach((li) => li.classList.toggle("cur", +li.dataset.k === k));
  }

  function buildLog() {
    const log = $("identLog"); log.innerHTML = "";
    seq.steps.forEach((s, k) => {
      if (!s.new.length) return;
      const li = document.createElement("li"); li.dataset.k = k;
      let txt = s.new.map((x) => `${x.sq} ${FRNAME[x.type]}`).join(", ");
      if (s.new.length > 4) {
        const cnt = {};
        s.new.forEach((x) => { cnt[x.type] = (cnt[x.type] || 0) + 1; });
        txt = `${s.new.length} marqueurs : ` + Object.entries(cnt).map(([t, n]) => `${n} ${FRNAME[t]}${n > 1 ? "s" : ""}`).join(", ");
      }
      li.textContent = (k === 0 ? "dès le début" : `demi-coup ${k}`) + " : " + txt;
      li.style.cursor = "pointer"; li.addEventListener("click", () => { stopSeq(); showStep(k); });
      log.appendChild(li);
    });
    if (!log.children.length) log.innerHTML = "<li>aucune pour l'instant</li>";
  }

  function stopSeq() { clearInterval(seqTimer); seqTimer = 0; $("playSeq").textContent = "Lecture"; }

  async function runReplay(body, keepStart) {
    stopSeq();
    setReplayStatus(loads.init ? "Calcul du rejeu…" : "Chargement de Python, puis calcul du rejeu…");
    try {
      const j = await call({ type: "replay", req: body }, "init");
      seq = j; req = body; total = j.total_plies; firstTurn = j.first_turn;
      $("startBar").hidden = false;
      $("startRange").max = Math.max(0, total - 1); $("startRange").value = j.start_ply; $("startLabel").textContent = moveLabel(j.start_ply);
      $("seqView").hidden = false; $("stepRange").max = j.steps.length - 1;
      buildLog(); showStep(0);
      setReplayStatus(`${j.label} : ${j.steps.length - 1} demi-coups observés à partir du ${moveLabel(j.start_ply)} (calcul ${ms(j.timing_ms)}).` +
        (j.resets ? ` Observation incohérente : remise à zéro ${j.resets} fois.` : ""));
    } catch (e) { setReplayStatus(`Impossible de rejouer : ${e.message}`, true); }
  }

  $("startRange").addEventListener("input", (e) => { $("startLabel").textContent = moveLabel(+e.target.value); });
  $("startRange").addEventListener("change", (e) => { if (req) runReplay(Object.assign({}, req, { start_ply: +e.target.value })); });
  $("importBtn").addEventListener("click", () => {
    document.querySelectorAll("#gameButtons button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    const pgn = $("pgnIn").value.trim(), fen = $("fenIn").value.trim(), moves = $("movesIn").value.trim();
    if (!pgn && !moves) { setReplayStatus("Collez un PGN, ou une FEN suivie de coups.", true); return; }
    runReplay(pgn ? { pgn, start_ply: 0 } : { fen, moves, start_ply: 0 });
  });
  $("showTruth").addEventListener("change", () => showStep(+$("stepRange").value));
  $("prev").addEventListener("click", () => { stopSeq(); showStep(Math.max(0, +$("stepRange").value - 1)); });
  $("next").addEventListener("click", () => { stopSeq(); showStep(Math.min(seq.steps.length - 1, +$("stepRange").value + 1)); });
  $("stepRange").addEventListener("input", (e) => { stopSeq(); showStep(+e.target.value); });
  $("playSeq").addEventListener("click", () => {
    if (seqTimer) { stopSeq(); return; }
    if (+$("stepRange").value >= seq.steps.length - 1) showStep(0);
    $("playSeq").textContent = "Pause";
    seqTimer = setInterval(() => { const i = +$("stepRange").value + 1; if (i >= seq.steps.length) { stopSeq(); return; } showStep(i); }, 800);
  });

  // ---------------- metrics
  function pct(v) { return v == null ? "–" : (100 * v).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + "\u00a0%"; }
  fetch("assets/metrics.json").then((r) => r.ok ? r.json() : Promise.reject()).then((m) => {
    const ev = m.evaluation, t = ev.results.test, syn = ev.results.test_synthetic_views;
    const best = t.v2_logreg ? "v2_logreg" : "v2_rule";
    const pi = m.piece_identification;
    const rows = [];
    if (pi) {
      const L = pi.lichess_test, starts = pi.protocol.observation_starts_at_move;
      const at = (k, meth) => starts.map((n) => { const v = L[String(n)][meth].curve_summary[String(k)]; return v == null ? "–" : pct(v); }).join(" · ");
      const all = pi.lichess_test_all_starts;
      rows.push([`Pièces identifiées avec certitude après 20 demi-coups observés, observation à partir du coup ${starts.join(" · ")} (${pi.protocol.test_games} parties lichess de test) : nouveau moteur`, at(20, "new")]);
      rows.push([`Idem, ancienne logique du dépôt (types déduits, faux compris)`, at(20, "old")]);
      rows.push([`Idem après 40 demi-coups : nouveau moteur`, at(40, "new")]);
      rows.push([`Idem après 40 demi-coups : ancienne logique`, at(40, "old")]);
      rows.push(["Identifications certaines fausses, tous départs confondus : nouveau moteur", `${all.new.certain_wrong_total.toLocaleString("fr-FR")} sur ${all.new.certain_total.toLocaleString("fr-FR")}`]);
      rows.push(["Types déduits faux, tous départs confondus : ancienne logique", `${all.old.certain_wrong_total.toLocaleString("fr-FR")} sur ${all.old.certain_total.toLocaleString("fr-FR")}`]);
      if (all.new.most_probable_accuracy_on_uncertain != null) rows.push(["Type « le plus probable » correct, pour les marqueurs non identifiés", pct(all.new.most_probable_accuracy_on_uncertain)]);
      rows.push(["Temps de calcul par demi-coup (moyenne, nouveau moteur avec a priori)", `${all.new.ms_per_ply.mean.toLocaleString("fr-FR", { maximumFractionDigits: 1 })}\u00a0ms`]);
      const ann = pi.annotated_games, a1 = Object.keys(ann).map((g) => ann[g].by_start["1"]).filter(Boolean);
      if (a1.length) rows.push(["Les 2 parties annotées du dépôt, observées dès le coup 1 : identifiées avec certitude à la fin (nouveau · ancien)",
        a1.map((x) => `${pct(x.new.curve[x.new.curve.length - 1].certain_share)} · ${pct(x.old.curve[x.old.curve.length - 1].certain_share)}`).join(" ; ")]);
    }
    rows.push(["Vision : calibration du plateau réussie (photos de test)", `${t[best].calibration_success}/${t[best].images}`]);
    rows.push(["Vision : calibration sur " + (syn ? syn.views : "–") + " vues synthétiques des photos de test", syn ? pct(syn.v2_success_rate) : "–"]);
    rows.push(["Vision : exactitude de l'occupation des cases, de bout en bout (test)", pct(t[best].occupancy_accuracy_end_to_end)]);
    rows.push(["Vision : exactitude vide / blanc / noir avec coins annotés (test)", pct(t[best].three_class_accuracy_gt_corners)]);
    $("metricsTable").innerHTML = "<thead><tr><th>Métrique</th><th>Résultat</th></tr></thead><tbody>" +
      rows.map((r) => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td></tr>`).join("") + "</tbody>";
    $("metricsNote").textContent = `Pièces inconnues : parties lichess (licence CC0) jamais vues à l'entraînement de l'a priori, observées pendant au plus 60 demi-coups ; le moteur ne reçoit que l'occupation des cases. Une identification certaine est une déduction logique : il n'y en a eu aucune de fausse. L'ancienne logique supposait les effectifs de départ (8 pions, 2 fous…), d'où ses erreurs quand l'observation commence en cours de partie. Vision : ${t.v1_original.images} photos de test (${t.v1_original.squares_evaluated} cases), petit échantillon, chiffres indicatifs.`;
  }).catch(() => { $("metricsTable").innerHTML = "<tbody><tr><td>Fichier de métriques indisponible.</td></tr></tbody>"; });

  // ---------------- init
  if (!hasWasm) { $("noWasm").hidden = false; }
  fetch("assets/catalog.json").then((r) => r.json()).then((d) => {
    const box = $("examples");
    d.examples.forEach((ex) => {
      const b = document.createElement("button");
      b.className = "ex"; b.type = "button"; b.setAttribute("aria-pressed", "false");
      b.innerHTML = `<img src="${ex.url}" alt="" loading="lazy"><span>${ex.label}</span>`;
      b.title = ex.note;
      b.addEventListener("click", () => {
        exButtons.forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true");
        analyse({ example: ex.id, url: ex.pixels }, ex.label);
      });
      box.appendChild(b); exButtons.push(b);
    });
    const gbox = $("gameButtons");
    d.games.forEach((g, i) => {
      const b = document.createElement("button"); b.className = "btn"; b.type = "button"; b.textContent = g.label; b.title = g.note;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => {
        gbox.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true");
        runReplay({ game: g.id, start_ply: 0 });
      });
      gbox.appendChild(b);
      if (i === 0 && hasWasm) b.click();
    });
    // OpenCV (11,4 Mo) n'est chargé qu'à la première analyse de photo
    setStatus(hasWasm ? "Choisissez une photo : le module de vision (OpenCV, 11,4 Mo) est chargé au premier clic, puis gardé en cache." : "WebAssembly indisponible : analyse impossible dans ce navigateur.", !hasWasm);
    if (!hasWasm) setReplayStatus("WebAssembly indisponible : rejeu impossible dans ce navigateur.", true);
  });
})();
