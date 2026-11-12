"""Entry points of the static site, called by assets/worker.js (Pyodide, in the visitor's browser).

Same answers as the routes of demo/app.py: the work is done by the functions of demo/app.py
(copied verbatim into demo_glue.py by site/sync_site.py) and by the pipeline modules of src/
(copied unchanged). Every entry point returns a JSON string: the route's body, or
{"error": message, "status": code} like the FastAPI demo.

- init(prior_json)               piece prior (piece_prior.json); replay needs NumPy only
- load_vision(model_json)        imports OpenCV + occupancy + robust_calibration (photo analysis)
- replay_game(req_json)          POST /api/replay
- analyze_bytes(data, example)   POST /api/analyze (image bytes, example photo or upload)
"""

import json

import demo_glue as G
import piece_inference

MODEL_SOURCE = PRIOR_SOURCE = None


def _wrap(fn, *args):
    try:
        return json.dumps(fn(*args))
    except G.HTTPException as exc:
        return json.dumps({"error": exc.detail, "status": exc.status_code})


def init(prior_json=None):
    global PRIOR_SOURCE
    if prior_json:
        G.PRIOR = piece_inference.Prior(json.loads(prior_json))
        PRIOR_SOURCE = "a priori chargé (piece_prior.json)"
    else:
        G.PRIOR = None
        PRIOR_SOURCE = "aucun a priori trouvé : seules les certitudes et les candidats restants sont affichés"
    return json.dumps({"prior": G.PRIOR is not None, "prior_source": PRIOR_SOURCE})


def load_vision(model_json=None):
    global MODEL_SOURCE
    import cv2
    import occupancy
    import robust_calibration
    G.cv2, G.occupancy, G.robust_calibration = cv2, occupancy, robust_calibration
    if model_json:
        G.MODEL = occupancy.LogReg(json.loads(model_json))
        MODEL_SOURCE = "régression logistique chargée (occupancy_logreg.json)"
    else:
        G.MODEL = None
        MODEL_SOURCE = "aucun poids de classifieur trouvé : détecteur à seuil adaptatif utilisé"
    return json.dumps({"opencv": cv2.__version__, "model": G.MODEL is not None, "model_source": MODEL_SOURCE})


def _replay(req):
    positions, moves, label = G.game_positions(req)
    try:
        start = int(req.get("start_ply", 0))
    except (TypeError, ValueError):
        raise G.HTTPException(400, "start_ply doit être un entier.")
    return G.build_replay(positions, moves, start, label)


def replay_game(req_json):
    try:
        req = json.loads(req_json)
    except ValueError:
        return json.dumps({"error": "Corps JSON attendu.", "status": 400})
    return _wrap(_replay, req)


def _analyze(data, example):
    cv2, np = G.cv2, G.np
    if example:
        if example not in {e["id"] for e in G.EXAMPLES}:
            raise G.HTTPException(404, f"Exemple inconnu '{example}'.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    else:
        if len(data) > G.MAX_UPLOAD_BYTES:
            raise G.HTTPException(413, "Image trop volumineuse : la démo accepte au plus 12 Mo.")
        if not data:
            raise G.HTTPException(400, "Fichier vide.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise G.HTTPException(415, "Impossible de décoder l'image (utilisez JPEG ou PNG).")
        if min(img.shape[:2]) < 200:
            raise G.HTTPException(400, "Image trop petite : au moins 200 px de chaque côté.")
    return G.analyze_image(img)


def analyze_bytes(data, example=None):
    """data: bytes, or a Uint8Array passed by the worker (JsProxy)."""
    if G.cv2 is None:
        return json.dumps({"error": "OpenCV n'est pas chargé.", "status": 503})
    if hasattr(data, "to_bytes"):      # JsProxy of a Uint8Array
        data = data.to_bytes()
    return _wrap(_analyze, bytes(data), example or None)
