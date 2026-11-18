"""
PathSnap API (runs on Laptop 1).

Run from the repo root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET /health              -> status, loaded model
    GET /trips               -> list of evaluation trip IDs
    GET /predict/{trip_id}   -> predicted route + accuracy for that trip

Ground truth (.route) is read only in src/scoring.py, after prediction,
and is never sent to the UI.
"""
import os
import sys
import json
import time
import logging

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import pipeline            # noqa: E402
import scoring             # noqa: E402

# ----------------------------------------------------------------------------- config
CONFIG_PATH = os.path.join(ROOT, "api", "config.json")
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)
DATA_DIR = os.path.join(ROOT, CONFIG["data_dir"])
MODEL_PATH = os.path.join(ROOT, CONFIG["model_path"])
USE_HMM = CONFIG.get("use_hmm", False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

log.info(f"Loading model from {MODEL_PATH}")
MODEL = joblib.load(MODEL_PATH)

DECODER = None
if USE_HMM:
    import hmm                                      # src/hmm.py (added later)
    DECODER = hmm.viterbi_decoder
    log.info("Decoder: HMM + Viterbi")
else:
    log.info("Decoder: classifier only (argmax)")


def list_trips():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))


# ----------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_PATH),
            "decoder": "hmm_viterbi" if USE_HMM else "argmax", "n_trips": len(list_trips())}


@app.get("/trips")
def trips():
    t = list_trips()
    log.info(f"GET /trips -> {len(t)} trips")
    return {"trips": t}


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in list_trips():
        raise HTTPException(status_code=404, detail=f"Unknown trip {trip_id}")
    trip_dir = os.path.join(DATA_DIR, trip_id)
    log.info(f"REQUEST predict trip={trip_id}")

    # 1) Prediction: uses only .track / .nodes / .arcs
    t0 = time.time()
    pred = pipeline.predict_trip(trip_dir, MODEL, decoder=DECODER)
    t_pred = time.time() - t0

    # 2) Scoring: server side only, reads .route
    t1 = time.time()
    metrics = scoring.score_trip(pred, trip_dir)
    t_score = time.time() - t1

    track, nodes, arcs = pred["track"], pred["nodes"], pred["arcs"]
    a, frac = pred["pred_arc"], pred["pred_frac"]
    has = a >= 0
    metrics["route_continuity"] = pipeline.route_continuity(a, track.segment.values, arcs)
    metrics["coverage"] = float(has.mean())
    metrics["prediction_time_s"] = round(t_pred, 3)
    metrics["scoring_time_s"] = round(t_score, 3)

    # Snapped positions on the predicted arc (for drawing)
    U, V = arcs.u.values, arcs.v.values
    au = np.where(has, a, 0)
    lat_s = nodes.lat.values[U[au]] + frac * (nodes.lat.values[V[au]] - nodes.lat.values[U[au]])
    lon_s = nodes.lon.values[U[au]] + frac * (nodes.lon.values[V[au]] - nodes.lon.values[U[au]])

    # Predicted route: distinct consecutive predicted arcs as [[lat, lon], [lat, lon]] segments
    seq = a[has]
    keep = np.r_[True, seq[1:] != seq[:-1]] if len(seq) else np.array([], bool)
    route_arcs = seq[keep]
    route_segments = [[[float(nodes.lat.values[U[x]]), float(nodes.lon.values[U[x]])],
                       [float(nodes.lat.values[V[x]]), float(nodes.lon.values[V[x]])]] for x in route_arcs]

    log.info(f"DONE trip={trip_id} points={metrics['n_points']} "
             f"top1={metrics['top1_accuracy']:.4f} continuity={metrics['route_continuity']:.4f} "
             f"pred_time={t_pred:.2f}s")

    return {
        "trip_id": trip_id,
        "decoder": "hmm_viterbi" if USE_HMM else "argmax",
        "metrics": metrics,
        "gps": track[["lat", "lon"]].round(7).values.tolist(),
        "snapped": [[float(la), float(lo)] if h else None for la, lo, h in zip(lat_s, lon_s, has)],
        "route_segments": route_segments,
        "n_route_arcs": int(len(route_arcs)),
    }
