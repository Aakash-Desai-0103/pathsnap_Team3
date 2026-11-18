"""
PathSnap scoring (SERVER SIDE ONLY).

This is the only module that reads a trip's .route file (reference route).
It is called AFTER prediction, so ground truth never enters the prediction
pipeline or the UI. Per-point reference arcs are derived with the same
monotone DP alignment used to build training labels (notebook 01).
"""
import os
import numpy as np
from pipeline import read_ws, point_to_segment

LABEL_MAX_M = 50   # reference label farther than this from the GPS point = unreliable, excluded


def load_reference_route(trip_dir):
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    return read_ws(os.path.join(trip_dir, trip_id + ".route"), ["arc_id"])


def label_points(track, arcs, route, cap=200.0):
    """Monotone DP alignment of GPS points to the ordered route arcs (same as notebook 01)."""
    r = route.arc_id.values.astype(int)
    A = arcs[["ax", "ay", "bx", "by"]].values[r]
    px, py = track.x.values, track.y.values
    N, M = len(px), len(r)
    back = np.empty((N, M), dtype=np.int32)
    idx = np.arange(M)
    cost = None
    for i in range(N):
        d, _ = point_to_segment(px[i], py[i], A[:, 0], A[:, 1], A[:, 2], A[:, 3])
        d = np.minimum(d, cap)
        if i == 0:
            cost, back[0] = d, idx
        else:
            run_min = np.minimum.accumulate(cost)
            back[i] = np.maximum.accumulate(np.where(cost == run_min, idx, 0))
            cost = d + run_min
    path = np.empty(N, dtype=np.int64)
    j = int(np.argmin(cost))
    for i in range(N - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    ldist, _ = point_to_segment(px, py, A[path, 0], A[path, 1], A[path, 2], A[path, 3])
    return r[path], ldist


def score_trip(pred, trip_dir):
    """Compare predicted arcs with the reference route. Returns a metrics dict."""
    track, arcs, pred_arc = pred["track"], pred["arcs"], pred["pred_arc"]
    route = load_reference_route(trip_dir)
    ref_arc, ref_dist = label_points(track, arcs, route)
    ok = ref_dist <= LABEL_MAX_M
    key = arcs.key.values
    has_pred = pred_arc >= 0
    correct = np.zeros(len(track), dtype=bool)
    correct[has_pred] = key[pred_arc[has_pred]] == key[ref_arc[has_pred]]   # duplicate arcs count as correct
    return {
        "top1_accuracy": float(correct[ok].mean()) if ok.any() else float("nan"),
        "n_points": int(len(track)),
        "n_points_scored": int(ok.sum()),
        "n_points_off_route": int((~ok).sum()),
    }
