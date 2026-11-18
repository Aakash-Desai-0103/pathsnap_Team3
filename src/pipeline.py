"""
PathSnap prediction pipeline (shared by the notebooks and the API).

Uses ONLY a trip's .track, .nodes and .arcs files. It never reads .route
(ground truth); scoring lives separately in src/scoring.py.

Steps: load -> project to metres -> segments/heading/speed -> candidate arcs
-> features -> classifier probabilities -> decode (argmax now, HMM/Viterbi later).
The feature code is the same as in notebooks/01_eda_preprocessing.ipynb.
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R_EARTH = 6371000
GAP_S = 60            # time gap (s) that starts a new segment
RADIUS_GEN = 100      # candidate generation radius (m), then filtered to R_FINAL
K_GEN = 20
DENSIFY_M = 10        # arc sampling step for the KD-tree (m)
R_FINAL = 50          # chosen from the candidate miss rate (train only)
K_FINAL = 10
CTX_OFFSETS = [-3, -2, -1, 1, 2, 3]

BASE_FEATURES = ["dist_m", "dist_rank", "dist_minus_min", "heading_diff", "proj_frac",
                 "at_endpoint", "arc_length_m", "n_cands", "is_twoway"]
CTX_FEATURES = ["dist_prev_m", "dist_next_m", "ctx_dist_m", "ctx_minus_min", "ctx_rank", "heading_diff_w3"]
FEATURES = BASE_FEATURES + CTX_FEATURES


# ----------------------------------------------------------------------------- loading
def read_ws(path, names):
    """Whitespace-separated file without header. Empty file -> empty DataFrame."""
    try:
        return pd.read_csv(path, sep=r"\s+", header=None, names=names)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=names)


def load_trip_inputs(trip_dir):
    """Load the prediction inputs of a trip folder (NOT the .route file)."""
    trip_id = os.path.basename(os.path.normpath(trip_dir))
    p = lambda ext: os.path.join(trip_dir, trip_id + ext)
    track = read_ws(p(".track"), ["lon", "lat", "t"])
    nodes = read_ws(p(".nodes"), ["lon", "lat"])
    arcs = read_ws(p(".arcs"), ["u", "v"])
    return track, nodes, arcs


# ----------------------------------------------------------------------------- geometry
def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres around the trip centre."""
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return x, y


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment(s) AB, and projection position t (0=A, 1=B)."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = ((px - ax) * dx + (py - ay) * dy) / np.where(L2 > 0, L2, 1)
    t = np.clip(np.where(L2 > 0, t, 0), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def preprocess(track, nodes, arcs):
    track, nodes, arcs = track.copy(), nodes.copy(), arcs.copy()
    lon0, lat0 = track.lon.mean(), track.lat.mean()
    track["x"], track["y"] = project(track.lon.values, track.lat.values, lon0, lat0)
    nodes["x"], nodes["y"] = project(nodes.lon.values, nodes.lat.values, lon0, lat0)

    track["dt"] = track.t.diff().fillna(0)
    track["segment"] = (track.dt > GAP_S).cumsum()
    g = track.groupby("segment")
    step = np.hypot(g.x.diff(), g.y.diff())
    track["speed_mps"] = (step / track.dt.where(track.dt > 0)).fillna(0)
    hx = g.x.shift(-1).fillna(track.x) - g.x.shift(1).fillna(track.x)
    hy = g.y.shift(-1).fillna(track.y) - g.y.shift(1).fillna(track.y)
    track["heading"] = np.degrees(np.arctan2(hx, hy)) % 360
    track["heading_valid"] = np.hypot(hx, hy) > 2.0

    U, V = arcs.u.values, arcs.v.values
    arcs["ax"], arcs["ay"] = nodes.x.values[U], nodes.y.values[U]
    arcs["bx"], arcs["by"] = nodes.x.values[V], nodes.y.values[V]
    arcs["length_m"] = np.hypot(arcs.bx - arcs.ax, arcs.by - arcs.ay)
    arcs["bearing"] = np.degrees(np.arctan2(arcs.bx - arcs.ax, arcs.by - arcs.ay)) % 360
    arcs["key"] = U.astype(np.int64) * len(nodes) + V
    arcs["is_twoway"] = np.isin(V.astype(np.int64) * len(nodes) + U, arcs["key"].values)
    return track, nodes, arcs


# ----------------------------------------------------------------------------- candidates
def generate_candidates(track, arcs, radius=RADIUS_GEN, k=K_GEN):
    xmin, xmax = track.x.min() - radius, track.x.max() + radius
    ymin, ymax = track.y.min() - radius, track.y.max() + radius
    near = arcs[(np.minimum(arcs.ax, arcs.bx) <= xmax) & (np.maximum(arcs.ax, arcs.bx) >= xmin) &
                (np.minimum(arcs.ay, arcs.by) <= ymax) & (np.maximum(arcs.ay, arcs.by) >= ymin)]
    empty = pd.DataFrame(columns=["point", "arc", "dist_m", "proj_frac", "dist_rank"])
    if len(near) == 0:
        return empty
    ax_, ay_, bx_, by_ = (near[c].values for c in ["ax", "ay", "bx", "by"])

    n = np.maximum(1, np.ceil(near.length_m.values / DENSIFY_M)).astype(int)
    cnt = n + 1
    owner = np.repeat(np.arange(len(near)), cnt)
    frac = (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)) / np.repeat(n, cnt)
    sx = ax_[owner] + frac * (bx_ - ax_)[owner]
    sy = ay_[owner] + frac * (by_ - ay_)[owner]
    tree = cKDTree(np.c_[sx, sy])
    hits = tree.query_ball_point(np.c_[track.x.values, track.y.values], r=radius + DENSIFY_M / 2)

    pts, cands = [], []
    for i, h in enumerate(hits):
        if h:
            u = np.unique(owner[h])
            pts.append(np.full(len(u), i))
            cands.append(u)
    if not pts:
        return empty
    pts, cands = np.concatenate(pts), np.concatenate(cands)
    d, tproj = point_to_segment(track.x.values[pts], track.y.values[pts],
                                ax_[cands], ay_[cands], bx_[cands], by_[cands])
    c = pd.DataFrame({"point": pts, "arc": near.index.values[cands], "dist_m": d, "proj_frac": tproj})
    c = c[c.dist_m <= radius].copy()
    c["dist_rank"] = c.groupby("point").dist_m.rank(method="first").astype(int) - 1
    return c[c.dist_rank < k].reset_index(drop=True)


# ----------------------------------------------------------------------------- features
def build_features(c, track, arcs):
    p, a = c.point.values, c.arc.values
    c["n_cands"] = c.groupby("point").arc.transform("size")
    c["dist_minus_min"] = c.dist_m - c.groupby("point").dist_m.transform("min")
    hd = np.abs(track.heading.values[p] - arcs.bearing.values[a]) % 360
    c["heading_diff"] = np.minimum(hd, 360 - hd)
    c["heading_valid"] = track.heading_valid.values[p].astype(int)
    c.loc[c.heading_valid == 0, "heading_diff"] = 90.0
    c["arc_length_m"] = arcs.length_m.values[a]
    c["is_twoway"] = arcs.is_twoway.values[a].astype(int)
    c["at_endpoint"] = ((c.proj_frac <= 0) | (c.proj_frac >= 1)).astype(int)
    c["speed_mps"] = track.speed_mps.values[p]
    c["dt"] = track.dt.values[p]
    c["segment"] = track.segment.values[p]
    return c


def add_context(c, track, arcs):
    px, py = track.x.values, track.y.values
    seg, n = track.segment.values, len(track)
    p, a = c.point.values, c.arc.values
    AX, AY = arcs["ax"].values[a], arcs["ay"].values[a]
    BX, BY = arcs["bx"].values[a], arcs["by"].values[a]

    total, count = c.dist_m.values.copy(), np.ones(len(c))
    for off in CTX_OFFSETS:
        q = p + off
        qc = np.clip(q, 0, n - 1)
        valid = (q >= 0) & (q < n) & (seg[qc] == seg[p])
        d, _ = point_to_segment(px[qc], py[qc], AX, AY, BX, BY)
        total = total + np.where(valid, d, 0)
        count = count + valid
        if off == -1:
            c["dist_prev_m"] = np.where(valid, d, c.dist_m.values)
        if off == 1:
            c["dist_next_m"] = np.where(valid, d, c.dist_m.values)
    c["ctx_dist_m"] = total / count

    lo, hi = np.clip(p - 3, 0, n - 1), np.clip(p + 3, 0, n - 1)
    lo = np.where(seg[lo] == seg[p], lo, p)
    hi = np.where(seg[hi] == seg[p], hi, p)
    hx, hy = px[hi] - px[lo], py[hi] - py[lo]
    h3 = np.degrees(np.arctan2(hx, hy)) % 360
    hd3 = np.abs(h3 - arcs["bearing"].values[a]) % 360
    hd3 = np.minimum(hd3, 360 - hd3)
    c["heading_diff_w3"] = np.where(np.hypot(hx, hy) > 5, hd3, 90.0)
    return c


def finalize(c):
    """Apply the chosen R/K and compute the per-point relative features."""
    c = c[(c.dist_m <= R_FINAL) & (c.dist_rank < K_FINAL)].copy()
    g = c.groupby("point")
    c["n_cands"] = g.arc.transform("size")
    c["ctx_minus_min"] = c.ctx_dist_m - g.ctx_dist_m.transform("min")
    c["ctx_rank"] = g.ctx_dist_m.rank(method="first").astype(int) - 1
    return c.reset_index(drop=True)


def make_features(track, nodes, arcs):
    """Raw trip inputs -> (preprocessed track, nodes, arcs, candidate feature table)."""
    track, nodes, arcs = preprocess(track, nodes, arcs)
    c = generate_candidates(track, arcs)
    if len(c) == 0:
        return track, nodes, arcs, c
    c = add_context(build_features(c, track, arcs), track, arcs)
    return track, nodes, arcs, finalize(c)


# ----------------------------------------------------------------------------- decoding
def decode_argmax(cands, n_points):
    """Classifier-only decoding: per GPS point, the candidate with the highest probability.
    Returns arrays (arc_id, proj_frac, prob) of length n_points; -1 where no candidate."""
    arc = np.full(n_points, -1, dtype=np.int64)
    frac = np.zeros(n_points)
    prob = np.zeros(n_points)
    if len(cands):
        best = cands.sort_values("p", ascending=False).groupby("point").head(1)
        arc[best.point.values] = best.arc.values
        frac[best.point.values] = best.proj_frac.values
        prob[best.point.values] = best.p.values
    return arc, frac, prob


def predict_trip(trip_dir, model, decoder=None):
    """Full prediction for one trip folder. Reads only .track/.nodes/.arcs.

    decoder: optional function(cands, track, arcs) -> (arc, proj_frac, prob) per point.
             None = classifier-only (argmax). The HMM/Viterbi decoder plugs in here.
    """
    track, nodes, arcs = load_trip_inputs(trip_dir)
    track, nodes, arcs, cands = make_features(track, nodes, arcs)
    if len(cands):
        cands["p"] = model.predict_proba(cands[FEATURES])[:, 1]
    if decoder is None:
        arc, frac, prob = decode_argmax(cands, len(track))
    else:
        arc, frac, prob = decoder(cands, track, arcs)
    return {"track": track, "nodes": nodes, "arcs": arcs, "cands": cands,
            "pred_arc": arc, "pred_frac": frac, "pred_prob": prob}


def route_continuity(pred_arc, segment, arcs):
    """Share of consecutive GPS points (same segment, both predicted) whose arcs are equal or connected."""
    a = pred_arc
    U, V = arcs.u.values, arcs.v.values
    both = (a[:-1] >= 0) & (a[1:] >= 0) & (np.diff(segment) == 0)
    if both.sum() == 0:
        return float("nan")
    a0, a1 = a[:-1][both], a[1:][both]
    return float(((a0 == a1) | (V[a0] == U[a1])).mean())
