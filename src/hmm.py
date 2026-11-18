"""
PathSnap HMM + Viterbi route reconstruction.

Hidden states : the candidate arcs of each GPS point (from the candidate generator).
Emissions     : log of the classifier probability for (GPS point, candidate arc).
Transitions   : from road-network connectivity and distance consistency
                (Newson & Krumm style): the distance travelled along the network between
                two candidates should match the straight-line distance between the
                two GPS points.
                  - same arc            : route = (frac_b - frac_a) * len(a)
                  - reachable in 1-2 hops: route = rest of a + intermediate arc + start of b
                  - U-turn onto the twin : extra penalty
                  - not reachable        : fixed large penalty (jump)
                  - new segment (time gap) or large GPS step : uniform (chain restarts)
Decoding      : Viterbi (max-product dynamic programming) over each chain of points.
Uses only the prediction inputs (no ground truth).
"""
from collections import defaultdict
import numpy as np

PARAMS = {
    "beta": 10.0,
    "w_emit": 1.0,
    "jump_logp": -20.0,
    "uturn_logp": -8.0,
    "reset_dist": 150.0,
    "eps": 1e-4,
}

class Network:
    """Directed road network with a cached 1-2 hop reachability lookup."""

    def __init__(self, arcs):
        self.U = arcs.u.values
        self.V = arcs.v.values
        self.L = arcs.length_m.values
        out = defaultdict(list)
        for i, u in enumerate(self.U):
            out[u].append(i)
        self.out = out
        self.cache = {}

    def reach(self, a):
        """Arcs reachable from the end of arc a within 2 hops -> {arc: length of the arc in between}."""
        r = self.cache.get(a)
        if r is not None:
            return r
        r = {b: 0.0 for b in self.out.get(self.V[a], [])}
        for x in list(r.keys()):
            for b in self.out.get(self.V[x], []):
                if b not in r or self.L[x] < r[b]:
                    r[b] = self.L[x]
        self.cache[a] = r
        return r

    def is_uturn(self, a, b):
        return self.U[b] == self.V[a] and self.V[b] == self.U[a]


def transition_matrix(net, arcs_a, frac_a, arcs_b, frac_b, gps_step, P):
    """Log transition scores, shape (len(arcs_a), len(arcs_b))."""
    T = np.empty((len(arcs_a), len(arcs_b)))
    for i, (a, fa) in enumerate(zip(arcs_a, frac_a)):
        reach = net.reach(a)
        for j, (b, fb) in enumerate(zip(arcs_b, frac_b)):
            if a == b:
                route = (fb - fa) * net.L[a]
            elif b in reach:
                route = (1 - fa) * net.L[a] + reach[b] + fb * net.L[b]
            else:
                T[i, j] = P["jump_logp"]
                continue
            score = -abs(route - gps_step) / P["beta"]
            if a != b and net.is_uturn(a, b):
                score += P["uturn_logp"]
            T[i, j] = score
    return T


def viterbi_decoder(cands, track, arcs, params=None):
    """Return (arc, proj_frac, prob) per GPS point; -1 where a point has no candidates."""
    P = dict(PARAMS, **(params or {}))
    n = len(track)
    out_arc = np.full(n, -1, dtype=np.int64)
    out_frac = np.zeros(n)
    out_prob = np.zeros(n)
    if len(cands) == 0:
        return out_arc, out_frac, out_prob

    net = Network(arcs)
    x, y, seg = track.x.values, track.y.values, track.segment.values
    groups = {pt: g for pt, g in cands.groupby("point")}

    def backtrack(chain, scores_last, backs):
        k = int(np.argmax(scores_last))
        for idx in range(len(chain) - 1, -1, -1):
            pt, a_arr, f_arr, p_arr = chain[idx]
            out_arc[pt], out_frac[pt], out_prob[pt] = a_arr[k], f_arr[k], p_arr[k]
            if idx > 0:
                k = int(backs[idx][k])

    chain, backs, score = [], [], None
    for t in range(n):
        g = groups.get(t)
        if g is None:                                  # no candidates: close the current chain
            if chain:
                backtrack(chain, score, backs)
            chain, backs, score = [], [], None
            continue
        a_arr, f_arr, p_arr = g.arc.values, g.proj_frac.values, g.p.values
        emit = P["w_emit"] * np.log(np.maximum(p_arr, P["eps"]))
        if not chain:
            score = emit
            backs.append(None)
        else:
            pt_prev, a_prev, f_prev, _ = chain[-1]
            step = float(np.hypot(x[t] - x[pt_prev], y[t] - y[pt_prev]))
            if seg[t] != seg[pt_prev] or step > P["reset_dist"]:
                T = np.zeros((len(a_prev), len(a_arr)))     # uniform: chain restarts softly
            else:
                T = transition_matrix(net, a_prev, f_prev, a_arr, f_arr, step, P)
            total = score[:, None] + T
            backs.append(np.argmax(total, axis=0))
            score = total.max(axis=0) + emit
            score = score - score.max()                     # keep numbers small
        chain.append((t, a_arr, f_arr, p_arr))
    if chain:
        backtrack(chain, score, backs)
    return out_arc, out_frac, out_prob
