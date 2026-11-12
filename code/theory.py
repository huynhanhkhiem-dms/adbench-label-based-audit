"""Constructions that make the paper's theorems checkable numerically.

Nothing here is a heuristic: each function realises, exactly, an object whose
existence a theorem asserts, so the test-suite can confirm the statements to
machine precision rather than by simulation alone.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Theorem 2 -- the evaluation target is not a functional of the observable law
# --------------------------------------------------------------------------- #
def labelling_with_auc(score, alpha, target, rng=None):
    """A labelling with contamination ~alpha whose AUC against ``score`` is
    ``target``, up to the 1/(n-m) grid imposed by having finitely many rows.

    Construction: put the anomalies on a contiguous block of the score-sorted
    rows.  If the block starts at sorted position j then every anomaly has
    exactly j negatives below it, so AUC = j / (n - m) exactly.
    """
    score = np.asarray(score, float).ravel()
    n = len(score)
    m = max(1, int(round(alpha * n)))
    m = min(m, n - 1)
    j = int(round(np.clip(target, 0.0, 1.0) * (n - m)))
    order = np.argsort(score, kind="mergesort")
    y = np.zeros(n, dtype=int)
    y[order[j:j + m]] = 1
    return y


def attainable_auc_grid(n, alpha):
    """The exact set of AUC values realisable by the construction above."""
    m = max(1, int(round(alpha * n)))
    return np.arange(0, n - m + 1) / (n - m)


# --------------------------------------------------------------------------- #
# AUC on a weighted grid, used for the exact statement of Theorem 7
# --------------------------------------------------------------------------- #
def auc_from_eta(score, eta):
    """AUC of ``score`` when each grid point carries anomaly posterior ``eta``.

    Positives get weight eta_i, negatives weight (1 - eta_i); ties in the score
    contribute one half, as in the rank definition of AUC.
    """
    s = np.asarray(score, float).ravel()
    e = np.asarray(eta, float).ravel()
    wp, wn = e, 1.0 - e
    gt = (s[:, None] > s[None, :]).astype(float)
    eq = (s[:, None] == s[None, :]).astype(float)
    num = (wp[:, None] * wn[None, :] * (gt + 0.5 * eq)).sum()
    den = wp.sum() * wn.sum()
    return float(num / den) if den > 0 else float("nan")


def is_auc_optimal(score, eta, tol=1e-12):
    """Whether ``score`` attains the Bayes AUC for posterior ``eta``."""
    return auc_from_eta(score, eta) >= auc_from_eta(eta, eta) - tol


def is_density_monotone(p, eta, tol=1e-12):
    """Whether eta is a non-increasing function of the density p."""
    order = np.argsort(-np.asarray(p, float).ravel(), kind="mergesort")
    return bool(np.all(np.diff(np.asarray(eta, float).ravel()[order]) >= -tol))


# --------------------------------------------------------------------------- #
# samples that satisfy / violate the low-density anomaly assumption
# --------------------------------------------------------------------------- #
def lda_sample(n=5000, d=2, alpha=0.05, rng=None):
    """A sample whose anomaly posterior is a strictly decreasing function of the
    true density: eta(x) = c / (1 + (p(x)/p0)^gamma)."""
    rng = rng or np.random.default_rng(0)
    X = rng.standard_normal((n, d))
    p = np.exp(-0.5 * (X ** 2).sum(1)) / (2 * np.pi) ** (d / 2)
    r = np.argsort(np.argsort(p)) / (n - 1.0)          # density rank in [0,1]
    eta = (1.0 - r) ** 6
    eta *= alpha / eta.mean()
    eta = np.clip(eta, 0.0, 1.0)
    y = (rng.random(n) < eta).astype(int)
    return X, y, p


def violating_sample(n=5000, alpha=0.05, rng=None):
    """Anomalies sit in their own dense cluster, so eta is *not* a function of
    the density and the negative density is far from optimal."""
    rng = rng or np.random.default_rng(0)
    m = max(2, int(round(alpha * n)))
    Xn = rng.standard_normal((n - m, 2))
    Xa = rng.standard_normal((m, 2)) * 0.15 + np.array([2.2, 0.0])
    X = np.vstack([Xn, Xa])
    y = np.concatenate([np.zeros(n - m, int), np.ones(m, int)])

    def gauss(Z, mu, sd):
        return np.exp(-0.5 * ((Z - mu) ** 2).sum(1) / sd ** 2) / (2 * np.pi * sd ** 2)

    p = (1 - alpha) * gauss(X, np.zeros(2), 1.0) + \
        alpha * gauss(X, np.array([2.2, 0.0]), 0.15)
    eta = alpha * gauss(X, np.array([2.2, 0.0]), 0.15) / p
    return X, y, p, eta


def random_scores(X, k=100, rng=None):
    """Random linear scores, used only as a sanity family in the tests."""
    rng = rng or np.random.default_rng(0)
    for _ in range(k):
        w = rng.standard_normal(X.shape[1])
        yield X @ w


# --------------------------------------------------------------------------- #
# label-free internal criteria (Proposition 10)
# --------------------------------------------------------------------------- #
def mass_volume_auc(score, X, n_mc=20000, t_grid=None, rng=None):
    """Area under the mass-volume curve of a score, estimated by Monte-Carlo
    volume on the bounding box (Clemencon & Jakubowicz; Goix).

    Lower is better.  The criterion is label-free: it is a functional of the
    observable law alone, which is exactly why Proposition 10 applies to it.
    """
    rng = rng or np.random.default_rng(0)
    s = np.asarray(score, float).ravel()
    lo, hi = X.min(0), X.max(0)
    U = rng.random((n_mc, X.shape[1])) * (hi - lo) + lo
    # nearest-neighbour extension of the score to the box
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(U)
    sU = s[idx.ravel()]
    t_grid = t_grid if t_grid is not None else np.linspace(0.05, 0.95, 19)
    out = []
    for a in t_grid:
        thr = np.quantile(s, 1 - a)            # top-a fraction flagged
        out.append((sU >= thr).mean())
    return float(np.trapezoid(out, t_grid) / (t_grid[-1] - t_grid[0]))


def excess_mass_auc(score, X, n_mc=20000, rng=None):
    """Excess-mass criterion (Polonik; Goix, Sabourin & Clemencon).  Higher is
    better.  Also a functional of the observable law alone."""
    rng = rng or np.random.default_rng(0)
    s = np.asarray(score, float).ravel()
    lo, hi = X.min(0), X.max(0)
    vol_box = float(np.prod(np.maximum(hi - lo, 1e-12)))
    U = rng.random((n_mc, X.shape[1])) * (hi - lo) + lo
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(X)
    _, idx = nn.kneighbors(U)
    sU = s[idx.ravel()]
    best = -np.inf
    for a in np.linspace(0.01, 0.99, 50):
        thr = np.quantile(s, 1 - a)
        mass = (s >= thr).mean()
        vol = (sU >= thr).mean() * vol_box
        best = max(best, mass - 0.9 * vol / max(vol_box, 1e-12))
    return float(best)
