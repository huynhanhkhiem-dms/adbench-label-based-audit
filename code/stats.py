"""Fast DeLong, paired bootstrap and multiple-testing adjustments.

The primary comparison measures the held-out AUC gap between a supervised
reference and a finite-family density score, both selected on outer training.
DeLong estimates their covariance on the paired test observations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# AUC and the fast DeLong covariance (Sun & Xu, 2014)
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved to their average ('midranks'), 1-based.

    ``scipy.stats.rankdata`` implements exactly this and does it in C; the
    equivalent Python loop is the single hottest line in the whole audit once
    bootstrap replicates are involved, so it is worth not writing one.
    """
    return stats.rankdata(x, method="average")


def delong_components(scores: np.ndarray, y: np.ndarray):
    """Return (auc, V10, V01) for one or more score vectors.

    ``scores`` has shape (k, n).  ``V10``/``V01`` are the DeLong structural
    components over positives / negatives, shape (k, m) and (k, n-m).
    """
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    y = np.asarray(y).ravel()
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    if m == 0 or n == 0:
        raise ValueError("need both classes present")
    tx = np.vstack([_midrank(pos[i]) for i in range(k)])
    ty = np.vstack([_midrank(neg[i]) for i in range(k)])
    tz = np.vstack([_midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    return auc, v10, v01


def delong_cov(scores: np.ndarray, y: np.ndarray):
    auc, v10, v01 = delong_components(scores, y)
    m, n = v10.shape[1], v01.shape[1]
    S = np.cov(v10) / m + np.cov(v01) / n
    S = np.atleast_2d(S)
    return auc, S


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC with ties handled by midranks."""
    return float(delong_components(score, y)[0][0])


def delong_test(score_a, score_b, y, alternative="greater"):
    """One-sided (default) DeLong test of AUC(a) - AUC(b).

    Returns (delta, se, z, p).  ``alternative='greater'`` tests H1: a beats b.
    """
    a, S = delong_cov(np.vstack([np.asarray(score_a, float),
                                 np.asarray(score_b, float)]), y)
    delta = float(a[0] - a[1])
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    if se <= 0:
        z = 0.0 if abs(delta) < 1e-12 else np.sign(delta) * np.inf
    else:
        z = delta / se
    if alternative == "greater":
        p = float(stats.norm.sf(z))
    elif alternative == "less":
        p = float(stats.norm.cdf(z))
    else:
        p = float(2 * stats.norm.sf(abs(z)))
    return delta, se, float(z), p


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def paired_bootstrap_auc_diff(score_a, score_b, y, n_boot=2000, seed=0):
    """Stratified paired bootstrap of AUC(a) - AUC(b); returns (delta, lo, hi)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    base = auc(a, y) - auc(b, y)
    out = np.empty(n_boot)
    for t in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        yy = y[i]
        out[t] = auc(a[i], yy) - auc(b[i], yy)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(base), float(lo), float(hi)


def benjamini_hochberg(pvals, q=0.05):
    """BH step-up.  Returns (rejected mask, adjusted p-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out <= q, out


# --------------------------------------------------------------------------- #
# rank agreement helpers
# --------------------------------------------------------------------------- #
def kendall_tau(a, b):
    return float(stats.kendalltau(a, b).statistic)


def spearman(a, b):
    return float(stats.spearmanr(a, b).statistic)


def average_precision(score, y):
    """Average precision (area under the precision-recall curve), ties broken
    by the standard step-wise estimator used throughout the benchmark."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, score))

def benjamini_yekutieli(pvals, q=0.05):
    """Benjamini-Yekutieli step-up control under arbitrary dependence."""
    p=np.asarray(pvals,dtype=float); n=len(p)
    c=float(np.sum(1.0/np.arange(1,n+1))) if n else 1.0
    order=np.argsort(p); ranked=p[order]
    adj=ranked*n*c/np.arange(1,n+1)
    adj=np.minimum.accumulate(adj[::-1])[::-1]
    adj=np.clip(adj,0,1); out=np.empty(n); out[order]=adj
    return out<=q,out

def bonferroni(pvals, alpha=0.05):
    p=np.asarray(pvals,dtype=float); adj=np.clip(p*len(p),0,1)
    return adj<=alpha,adj
