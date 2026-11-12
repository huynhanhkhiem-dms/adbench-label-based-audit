"""Numerical verification of the theorems, and the demonstration behind
Proposition 10 (internal criteria cannot separate two worlds that share an
observable law).

Writes ``results/theory.json``.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stats as ST                                           # noqa: E402
import theory as TH                                          # noqa: E402
import detectors as DET                                      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


# --------------------------------------------------------------------------- #
def theorem2(n=5000, alpha=0.05, rng=None):
    """The AUC of a *fixed* score is unrestricted by the observable law."""
    rng = rng or np.random.default_rng(0)
    s = rng.standard_normal(n)
    targets = np.linspace(0.0, 1.0, 21)
    errs, achieved = [], []
    for t in targets:
        y = TH.labelling_with_auc(s, alpha, t, rng)
        a = ST.auc(s, y)
        achieved.append(a)
        errs.append(abs(a - t))
    grid = TH.attainable_auc_grid(n, alpha)
    return dict(max_abs_error=float(max(errs)),
                grid_spacing=float(np.diff(grid).max()),
                identified_set_width=float(max(achieved) - min(achieved)),
                min_auc=float(min(achieved)), max_auc=float(max(achieved)),
                n=n, alpha=alpha)


def theorem7(n_grids=2000, m=10, rng=None):
    """Exact equivalence: -p is AUC-optimal  <=>  eta is density-monotone."""
    rng = rng or np.random.default_rng(1)
    mismatch = 0
    gaps_mono, gaps_non = [], []
    for _ in range(n_grids):
        p = rng.random(m)
        eta = rng.random(m)
        if rng.random() < 0.5:                       # half the grids monotone
            order = np.argsort(-p)
            eta = np.sort(eta)[np.argsort(order)]
        mono = TH.is_density_monotone(p, eta)
        gap = TH.auc_from_eta(eta, eta) - TH.auc_from_eta(-p, eta)
        opt = gap <= 1e-12
        mismatch += int(mono != opt)
        (gaps_mono if mono else gaps_non).append(gap)
    return dict(n_grids=n_grids, grid_size=m, mismatches=int(mismatch),
                max_gap_when_monotone=float(np.max(gaps_mono)),
                mean_gap_when_not_monotone=float(np.mean(gaps_non)),
                min_gap_when_not_monotone=float(np.min(gaps_non)))


def proposition10(n=4000, rng=None):
    """Two worlds, one observable law.

    The design matrix and every unsupervised score are held *identical*; only
    the latent labelling changes.  Any label-free criterion therefore returns
    the identical ranking in both worlds, while the AUC-optimal detector is
    different -- so no label-free criterion can be right in both.
    """
    rng = rng or np.random.default_rng(2)
    X = np.vstack([rng.standard_normal((n - 200, 2)),
                   rng.standard_normal((200, 2)) * 0.12 + np.array([2.6, 0.0])])
    scores = {k: np.asarray(f(X, np.random.default_rng(0)), float)
              for k, f in DET.DETECTORS.items()}

    # world A: anomalies are the isolated low-density tail of the bulk
    dens = DET.knn_distance(X, k=10, agg="max")
    yA = np.zeros(len(X), int)
    yA[np.argsort(-dens)[:200]] = 1
    # world B: anomalies are the tight satellite cluster (same X, same scores)
    yB = np.zeros(len(X), int)
    yB[n - 200:] = 1

    aucA = {k: ST.auc(v, yA) for k, v in scores.items()}
    aucB = {k: ST.auc(v, yB) for k, v in scores.items()}

    mv = {k: TH.mass_volume_auc(v, X, rng=np.random.default_rng(3))
          for k, v in scores.items()}
    em = {k: TH.excess_mass_auc(v, X, rng=np.random.default_rng(3))
          for k, v in scores.items()}

    pick_mv = min(mv, key=mv.get)          # mass-volume: lower is better
    pick_em = max(em, key=em.get)          # excess-mass: higher is better
    return dict(
        best_in_world_A=max(aucA, key=aucA.get),
        best_in_world_B=max(aucB, key=aucB.get),
        auc_A=aucA, auc_B=aucB,
        internal_pick_mass_volume=pick_mv,
        internal_pick_excess_mass=pick_em,
        regret_A_mass_volume=float(max(aucA.values()) - aucA[pick_mv]),
        regret_B_mass_volume=float(max(aucB.values()) - aucB[pick_mv]),
        regret_A_excess_mass=float(max(aucA.values()) - aucA[pick_em]),
        regret_B_excess_mass=float(max(aucB.values()) - aucB[pick_em]),
        identical_scores=True, n=n)


def proposition12(rng=None):
    """Check the rare-anomaly scaling of the paired DeLong standard error."""
    rng = rng or np.random.default_rng(4)
    out = []
    for m in [10, 20, 50, 100, 200, 500, 1000]:
        n = 20 * m
        y = np.concatenate([np.ones(m, int), np.zeros(n, int)])
        a = np.concatenate([rng.standard_normal(m) + 1.6, rng.standard_normal(n)])
        b = a + 0.9 * rng.standard_normal(m + n)
        delta, se, z, p = ST.delong_test(a, b, y, "greater")
        out.append(dict(m=m, n=n, delta=delta, se=se))
    ses = np.array([o["se"] for o in out])
    ms = np.array([o["m"] for o in out], float)
    slope = float(np.polyfit(np.log(ms), np.log(ses), 1)[0])
    return dict(rows=out, log_log_slope=slope)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    out = dict(theorem2=theorem2(), theorem7=theorem7(),
               proposition10=proposition10(), proposition12=proposition12())
    with open(os.path.join(RESULTS, "theory.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: (v if k != "proposition10" else
                          {kk: vv for kk, vv in v.items()
                           if kk not in ("auc_A", "auc_B")})
                      for k, v in out.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
