"""Unsupervised anomaly detectors, grouped by the inductive bias they encode.

Every detector here is a map from an unlabelled design matrix to a real-valued
score, larger = more anomalous.  Implementations are deliberately self-contained
(NumPy + scikit-learn only) so that the audit does not inherit version drift
from a third-party detector library, and so that the *bias family* of each
method is explicit in the code rather than buried in a package.

Families
--------
``global``   monotone (up to estimation error) in an estimate of the joint
             density  p(x)  -- kNN distance, KDE, GMM, Gaussian/robust
             Mahalanobis, PCA reconstruction, one-class SVM.
``local``    monotone in a *ratio* of local densities -- LOF, CBLOF.
``isolation``partition-depth surrogates for density -- Isolation Forest, LODA.
``marginal`` functions of the one-dimensional marginals only -- HBOS, ECOD,
             COPOD, MaxQuantile.

The families matter because Theorem 3 of the paper says a detector is
AUC-optimal exactly on the data-generating processes whose anomaly posterior is
co-monotone with that detector's own functional; grouping by family is what
makes the resulting leaderboard-composition effect visible.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import (KernelDensity, LocalOutlierFactor,
                               NearestNeighbors)
from sklearn.svm import OneClassSVM

FIT_CAP = 5_000          # cap on the fitting set for O(n^2)-ish learners
_EPS = 1e-12


def _fit_subset(X, rng, cap=FIT_CAP):
    if X.shape[0] <= cap:
        return X
    idx = rng.choice(X.shape[0], cap, replace=False)
    return X[idx]


# --------------------------------------------------------------------------- #
# global-density family
# --------------------------------------------------------------------------- #
def knn_distance(X, k=10, agg="max"):
    """Distance to the k-th nearest neighbour (self excluded).

    A monotone decreasing function of the classical Loftsgaarden-Quesenberry
    k-NN density estimate, hence a member of the global-density family.
    """
    k = int(min(k, X.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    dist, _ = nn.kneighbors(X)
    dist = dist[:, 1:]
    if agg == "max":
        return dist[:, -1]
    if agg == "mean":
        return dist.mean(axis=1)
    if agg == "median":
        return np.median(dist, axis=1)
    raise ValueError(agg)


def knn_log_density(X, k=10):
    """log p̂(x) for the k-NN density estimate, up to an additive constant."""
    n, d = X.shape
    r = knn_distance(X, k=k, agg="max")
    return np.log(k / n + _EPS) - d * np.log(r + _EPS)


def kde_score(X, bandwidth=1.0, rng=None, cap=FIT_CAP):
    rng = rng or np.random.default_rng(0)
    Xf = _fit_subset(X, rng, cap)
    kde = KernelDensity(bandwidth=bandwidth).fit(Xf)
    out = np.empty(X.shape[0])
    for s in range(0, X.shape[0], 4096):
        out[s:s + 4096] = kde.score_samples(X[s:s + 4096])
    return -out


HIGH_D = 50      # above this a full d x d covariance is not estimable here


def gmm_score(X, n_components=4, rng=None):
    """Negative mixture log-likelihood.

    A full covariance needs $n \\gg d(d+1)/2$ observations per component; above
    ``HIGH_D`` that is not available on these tables, so the mixture switches to
    diagonal covariances rather than being regularised into a near-singular fit.
    """
    seed = 0 if rng is None else int(rng.integers(1 << 30))
    n_components = int(min(n_components, max(1, X.shape[0] // 20)))
    cov = "full" if X.shape[1] <= HIGH_D else "diag"
    gm = GaussianMixture(n_components=n_components, covariance_type=cov,
                         reg_covar=1e-4, random_state=seed, max_iter=100).fit(X)
    return -gm.score_samples(X)


def mahalanobis(X):
    S = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
    Si = np.linalg.pinv(S)
    Z = X - X.mean(axis=0)
    return np.einsum("ij,jk,ik->i", Z, Si, Z)


def mcd_mahalanobis(X, rng=None, cap=FIT_CAP):
    """Robust Mahalanobis distance from a minimum-covariance-determinant fit.

    The descriptive fit uses a support fraction of 0.75 and falls back to
    empirical Mahalanobis above HIGH_D or if the robust fit fails.
    """
    rng = rng or np.random.default_rng(0)
    if X.shape[1] > HIGH_D:
        return mahalanobis(X)
    try:
        Xf = _fit_subset(X, rng, min(cap, 2000))
        mcd = MinCovDet(random_state=0, support_fraction=0.75).fit(Xf)
        return mcd.mahalanobis(X)
    except Exception:                                    # ill-conditioned
        return mahalanobis(X)


def pca_reconstruction(X, q=None):
    q = q or max(1, min(X.shape[1] - 1, int(np.ceil(0.5 * X.shape[1]))))
    q = int(min(q, min(X.shape) - 1))
    if q < 1:
        return mahalanobis(X)
    p = PCA(n_components=q, random_state=0).fit(X)
    R = X - p.inverse_transform(p.transform(X))
    return (R ** 2).sum(axis=1)


def ocsvm(X, nu=0.1, rng=None, cap=FIT_CAP):
    rng = rng or np.random.default_rng(0)
    Xf = _fit_subset(X, rng, min(cap, 4000))
    gamma = 1.0 / max(1, X.shape[1])
    m = OneClassSVM(nu=nu, gamma=gamma, kernel="rbf").fit(Xf)
    return -m.decision_function(X)


# --------------------------------------------------------------------------- #
# local-density family
# --------------------------------------------------------------------------- #
def lof(X, k=20):
    k = int(min(k, X.shape[0] - 1))
    m = LocalOutlierFactor(n_neighbors=k, n_jobs=-1)
    m.fit(X)
    return -m.negative_outlier_factor_


def cblof(X, n_clusters=8, rng=None, alpha_big=0.9):
    """Cluster-based local outlier factor (He, Xu & Deng, 2003).

    Points in a *large* cluster are scored by the distance to their own centre;
    points in a *small* cluster by the distance to the nearest large centre.
    Cluster-size weighting is left off, matching the default of the reference
    implementation, because the weighting inverts the ranking whenever the
    anomalies form a tight cluster of their own.
    """
    seed = 0 if rng is None else int(rng.integers(1 << 30))
    n_clusters = int(min(n_clusters, max(2, X.shape[0] // 20)))
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(X)
    sizes = np.bincount(km.labels_, minlength=n_clusters)
    order = np.argsort(-sizes)
    cum = np.cumsum(sizes[order]) / sizes.sum()
    n_big = int(np.searchsorted(cum, alpha_big) + 1)
    n_big = int(min(max(n_big, 1), n_clusters - 1)) if n_clusters > 1 else 1
    big = sorted(order[:n_big].tolist())
    C = km.cluster_centers_
    d_all = np.empty((X.shape[0], n_clusters))
    for j in range(n_clusters):
        d_all[:, j] = np.linalg.norm(X - C[j], axis=1)
    d_big = d_all[:, big].min(axis=1)
    own = d_all[np.arange(X.shape[0]), km.labels_]
    in_big = np.isin(km.labels_, big)
    return np.where(in_big, own, d_big)


# --------------------------------------------------------------------------- #
# isolation family
# --------------------------------------------------------------------------- #
def iforest(X, n_estimators=200, rng=None):
    seed = 0 if rng is None else int(rng.integers(1 << 30))
    m = IsolationForest(n_estimators=n_estimators, random_state=seed,
                        n_jobs=-1).fit(X)
    return -m.score_samples(X)


def loda(X, n_projections=100, n_bins=None, rng=None):
    """Lightweight on-line detector of anomalies (Pevny, 2016).

    Sparse random projections, each summarised by a histogram; the score is the
    negative mean log-density over projections.
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    n_bins = n_bins or max(10, int(np.sqrt(n)) // 2)
    n_nonzero = max(1, int(round(np.sqrt(d))))
    out = np.zeros(n)
    for _ in range(n_projections):
        w = np.zeros(d)
        idx = rng.choice(d, n_nonzero, replace=False)
        w[idx] = rng.standard_normal(n_nonzero)
        z = X @ w
        hist, edges = np.histogram(z, bins=n_bins, density=True)
        b = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n_bins - 1)
        out += -np.log(hist[b] + _EPS)
    return out / n_projections


# --------------------------------------------------------------------------- #
# marginal-tail family
# --------------------------------------------------------------------------- #
def _ecdf(col):
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty(len(col))
    ranks[order] = np.arange(1, len(col) + 1)
    return ranks / (len(col) + 1.0)


def hbos(X, n_bins=None):
    n, d = X.shape
    n_bins = n_bins or max(5, int(np.sqrt(n)))
    out = np.zeros(n)
    for j in range(d):
        hist, edges = np.histogram(X[:, j], bins=n_bins, density=True)
        b = np.clip(np.searchsorted(edges, X[:, j], side="right") - 1,
                    0, n_bins - 1)
        out += -np.log(hist[b] + _EPS)
    return out


def ecod(X):
    """Empirical-CDF outlier detection (Li et al., 2022), skewness-aware."""
    from scipy.stats import skew
    n, d = X.shape
    left = np.zeros((n, d))
    right = np.zeros((n, d))
    for j in range(d):
        u = _ecdf(X[:, j])
        left[:, j] = -np.log(u + _EPS)
        right[:, j] = -np.log(1.0 - u + _EPS)
    sk = skew(X, axis=0)
    chosen = np.where(sk[None, :] < 0, left, right)
    return np.maximum.reduce([left.sum(1), right.sum(1), chosen.sum(1)])


def copod(X):
    """Copula-based outlier detection (Li et al., 2020): the ECOD tail sum
    without the skewness-selected branch, i.e. the empirical copula tail."""
    n, d = X.shape
    left = np.zeros(n)
    right = np.zeros(n)
    for j in range(d):
        u = _ecdf(X[:, j])
        left += -np.log(u + _EPS)
        right += -np.log(1.0 - u + _EPS)
    return np.maximum(left, right)


def quantiles(X):
    """The two-line baseline of the 2025 position paper: how extreme is the
    most extreme coordinate, measured on the empirical quantile scale."""
    n, d = X.shape
    out = np.zeros(n)
    for j in range(d):
        u = _ecdf(X[:, j])
        out = np.maximum(out, np.maximum(u, 1.0 - u))
    return out


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
FAMILY = {}


def _reg(name, family, fn):
    FAMILY[name] = family
    return name, fn


DETECTORS = dict([
    _reg("KNN",        "global",    lambda X, rng: knn_distance(X, 10, "max")),
    _reg("AvgKNN",     "global",    lambda X, rng: knn_distance(X, 10, "mean")),
    _reg("KDE",        "global",    lambda X, rng: kde_score(X, 1.0, rng)),
    _reg("GMM",        "global",    lambda X, rng: gmm_score(X, 4, rng)),
    _reg("Mahalanobis", "global",   lambda X, rng: mahalanobis(X)),
    _reg("MCD",        "global",    lambda X, rng: mcd_mahalanobis(X, rng)),
    _reg("PCA",        "global",    lambda X, rng: pca_reconstruction(X)),
    _reg("OCSVM",      "global",    lambda X, rng: ocsvm(X, 0.1, rng)),
    _reg("LOF",        "local",     lambda X, rng: lof(X, 20)),
    _reg("CBLOF",      "local",     lambda X, rng: cblof(X, 8, rng)),
    _reg("IForest",    "isolation", lambda X, rng: iforest(X, 200, rng)),
    _reg("LODA",       "isolation", lambda X, rng: loda(X, 100, None, rng)),
    _reg("HBOS",       "marginal",  lambda X, rng: hbos(X)),
    _reg("ECOD",       "marginal",  lambda X, rng: ecod(X)),
    _reg("COPOD",      "marginal",  lambda X, rng: copod(X)),
    _reg("MaxQuantile", "marginal", lambda X, rng: quantiles(X)),
])


# --------------------------------------------------------------------------- #
# descriptive, full-sample density scores
# --------------------------------------------------------------------------- #
def density_family(X, rng=None):
    """Return full-sample density rankings for descriptive comparisons.

    The inductive audit uses fit_select_density_score to select and fit
    one density score inside outer training.
    """
    rng = rng or np.random.default_rng(0)
    n, d = X.shape
    out = {}
    ks = [1, 2, 5, 10, 20, 50, 100, 200]
    ks = [k for k in ks if k < n]
    for k in ks:
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
        dist, _ = nn.kneighbors(X)
        dist = dist[:, 1:]
        out[f"knn_max_k{k}"] = dist[:, -1]
        out[f"knn_mean_k{k}"] = dist.mean(axis=1)
    # kernel density is only informative in modest dimension and is by far the
    # most expensive member, so the grid thins as d grows.  The k-NN members
    # above already span the same bias family at every dimension.
    bws = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0] if d <= 50 else [0.5, 1.0, 2.0]
    kcap = 5_000 if d <= 50 else 1_500
    for bw in bws:
        try:
            out[f"kde_bw{bw}"] = kde_score(X, bw, rng, cap=kcap)
        except Exception:
            pass
    for c in [1, 2, 4, 8, 16]:
        if c <= max(1, n // 20):
            try:
                out[f"gmm_c{c}"] = gmm_score(X, c, rng)
            except Exception:
                pass
    out["mahalanobis"] = mahalanobis(X)
    try:
        out["mcd"] = mcd_mahalanobis(X, rng)
    except Exception:
        pass
    return out

# --------------------------------------------------------------------------- #
# leakage-safe inductive density family for the inductive inferential audit
# --------------------------------------------------------------------------- #
def density_candidate_names(n_train, d):
    """Candidate names that may be selected *inside* the outer training set."""
    out=[]
    for k in [1,2,5,10,20,50,100,200]:
        if k < max(2, n_train//3):
            out += [f"knn_max_k{k}", f"knn_mean_k{k}"]
    bws=[0.1,0.25,0.5,1.0,2.0,4.0] if d<=50 else [0.5,1.0,2.0]
    out += [f"kde_bw{bw}" for bw in bws]
    for c in [1,2,4,8,16]:
        if c <= max(1,n_train//40): out.append(f"gmm_c{c}")
    out += ["mahalanobis","mcd"]
    return out

def _mahalanobis_fit_score(Xtr, Xev, robust=False, seed=0):
    if robust and Xtr.shape[1] <= HIGH_D:
        try:
            rng=np.random.default_rng(seed)
            Xf=_fit_subset(Xtr,rng,min(FIT_CAP,2000))
            m=MinCovDet(random_state=seed,support_fraction=0.75).fit(Xf)
            return m.mahalanobis(Xev)
        except Exception:
            pass
    mu=Xtr.mean(axis=0)
    S=np.cov(Xtr,rowvar=False)+1e-6*np.eye(Xtr.shape[1])
    Si=np.linalg.pinv(S)
    Z=Xev-mu
    return np.einsum("ij,jk,ik->i",Z,Si,Z)

def score_density_candidate(name, Xtr, Xev, seed=0):
    """Fit one density ranking on Xtr and score Xev without using Xev in fit."""
    Xtr=np.asarray(Xtr,float); Xev=np.asarray(Xev,float)
    rng=np.random.default_rng(seed)
    if name.startswith('knn_'):
        kind='max' if '_max_' in name else 'mean'
        k=int(name.rsplit('k',1)[1])
        if k >= len(Xtr): raise ValueError('k >= training size')
        nn=NearestNeighbors(n_neighbors=k,n_jobs=-1).fit(Xtr)
        dist,_=nn.kneighbors(Xev)
        return dist[:,-1] if kind=='max' else dist.mean(axis=1)
    if name.startswith('kde_bw'):
        bw=float(name.split('bw',1)[1])
        cap=5000 if Xtr.shape[1]<=50 else 1500
        Xf=_fit_subset(Xtr,rng,cap)
        m=KernelDensity(bandwidth=bw).fit(Xf)
        return -m.score_samples(Xev)
    if name.startswith('gmm_c'):
        c=int(name.split('c',1)[1]); c=min(c,max(1,len(Xtr)//20))
        cov='full' if Xtr.shape[1]<=HIGH_D else 'diag'
        m=GaussianMixture(n_components=c,covariance_type=cov,reg_covar=1e-4,
                          random_state=seed,max_iter=100).fit(Xtr)
        return -m.score_samples(Xev)
    if name=='mahalanobis': return _mahalanobis_fit_score(Xtr,Xev,False,seed)
    if name=='mcd': return _mahalanobis_fit_score(Xtr,Xev,True,seed)
    raise KeyError(name)

def fit_select_density_score(X_tr_raw, y_tr, X_te_raw, seed=0, n_splits=3):
    """Nested, leakage-safe density-family selection.

    Candidate choice uses labels only inside the outer training split.  Each
    inner fold fits its own standardizer and density estimator.  After choosing
    the candidate, preprocessing and the density estimator are refit on all
    outer-training rows and the outer test is scored exactly once.
    """
    from sklearn.model_selection import StratifiedKFold
    import data as D
    import stats as ST
    X_tr_raw=np.asarray(X_tr_raw,float); y_tr=np.asarray(y_tr,int)
    n_pos=int(y_tr.sum()); n_neg=len(y_tr)-n_pos
    n_splits=int(max(2,min(n_splits,n_pos,n_neg)))
    names=density_candidate_names(len(y_tr),X_tr_raw.shape[1])
    if n_pos<2 or n_neg<2:
        names=[n for n in names if n.startswith('knn_')] or ['mahalanobis']
        chosen=names[0]
        Xtr,Xte,_=D.train_standardise(X_tr_raw,X_te_raw)
        return score_density_candidate(chosen,Xtr,Xte,seed),chosen,{chosen:float('nan')}
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    folds=list(skf.split(X_tr_raw,y_tr)); table={}
    for name in names:
        oof=np.full(len(y_tr),np.nan); ok=True
        for j,(itr,iva) in enumerate(folds):
            try:
                Xf,Xv,_=D.train_standardise(X_tr_raw[itr],X_tr_raw[iva])
                oof[iva]=score_density_candidate(name,Xf,Xv,seed+1009*j)
            except Exception:
                ok=False; break
        table[name]=ST.auc(oof,y_tr) if ok and np.isfinite(oof).all() else float('nan')
    valid={k:v for k,v in table.items() if np.isfinite(v)}
    if not valid:
        chosen='mahalanobis'
    else:
        chosen=max(valid,key=valid.get)
    Xtr,Xte,_=D.train_standardise(X_tr_raw,X_te_raw)
    scores=score_density_candidate(chosen,Xtr,Xte,seed+99991)
    return np.asarray(scores,float),chosen,table
