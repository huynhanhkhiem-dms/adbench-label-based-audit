"""ADBench Classical data loading with leakage-safe preprocessing.

The inductive audit splits rows before any feature transformation used by an
inferential comparison.  ``load`` therefore returns raw (or raw stratified-
subsampled) features.  Scaling is fitted only on the relevant training fold.
"""
from __future__ import annotations

import hashlib, os
from dataclasses import dataclass, field
import numpy as np

DEFAULT_ROOT = os.environ.get(
    "ADBENCH_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "..", "ADBench", "adbench", "datasets", "Classical"),
)
N_MAX = 20_000
N_REPLICATES = 5

@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    n_original: int
    d_original: int
    subsampled: bool
    replicate: int = 0
    duplicate_rows: int = 0
    meta: dict = field(default_factory=dict)
    @property
    def n(self): return int(self.X.shape[0])
    @property
    def d(self): return int(self.X.shape[1])
    @property
    def alpha(self): return float(self.y.mean())
    def digest(self):
        h=hashlib.blake2b(digest_size=8)
        h.update(np.ascontiguousarray(self.X).tobytes())
        h.update(np.ascontiguousarray(self.y).tobytes())
        return h.hexdigest()

@dataclass
class Standardizer:
    keep: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    def transform(self, X):
        Z=np.asarray(X, dtype=np.float64)[:, self.keep]
        return np.ascontiguousarray((Z-self.mean)/self.scale)
    @property
    def dropped(self): return int((~self.keep).sum())

def list_datasets(root=DEFAULT_ROOT):
    root=os.path.abspath(root)
    files=[f for f in os.listdir(root) if f.endswith('.npz')]
    def key(f):
        try: return int(f.split('_')[0])
        except Exception: return 10**9
    files.sort(key=lambda f:(key(f),f))
    return [f[:-4] for f in files]

def _raw(name, root=DEFAULT_ROOT):
    path=os.path.join(os.path.abspath(root), name+'.npz')
    with np.load(path) as arch:
        return np.asarray(arch['X'],np.float64), np.asarray(arch['y'],np.int64).ravel()

def _count_duplicates(X):
    if X.shape[0] > 200_000: return -1
    X=np.ascontiguousarray(X)
    view=X.view(np.dtype((np.void, X.dtype.itemsize*X.shape[1])))
    return int(X.shape[0]-np.unique(view).shape[0])

def _stratified_subsample(X,y,n_max,seed):
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    frac=n_max/len(y)
    n_pos=min(len(pos), max(2,int(round(len(pos)*frac))))
    n_neg=min(len(neg), max(2,n_max-n_pos))
    idx=np.concatenate([rng.choice(pos,n_pos,False), rng.choice(neg,n_neg,False)])
    rng.shuffle(idx)
    return X[idx],y[idx]

def load(name, replicate=0, root=DEFAULT_ROOT, n_max=N_MAX):
    """Load raw features; subsample if needed; do not standardize here."""
    X,y=_raw(name,root); n0,d0=X.shape; dup=_count_duplicates(X)
    subsampled=n0>n_max
    if subsampled:
        X,y=_stratified_subsample(X,y,n_max,1000*(replicate+1)+7)
    return Dataset(name,np.ascontiguousarray(X),y,n0,d0,subsampled,replicate,dup)

def fit_standardizer(X):
    X=np.asarray(X,np.float64)
    sd=X.std(axis=0)
    keep=np.isfinite(sd) & (sd>1e-12)
    if not keep.any():
        keep=np.ones(X.shape[1],dtype=bool)
    Z=X[:,keep]
    mu=np.nanmean(Z,axis=0)
    sc=np.nanstd(Z,axis=0)
    sc=np.where(np.isfinite(sc)&(sc>1e-12),sc,1.0)
    return Standardizer(keep,mu,sc)

def train_standardise(X_train, X_eval):
    prep=fit_standardizer(X_train)
    return prep.transform(X_train), prep.transform(X_eval), prep

def standardise_all(X):
    """Descriptive-only full-sample transform. Never use for held-out inference."""
    p=fit_standardizer(X)
    return p.transform(X), p

def n_replicates(name,root=DEFAULT_ROOT,n_max=N_MAX):
    path=os.path.join(os.path.abspath(root),name+'.npz')
    with np.load(path) as arch: n=arch['X'].shape[0]
    return N_REPLICATES if n>n_max else 1

def inventory(root=DEFAULT_ROOT):
    rows=[]
    for name in list_datasets(root):
        X,y=_raw(name,root); p=fit_standardizer(X)
        rows.append(dict(dataset=name,n=X.shape[0],d=X.shape[1],
                         d_effective=int(p.keep.sum()),dropped=p.dropped,
                         n_anomalies=int(y.sum()),alpha=float(y.mean()),
                         duplicates=_count_duplicates(X)))
    return rows
