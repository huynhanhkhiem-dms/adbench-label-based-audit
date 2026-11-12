"""ADBench audit with strict outer-test isolation.

Raw rows are split before preprocessing. Every transformation, density estimator,
and supervised learner used in the inferential comparison is fitted inside outer
training data; candidate selection uses nested inner folds; the outer test is
scored exactly once. DeLong and paired bootstrap inference are interpreted
conditional on the fixed train-fitted scoring rules. Additional outer seeds are
descriptive robustness checks and are not pooled into the primary p-value.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import json, os, sys, time, traceback
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import data as D, detectors as DET, stats as ST
from supervised import fit_select_score

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS=os.path.join(ROOT,'results'); PER_DATASET=os.path.join(RESULTS,'per_dataset')
PROTOCOL_VERSION='inductive_outer_test'
PRIMARY_SEED=0; TEST_FRACTION=0.5
ROBUSTNESS_SEEDS=(11,23,37,53,71)

def _split(y,seed):
    s=StratifiedShuffleSplit(n_splits=1,test_size=TEST_FRACTION,random_state=seed)
    return next(s.split(np.zeros(len(y)),y))

def _clean_scores(s):
    s=np.asarray(s,float).ravel()
    if np.isfinite(s).all(): return s
    finite=s[np.isfinite(s)]
    fill=float(np.median(finite)) if len(finite) else 0.0
    return np.nan_to_num(s,nan=fill,posinf=float(np.max(finite)) if len(finite) else 1.0,
                         neginf=float(np.min(finite)) if len(finite) else -1.0)

def descriptive_scores(ds,rng):
    """Conventional transductive benchmark scores; descriptive only, no p-values."""
    Xs,prep=D.standardise_all(ds.X); out={}; timing={}
    for name,fn in DET.DETECTORS.items():
        t=time.time()
        try: out[name]=_clean_scores(fn(Xs,rng))
        except Exception:
            out[name]=None; timing[name+'_error']=traceback.format_exc(limit=1)
        timing[name]=round(time.time()-t,3)
    return out,timing,prep

def _outer_comparison(ds,seed):
    tr,te=_split(ds.y,seed); ytr,yte=ds.y[tr],ds.y[te]
    if ytr.sum()<2 or (len(ytr)-ytr.sum())<2 or yte.sum()<2 or (len(yte)-yte.sum())<2:
        return None
    s_dens,dens_name,dens_cv=DET.fit_select_density_score(ds.X[tr],ytr,ds.X[te],seed=seed)
    s_sup,sup_name,sup_cv=fit_select_score(ds.X[tr],ytr,ds.X[te],seed=seed)
    delta,se,z,p=ST.delong_test(s_sup,s_dens,yte,'greater')
    b_delta,b_lo,b_hi=ST.paired_bootstrap_auc_diff(s_sup,s_dens,yte,n_boot=2000,seed=seed)
    return dict(
        split_seed=int(seed), n_train=int(len(tr)), n_test=int(len(te)),
        n_anom_train=int(ytr.sum()), n_anom_test=int(yte.sum()),
        best_density_member=dens_name, density_cv_auc=dens_cv,
        supervised_model=sup_name, supervised_cv_auc=sup_cv,
        auc_density_test=ST.auc(s_dens,yte), auc_supervised_test=ST.auc(s_sup,yte),
        delta=float(delta), se=float(se), z=float(z), p_one_sided=float(p),
        boot_delta=float(b_delta), boot_lo=float(b_lo), boot_hi=float(b_hi),
        train_idx=tr.astype(np.int32), test_idx=te.astype(np.int32),
        s_density_test=np.asarray(s_dens,np.float32),
        s_supervised_test=np.asarray(s_sup,np.float32), y_test=yte.astype(np.int8),
    )

def _jsonable_outer(o):
    if o is None: return None
    return {k:v for k,v in o.items() if k not in ('train_idx','test_idx','s_density_test','s_supervised_test','y_test')}

def run_dataset(name,replicate=0,force=False,robustness_splits=0):
    os.makedirs(PER_DATASET,exist_ok=True)
    path=os.path.join(PER_DATASET,f'{name}__rep{replicate}.json')
    if os.path.exists(path) and not force:
        with open(path) as fh:
            old=json.load(fh)
        if old.get('protocol_version')==PROTOCOL_VERSION:
            return old
    t0=time.time(); rng=np.random.default_rng(1234+17*replicate)
    ds=D.load(name,replicate=replicate)
    rec=dict(protocol_version=PROTOCOL_VERSION,dataset=name,replicate=int(replicate),
             n=ds.n,d=ds.d,n_original=ds.n_original,d_original=ds.d_original,
             alpha=ds.alpha,subsampled=ds.subsampled,duplicate_rows=ds.duplicate_rows,
             digest=ds.digest())

    # Conventional full-sample benchmark scores are retained as descriptive
    # context only. They are not used by the inductive inferential audit.
    scores,timing,prep=descriptive_scores(ds,rng)
    keys=[k for k,v in scores.items() if v is not None]
    rec['descriptive_transductive']=True
    rec['dropped_columns_descriptive']=prep.dropped
    rec['detector_auc']={k:(ST.auc(v,ds.y) if v is not None else None) for k,v in scores.items()}
    rec['detector_ap']={k:(ST.average_precision(v,ds.y) if v is not None else None) for k,v in scores.items()}
    rec['timing']=timing
    base=scores.get('KNN')
    rec['vs_knn']={k:float(ST.auc(v,ds.y)-ST.auc(base,ds.y)) for k,v in scores.items()
                   if base is not None and v is not None and k!='KNN'}
    from scipy.stats import kendalltau
    sub=np.random.default_rng(11).choice(ds.n,4000,replace=False) if ds.n>4000 else np.arange(ds.n)
    rec['kendall_tau']={}
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            rec['kendall_tau'][f'{a}|{b}']=float(kendalltau(scores[a][sub],scores[b][sub]).statistic)

    primary=_outer_comparison(ds,PRIMARY_SEED)
    rec['density_gap_primary']=_jsonable_outer(primary)
    extra=[]
    if replicate==0:
        for seed in ROBUSTNESS_SEEDS[:max(0,int(robustness_splits))]:
            extra.append(_jsonable_outer(_outer_comparison(ds,seed)))
    rec['density_gap_robustness']=[x for x in extra if x is not None]

    # Archive untouched primary-test score vectors for label-budget analysis.
    if replicate==0 and primary is not None:
        sdir=os.path.join(RESULTS,'scores'); os.makedirs(sdir,exist_ok=True)
        np.savez_compressed(os.path.join(sdir,f'{name}.npz'),
            protocol_version=np.array(PROTOCOL_VERSION),
            train_idx=primary['train_idx'],test_idx=primary['test_idx'],
            y_test=primary['y_test'],
            s_density_test=primary['s_density_test'],
            s_supervised_test=primary['s_supervised_test'])

    rec['seconds']=round(time.time()-t0,2)
    with open(path,'w') as fh: json.dump(rec,fh,indent=1,default=float)
    return rec

def main(argv):
    os.makedirs(PER_DATASET,exist_ok=True)
    force='--force' in argv
    only=[a for a in argv[1:] if not a.startswith('--')]
    budget=float(os.environ.get('AUDIT_BUDGET_S','1e9'))
    robustness=int(os.environ.get('AUDIT_ROBUSTNESS_SPLITS','0'))
    names=D.list_datasets()
    if only:
        known=set(names); names=[x for x in only if x in known]
    begin=time.time()
    for name in names:
        for rep in range(D.n_replicates(name)):
            if time.time()-begin>budget:
                print('budget reached; safe to rerun the same command to resume',flush=True); return
            r=run_dataset(name,rep,force,robustness)
            c=r.get('density_gap_primary') or {}
            print(f"{name:26s} rep{rep} n={r['n']:6d} d={r['d']:4d} "
                  f"dens={c.get('auc_density_test',float('nan')):.4f} "
                  f"sup={c.get('auc_supervised_test',float('nan')):.4f} "
                  f"D={c.get('delta',float('nan')):+.4f} p={c.get('p_one_sided',float('nan')):.2e} "
                  f"[{r['seconds']:.1f}s]",flush=True)

if __name__=='__main__': main(sys.argv)
