"""Reproduce the post-hoc sensitivity analyses reported in the manuscript.

Run from the root of Online Resource 1:
    python code/additional/run_additional_sensitivity.py

Outputs are written to results/additional_sensitivity/.
The SynBench-aligned experiment reproduces a small public-construction-aligned
GMM stress test; it does NOT download or claim to evaluate released MacrOData files.
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.special import logsumexp, gammaln

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
OUT = ROOT / "results" / "additional_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

import run_audit as RA
import stats as ST
from data import Dataset


def bh(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p)
    vals=p[order]*m/np.arange(1,m+1)
    vals=np.minimum.accumulate(vals[::-1])[::-1]
    q=np.empty(m); q[order]=np.minimum(vals,1.0)
    return q


def by(p):
    p=np.asarray(p,float); m=len(p); c=np.sum(1/np.arange(1,m+1))
    order=np.argsort(p); vals=p[order]*m*c/np.arange(1,m+1)
    vals=np.minimum.accumulate(vals[::-1])[::-1]
    q=np.empty(m); q[order]=np.minimum(vals,1.0)
    return q


def known_lda_case(d,n,seed):
    rng=np.random.default_rng(seed); alpha=.10
    scales=np.exp(np.linspace(-0.8,0.8,d))
    Q,_=np.linalg.qr(rng.normal(size=(d,d)))
    cov=Q@np.diag(scales**2)@Q.T
    X=rng.multivariate_normal(np.zeros(d),cov,size=n)
    inv=np.linalg.inv(cov); sign,ld=np.linalg.slogdet(cov)
    r2=np.einsum('ij,jk,ik->i',X,inv,X)
    logp=-.5*(d*np.log(2*np.pi)+ld+r2)
    y=(r2>=chi2.ppf(1-alpha,d)).astype(int)
    ds=Dataset(f'gaussian_d{d}_s{seed}',X,y,n,d,False,0,0)
    t=time.time(); o=RA._outer_comparison(ds,0); _,te=RA._split(y,0)
    return dict(d=d,n=n,seed=seed,alpha=float(y.mean()),n_test_anom=int(y[te].sum()),
                oracle_auc=float(ST.auc(-logp[te],y[te])),
                density_auc=float(o['auc_density_test']),
                supervised_auc=float(o['auc_supervised_test']),delta=float(o['delta']),
                p=float(o['p_one_sided']),density_member=o['best_density_member'],
                supervised_model=o['supervised_model'],seconds=time.time()-t)


def run_known_lda():
    cases=[(2,300,s) for s in [201,202,203,204,205,206]] + [(5,300,s) for s in [211,212,213]] + [(10,600,401),(25,800,402),(50,1000,403),(100,1500,404)]
    rows=[known_lda_case(*c) for c in cases]
    df=pd.DataFrame(rows); df['q_bh']=bh(df.p); df['flag']=df.q_bh<.05
    df.to_csv(OUT/'known_lda_calibration.csv',index=False)
    sm=dict(n_cases=len(df),dimensions=sorted(df.d.unique().tolist()),n_range=[int(df.n.min()),int(df.n.max())],
            bh_flags=int(df.flag.sum()),raw_p_lt_0_05=int((df.p<.05).sum()),
            median_delta=float(df.delta.median()),delta_range=[float(df.delta.min()),float(df.delta.max())],
            median_density_auc=float(df.density_auc.median()),median_supervised_auc=float(df.supervised_auc.median()),
            oracle_auc_min=float(df.oracle_auc.min()))
    (OUT/'known_lda_calibration_summary.json').write_text(json.dumps(sm,indent=2))
    return sm



def _orthogonal(rng,d):
    Q,R=np.linalg.qr(rng.normal(size=(d,d)))
    signs=np.sign(np.diag(R)); signs[signs==0]=1
    return Q*signs[None,:]


def _non_gaussian_lda_sample(family,d,n,seed,alpha=.10,n_reference=60000):
    """Generate an exact-LDA calibration sample from a known non-Gaussian density.

    A separate Monte Carlo reference sample fixes a low-density threshold before
    the audited sample is drawn. Labels are then deterministic functions of the
    *known population density*: y=1{log p(x) <= c}. Thus the oracle negative-density
    ranking has AUC 1 by construction, while the operational density family remains
    unchanged. The families deliberately include multimodality, heavy tails, and a
    nonlinear banana transformation to move beyond Gaussian calibration.
    """
    rng=np.random.default_rng(seed)
    if family=='gmm':
        m=3; Q=_orthogonal(rng,d)
        means=np.vstack([
            np.zeros(d),
            np.r_[np.full(min(2,d),2.5),np.zeros(max(0,d-2))],
            np.r_[np.full(min(2,d),-2.0),np.zeros(max(0,d-2))]
        ]) @ Q.T
        scales=[np.linspace(.6,1.5,d),np.linspace(1.2,.5,d),np.linspace(.8,1.8,d)]
        covs=[Q@np.diag(z*z)@Q.T for z in scales]
        invs=[np.linalg.inv(c) for c in covs]
        logdets=[np.linalg.slogdet(c)[1] for c in covs]
        weights=np.array([.45,.35,.20]); logw=np.log(weights)
        def draw(k):
            z=rng.choice(m,size=k,p=weights); X=np.empty((k,d))
            for j in range(m):
                ix=np.flatnonzero(z==j)
                if len(ix): X[ix]=rng.multivariate_normal(means[j],covs[j],size=len(ix))
            return X
        def logp(X):
            parts=[]
            for j in range(m):
                D=X-means[j]; r2=np.einsum('ij,jk,ik->i',D,invs[j],D)
                parts.append(logw[j]-.5*(d*np.log(2*np.pi)+logdets[j]+r2))
            return logsumexp(np.vstack(parts),axis=0)
    elif family=='student_t':
        nu=3.0; Q=_orthogonal(rng,d); scales=np.linspace(.7,1.6,d)
        S=Q@np.diag(scales**2)@Q.T; Si=np.linalg.inv(S); ld=np.linalg.slogdet(S)[1]
        const=gammaln((nu+d)/2)-gammaln(nu/2)-.5*(d*np.log(nu*np.pi)+ld)
        def draw(k):
            z=rng.multivariate_normal(np.zeros(d),S,size=k)
            u=rng.chisquare(nu,size=k)/nu
            return z/np.sqrt(u)[:,None]
        def logp(X):
            q=np.einsum('ij,jk,ik->i',X,Si,X)
            return const-.5*(nu+d)*np.log1p(q/nu)
    elif family=='banana':
        b=.35; Q=_orthogonal(rng,d)
        def draw(k):
            Z=rng.normal(size=(k,d)); W=Z.copy()
            if d>=2: W[:,1]=W[:,1]+b*(W[:,0]**2-1.0)
            return W@Q.T
        def logp(X):
            W=X@Q; Z=W.copy()
            if d>=2: Z[:,1]=W[:,1]-b*(W[:,0]**2-1.0)
            # The triangular banana transform has unit Jacobian.
            return -.5*(d*np.log(2*np.pi)+(Z*Z).sum(1))
    else:
        raise ValueError(family)

    reference=draw(n_reference)
    threshold=float(np.quantile(logp(reference),alpha))
    X=draw(n); lp=logp(X); y=(lp<=threshold).astype(int)
    # Extremely unlikely guard for a stratified outer split; redraw the whole
    # audited sample without changing the pre-fixed threshold.
    for _ in range(5):
        if 20 <= int(y.sum()) <= n-20: break
        X=draw(n); lp=logp(X); y=(lp<=threshold).astype(int)
    return X,y,lp,threshold


def run_non_gaussian_lda():
    cases=[]
    for family,base in [('gmm',510),('student_t',610),('banana',710)]:
        for j,d in enumerate([2,5,10]):
            cases.append((family,d,1000 if d<10 else 1200,base+j))
    rows=[]
    for family,d,n,seed in cases:
        print(f'non_gaussian_lda {family} d={d} seed={seed}', flush=True)
        X,y,lp,threshold=_non_gaussian_lda_sample(family,d,n,seed)
        ds=Dataset(f'{family}_d{d}_s{seed}',X,y,n,d,False,0,0)
        t=time.time(); o=RA._outer_comparison(ds,0); _,te=RA._split(y,0)
        rows.append(dict(family=family,d=d,n=n,seed=seed,alpha=float(y.mean()),
                         n_test_anom=int(y[te].sum()),log_density_threshold=threshold,
                         oracle_auc=float(ST.auc(-lp[te],y[te])),
                         density_auc=float(o['auc_density_test']),
                         supervised_auc=float(o['auc_supervised_test']),delta=float(o['delta']),
                         p=float(o['p_one_sided']),density_member=o['best_density_member'],
                         supervised_model=o['supervised_model'],seconds=time.time()-t))
    df=pd.DataFrame(rows); df['q_bh']=bh(df.p); df['flag_bh']=df.q_bh<.05
    df.to_csv(OUT/'non_gaussian_lda_calibration.csv',index=False)
    sm=dict(n_cases=len(df),families=sorted(df.family.unique().tolist()),
            dimensions=sorted(df.d.unique().tolist()),bh_flags=int(df.flag_bh.sum()),
            raw_p_lt_0_05=int((df.p<.05).sum()),median_delta=float(df.delta.median()),
            delta_range=[float(df.delta.min()),float(df.delta.max())],
            median_density_auc=float(df.density_auc.median()),
            median_supervised_auc=float(df.supervised_auc.median()),
            oracle_auc_min=float(df.oracle_auc.min()))
    (OUT/'non_gaussian_lda_calibration_summary.json').write_text(json.dumps(sm,indent=2))
    return sm


def run_duplicate_exclusion():
    audit=pd.read_csv(ROOT/'results'/'density_gap_audit.csv')
    # one rep0 JSON per dataset records duplicate_rows on the ORIGINAL matrix.
    inv=[]
    for f in sorted((ROOT/'results'/'per_dataset').glob('*__rep0.json')):
        r=json.loads(f.read_text())
        dup=int(r.get('duplicate_rows',-1)); n=int(r['n_original'])
        inv.append(dict(dataset=r['dataset'],n_original=n,duplicate_rows=dup,
                        duplicate_rate=(dup/n if dup>=0 else np.nan)))
    inv=pd.DataFrame(inv).drop_duplicates('dataset')
    inv.to_csv(OUT/'duplicate_inventory.csv',index=False)
    # The frozen audit table already contains n_original and duplicate_rows.
    # Drop overlapping inventory fields before merging so this script works
    # both with the archived table and with a fresh rerun of the pipeline.
    overlap=[c for c in ('n_original','duplicate_rows','duplicate_rate') if c in audit.columns]
    base=audit.drop(columns=overlap).merge(inv,on='dataset',how='left',validate='one_to_one')
    pcol='p_one_sided' if 'p_one_sided' in base.columns else 'p'
    if pcol not in base.columns:
        raise KeyError('density_gap_audit.csv must contain p or p_one_sided')
    specs=[('all',np.ones(len(base),dtype=bool)),
           ('exclude_dup_ge_1pct',~(base.duplicate_rate>=.01)),
           ('exclude_dup_ge_5pct',~(base.duplicate_rate>=.05)),
           ('exclude_dup_ge_10pct',~(base.duplicate_rate>=.10)),
           ('exclude_any_recorded_duplicate',~(base.duplicate_rows>0)),
           # Strict sensitivity: retain only datasets whose ORIGINAL matrix was
           # exhaustively checked and contained zero exact duplicate feature rows.
           # This also excludes duplicate status = -1 (not exhaustively counted).
           ('confirmed_zero_duplicates_only',base.duplicate_rows==0)]
    rows=[]
    for name,keep in specs:
        d=base.loc[keep].copy(); p=d[pcol].to_numpy(float)
        qbh=bh(p); qby=by(p); qbon=np.minimum(p*len(p),1)
        rows.append(dict(sensitivity=name,n_datasets=len(d),excluded=len(base)-len(d),
                         bh_flags=int((qbh<.05).sum()),by_flags=int((qby<.05).sum()),
                         bonf_flags=int((qbon<.05).sum()),median_delta=float(d.delta.median()),
                         positive_delta=int((d.delta>0).sum())))
    out=pd.DataFrame(rows); out.to_csv(OUT/'duplicate_exclusion_sensitivity.csv',index=False)
    return rows



def run_low_count_sensitivity():
    """Recompute multiplicity correction after minimum outer-test anomaly filters.

    This addresses small-positive-sample sensitivity of the asymptotic DeLong
    approximation without reusing or pooling additional splits.
    """
    audit=pd.read_csv(ROOT/'results'/'density_gap_audit.csv')
    pcol='p' if 'p' in audit.columns else 'p_one_sided'
    ncol='n_anom_test' if 'n_anom_test' in audit.columns else 'n_test_anom'
    rows=[]
    for minimum in [2,5,10,20,30,50]:
        d=audit.loc[audit[ncol]>=minimum].copy(); p=d[pcol].to_numpy(float)
        qbh=bh(p); qby=by(p); qbon=np.minimum(p*len(p),1)
        rows.append(dict(min_test_anomalies=minimum,n_datasets=len(d),
                         excluded=len(audit)-len(d),bh_flags=int((qbh<.05).sum()),
                         by_flags=int((qby<.05).sum()),bonf_flags=int((qbon<.05).sum()),
                         median_delta=float(d.delta.median()),positive_delta=int((d.delta>0).sum())))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'small_anomaly_sensitivity.csv',index=False)
    return rows

def make_synbench_aligned(seed,d=2,n=300,alpha=.10):
    """Compact GMM local-anomaly construction aligned to public SynBench/OutFormer logic."""
    rng=np.random.default_rng(seed); m=3
    means=[]; covs=[]
    for _ in range(m):
        means.append(rng.random(d)*rng.integers(-5,5,size=d))
        diag=rng.random(d)*rng.integers(1,6,size=d); diag[diag<=1e-8]=2.5
        covs.append(np.diag(diag))
    sub=np.sort(rng.choice(d,size=int(rng.integers(1,d+1)),replace=False))
    invsubs=[np.linalg.inv(c[np.ix_(sub,sub)]) for c in covs]
    inflated=[]
    for c in covs:
        cc=c.copy(); cc[np.ix_(sub,sub)]*=5.0; inflated.append(cc)
    lo=chi2.ppf(.85,df=len(sub)); hi=chi2.ppf(.95,df=len(sub))
    def mind2(X):
        vals=[]
        for mu,iv in zip(means,invsubs):
            D=X[:,sub]-mu[sub]; vals.append(np.einsum('ij,jk,ik->i',D,iv,D))
        return np.min(np.vstack(vals),axis=0)
    def draw(target,outs=False):
        acc=[]; covlist=inflated if outs else covs
        while sum(len(a) for a in acc)<target:
            k=max(128,2*(target-sum(len(a) for a in acc)))
            z=rng.integers(0,m,size=k); X=np.empty((k,d))
            for j in range(m):
                ix=np.where(z==j)[0]
                if len(ix): X[ix]=rng.multivariate_normal(means[j],covlist[j],size=len(ix))
            md=mind2(X); mask=(md>hi) if outs else (md<lo)
            if mask.any(): acc.append(X[mask])
        return np.vstack(acc)[:target]
    no=int(round(n*alpha)); ni=n-no
    Xin=draw(ni,False); Xout=draw(no,True)
    X=np.vstack([Xin,Xout]); y=np.r_[np.zeros(ni,int),np.ones(no,int)]
    perm=rng.permutation(n)
    return X[perm],y[perm],len(sub)


def run_synbench_aligned():
    # Exact six frozen cases used by the manuscript output.
    cases=[(2,s) for s in [301,302,303,304,305]] + [(5,311)]
    rows=[]
    for d,seed in cases:
        X,y,k=make_synbench_aligned(seed,d=d)
        ds=Dataset(f'synbench_gmm_d{d}_s{seed}',X,y,len(y),d,False,0,0)
        t=time.time(); o=RA._outer_comparison(ds,0)
        rows.append(dict(d=d,seed=seed,n=len(y),alpha=float(y.mean()),inflated_dims=k,
                         density_auc=float(o['auc_density_test']),supervised_auc=float(o['auc_supervised_test']),
                         delta=float(o['delta']),p=float(o['p_one_sided']),
                         density_member=o['best_density_member'],supervised_model=o['supervised_model'],seconds=time.time()-t))
    df=pd.DataFrame(rows); df['q_bh']=bh(df.p); df['flag_bh']=df.q_bh<.05
    df.to_csv(OUT/'synbench_aligned_gmm_stress.csv',index=False)
    sm=dict(n_cases=len(df),bh_flags=int(df.flag_bh.sum()),raw_p_lt_0_05=int((df.p<.05).sum()),
            median_delta=float(df.delta.median()),delta_range=[float(df.delta.min()),float(df.delta.max())],
            median_density_auc=float(df.density_auc.median()),median_supervised_auc=float(df.supervised_auc.median()))
    (OUT/'synbench_aligned_gmm_stress_summary.json').write_text(json.dumps(sm,indent=2))
    return sm

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['known_lda','non_gaussian_lda','duplicates','small_counts','synbench'],default=None,
                    help='sensitivity stage to reproduce')
    args=ap.parse_args()
    if args.stage is None:
        print('Run the four reproducibility stages separately:')
        print('  python code/additional/run_additional_sensitivity.py --stage known_lda')
        print('  python code/additional/run_additional_sensitivity.py --stage non_gaussian_lda')
        print('  python code/additional/run_additional_sensitivity.py --stage duplicates')
        print('  python code/additional/run_additional_sensitivity.py --stage small_counts')
        print('  python code/additional/run_additional_sensitivity.py --stage synbench')
        raise SystemExit(0)
    if args.stage=='known_lda':
        print('Known LDA:',json.dumps(run_known_lda(),indent=2))
    elif args.stage=='non_gaussian_lda':
        print('Non-Gaussian LDA:',json.dumps(run_non_gaussian_lda(),indent=2))
    elif args.stage=='duplicates':
        print('Duplicate exclusion:',json.dumps(run_duplicate_exclusion(),indent=2))
    elif args.stage=='small_counts':
        print('Small-anomaly sensitivity:',json.dumps(run_low_count_sensitivity(),indent=2))
    else:
        print('SynBench-aligned:',json.dumps(run_synbench_aligned(),indent=2))
