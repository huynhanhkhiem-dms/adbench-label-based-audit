"""VJCS Reviewer R1.4 / R2.4.2: exact-float64 ties and density candidate test AUC audit.

Run only with the original public ADBench Classical .npz datasets. Writes checkpointed
CSV artifacts; never reuses outer-test values for model selection. All methods and
seed conventions match the archived run_audit.py, which remains authoritative.

Usage:
    python code/reviewer_reanalysis.py --data-root /path/to/Classical --out results/reviewer_raw
    python code/reviewer_reanalysis.py --data-root /path/to/Classical --only 14_glass

This is a reviewer-specific diagnostic tool, not a replacement for the primary audit.
"""
from __future__ import annotations
import argparse, csv, json, os, sys, traceback
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0,str(Path(__file__).resolve().parent))
import data as D
import detectors as DET
import stats as ST
from supervised import fit_select_score
from run_audit import _split


def tie_stats(s,y):
    s=np.asarray(s,dtype=np.float64);y=np.asarray(y,dtype=int)
    uniq,idx,cnt=np.unique(s,return_inverse=True,return_counts=True)
    np1=int(y.sum());nn=len(y)-np1
    pc=np.bincount(idx,weights=(y==1).astype(int),minlength=len(uniq))
    return {'n':int(len(y)),
            'duplicate_score_fraction':float(1-len(uniq)/len(y)),
            'cross_class_tie_pair_fraction':float((pc*(cnt-pc)).sum()/(np1*nn)) if np1>0 and nn>0 else np.nan}


def _write(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)


def one(name,root,out,all_density=False):
    ds=D.load(name,replicate=0,root=root)
    # Section 1: 16 descriptor score ties on the conventional full-sample
    # transductive benchmark, not inferential scores.
    Xdesc,_=D.standardise_all(ds.X)
    rr=np.random.default_rng(1234)
    rows=[]
    for label,fn in DET.DETECTORS.items():
        try:
            s=np.asarray(fn(Xdesc,rr),dtype=np.float64)
            a=ST.auc(s,ds.y)
            rows.append({'dataset':name,'detector':label,'auc':float(a),**tie_stats(s,ds.y),'status':'ok','error':''})
        except Exception as e:
            rows.append({'dataset':name,'detector':label,'auc':np.nan,'n':int(len(ds.y)),
                         'duplicate_score_fraction':np.nan,'cross_class_tie_pair_fraction':np.nan,
                         'status':'error','error':str(e)[:180]})
    _write(out/f'{name}__16_detector_ties.csv',rows)

    # Section 2: real float64 inductive scores and tied-pair DeLong test.
    tr,te=_split(ds.y,0); yt=ds.y[te]
    sd,selected,cv=DET.fit_select_density_score(ds.X[tr],ds.y[tr],ds.X[te],seed=0)
    ss,sup,supcv=fit_select_score(ds.X[tr],ds.y[tr],ds.X[te],seed=0)
    delta,se,z,p=ST.delong_test(ss,sd,yt,'greater')
    main=[{'dataset':name,'selected_density':selected,'supervised_model':sup,
           'auc_density_float64':ST.auc(sd,yt),'auc_supervised_float64':ST.auc(ss,yt),
           'delta':delta,'se':se,'delong_z':z,'one_sided_p':p,
           'density_duplicate_score_fraction':tie_stats(sd,yt)['duplicate_score_fraction'],
           'supervised_duplicate_score_fraction':tie_stats(ss,yt)['duplicate_score_fraction'],
           'density_cross_class_tie_pair_fraction':tie_stats(sd,yt)['cross_class_tie_pair_fraction'],
           'supervised_cross_class_tie_pair_fraction':tie_stats(ss,yt)['cross_class_tie_pair_fraction']}]
    _write(out/f'{name}__outer_float64_ties.csv',main)

    if all_density:
        # Every eligible candidate evaluated on outer test for diagnostic
        # plotting *after* model selection, NOT for choosing models or p-values.
        Xtr,Xte,_=D.train_standardise(ds.X[tr],ds.X[te]); vals=[]
        for candidate in DET.density_candidate_names(len(tr),Xtr.shape[1]):
            try:
                scores=DET.score_density_candidate(candidate,Xtr,Xte,seed=99991)
                vals.append({'dataset':name,'candidate':candidate,'selected_in_train_cv':candidate==selected,
                             'heldout_auc_post_selection_diagnostic':ST.auc(scores,yt),'status':'ok','error':''})
            except Exception as e:
                vals.append({'dataset':name,'candidate':candidate,'selected_in_train_cv':candidate==selected,
                             'heldout_auc_post_selection_diagnostic':np.nan,'status':'error','error':str(e)[:180]})
        _write(out/f'{name}__all_density_candidates_posthoc.csv',vals)
    print(f'{name}: selected={selected}, p={p:.4g}; 16-detector and float64 tie summaries saved',flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True)
    ap.add_argument('--out',default='results/reviewer_raw')
    ap.add_argument('--only',nargs='*',default=None)
    ap.add_argument('--all-density',action='store_true')
    args=ap.parse_args()
    root=Path(args.data_root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    names=args.only or D.list_datasets(str(root))
    for name in names:
        if (out/f'{name}__outer_float64_ties.csv').exists() and not args.all_density:
            print('skip already completed',name,flush=True);continue
        try:one(name,str(root),out,args.all_density)
        except Exception as e:
            (out/f'{name}__error.txt').write_text(traceback.format_exc())
            print('FAIL',name,str(e),flush=True)

if __name__=='__main__':main()
