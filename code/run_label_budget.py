"""Descriptive label-budget curves using untouched primary-test scores.

The scoring rules are fixed by outer training.  We repeatedly subsample labelled
anomalies from the archived outer test while retaining all available normal
rows.  This is a conditional-on-training power description, not a universal
sample-size guarantee.
"""
from __future__ import annotations
import glob,json,os,sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import stats as ST
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS=os.path.join(ROOT,'results'); PROTOCOL='inductive_outer_test'
GRID=(3,5,10,20,50,100,200,500); N_REP=200

def main():
    out={}; rng=np.random.default_rng(20260818)
    for f in sorted(glob.glob(os.path.join(RESULTS,'scores','*.npz'))):
        with np.load(f,allow_pickle=True) as z:
            pv=str(z['protocol_version'].item())
            if pv!=PROTOCOL: continue
            y=z['y_test'].astype(int); a=z['s_supervised_test'].astype(float); b=z['s_density_test'].astype(float)
        pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0); curve={}
        for m in GRID:
            if m>len(pos): continue
            hits=0
            for _ in range(N_REP):
                pp=rng.choice(pos,m,replace=False); idx=np.concatenate([pp,neg])
                _,_,_,p=ST.delong_test(a[idx],b[idx],y[idx],'greater')
                hits += int(p<0.05)
            curve[str(m)]=hits/N_REP
        out[os.path.splitext(os.path.basename(f))[0]]=curve
    with open(os.path.join(RESULTS,'label_budget.json'),'w') as fh: json.dump(out,fh,indent=1)
    print('wrote curves for',len(out),'datasets')
if __name__=='__main__': main()
