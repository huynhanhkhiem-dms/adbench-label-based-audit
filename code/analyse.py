"""Aggregate outputs from the inductive outer-test protocol."""
from __future__ import annotations
import glob,json,os,sys
import numpy as np, pandas as pd
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import detectors as DET, stats as ST
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS=os.path.join(ROOT,'results'); PER=os.path.join(RESULTS,'per_dataset')
PROTOCOL='inductive_outer_test'; Q=0.05; DETS=list(DET.DETECTORS)

def load_records():
    out=[]
    for f in sorted(glob.glob(os.path.join(PER,'*.json'))):
        with open(f) as fh:r=json.load(fh)
        if r.get('protocol_version')==PROTOCOL: out.append(r)
    return out

def build(recs):
    rows=[]
    for r in recs:
        c=r.get('density_gap_primary') or {}
        row={k:r.get(k) for k in ['dataset','replicate','n','d','n_original','d_original','alpha','subsampled','duplicate_rows']}
        row.update(protocol_version=r.get('protocol_version'),n_test=c.get('n_test'),n_anom_test=c.get('n_anom_test'),
                   auc_dens=c.get('auc_density_test'),auc_sup=c.get('auc_supervised_test'),delta=c.get('delta'),se=c.get('se'),
                   p=c.get('p_one_sided'),boot_lo=c.get('boot_lo'),boot_hi=c.get('boot_hi'),
                   sup_model=c.get('supervised_model'),dens_member=c.get('best_density_member'),seconds=r.get('seconds'))
        for k,v in (r.get('detector_auc') or {}).items(): row['auc_'+k]=v
        rows.append(row)
    return pd.DataFrame(rows)

def density_gap_table(df):
    prim=df[df.replicate==0].copy().sort_values('dataset').reset_index(drop=True)
    ok=prim.p.notna().to_numpy(); p=prim.loc[ok,'p'].to_numpy()
    bh,qbh=ST.benjamini_hochberg(p,Q); by,qby=ST.benjamini_yekutieli(p,Q); bo,qbo=ST.bonferroni(p,Q)
    for name,rej,adj in [('bh',bh,qbh),('by',by,qby),('bonf',bo,qbo)]:
        rr=np.zeros(len(prim),bool); qq=np.full(len(prim),np.nan); rr[ok]=rej; qq[ok]=adj
        prim['flag_'+name]=rr; prim['q_'+name]=qq
    prim['density_gap_flagged']=prim.flag_bh
    return prim

def leaderboard(df,subset=None):
    d=df if subset is None else df[df.dataset.isin(subset)]
    cols=[f'auc_{k}' for k in DETS if f'auc_{k}' in d]
    per=d.groupby('dataset')[cols].mean(); ranks=per.rank(axis=1,ascending=False)
    rows=[]
    for k in DETS:
        c='auc_'+k
        if c in per:
            rows.append(dict(detector=k,family=DET.FAMILY[k],mean_auc=float(per[c].mean()),median_auc=float(per[c].median()),
                             mean_rank=float(ranks[c].mean()),n_wins=int((ranks[c]==1).sum()),n_datasets=int(len(per))))
    return pd.DataFrame(rows).sort_values('mean_auc',ascending=False).reset_index(drop=True)

def vs_knn_table(recs):
    return pd.DataFrame([dict(dataset=r['dataset'],replicate=r['replicate'],detector=k,delta=float(v))
                         for r in recs if r['replicate']==0 for k,v in (r.get('vs_knn') or {}).items()])

def agreement_table(recs,audit):
    fmap=dict(zip(audit.dataset,audit.flag_bh)); rows=[]
    for r in recs:
        if r['replicate']!=0 or not r.get('kendall_tau'): continue
        within=[]; between=[]
        for key,v in r['kendall_tau'].items():
            a,b=key.split('|'); (within if DET.FAMILY[a]==DET.FAMILY[b] else between).append(v)
        aucs=np.array([v for v in (r.get('detector_auc') or {}).values() if v is not None],float)
        rows.append(dict(dataset=r['dataset'],tau_within=float(np.mean(within)),tau_between=float(np.mean(between)),
                         auc_spread=float(aucs.max()-aucs.min()),density_gap_flagged=bool(fmap.get(r['dataset'],False))))
    return pd.DataFrame(rows)

def split_robustness(recs):
    rows=[]
    for r in recs:
        if r['replicate']!=0: continue
        primary=r.get('density_gap_primary') or {}
        for c in r.get('density_gap_robustness') or []:
            rows.append(dict(dataset=r['dataset'],seed=c['split_seed'],delta=c['delta'],p=c['p_one_sided'],
                             primary_delta=primary.get('delta'),delta_sign_agree=(np.sign(c['delta'])==np.sign(primary.get('delta',0)))))
    return pd.DataFrame(rows)

def subsample_robustness(df):
    rows=[]
    for name,g in df.groupby('dataset'):
        if len(g)<=1: continue
        g=g[g.delta.notna()]
        if not len(g): continue
        d0=float(g.loc[g.replicate==0,'delta'].iloc[0]) if (g.replicate==0).any() else float(g.delta.iloc[0])
        rows.append(dict(dataset=name,n_replicates=int(len(g)),positive_fraction=float((g.delta>0).mean()),
                         median_delta=float(g.delta.median()),median_abs_delta_deviation=float(np.median(np.abs(g.delta-d0)))))
    return pd.DataFrame(rows)

def composition(df):
    per=df.groupby('dataset')[[f'auc_{k}' for k in DETS if f'auc_{k}' in df]].mean(); A=per.to_numpy(); names=np.array([c[4:] for c in per.columns])
    rng=np.random.default_rng(0); wins=np.zeros(len(names),int)
    for _ in range(5000):
        idx=rng.integers(0,len(A),len(A)); wins[np.argmax(A[idx].mean(axis=0))]+=1
    best=names[A.argmax(axis=1)]
    return dict(bootstrap_win_share=dict(zip(names,(wins/5000).tolist())),n_detectors_that_can_be_made_winner=len(set(best)),
                detectors_that_can_be_made_winner=sorted(set(best.tolist())))

def main():
    recs=load_records()
    if not recs: print('no protocol-matched records found'); return
    df=build(recs); audit=density_gap_table(df)
    df.to_csv(os.path.join(RESULTS,'raw_per_run.csv'),index=False); audit.to_csv(os.path.join(RESULTS,'density_gap_audit.csv'),index=False)
    flagged=audit.loc[audit.flag_bh,'dataset'].tolist(); non=audit.loc[~audit.flag_bh,'dataset'].tolist()
    lb=leaderboard(df); lbf=leaderboard(df,flagged); lbn=leaderboard(df,non)
    lb.to_csv(os.path.join(RESULTS,'leaderboard_all.csv'),index=False); lbf.to_csv(os.path.join(RESULTS,'leaderboard_flagged.csv'),index=False); lbn.to_csv(os.path.join(RESULTS,'leaderboard_not_flagged.csv'),index=False)
    vk=vs_knn_table(recs); vk.to_csv(os.path.join(RESULTS,'vs_knn.csv'),index=False)
    ag=agreement_table(recs,audit); ag.to_csv(os.path.join(RESULTS,'agreement.csv'),index=False)
    sr=split_robustness(recs); sr.to_csv(os.path.join(RESULTS,'split_robustness.csv'),index=False)
    sub=subsample_robustness(df); sub.to_csv(os.path.join(RESULTS,'subsample_robustness.csv'),index=False)
    def order(x): return x.set_index('detector').mean_auc.reindex(DETS).to_numpy()
    tau=float(ST.kendall_tau(order(lbf),order(lbn))) if len(lbf) and len(lbn) else float('nan')
    # effect-size-only headroom above untuned transductive KNN; exploratory descriptive
    maxgain=vk.groupby('dataset').delta.max() if len(vk) else pd.Series(dtype=float)
    g0=maxgain.reindex(non).dropna(); g1=maxgain.reindex(flagged).dropna()
    mw=float('nan')
    if len(g0) and len(g1):
        from scipy.stats import mannwhitneyu; mw=float(mannwhitneyu(g0,g1,alternative='two-sided').pvalue)
    summary=dict(protocol_version=PROTOCOL,n_datasets=int(audit.dataset.nunique()),n_runs=int(len(df)),
        n_density_gap_flagged=int(audit.flag_bh.sum()),n_flagged_by=int(audit.flag_by.sum()),n_flagged_bonf=int(audit.flag_bonf.sum()),
        median_delta=float(audit.delta.median()),max_delta=float(audit.delta.max()),n_delta_positive=int((audit.delta>0).sum()),
        delta_quartiles=[float(x) for x in audit.delta.quantile([.25,.5,.75])],median_auc_dens=float(audit.auc_dens.median()),median_auc_sup=float(audit.auc_sup.median()),
        n_flagged=len(flagged),n_not_flagged=len(non),kendall_tau_between_regimes=tau,composition=composition(df),
        median_max_gain_vs_knn_not_flagged=float(g0.median()) if len(g0) else float('nan'),median_max_gain_vs_knn_flagged=float(g1.median()) if len(g1) else float('nan'),
        max_gain_mannwhitney_p=mw,
        agreement_spearman=float(ST.spearman(ag.tau_between,ag.auc_spread)) if len(ag)>2 else float('nan'),
        agreement_pearson=float(np.corrcoef(ag.tau_between,ag.auc_spread)[0,1]) if len(ag)>2 else float('nan'),
        split_robustness_n=int(len(sr)),split_robustness_sign_agreement=float(sr.delta_sign_agree.mean()) if len(sr) else float('nan'),
        subsample_n_datasets=int(len(sub)),subsample_median_positive_fraction=float(sub.positive_fraction.median()) if len(sub) else float('nan'),
        subsample_median_abs_delta_deviation=float(sub.median_abs_delta_deviation.median()) if len(sub) else float('nan'))
    with open(os.path.join(RESULTS,'summary.json'),'w') as fh: json.dump(summary,fh,indent=1)
    print(json.dumps({k:v for k,v in summary.items() if k!='composition'},indent=1))

if __name__=='__main__': main()
