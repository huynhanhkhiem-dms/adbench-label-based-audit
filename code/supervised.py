"""Leakage-safe supervised reference for the inductive audit.

All feature preprocessing and model choice occur strictly inside the outer
training split.  The untouched outer test is transformed with parameters fitted
on outer training and scored once.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


def candidate_models(seed=0):
    return {
        'hgb': HistGradientBoostingClassifier(max_iter=300,learning_rate=0.1,random_state=seed),
        'rf': RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=seed,min_samples_leaf=2),
        'logit': LogisticRegression(max_iter=2000,C=1.0),
        'knn': KNeighborsClassifier(n_neighbors=10,n_jobs=-1),
    }

def _scores(model,X):
    return model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X)

def fit_select_score(X_tr_raw,y_tr,X_te_raw,seed=0,n_splits=3):
    import data as D
    from stats import auc
    X_tr_raw=np.asarray(X_tr_raw,float); y_tr=np.asarray(y_tr,int)
    n_pos=int(y_tr.sum()); n_neg=len(y_tr)-n_pos
    n_splits=int(max(2,min(n_splits,n_pos,n_neg)))
    if n_pos<2 or n_neg<2:
        Xtr,Xte,_=D.train_standardise(X_tr_raw,X_te_raw)
        m=candidate_models(seed)['logit'].fit(Xtr,y_tr)
        return _scores(m,Xte),'logit',{'logit':float('nan')}
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    folds=list(skf.split(X_tr_raw,y_tr)); table={}
    for name in candidate_models(seed):
        oof=np.full(len(y_tr),np.nan); ok=True
        for j,(itr,iva) in enumerate(folds):
            try:
                Xf,Xv,_=D.train_standardise(X_tr_raw[itr],X_tr_raw[iva])
                m=candidate_models(seed+1009*j)[name].fit(Xf,y_tr[itr])
                oof[iva]=_scores(m,Xv)
            except Exception:
                ok=False; break
        table[name]=auc(oof,y_tr) if ok and np.isfinite(oof).all() else float('nan')
    valid={k:v for k,v in table.items() if np.isfinite(v)}
    chosen=max(valid,key=valid.get) if valid else 'logit'
    Xtr,Xte,_=D.train_standardise(X_tr_raw,X_te_raw)
    m=candidate_models(seed+99991)[chosen].fit(Xtr,y_tr)
    return np.asarray(_scores(m,Xte),float),chosen,table
