"""Regenerate manuscript figures from the archived analysis outputs."""
from __future__ import annotations
import json,os,sys
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=os.path.join(ROOT,'results')
F=os.environ.get('FIGURE_OUT',os.path.join(ROOT,'figures')); os.makedirs(F,exist_ok=True)
PROTOCOL='inductive_outer_test'

def two_worlds():
    t=json.load(open(os.path.join(R,'theory.json')))['proposition10']; A=t['auc_A']; B=t['auc_B']; names=list(A)
    fig,ax=plt.subplots(figsize=(5.7,3.1))
    for k in names: ax.plot([0,1],[A[k],B[k]],marker='o',ms=2.8,lw=.8,alpha=.8)
    for k, label_y in [('KDE', .98), ('MaxQuantile', .91), ('IForest', .82), ('KNN', .16), ('PCA', .06)]:
        if k in B:
            ax.annotate(k, (1, B[k]), xytext=(1.04, label_y), fontsize=7, va='center',
                        arrowprops={'arrowstyle': '-', 'lw': .5, 'color': '.4'})
    ax.set_xlim(-.08,1.38); ax.set_ylim(-.02,1.05); ax.set_xticks([0,1],['World A','World B']); ax.set_ylabel('AUC')
    ax.axhline(.5,lw=.6,ls=':'); ax.set_title('Same numerical scores, different label mechanisms')
    ax.spines[['top','right']].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(F,'fig_two_worlds.pdf')); plt.close(fig)

def certificate():
    p=os.path.join(R,'density_gap_audit.csv')
    if not os.path.exists(p): return False
    d=pd.read_csv(p).sort_values('delta');
    if 'protocol_version' in d and not (d.protocol_version==PROTOCOL).all(): raise RuntimeError('records from a different protocol')
    y=np.arange(len(d)); fig,ax=plt.subplots(figsize=(5.7,8.0))
    flag=d.flag_bh.astype(bool).to_numpy()
    for mask, lab, color, face in [(flag, 'BH flagged', '#254e70', '#254e70'),
                                   (~flag, 'Not flagged', '#515151', 'white')]:
        ax.hlines(y[mask], d.boot_lo[mask], d.boot_hi[mask], color=color, linewidth=.75)
        ax.scatter(d.delta[mask], y[mask], s=15, edgecolors=color, facecolors=face, linewidths=.8, label=lab, zorder=3)
    ax.axvline(0,lw=.75,color='.45'); ax.set_yticks(y,d.dataset,fontsize=7.3); ax.set_xlabel(r'$\Delta=$ AUC(supervised) - AUC(selected density)')
    ax.legend(frameon=False,fontsize=7); ax.spines[['top','right']].set_visible(False); fig.tight_layout(); fig.savefig(os.path.join(F,'fig_certificate.pdf')); plt.close(fig); return True

def label_budget():
    p=os.path.join(R,'label_budget.json')
    if not os.path.exists(p): return False
    L=json.load(open(p));
    if not L: return False
    fig,ax=plt.subplots(figsize=(3.6,2.8)); grids=sorted({int(k) for c in L.values() for k in c})
    curves=[]
    for c in L.values():
        cc={int(k):float(v) for k,v in c.items()}; xs=sorted(cc); ax.plot(xs,[cc[x] for x in xs],lw=.6,alpha=.35); curves.append(cc)
    med=[np.median([c[m] for c in curves if m in c]) for m in grids]
    ax.plot(grids,med,marker='o',ms=3,lw=1.5,label='median'); ax.axhline(.8,lw=.7,ls=':'); ax.set_xscale('log'); ax.set_ylim(-.02,1.02)
    ax.set_xlabel('Labeled anomalies in outer test'); ax.set_ylabel('Rejection probability'); ax.legend(frameon=False,fontsize=7); ax.spines[['top','right']].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(F,'fig_label_budget.pdf')); plt.close(fig); return True

def main():
    two_worlds(); emp=certificate(); lb=label_budget(); print('generated theory figure; empirical=',emp,'label_budget=',lb,'in',F)
if __name__=='__main__': main()
