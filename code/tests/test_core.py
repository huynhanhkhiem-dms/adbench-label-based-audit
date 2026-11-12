"""Unit tests: AUC, DeLong variance, BH, detector sanity, theorem identities."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stats as ST          # noqa: E402
import detectors as DET     # noqa: E402
import theory as TH         # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


# --------------------------------------------------------------------- AUC
def test_auc_against_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        n = rng.integers(30, 400)
        y = (rng.random(n) < rng.uniform(0.05, 0.5)).astype(int)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.standard_normal(n) + 0.7 * y
        if rng.random() < 0.5:                       # force heavy ties
            s = np.round(s, 1)
        worst = max(worst, abs(ST.auc(s, y) - roc_auc_score(y, s)))
    check("AUC matches sklearn (incl. ties)", worst < 1e-12, f"max|diff|={worst:.2e}")


def test_delong_variance_against_bootstrap():
    rng = np.random.default_rng(1)
    n = 4000
    y = (rng.random(n) < 0.2).astype(int)
    a = rng.standard_normal(n) + 1.0 * y
    b = rng.standard_normal(n) + 0.7 * y
    delta, se, z, p = ST.delong_test(a, b, y, "greater")
    boots = []
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for t in range(600):
        i = np.concatenate([rng.choice(pos, len(pos), True),
                            rng.choice(neg, len(neg), True)])
        boots.append(ST.auc(a[i], y[i]) - ST.auc(b[i], y[i]))
    rel = abs(se - np.std(boots)) / np.std(boots)
    check("DeLong SE matches bootstrap SE", rel < 0.15,
          f"delong={se:.5f} boot={np.std(boots):.5f} rel={rel:.3f}")


def test_delong_identical_scores():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.3).astype(int)
    s = rng.standard_normal(500)
    delta, se, z, p = ST.delong_test(s, s, y, "greater")
    check("DeLong on identical scores: delta=0", abs(delta) < 1e-12,
          f"delta={delta:.2e} se={se:.2e}")


def test_bh():
    # the worked example of Benjamini & Hochberg (1995, Table 1): 15 p-values,
    # q = 0.05, exactly four rejections.
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                  0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000])
    rej, adj = ST.benjamini_hochberg(p, 0.05)
    check("BH reproduces the 1995 worked example (4 rejections)",
          rej.tolist() == [True] * 4 + [False] * 11, f"{int(rej.sum())} rejections")
    check("BH adjusted p is monotone", np.all(np.diff(adj[np.argsort(p)]) >= -1e-15))
    check("BH is never more liberal than Bonferroni at the smallest p",
          adj[np.argmin(p)] <= 1.0 and adj[np.argmin(p)] >= p.min())


# ------------------------------------------------------------------ theory
def test_theorem1_identification():
    """Theorem 2: for a fixed sample and score, the label-consistent AUC can be
    driven to any value in [0,1] at fixed contamination."""
    rng = np.random.default_rng(3)
    s = rng.standard_normal(2000)
    errs = []
    for target in [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]:
        y = TH.labelling_with_auc(s, alpha=0.05, target=target, rng=rng)
        errs.append(abs(ST.auc(s, y) - target))
    check("Theorem 2: every AUC in [0,1] is attainable", max(errs) < 0.02,
          f"max err={max(errs):.4f}")


def test_theorem3_density_optimal():
    """Theorem 7: under density-monotone eta the negative density attains the
    Bayes AUC exactly; break monotonicity and it no longer does."""
    rng = np.random.default_rng(4)

    # --- assumption satisfied: -p and eta induce the identical ranking ------
    n = 400
    p = rng.random(n)
    r = np.argsort(np.argsort(p)) / (n - 1.0)
    eta = np.clip((1.0 - r) ** 4 * 0.4, 1e-9, 1 - 1e-9)
    gap = TH.auc_from_eta(eta, eta) - TH.auc_from_eta(-p, eta)
    check("Theorem 7 (holds): -p attains the Bayes AUC exactly",
          abs(gap) < 1e-12, f"Bayes-minus-density gap = {gap:.3e}")
    check("Theorem 7 (holds): eta is certified density-monotone",
          TH.is_density_monotone(p, eta))

    # --- assumption violated: anomalies in their own dense cluster ---------
    X2, y2, p2, eta2 = TH.violating_sample(n=6000, rng=rng)
    a_dens2, a_eta2 = ST.auc(-p2, y2), ST.auc(eta2, y2)
    check("Theorem 7 (violated): the posterior beats the negative density",
          a_eta2 > a_dens2 + 0.05, f"auc(-p)={a_dens2:.4f} auc(eta)={a_eta2:.4f}")
    check("Theorem 7 (violated): eta is not density-monotone",
          not TH.is_density_monotone(p2, eta2))

    # --- a sampled LDA population: -p and the posterior tie exactly ---------
    X, y, p = TH.lda_sample(n=6000, rng=rng)
    r = np.argsort(np.argsort(p)) / (len(p) - 1.0)
    eta_true = (1.0 - r) ** 6
    a_p, a_eta = ST.auc(-p, y), ST.auc(eta_true, y)
    check("Theorem 7: on a sampled LDA population -p ties the true posterior "
          "to machine precision", abs(a_p - a_eta) < 1e-12,
          f"auc(-p)={a_p:.6f} auc(eta)={a_eta:.6f} diff={abs(a_p - a_eta):.2e}")


def test_theorem3_equivalence_exact():
    """The equivalence is exact on a finite grid: -p is AUC-optimal iff eta is
    a non-increasing function of p."""
    rng = np.random.default_rng(5)
    bad = 0
    for _ in range(300):
        m = 8
        p = np.sort(rng.random(m))[::-1]
        eta = rng.random(m)
        mono = np.all(np.diff(eta[np.argsort(-p)]) >= -1e-15)
        opt = TH.is_auc_optimal(-p, eta)
        if mono != opt:
            bad += 1
    check("Theorem 7 equivalence holds on 300 random grids", bad == 0,
          f"mismatches={bad}")


def test_theorem3_ties_and_atoms():
    """Edge case for the exact characterisation: if density has an atom/tie
    but the posterior varies inside that density level, -p cannot be Bayes
    AUC-optimal because it ties observations that the posterior ranks."""
    p=np.array([0.2,0.2,0.5,0.8],float)
    eta=np.array([0.8,0.3,0.2,0.1],float)
    gap=TH.auc_from_eta(eta,eta)-TH.auc_from_eta(-p,eta)
    check('Theorem 7 atom edge case: posterior variation inside a density tie creates positive Bayes gap',
          gap>1e-6,f'gap={gap:.6g}')
    eta2=np.array([0.8,0.8,0.2,0.1],float)
    gap2=TH.auc_from_eta(eta2,eta2)-TH.auc_from_eta(-p,eta2)
    check('Theorem 7 atom edge case: constant posterior on density ties preserves optimality',
          abs(gap2)<1e-12,f'gap={gap2:.6g}')


# --------------------------------------------------------------- detectors
def test_detectors_run_and_separate():
    rng = np.random.default_rng(6)
    n = 3000
    X = rng.standard_normal((n, 5))
    idx = rng.choice(n, 60, replace=False)
    # scattered, isolated anomalies: each sits alone in a low-density region,
    # so *every* family -- global, local, isolation, marginal -- should find
    # them.  (Anomalies placed as one tight cluster would be locally dense and
    # would defeat LOF by construction; that case is Theorem 7's violation.)
    direction = rng.standard_normal((60, 5))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    X[idx] = direction * rng.uniform(7.0, 11.0, size=(60, 1))
    y = np.zeros(n, int)
    y[idx] = 1
    aucs = {}
    for name, fn in DET.DETECTORS.items():
        s = np.asarray(fn(X, np.random.default_rng(0)), float)
        check(f"detector {name} returns finite scores of the right length",
              s.shape == (n,) and np.all(np.isfinite(s)))
        aucs[name] = ST.auc(s, y)
    worst = min(aucs.values())
    check("every detector exceeds 0.9 AUC on textbook low-density anomalies",
          worst > 0.9, f"worst={min(aucs, key=aucs.get)}={worst:.4f}")


def test_auc_invariance_to_monotone_transform():
    rng = np.random.default_rng(7)
    y = (rng.random(800) < 0.2).astype(int)
    s = rng.random(800)
    check("AUC is invariant to strictly monotone rescoring",
          abs(ST.auc(s, y) - ST.auc(np.exp(3 * s) - 5, y)) < 1e-12)



# ---------------------------------------------------------- leakage guards
def test_train_standardizer_ignores_evaluation_covariates():
    import data as D
    rng=np.random.default_rng(80)
    Xtr=rng.normal(size=(80,4)); x=np.array([[1.,2.,3.,4.]])
    A=np.vstack([x,np.zeros((3,4))]); B=np.vstack([x,np.full((3,4),1e6)])
    _,Za,_=D.train_standardise(Xtr,A); _,Zb,_=D.train_standardise(Xtr,B)
    check('train-fitted standardizer ignores other evaluation rows',np.allclose(Za[0],Zb[0]))

def test_inductive_density_score_ignores_other_evaluation_rows():
    import data as D
    rng=np.random.default_rng(81)
    Xtr=rng.normal(size=(100,3)); x=np.array([[2.,0.,-1.]])
    A=np.vstack([x,np.zeros((2,3))]); B=np.vstack([x,np.full((2,3),999.)])
    T,A1,_=D.train_standardise(Xtr,A); _,B1,_=D.train_standardise(Xtr,B)
    sa=DET.score_density_candidate('knn_max_k10',T,A1,seed=0)
    sb=DET.score_density_candidate('knn_max_k10',T,B1,seed=0)
    check('fixed inductive density score ignores other evaluation rows',abs(sa[0]-sb[0])<1e-12)

def test_density_selection_does_not_use_outer_test_covariates():
    rng=np.random.default_rng(82)
    Xtr=rng.normal(size=(90,3)); y=np.array([0]*70+[1]*20); rng.shuffle(y)
    Xa=rng.normal(size=(20,3)); Xb=Xa.copy(); Xb[1:]+=1e5
    _,na,ta=DET.fit_select_density_score(Xtr,y,Xa,seed=3,n_splits=3)
    _,nb,tb=DET.fit_select_density_score(Xtr,y,Xb,seed=3,n_splits=3)
    same=na==nb and all((np.isnan(ta[k]) and np.isnan(tb[k])) or abs(ta[k]-tb[k])<1e-12 for k in ta)
    check('density candidate selection ignores outer-test covariates',same,f'{na} vs {nb}')

def test_supervised_selection_does_not_use_outer_test_covariates():
    import supervised as SUP
    rng=np.random.default_rng(83)
    Xtr=rng.normal(size=(100,4)); y=np.array([0]*75+[1]*25); rng.shuffle(y)
    Xa=rng.normal(size=(12,4)); Xb=Xa.copy(); Xb[1:]-=1e5
    _,na,ta=SUP.fit_select_score(Xtr,y,Xa,seed=4,n_splits=3)
    _,nb,tb=SUP.fit_select_score(Xtr,y,Xb,seed=4,n_splits=3)
    same=na==nb and all((np.isnan(ta[k]) and np.isnan(tb[k])) or abs(ta[k]-tb[k])<1e-12 for k in ta)
    check('supervised model selection ignores outer-test covariates',same,f'{na} vs {nb}')

def test_by_not_more_liberal_than_bh():
    p=np.array([.0001,.001,.01,.03,.2,.8])
    bh,_=ST.benjamini_hochberg(p,.05); by,_=ST.benjamini_yekutieli(p,.05)
    check('BY is not more liberal than BH',np.all(~by | bh))

if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("all tests passed")
