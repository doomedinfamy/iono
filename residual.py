
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr

from models import lol_weight, sigma, metrics


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def pt_dv(F, o, alpha_v, gamma, monetary):
    """dV = V(B) - V(A) under a one-parameter LOL weighting."""
    w = lambda p: lol_weight(p, gamma, 1.0)
    if monetary:
        outc = o                                   # (n,3) monetary outcomes
    else:
        outc = np.tile(np.array([0.0, 1.0, 2.0]), (o.shape[0], 1))
    v = np.abs(outc) ** alpha_v
    VA = np.sum(w(F["RA"]) * v, axis=1)
    VB = np.sum(w(F["RB"]) * v, axis=1)
    return VB - VA


def residual_correlations(F, resid):
    """Correlation of transfer residuals with each PMF feature (Section 7)."""
    out = {}
    for j, R in (("A", F["RA"]), ("B", F["RB"])):
        for k in range(3):
            r, p = pearsonr(R[:, k], resid)
            out[f"R_{j}{k}"] = (r, p)
    return out


def fit_residual_model(pc_train, F_tr, o_tr, nb_tr, n_per, monetary, seed=0):
    """MLE of (alpha, beta0, beta1, alpha_v, gamma) on binomial counts."""
    base = logit(pc_train)
    rng = np.random.default_rng(seed)

    def nll(th):
        a, b0, b1, av, g = th
        if av <= 0 or g <= 0:
            return 1e9
        with np.errstate(all="ignore"):
            dv = pt_dv(F_tr, o_tr, av, g, monetary)
            q = np.clip(sigma(a * base + b0 + b1 * dv), 1e-9, 1 - 1e-9)
        if not np.all(np.isfinite(q)):
            return 1e9
        return float(-np.sum(nb_tr * np.log(q) + (n_per - nb_tr) * np.log(1 - q)))

    best, best_v = None, np.inf
    for _ in range(10):
        x0 = np.array([rng.uniform(0, 1.5), rng.normal(0, 0.3),
                       rng.uniform(0, 2), rng.uniform(0.2, 1.5),
                       rng.uniform(0.3, 1.5)])
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "fatol": 1e-8})
        if res.fun < best_v:
            best_v, best = res.fun, res.x
    return best


def predict_residual_model(th, pc, F, o, monetary):
    a, b0, b1, av, g = th
    dv = pt_dv(F, o, av, g, monetary)
    return sigma(a * logit(pc) + b0 + b1 * dv)


def bootstrap_metrics(p_emp, q, B=1000, seed=0):
    """Test-set bootstrap SEs for (CE, MSE, Acc), eq. (9) of the paper."""
    rng = np.random.default_rng(seed)
    n = len(p_emp)
    vals = np.empty((B, 3))
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = metrics(p_emp[idx], q[idx])
    mean = vals.mean(axis=0)
    se = vals.std(axis=0, ddof=1)
    return mean, se
