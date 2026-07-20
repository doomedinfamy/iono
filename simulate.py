"""Synthetic behavioral data for the combinatorial-risk experiment.

The human dataset (N = 2640 on Credamo) is not public, so we simulate agents
whose generating processes mirror the paper's findings (Sections 4, 6.2, 7):

Control condition   -- choices driven by combinatorial-risk features only:
    logit = a1 (dP_B - dP_A) + a2 sgp(P_aft_B - P_aft_A, P_ini-modulated power)
    with larger sensitivity in the high-magnitude cell (Section 4.2).

Treatment condition -- the control rule, dampened, plus a prospect-theoretic
    evaluation of the displayed PMFs (Section 7), giving compressed choice
    variance and weaker response to combinatorial-risk features (Section 4.3).

Each (problem, magnitude) cell records N=20 binomial choices (paper Section 3.3).
"""
from __future__ import annotations

import numpy as np

from models import sgp, lol_weight
from task import make_stimuli, outcome_vector, PAYOFFS

N_PER_CELL = 20


def _pt_value(R, alpha=0.75, gamma=0.65):
    """PT valuation of an induced lottery on normalized outcomes {0,1,2}."""
    w = lambda p: lol_weight(p, gamma, 1.0)
    return w(R[:, 1]) * 1.0 + w(R[:, 2]) * 2.0 ** alpha


def true_logit_control(F, magnitude):
    a1 = 3.2 if magnitude == "low" else 3.6
    a2 = 2.6 if magnitude == "low" else 3.4     # magnitude boosts P-sensitivity
    power = np.clip(0.4 + 0.5 * F["P_ini_A"], 0.1, None)
    return (a1 * (F["dP_B"] - F["dP_A"])
            + a2 * sgp(F["P_aft_B"] - F["P_aft_A"], power))


def true_logit_treatment(F, magnitude):
    base = 0.45 * true_logit_control(F, magnitude)     # dampened feature use
    dv = _pt_value(F["RB"]) - _pt_value(F["RA"])       # PMF-based valuation
    pay = PAYOFFS[magnitude]
    ev_a = pay * (F["RA"] @ np.array([0.0, 1.0, 2.0]))
    ev_b = pay * (F["RB"] @ np.array([0.0, 1.0, 2.0]))
    mag_term = 0.35 * sgp(ev_b - ev_a, 0.35) / (30.0 ** 0.35)
    return base + 1.7 * dv + mag_term


def simulate(seed=7, n_problems=None):
    """Returns dict: data[(g, c)] = dict(F=..., o=..., p_emp=..., n_B=...)."""
    rng = np.random.default_rng(seed)
    F = make_stimuli(seed=seed) if n_problems is None else make_stimuli(n_problems, seed)
    n = len(F["P_ini_A"])
    data = {}
    for g in ("C", "T"):
        for c in ("low", "high"):
            logit = (true_logit_control if g == "C" else true_logit_treatment)(F, c)
            # problem-level heterogeneity across participants
            noise = rng.normal(0, 0.55, n)
            p_true = 1.0 / (1.0 + np.exp(-(logit + noise)))
            n_b = rng.binomial(N_PER_CELL, p_true)
            data[(g, c)] = dict(F=F, o=outcome_vector(c, n),
                                p_emp=n_b / N_PER_CELL, n_B=n_b,
                                p_true=p_true)
    return F, data


def pooled_condition(data, g):
    """Stack low+high magnitude cells of condition g for model fitting."""
    lo, hi = data[(g, "low")], data[(g, "high")]
    F = {k: (np.concatenate([lo["F"][k], hi["F"][k]])
             if lo["F"][k].ndim == 1 else np.vstack([lo["F"][k], hi["F"][k]]))
         for k in lo["F"]}
    o = np.vstack([lo["o"], hi["o"]])
    p = np.concatenate([lo["p_emp"], hi["p_emp"]])
    return F, o, p


def train_test_split(n, seed=123, frac=0.8):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(frac * n)
    return perm[:k], perm[k:]


def subset(F, o, p, idx):
    Fs = {k: v[idx] for k, v in F.items()}
    return Fs, o[idx], p[idx]
