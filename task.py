
from __future__ import annotations

import numpy as np

# Discrete probability grid used to generate stimuli (Section 3.3)
GRID = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0])

N_PROBLEMS = 1873          # size of the stimulus pool in the paper
PAYOFFS = {"low": 30.0, "high": 100.0}
EPS = 1e-9


def induced_pmf(p1, p2):
    """PMF of U = X1 + X2 for independent Bernoulli(p1), Bernoulli(p2).

    Returns (R0, R1, R2) = (Pr(U=0), Pr(U=1), Pr(U=2)), Section 5.1.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    r0 = (1 - p1) * (1 - p2)
    r1 = p1 * (1 - p2) + (1 - p1) * p2
    r2 = p1 * p2
    return r0, r1, r2


def component_pool():
    """All valid components (P_ini, dP): P_ini in GRID, dP in GRID\\{0}, sum <= 1."""
    out = []
    for pi in GRID:
        for dp in GRID[GRID > 0]:
            if pi + dp <= 1 + EPS:
                out.append((float(pi), float(dp)))
    return out


def full_problem_pool():
    """All ordered problems (A, B) with comparable increments:
    dP_B in [dP_A, dP_A + 0.10] (Section 3.3)."""
    comps = component_pool()
    problems = []
    for (pa, da) in comps:
        for (pb, db) in comps:
            if da - EPS <= db <= da + 0.10 + EPS:
                problems.append((pa, da, pb, db))
    return problems


def make_stimuli(n=N_PROBLEMS, seed=0):
    """Sample the stimulus pool and randomly swap A/B labels (position debias).

    Returns a structured dict of numpy arrays with all task features.
    """
    rng = np.random.default_rng(seed)
    pool = np.array(full_problem_pool())
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    pa, da, pb, db = pool[idx].T
    # swap labels with probability 0.5
    swap = rng.random(len(idx)) < 0.5
    pa, pb = np.where(swap, pb, pa), np.where(swap, pa, pb)
    da, db = np.where(swap, db, da), np.where(swap, da, db)
    return problem_features(pa, da, pb, db)


def problem_features(pa, da, pb, db):
    """Compute every derived quantity of a problem (Table 1)."""
    paft_a, paft_b = pa + da, pb + db
    # induced lotteries: invest in A -> (P_aft_A, P_ini_B); invest in B -> (P_ini_A, P_aft_B)
    RA = np.stack(induced_pmf(paft_a, pb), axis=-1)   # (n, 3)
    RB = np.stack(induced_pmf(pa, paft_b), axis=-1)
    u = np.array([0.0, 1.0, 2.0])
    ev_a, ev_b = RA @ u, RB @ u
    var_a = RA @ (u ** 2) - ev_a ** 2
    var_b = RB @ (u ** 2) - ev_b ** 2
    return dict(P_ini_A=pa, P_ini_B=pb, dP_A=da, dP_B=db,
                P_aft_A=paft_a, P_aft_B=paft_b,
                RA=RA, RB=RB, EV_A=ev_a, EV_B=ev_b,
                Var_A=var_a, Var_B=var_b)


def outcome_vector(magnitude, n):
    """Monetary outcomes o = payoff * (0, 1, 2) for a magnitude condition."""
    pay = PAYOFFS[magnitude]
    return np.tile(pay * np.array([0.0, 1.0, 2.0]), (n, 1))
