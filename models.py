
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# primitives
# ----------------------------------------------------------------------------

def sigma(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sgp(x, y):
    """Sign-preserving power sign(x)|x|^y (the paper's `signed_pow`)."""
    return np.sign(x) * np.abs(x) ** y


def lol_weight(p, aw, bw):
    """Log Odds Linear probability weighting (Gonzalez & Wu, 1999)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    num = bw * p ** aw
    return num / (num + (1 - p) ** aw)


def value_fn(x, alpha):
    return np.abs(x) ** alpha




def _lottery_ev(F, o):
    """Expected monetary payoff dot(o, R) of each induced lottery."""
    return np.sum(o * F["RA"], axis=1), np.sum(o * F["RB"], axis=1)


# --- Section 5.1: heuristics -------------------------------------------------

def score_delta_diff(F, o, th):
    return th[0] * (F["dP_B"] - F["dP_A"])


def score_aft_prob(F, o, th):
    return th[0] * (F["P_aft_B"] - F["P_aft_A"])


def _tail_scores(F, w0, w2):
    ua = w0 * (1 - F["RA"][:, 0]) + w2 * F["RA"][:, 2]
    ub = w0 * (1 - F["RB"][:, 0]) + w2 * F["RB"][:, 2]
    return ub - ua


def score_tail_10(F, o, th):
    return th[0] * _tail_scores(F, 1.0, 0.0)


def score_tail_01(F, o, th):
    return th[0] * _tail_scores(F, 0.0, 1.0)


def score_tail_free(F, o, th):
    return th[0] * _tail_scores(F, th[1], th[2])


# --- Section 5.2: state-space utility models --------------------------------

def score_risk_neutral(F, o, th):
    # u_RN(m_B) - u_RN(m_A) = b2 dP_B - b1 dP_A
    return th[0] * (th[2] * F["dP_B"] - th[1] * F["dP_A"])


def score_cra(F, o, th):
    invT, A, b1, b2, eta1 = th
    z = b1 * F["P_ini_A"] + b2 * F["P_ini_B"]
    du = (A * np.exp(-z) * (np.exp(-b2 * F["dP_B"]) - np.exp(-b1 * F["dP_A"]))
          - eta1 * (F["dP_A"] + F["dP_B"]))
    return invT * du


def _F_from_rho(rho_fn, zmax, n=2001):
    """Build F with -F''/F' = rho by numeric integration; returns interp fn."""
    z = np.linspace(0.0, max(zmax, 1e-6), n)
    # F'(z) = exp(-int_0^z rho), F(z) = int_0^z F'
    integral_rho = np.concatenate([[0.0], np.cumsum(
        0.5 * (rho_fn(z[1:]) + rho_fn(z[:-1])) * np.diff(z))])
    fp = np.exp(np.clip(-integral_rho, -60, 60))
    Fv = np.concatenate([[0.0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(z))])
    return lambda x: np.interp(x, z, Fv)


def score_dra_linear(F, o, th):
    invT, b1, b2, eta1, r0, r1 = th
    z = b1 * F["P_ini_A"] + b2 * F["P_ini_B"]
    zb = z + b2 * F["dP_B"]
    za = z + b1 * F["dP_A"]
    Ffun = _F_from_rho(lambda t: r0 - r1 * t, float(max(zb.max(), za.max())))
    du = Ffun(zb) - Ffun(za) - eta1 * (F["dP_A"] + F["dP_B"])
    return invT * du


def score_dra_exp(F, o, th):
    invT, b1, b2, eta1, r0, g = th
    z = b1 * F["P_ini_A"] + b2 * F["P_ini_B"]
    zb = z + b2 * F["dP_B"]
    za = z + b1 * F["dP_A"]
    Ffun = _F_from_rho(lambda t: r0 * np.exp(-g * t), float(max(zb.max(), za.max())))
    du = Ffun(zb) - Ffun(za) - eta1 * (F["dP_A"] + F["dP_B"])
    return invT * du


# --- Section 5.3: risky-choice models on the induced lotteries ---------------
# Outcomes are the normalized success counts {0, 1, 2} (V_EU = R1 + R2 2^a).

def score_eu(F, o, th):
    invT, a = th
    va = F["RA"][:, 1] + F["RA"][:, 2] * 2.0 ** a
    vb = F["RB"][:, 1] + F["RB"][:, 2] * 2.0 ** a
    return invT * (vb - va)


def score_pt(F, o, th):
    invT, a, aw, bw = th
    w = lambda p: lol_weight(p, aw, bw)
    va = w(F["RA"][:, 1]) + w(F["RA"][:, 2]) * 2.0 ** a
    vb = w(F["RB"][:, 1]) + w(F["RB"][:, 2]) * 2.0 ** a
    return invT * (vb - va)


def score_cpt(F, o, th):
    invT, a, aw, bw = th
    w = lambda p: lol_weight(p, aw, bw)

    def V(R):
        pi2 = w(R[:, 2])
        pi1 = w(R[:, 1] + R[:, 2]) - pi2
        return pi1 + pi2 * 2.0 ** a

    return invT * (V(F["RB"]) - V(F["RA"]))


# --- Section 6.2: SR-discovered models (Tables 3 and 4) ----------------------
# Expressions score option A vs B; a free signed scale absorbs orientation.

def score_ini_scaled_aft(F, o, th):
    # sgp(P_aft_A - P_aft_B, P_ini_A), complexity 6 (control)
    return th[0] * sgp(F["P_aft_A"] - F["P_aft_B"], F["P_ini_A"])


def score_ini_scaled_aft8(F, o, th):
    # sgp(P_aft_A - P_aft_B, sgp(P_ini_A, c1)), complexity 8 (control)
    return th[0] * sgp(F["P_aft_A"] - F["P_aft_B"], sgp(F["P_ini_A"], th[1]))


def score_hybrid10(F, o, th):
    # dP_A - dP_B + sgp(P_aft_A - P_aft_B, P_ini_A), complexity 10 (control)
    expr = (F["dP_A"] - F["dP_B"]
            + th[1] * sgp(F["P_aft_A"] - F["P_aft_B"], F["P_ini_A"]))
    return th[0] * expr


def score_hybrid12(F, o, th):
    # dP_A - dP_B + sgp(P_aft_A - P_aft_B, sgp(P_ini_A, c1)), complexity 12
    expr = (F["dP_A"] - F["dP_B"]
            + th[2] * sgp(F["P_aft_A"] - F["P_aft_B"], sgp(F["P_ini_A"], th[1])))
    return th[0] * expr


def score_hybrid17(F, o, th):
    # c1 (dot(oA,RA) - dot(oB,RB)) - c2 sgp(P_aft_B - P_aft_A, c3), complexity 17
    eva, evb = _lottery_ev(F, o)
    expr = th[1] * (eva - evb) - th[2] * sgp(F["P_aft_B"] - F["P_aft_A"], th[3])
    return th[0] * expr


def score_t_hybrid8(F, o, th):
    # P_aft_A - P_aft_B + dP_A - dP_B (treatment, complexity 8)
    return th[0] * (F["P_aft_A"] - F["P_aft_B"] + F["dP_A"] - F["dP_B"])


def score_log_ev(F, o, th):
    # ln(dot(RA,oA) / dot(RB,oB)) (treatment, complexity 9)
    eva, evb = _lottery_ev(F, o)
    return th[0] * np.log(np.maximum(eva, 1e-9) / np.maximum(evb, 1e-9))


def score_power_ev14(F, o, th):
    # P_ini_A - sgp(dot(oB,RB) - dot(oA,RA), c1) - c2 (treatment, complexity 14)
    eva, evb = _lottery_ev(F, o)
    return th[0] * (F["P_ini_A"] - sgp(evb - eva, th[1]) - th[2])


def score_t_hybrid16(F, o, th):
    # P_aft_A - P_aft_B - c1 sgp(dot(oB,RB) - dot(oA,RA), c2) (complexity 16)
    eva, evb = _lottery_ev(F, o)
    expr = F["P_aft_A"] - F["P_aft_B"] - th[1] * sgp(evb - eva, th[2])
    return th[0] * expr


def score_t_hybrid27(F, o, th):
    # P_aft_A - P_aft_B - c1 sgp(dot(oB,RB) - dot(oA,RA), c2)
    #   + ln(dot(oA,RA)/dot(oB,RB)) + c3 (complexity 27)
    eva, evb = _lottery_ev(F, o)
    expr = (F["P_aft_A"] - F["P_aft_B"] - th[1] * sgp(evb - eva, th[2])
            + np.log(np.maximum(eva, 1e-9) / np.maximum(evb, 1e-9)) + th[3])
    return th[0] * expr


# ----------------------------------------------------------------------------
# registry: name -> (score_fn, bounds list [(lo, hi), ...])
# first parameter is a signed scale (inverse temperature, orientation-free).
# ----------------------------------------------------------------------------

SCALE = (-80.0, 80.0)
POS = (1e-3, 80.0)

BENCHMARKS = {
    "Delta-diff":            (score_delta_diff, [SCALE]),
    "Aft-prob":              (score_aft_prob, [SCALE]),
    "Tail(1,0)":             (score_tail_10, [SCALE]),
    "Tail(0,1)":             (score_tail_01, [SCALE]),
    "Tail(w0,w2)":           (score_tail_free, [SCALE, (0.0, 1.0), (0.0, 1.0)]),
    "Risk Neutral":          (score_risk_neutral, [SCALE, (0.0, 5.0), (0.0, 5.0)]),
    "Const. Risk Averse":    (score_cra, [SCALE, (-10.0, 10.0), (0.0, 6.0),
                                          (0.0, 6.0), (-5.0, 5.0)]),
    "Lin. Decr. Risk Averse": (score_dra_linear, [SCALE, (0.0, 6.0), (0.0, 6.0),
                                                  (-5.0, 5.0), (0.0, 8.0), (0.0, 8.0)]),
    "Exp. Decr. Risk Averse": (score_dra_exp, [SCALE, (0.0, 6.0), (0.0, 6.0),
                                               (-5.0, 5.0), (0.0, 8.0), (0.0, 8.0)]),
    "EU":                    (score_eu, [SCALE, (0.05, 3.0)]),
    "PT":                    (score_pt, [SCALE, (0.05, 3.0), (0.05, 3.0), (0.05, 5.0)]),
    "CPT":                   (score_cpt, [SCALE, (0.05, 3.0), (0.05, 3.0), (0.05, 5.0)]),
}

DISCOVERED_CONTROL = {
    "SR: Aft-prob (c=4)":        (score_aft_prob, [SCALE]),
    "SR: Ini-scaled Aft (c=6)":  (score_ini_scaled_aft, [SCALE]),
    "SR: Ini-scaled Aft (c=8)":  (score_ini_scaled_aft8, [SCALE, (0.05, 4.0)]),
    "SR: Hybrid (c=10)":         (score_hybrid10, [SCALE, (-10.0, 10.0)]),
    "SR: Hybrid (c=12)":         (score_hybrid12, [SCALE, (0.05, 4.0), (-10.0, 10.0)]),
    "SR: Hybrid (c=17)":         (score_hybrid17, [SCALE, (-2.0, 2.0), (-10.0, 10.0),
                                                   (0.05, 4.0)]),
}

DISCOVERED_TREATMENT = {
    "SR: Aft-prob (c=4)":    (score_aft_prob, [SCALE]),
    "SR: Hybrid (c=8)":      (score_t_hybrid8, [SCALE]),
    "SR: Log-EV (c=9)":      (score_log_ev, [SCALE]),
    "SR: Power-EV (c=14)":   (score_power_ev14, [SCALE, (0.05, 4.0), (-2.0, 2.0)]),
    "SR: Hybrid (c=16)":     (score_t_hybrid16, [SCALE, (-10.0, 10.0), (0.05, 4.0)]),
    "SR: Hybrid (c=27)":     (score_t_hybrid27, [SCALE, (-10.0, 10.0), (0.05, 4.0),
                                                 (-2.0, 2.0)]),
}


# ----------------------------------------------------------------------------
# fitting and evaluation (Section 5.4)
# ----------------------------------------------------------------------------

def predict(score_fn, F, o, th):
    return sigma(score_fn(F, o, th))


def metrics(p_emp, q):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    ce = float(-np.mean(p_emp * np.log(q) + (1 - p_emp) * np.log(1 - q)))
    mse = float(np.mean((q - p_emp) ** 2))
    acc = float(np.mean((q > 0.5) == (p_emp > 0.5)))
    return ce, mse, acc


def baseline_ce(p_emp):
    """Irreducible cross-entropy: entropy of the empirical rates."""
    p = np.clip(p_emp, 1e-9, 1 - 1e-9)
    return float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))


def fit_model(score_fn, bounds, F, o, p_emp, seed=0, n_starts=12):
    """Fit free parameters by minimizing train MSE (paper Section 5.4)."""
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def loss(th):
        with np.errstate(all="ignore"):
            q = predict(score_fn, F, o, th)
        if not np.all(np.isfinite(q)):
            return 1e6
        return float(np.mean((q - p_emp) ** 2))

    best_th, best_l = None, np.inf
    starts = [lo + (hi - lo) * rng.random(len(bounds)) for _ in range(n_starts)]
    starts.append(np.clip(np.ones(len(bounds)), lo, hi))
    for x0 in starts:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 2500, "xatol": 1e-8, "fatol": 1e-10})
        if res.fun < best_l:
            best_l, best_th = res.fun, np.clip(res.x, lo, hi)
    return best_th, best_l
