"""Numerical verification of every closed-form calculation in the paper.

Checks (all from Sections 3.2, 5.1, 5.2, 5.3):
  1. E[S_A] - E[S_B] = dP_A - dP_B
  2. Var(S_A) - Var(S_B) = dP_A(1 - 2P_ini_A - dP_A) - dP_B(1 - 2P_ini_B - dP_B)
  3. equal increments dP_A = dP_B = dP:  Var(S_A) - Var(S_B) = 2 dP (P_ini_B - P_ini_A)
  4. tail-score identity:
     U_tail(B) - U_tail(A) = w2 (P_ini_A dP_B - P_ini_B dP_A)
                             - w0 ((1-P_ini_A) dP_B - (1-P_ini_B) dP_A)
  5. EU with alpha = 1 reduces to the probability-gain (expected-value) model
  6. CRA closed form: u(m_B) - u(m_A) = A e^{-z_ini}(e^{-b2 dP_B} - e^{-b1 dP_A})
                                        - eta1 (dP_A + dP_B)
  7. DRA closed form: u(m_B) - u(m_A) = F(z_ini + b2 dP_B) - F(z_ini + b1 dP_A)
                                        - eta1 (dP_A + dP_B)
  8. CPT rank-dependent weights sum to 1
"""
from __future__ import annotations

import numpy as np

from task import induced_pmf, component_pool, full_problem_pool, make_stimuli


def random_states(rng, n=20000):
    pa = rng.random(n)
    pb = rng.random(n)
    da = rng.random(n) * (1 - pa)
    db = rng.random(n) * (1 - pb)
    return pa, da, pb, db


def moments(p1, p2):
    r0, r1, r2 = induced_pmf(p1, p2)
    ev = r1 + 2 * r2
    var = r1 + 4 * r2 - ev ** 2
    return ev, var


def run_checks(verbose=True):
    rng = np.random.default_rng(42)
    pa, da, pb, db = random_states(rng)
    results = {}

    ev_a, var_a = moments(pa + da, pb)   # invest in A
    ev_b, var_b = moments(pa, pb + db)   # invest in B

    # 1) expected-value identity
    results["EV identity"] = np.max(np.abs((ev_a - ev_b) - (da - db)))

    # 2) variance identity
    rhs = da * (1 - 2 * pa - da) - db * (1 - 2 * pb - db)
    results["Var identity"] = np.max(np.abs((var_a - var_b) - rhs))

    # 3) equal-increment variance identity
    d = np.minimum(rng.random(len(pa)) * (1 - pa), (1 - pb))
    _, va = moments(pa + d, pb)
    _, vb = moments(pa, pb + d)
    results["Var identity (equal dP)"] = np.max(np.abs((va - vb) - 2 * d * (pb - pa)))

    # 4) tail-score identity.  NOTE: the paper prints the omega_0 term with a
    # MINUS sign; direct expansion shows it must be PLUS:
    #   Pr(U=0|A) - Pr(U=0|B) = (1-P_ini_A) dP_B - (1-P_ini_B) dP_A
    # so U_tail(B) - U_tail(A) = w2 (P_ini_A dP_B - P_ini_B dP_A)
    #                            + w0 ((1-P_ini_A) dP_B - (1-P_ini_B) dP_A)
    w0, w2 = rng.random(2)
    ra0, _, ra2 = induced_pmf(pa + da, pb)
    rb0, _, rb2 = induced_pmf(pa, pb + db)
    u_tail_a = w0 * (1 - ra0) + w2 * ra2
    u_tail_b = w0 * (1 - rb0) + w2 * rb2
    rhs_paper = w2 * (pa * db - pb * da) - w0 * ((1 - pa) * db - (1 - pb) * da)
    rhs_fixed = w2 * (pa * db - pb * da) + w0 * ((1 - pa) * db - (1 - pb) * da)
    results["Tail identity (as printed)"] = np.max(np.abs((u_tail_b - u_tail_a) - rhs_paper))
    results["Tail identity (sign-fixed)"] = np.max(np.abs((u_tail_b - u_tail_a) - rhs_fixed))

    # 5) EU(alpha=1) == probability gain (expected value)
    v_eu_a = (lambda r: r[1] + r[2] * 2.0)(induced_pmf(pa + da, pb))
    v_eu_b = (lambda r: r[1] + r[2] * 2.0)(induced_pmf(pa, pb + db))
    results["EU(alpha=1) = prob-gain"] = np.max(np.abs((v_eu_b - v_eu_a) - (db - da)))

    # 6) CRA closed form
    A, b1, b2, eta1, D = 1.3, 0.7, 1.9, 0.4, 2.0
    u_cra = lambda m1, m2: A * np.exp(-(b1 * m1 + b2 * m2)) + eta1 * m1 - eta1 * m2 + D
    lhs = u_cra(pa, pb + db) - u_cra(pa + da, pb)
    z_ini = b1 * pa + b2 * pb
    rhs = A * np.exp(-z_ini) * (np.exp(-b2 * db) - np.exp(-b1 * da)) - eta1 * (da + db)
    results["CRA closed form"] = np.max(np.abs(lhs - rhs))

    # 7) DRA closed form (any F; use F(z) = log(1+z))
    F = np.log1p
    u_dra = lambda m1, m2: F(b1 * m1 + b2 * m2) + eta1 * m1 - eta1 * m2 + D
    lhs = u_dra(pa, pb + db) - u_dra(pa + da, pb)
    rhs = F(z_ini + b2 * db) - F(z_ini + b1 * da) - eta1 * (da + db)
    results["DRA closed form"] = np.max(np.abs(lhs - rhs))

    # 8) CPT decision weights sum to one
    aw, bw = 0.6, 0.8
    w = lambda p: bw * p ** aw / (bw * p ** aw + (1 - p) ** aw)
    r0, r1, r2 = induced_pmf(pa + da, pb)
    pi2 = w(r2)
    pi1 = w(r1 + r2) - w(r2)
    pi0 = 1 - w(r1 + r2)
    results["CPT weights sum to 1"] = np.max(np.abs(pi0 + pi1 + pi2 - 1))

    if verbose:
        print("=" * 64)
        print("VERIFICATION OF CLOSED-FORM CALCULATIONS (max abs error)")
        print("=" * 64)
        for k, v in results.items():
            if k == "Tail identity (as printed)":
                status = "TYPO" if v > 1e-10 else "PASS"
            else:
                status = "PASS" if v < 1e-10 else "FAIL"
            print(f"  [{status}] {k:<28s} err = {v:.2e}")
        print("  NOTE: the printed tail identity in Section 5.1 carries a sign typo"
              "\n        on the omega_0 term; the sign-fixed version is exact.")
    return results


def stimulus_pool_stats(verbose=True):
    comps = component_pool()
    pool = full_problem_pool()
    stim = make_stimuli()
    if verbose:
        print("\nSTIMULUS POOL CALCULATIONS (Section 3.3)")
        print(f"  valid components (P_ini, dP) on the grid : {len(comps)}")
        print(f"  valid ordered problems with dP_B in [dP_A, dP_A+0.10]: {len(pool)}")
        print(f"  sampled stimulus pool size (paper: 1,873): {len(stim['P_ini_A'])}")
        print(f"  problems with equal increments dP_A = dP_B: "
              f"{int(np.sum(np.isclose(stim['dP_A'], stim['dP_B'])))}")
    return len(comps), len(pool)


if __name__ == "__main__":
    run_checks()
    stimulus_pool_stats()
