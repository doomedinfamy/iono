"""End-to-end reproduction of 'Decision-Making under Combinatorial Risk'
(arXiv:2606.10092). Runs every calculation of the paper on simulated data:

  1. verification of all closed-form identities (Sections 3.2, 5.1-5.3)
  2. stimulus-pool construction (Section 3.3)
  3. behavioral analyses (Section 4)
  4. benchmark model fitting/evaluation (Section 5, Table 2)
  5. SR-discovered model evaluation (Section 6.2, Tables 3-4)
  6. miniature symbolic-regression search (Section 6.1)
  7. PT residual analysis with bootstrap SEs (Section 7)

Usage: python main.py [--fast]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import verify_math
from task import make_stimuli
from models import (BENCHMARKS, DISCOVERED_CONTROL, DISCOVERED_TREATMENT,
                    fit_model, predict, metrics, baseline_ce, sigma)
from simulate import simulate, pooled_condition, train_test_split, subset
from srmini import run_sr, to_str
from residual import (fit_residual_model, predict_residual_model,
                      residual_correlations, bootstrap_metrics, logit)

FAST = "--fast" in sys.argv
OUT = "outputs"


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
# Section 4: behavioral analyses
# ---------------------------------------------------------------------------

def behavioral_analysis(F, data):
    hr("SECTION 4 - BEHAVIORAL PATTERNS (simulated data)")
    neq = ~np.isclose(F["dP_A"], F["dP_B"])
    eq = ~neq & ~np.isclose(F["P_ini_A"], F["P_ini_B"])

    print("\n4.1 Preference for the dominant option")
    print("  (a) problems with unequal increments: P(choose higher-dP option)")
    for g in ("C", "T"):
        for c in ("low", "high"):
            p = data[(g, c)]["p_emp"][neq]
            b_dom = F["dP_B"][neq] > F["dP_A"][neq]
            p_dom = np.where(b_dom, p, 1 - p)
            t, pv = stats.ttest_1samp(p_dom, 0.5)
            d = (p_dom.mean() - 0.5) / p_dom.std(ddof=1)
            print(f"    {g}/{c:<4s}: mean={p_dom.mean():.3f}  "
                  f"t({len(p_dom)-1})={t:.2f}, p={pv:.2e}, d={d:.3f}")

    print("  (b) equal increments: P(choose higher initial-probability option)")
    for g in ("C", "T"):
        for c in ("low", "high"):
            p = data[(g, c)]["p_emp"][eq]
            b_dom = F["P_ini_B"][eq] > F["P_ini_A"][eq]
            p_dom = np.where(b_dom, p, 1 - p)
            t, pv = stats.ttest_1samp(p_dom, 0.5)
            d = (p_dom.mean() - 0.5) / p_dom.std(ddof=1)
            print(f"    {g}/{c:<4s}: mean={p_dom.mean():.3f}  "
                  f"t({len(p_dom)-1})={t:.2f}, p={pv:.2e}, d={d:.3f}")

    print("\n4.2 Effect of payoff magnitude (change in bRate, high - low)")
    for g in ("C", "T"):
        dch = data[(g, "high")]["p_emp"] - data[(g, "low")]["p_emp"]
        frac = np.mean(~np.isclose(dch, 0))
        r1, p1 = stats.pearsonr(F["P_ini_B"] - F["P_ini_A"], dch)
        r2, p2 = stats.pearsonr(F["P_aft_B"] - F["P_aft_A"], dch)
        r3, p3 = stats.pearsonr(F["Var_B"] - F["Var_A"], dch)
        print(f"  {g}: changed in {frac*100:.1f}% of problems | "
              f"corr(dPini)={r1:.3f} (p={p1:.1e}), corr(dPaft)={r2:.3f} "
              f"(p={p2:.1e}), corr(dVar)={r3:.3f} (p={p3:.1e})")
    dc = data[("C", "high")]["p_emp"] - data[("C", "low")]["p_emp"]
    dt = data[("T", "high")]["p_emp"] - data[("T", "low")]["p_emp"]
    r, p = stats.pearsonr(dc, dt)
    print(f"  corr of magnitude effects across conditions: r={r:.4f}, p={p:.3f}")

    print("\n4.3 Effect of information treatment")
    for c in ("low", "high"):
        pc, pt_ = data[("C", c)]["p_emp"][eq], data[("T", c)]["p_emp"][eq]
        b_dom = F["P_ini_B"][eq] > F["P_ini_A"][eq]
        dc_, dt_ = np.where(b_dom, pc, 1 - pc), np.where(b_dom, pt_, 1 - pt_)
        t, pv = stats.ttest_ind(dc_, dt_)
        print(f"  {c:<4s}: control pref={dc_.mean():.3f} vs treatment="
              f"{dt_.mean():.3f}  t={t:.2f}, p={pv:.2e}")
        # variance compression of choice probabilities (unequal-increment set)
        vc = np.var(data[("C", c)]["p_emp"][neq], ddof=1)
        vt = np.var(data[("T", c)]["p_emp"][neq], ddof=1)
        Fst = vc / vt
        print(f"        choice-prob variance: control={vc:.4f}, "
              f"treatment={vt:.4f} (F={Fst:.2f})")
    # achieved expected successes
    for c in ("low", "high"):
        out = {}
        for g in ("C", "T"):
            p = data[(g, c)]["p_emp"]
            ev = p * F["EV_B"] + (1 - p) * F["EV_A"]
            out[g] = ev
        t, pv = stats.ttest_rel(out["T"], out["C"])
        print(f"  {c:<4s}: expected successes  C={out['C'].mean():.4f}  "
              f"T={out['T'].mean():.4f}  paired t={t:.2f}, p={pv:.3f}")
    return neq, eq


# ---------------------------------------------------------------------------
# Sections 5 & 6.2: model fitting tables
# ---------------------------------------------------------------------------

def fit_table(registry, F_tr, o_tr, p_tr, F_te, o_te, p_te, label):
    print(f"\n  {label}")
    print(f"  {'Model':<26s} {'CE_test':>8s} {'MSE_test':>9s} {'Acc_test':>9s}")
    rows = {}
    for name, (fn, bounds) in registry.items():
        t0 = time.time()
        th, _ = fit_model(fn, bounds, F_tr, o_tr, p_tr,
                          n_starts=(4 if FAST else 12))
        q = predict(fn, F_te, o_te, th)
        ce, mse, acc = metrics(p_te, q)
        rows[name] = (ce, mse, acc, th)
        print(f"  {name:<26s} {ce:8.4f} {mse:9.4f} {acc:9.4f}"
              f"   [{time.time()-t0:.1f}s]")
    return rows


def model_evaluation(data):
    results = {}
    splits = {}
    for g, gname in (("C", "control"), ("T", "treatment")):
        F, o, p = pooled_condition(data, g)
        tr, te = train_test_split(len(p))
        F_tr, o_tr, p_tr = subset(F, o, p, tr)
        F_te, o_te, p_te = subset(F, o, p, te)
        splits[g] = (F_tr, o_tr, p_tr, F_te, o_te, p_te)
        hr(f"SECTION 5 - BENCHMARK MODELS ({gname} condition, Table 2)")
        print(f"  baseline CE (empirical entropy): {baseline_ce(p_te):.4f}")
        results[(g, "bench")] = fit_table(
            BENCHMARKS, F_tr, o_tr, p_tr, F_te, o_te, p_te,
            f"benchmarks / {gname}")
        reg = DISCOVERED_CONTROL if g == "C" else DISCOVERED_TREATMENT
        hr(f"SECTION 6.2 - SR-DISCOVERED MODELS ({gname}, "
           f"Table {'3' if g == 'C' else '4'})")
        results[(g, "sr")] = fit_table(
            reg, F_tr, o_tr, p_tr, F_te, o_te, p_te,
            f"discovered models / {gname}")
    return results, splits


# ---------------------------------------------------------------------------
# Section 6.1: run the miniature symbolic regression
# ---------------------------------------------------------------------------

def run_symbolic_regression(splits):
    hr("SECTION 6.1 - MINIATURE SYMBOLIC REGRESSION (GP + Pareto search)")
    fronts = {}
    for g, gname in (("C", "control"), ("T", "treatment")):
        F_tr, o_tr, p_tr, F_te, o_te, p_te = splits[g]
        t0 = time.time()
        front = run_sr(F_tr, p_tr, seed=1,
                       pop_size=(120 if FAST else 250),
                       generations=(12 if FAST else 35))
        print(f"\n  Pareto frontier ({gname}), search took {time.time()-t0:.0f}s")
        print(f"  {'Cmplx':>5s} {'MSE_tr':>8s} {'CE_te':>7s} {'MSE_te':>7s} "
              f"{'Acc_te':>7s}  Expression")
        rows = []
        for c, mse_tr, tree, k in front:
            if mse_tr >= 1e6:
                continue
            from srmini import evaluate
            with np.errstate(all="ignore"):
                q = sigma(k * np.clip(evaluate(tree, F_te), -1e6, 1e6))
            ce, mse, acc = metrics(p_te, q)
            rows.append((c, mse_tr, ce, mse, acc, to_str(tree)))
            print(f"  {c:5d} {mse_tr:8.4f} {ce:7.4f} {mse:7.4f} {acc:7.4f}  "
                  f"{to_str(tree)[:70]}")
        fronts[g] = rows
    return fronts


# ---------------------------------------------------------------------------
# Section 7: residual analysis
# ---------------------------------------------------------------------------

def residual_analysis(results, splits, data):
    hr("SECTION 7 - RESIDUAL ANALYSIS FOR THE TREATMENT EFFECT")
    # control model (best without expected payoff): SR Hybrid (c=12)
    from models import DISCOVERED_CONTROL, DISCOVERED_TREATMENT
    fn_c, _ = DISCOVERED_CONTROL["SR: Hybrid (c=12)"]
    th_c = results[("C", "sr")]["SR: Hybrid (c=12)"][3]

    F_tr, o_tr, p_tr, F_te, o_te, p_te = splits["T"]
    pc_tr = np.clip(predict(fn_c, F_tr, o_tr, th_c), 1e-6, 1 - 1e-6)
    pc_te = np.clip(predict(fn_c, F_te, o_te, th_c), 1e-6, 1 - 1e-6)

    # transfer performance & residual correlations on the full treatment set
    ce, mse, acc = metrics(p_te, pc_te)
    print(f"\n  control model transferred to treatment data: "
          f"CE={ce:.4f} MSE={mse:.4f} Acc={acc:.4f}")
    resid = p_te - pc_te
    print("  residual correlations with PMF features:")
    for k, (r, p) in residual_correlations(F_te, resid).items():
        flag = "*" if p < 0.05 else " "
        print(f"    corr(r, {k}) = {r:+.3f} (p={p:.3f}){flag}")

    # binomial counts for MLE fitting on the training split
    n_per = 20
    nb_tr = np.round(p_tr * n_per).astype(int)

    variants = {}
    for label, monetary in (("C+PT (w/)", True), ("C+PT (w/o)", False)):
        th = fit_residual_model(pc_tr, F_tr, o_tr, nb_tr, n_per, monetary)
        q = predict_residual_model(th, pc_te, F_te, o_te, monetary)
        variants[label] = q
        a, b0, b1, av, g = th
        print(f"\n  {label}: alpha={a:.3f} beta0={b0:.3f} beta1={b1:.3f} "
              f"alpha_v={av:.3f} gamma={g:.3f}")

    # comparison set: C alone, C+PT variants, PT-only benchmark, T model
    fn_t, _ = DISCOVERED_TREATMENT["SR: Hybrid (c=27)"]
    th_t = results[("T", "sr")]["SR: Hybrid (c=27)"][3]
    fn_pt, _ = BENCHMARKS["PT"]
    th_pt = results[("T", "bench")]["PT"][3]
    preds = {"C (transfer)": pc_te,
             "C+PT (w/)": variants["C+PT (w/)"],
             "C+PT (w/o)": variants["C+PT (w/o)"],
             "PT only": predict(fn_pt, F_te, o_te, th_pt),
             "T (SR c=27)": predict(fn_t, F_te, o_te, th_t)}

    print(f"\n  bootstrap comparison on treatment test set "
          f"(B={100 if FAST else 1000}):")
    print(f"  {'Model':<14s} {'CE':>16s} {'MSE':>16s} {'Acc':>16s}")
    bars = {}
    for name, q in preds.items():
        mean, se = bootstrap_metrics(p_te, q, B=(100 if FAST else 1000))
        bars[name] = (mean, se)
        print(f"  {name:<14s} {mean[0]:8.4f}±{se[0]:.4f} "
              f"{mean[1]:8.4f}±{se[1]:.4f} {mean[2]:8.4f}±{se[2]:.4f}")
    return bars


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(F, data, neq, eq, fronts, bars, results):
    import os
    os.makedirs(OUT, exist_ok=True)

    # Figure 3 analogue: dominant-option preference boxplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mask, dom_mask, title in (
            (axes[0], neq, F["dP_B"] > F["dP_A"],
             "(a) higher-increment option, dP_A != dP_B"),
            (axes[1], eq, F["P_ini_B"] > F["P_ini_A"],
             "(b) higher initial-prob option, dP_A = dP_B")):
        groups, labels = [], []
        for g in ("C", "T"):
            for c in ("low", "high"):
                p = data[(g, c)]["p_emp"][mask]
                groups.append(np.where(dom_mask[mask], p, 1 - p))
                labels.append(f"{g}/{c}")
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("P(choose dominant)")
    fig.suptitle("Preference for the dominant option (Figure 3 analogue)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_dominant_option.png", dpi=130)
    plt.close(fig)

    # Figure 6 analogue: Pareto frontier vs benchmarks
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, g, gname in ((axes[0], "C", "control"), (axes[1], "T", "treatment")):
        rows = fronts[g]
        ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-",
                label="SR Pareto frontier")
        bench = results[(g, "bench")]
        for name in ("Delta-diff", "Aft-prob", "Tail(w0,w2)", "EU", "PT", "CPT"):
            ce = bench[name][0]
            ax.scatter([14], [ce], marker="x")
            ax.annotate(name, (14, ce), fontsize=7,
                        textcoords="offset points", xytext=(4, 0))
        ax.set_xlabel("complexity")
        ax.set_ylabel("test cross-entropy")
        ax.set_title(f"{gname} condition")
        ax.legend(fontsize=8)
    fig.suptitle("Discovered models vs benchmarks (Figure 6 analogue)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_pareto.png", dpi=130)
    plt.close(fig)

    # Figure 8 analogue: residual-model comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    names = list(bars.keys())
    for i, (ax, metric) in enumerate(zip(axes, ("CE", "MSE", "Acc"))):
        vals = [bars[n][0][i] for n in names]
        errs = [bars[n][1][i] for n in names]
        ax.bar(range(len(names)), vals, yerr=errs, capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
    fig.suptitle("Residual augmentation on treatment data (Figure 8 analogue)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_residual.png", dpi=130)
    plt.close(fig)
    print(f"\n  figures saved to {OUT}/fig3_dominant_option.png, "
          f"fig6_pareto.png, fig8_residual.png")


def main():
    t0 = time.time()
    hr("STEP 1 - CLOSED-FORM VERIFICATION")
    verify_math.run_checks()
    verify_math.stimulus_pool_stats()

    hr("STEP 2 - SIMULATED EXPERIMENT")
    F, data = simulate(seed=7)
    n = len(F["P_ini_A"])
    print(f"  {n} problems x 2 magnitudes x 2 conditions, 20 choices per cell"
          f" -> {n * 4 * 20} simulated decisions")

    neq, eq = behavioral_analysis(F, data)
    results, splits = model_evaluation(data)
    fronts = run_symbolic_regression(splits)
    bars = residual_analysis(results, splits, data)
    make_figures(F, data, neq, eq, fronts, bars, results)
    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
