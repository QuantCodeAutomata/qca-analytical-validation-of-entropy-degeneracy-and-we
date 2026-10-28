"""
Experiment 3: Date-wise exact-fit entropy calibration.

At each quarter-end quote date before maturity, solve the constrained problem:
  maximise H_V(Q_vartheta) = -K_C(vartheta)
  subject to V_model(t_k; vartheta) = 100 exactly.

This selects the entropy-maximising (closest-to-physical-measure) point on the
weakly-identified feasible set for each date.
"""

import sys
import os
import warnings
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from physical_model import (
    gpd_cdf, gpd_survival, LAMBDA_PHYS, A_SHAPE, C_SCALE, D_THRESHOLD,
)
from measure_change import (
    wang_distorted_cdf, trigger_intensity, eta_f_to_lambda_prime, q_intensity,
)
from entropy_geometry import (
    kl_divergence, financial_entropy, riemannian_gradient, entropy_gradient,
)
from compound_poisson_fft import (
    aggregate_loss_prob_less_than,
    aggregate_loss_prob_less_than_from_pmf,
    make_severity_pmf,
)
from cat_bond_pricer import price_cat_bond
from treasury_curve import (
    get_treasury_data, build_curve, get_discount_factor, compute_par_yield,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA = LAMBDA_PHYS
A = A_SHAPE
C = C_SCALE
D = D_THRESHOLD
T = 4.25
N_NOMINAL = 100.0
ZETA = 0.5
DELTA = 0.25
PAYMENT_SCHEDULE = np.array([j / 4 for j in range(18)])

BAR_F_D = gpd_survival(D, A, C)
Z_D = norm.ppf(gpd_cdf(D, A, C))  # Phi^{-1}(F(D))

QUARTER_DATES = [
    "2014-09-30", "2014-12-31", "2015-03-31", "2015-06-30",
    "2015-09-30", "2015-12-31", "2016-03-31", "2016-06-30",
    "2016-09-30", "2016-12-31", "2017-03-31", "2017-06-30",
    "2017-09-30", "2017-12-31", "2018-03-31", "2018-06-30",
    "2018-09-30", "2018-12-31",
]
SPREADS_PCT = [
    2.16, 2.23, 2.26, 2.32, 2.18, 4.56, 2.60, 2.50,
    2.36, 2.27, 2.33, 2.38, 2.41, 1.83, 1.65, 1.96,
    1.69, 3.11,
]

FFT_GRID = 2 ** 12    # 4096 — fast and accurate enough
X_MAX   = 5e9         # aggregate-loss truncation point

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Market-input helpers (same convention as exp_2)
# ---------------------------------------------------------------------------

def build_market_inputs(treasury_data: dict) -> list[dict]:
    """Build per-date market inputs (spreads, coupons, discount factors)."""
    inputs = []
    for k, (date, spread_pct) in enumerate(zip(QUARTER_DATES, SPREADS_PCT)):
        spread = spread_pct / 100.0
        curve = build_curve(date, treasury_data)
        remaining_j = np.arange(k + 1, 18)
        remaining_times_from_tk = (remaining_j - k) * 0.25

        if len(remaining_times_from_tk) == 0:
            disc_factors = np.array([])
            coupon = spread
        else:
            disc_factors = get_discount_factor(remaining_times_from_tk, curve)
            par_yield = compute_par_yield(remaining_times_from_tk, disc_factors, DELTA)
            coupon = spread + par_yield

        inputs.append({
            "date": date,
            "k": k,
            "spread": spread,
            "coupon": coupon,
            "disc_factors": disc_factors,
            "remaining_times_from_tk": remaining_times_from_tk,
        })
    return inputs


# ---------------------------------------------------------------------------
# Pricing helper
# ---------------------------------------------------------------------------

def price_at_date(
    vartheta: tuple,
    market_input: dict,
    severity_pmf: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    """Compute model CAT bond price at a single valuation date.

    Parameters
    ----------
    vartheta : (eta_f, kappa)
    market_input : dict with 'k', 'coupon', 'disc_factors', 'remaining_times_from_tk'
    severity_pmf, dx : optional pre-built discretised severity PMF (Q-measure).
        If supplied the FFT convolution reuses the PMF directly; otherwise it
        is built on the fly.
    """
    k = market_input["k"]
    if k == 17:
        return N_NOMINAL

    eta_f, kappa = vartheta
    coupon = market_input["coupon"]
    disc_factors = market_input["disc_factors"]
    times_from_tk = market_input["remaining_times_from_tk"]

    hat_lambda = q_intensity(eta_f, lambda_phys=LAMBDA)

    if severity_pmf is None or dx is None:
        sev_cdf_q: Callable = lambda x: wang_distorted_cdf(x, kappa, A, C)
        severity_pmf, dx = make_severity_pmf(sev_cdf_q, X_MAX, FFT_GRID)

    times_with_zero = np.concatenate([[0.0], times_from_tk])
    q_surv = np.array([
        1.0 if t == 0.0 else aggregate_loss_prob_less_than_from_pmf(
            D, hat_lambda, float(t), severity_pmf, dx
        )
        for t in times_with_zero
    ])

    return price_cat_bond(
        coupon=coupon,
        discount_factors=disc_factors,
        payment_times=times_from_tk,
        q_survival=q_surv,
        N_nominal=N_NOMINAL,
        zeta=ZETA,
        Delta=DELTA,
    )


def price_residual(
    vartheta: tuple,
    market_input: dict,
    severity_pmf: np.ndarray | None = None,
    dx: float | None = None,
) -> float:
    """Price residual: V_model - 100."""
    return price_at_date(vartheta, market_input, severity_pmf, dx) - N_NOMINAL


# ---------------------------------------------------------------------------
# Feasible-curve tracing: for fixed eta_f, solve for kappa
# ---------------------------------------------------------------------------

def solve_kappa_exact_fit(
    eta_f: float,
    market_input: dict,
    kappa_lo: float = -4.9,
    kappa_hi: float = 4.9,
) -> float | None:
    """Find kappa in [kappa_lo, kappa_hi] such that V_model = 100.

    The PMF is rebuilt for each kappa trial (kappa changes the severity distortion).
    Returns the kappa value, or None if no solution exists in the bracket.
    """
    def f(kappa: float) -> float:
        sev_cdf_q = lambda x: wang_distorted_cdf(x, kappa, A, C)
        pmf, dxx = make_severity_pmf(sev_cdf_q, X_MAX, FFT_GRID)
        return price_residual((eta_f, kappa), market_input, pmf, dxx)

    try:
        f_lo = f(kappa_lo)
        f_hi = f(kappa_hi)
        if f_lo * f_hi > 0:
            return None  # no sign change — no feasible kappa in bracket
        kappa_sol = optimize.brentq(f, kappa_lo, kappa_hi, xtol=1e-8, rtol=1e-8, maxiter=100)
        return float(kappa_sol)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Algorithm 2: Constrained Riemannian gradient ascent
# ---------------------------------------------------------------------------

def datewise_constrained_entropy_max(
    market_input: dict,
    n_ef_grid: int = 60,
    fd_bump: float = 5e-4,
    tol_H: float = 1e-6,
    tol_c: float = 1e-6,
) -> dict | None:
    """Maximise entropy subject to exact price fit for one valuation date.

    Strategy (Algorithm 2 variant):
      1. Grid over eta_f, solve exact-fit constraint for kappa using brentq.
      2. Collect all feasible (eta_f, kappa) pairs and compute entropy.
      3. Return the pair with maximum entropy (minimum KL).

    Also attempts a constrained SLSQP optimisation from the best grid point
    as a refinement step.

    Parameters
    ----------
    market_input : dict
        Per-date market data from build_market_inputs().
    n_ef_grid : int
        Number of eta_f grid points for feasible-curve tracing.

    Returns
    -------
    dict with keys: eta_f, kappa, kl, ell, risk_multiple, feasible,
                    eta_f_feasible, kappa_feasible, kl_feasible
    """
    k = market_input["k"]
    if k == 17:
        return None  # maturity — uninformative

    eta_f_grid = np.linspace(-4.5, 4.5, n_ef_grid)
    feasible_ef = []
    feasible_ka = []
    feasible_kl = []

    for ef in eta_f_grid:
        ka = solve_kappa_exact_fit(ef, market_input)
        if ka is not None:
            kl = kl_divergence(ef, ka, LAMBDA, T)
            feasible_ef.append(ef)
            feasible_ka.append(ka)
            feasible_kl.append(kl)

    if len(feasible_ef) == 0:
        return {
            "feasible": False,
            "date": market_input["date"],
            "k": k,
        }

    feasible_ef = np.array(feasible_ef)
    feasible_ka = np.array(feasible_ka)
    feasible_kl = np.array(feasible_kl)

    # Grid best: minimum KL = maximum entropy
    idx_best = np.argmin(feasible_kl)
    ef_best = feasible_ef[idx_best]
    ka_best = feasible_ka[idx_best]

    # SLSQP refinement
    try:
        res = optimize.minimize(
            fun=lambda v: kl_divergence(v[0], v[1], LAMBDA, T),
            x0=[ef_best, ka_best],
            method="SLSQP",
            bounds=[(-5, 5), (-5, 5)],
            constraints={"type": "eq", "fun": lambda v: price_residual(v, market_input)},
            options={"ftol": 1e-10, "maxiter": 200},
        )
        if res.success or abs(price_residual(res.x, market_input)) < 0.05:
            ef_refined = res.x[0]
            ka_refined = res.x[1]
            # Verify feasibility
            resid = price_residual((ef_refined, ka_refined), market_input)
            if abs(resid) < 0.1:
                ef_best = ef_refined
                ka_best = ka_refined
    except Exception:
        pass

    # Ensure ka_best satisfies constraint
    ka_check = solve_kappa_exact_fit(ef_best, market_input)
    if ka_check is not None:
        ka_best = ka_check

    kl_best = kl_divergence(ef_best, ka_best, LAMBDA, T)
    ell_best = trigger_intensity(ef_best, ka_best, D, LAMBDA, A, C)
    risk_mult = ell_best / (LAMBDA * BAR_F_D)
    lp_best = eta_f_to_lambda_prime(ef_best)

    # Frequency share: 100 * eta_f / ln(ell / (lambda * barF_D))
    ell_phys = LAMBDA * BAR_F_D
    denom = np.log(risk_mult) if risk_mult > 1 else 1e-8
    freq_share_pct = 100.0 * ef_best / denom if abs(denom) > 1e-6 else np.nan

    return {
        "feasible": True,
        "date": market_input["date"],
        "k": k,
        "eta_f": float(ef_best),
        "kappa": float(ka_best),
        "lambda_prime": float(lp_best),
        "kl": float(kl_best),
        "ell": float(ell_best),
        "risk_multiple": float(risk_mult),
        "freq_share_pct": float(freq_share_pct),
        "price_check": float(price_at_date((ef_best, ka_best), market_input)),
        "eta_f_feasible": feasible_ef,
        "kappa_feasible": feasible_ka,
        "kl_feasible": feasible_kl,
    }


# ---------------------------------------------------------------------------
# Actuarial benchmark
# ---------------------------------------------------------------------------

def actuarial_price(market_input: dict) -> float:
    """Price with physical survival and market discounting (kappa=0, lambda'=0)."""
    return price_at_date((0.0, 0.0), market_input)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(results: list[dict]) -> None:
    """Diagnostic plots for date-wise calibration."""
    feasible_results = [r for r in results if r.get("feasible", False)]
    dates = [r["date"] for r in feasible_results]
    multiples = [r["risk_multiple"] for r in feasible_results]
    spreads_pct = [SPREADS_PCT[r["k"]] for r in feasible_results]
    lambdas = [r["lambda_prime"] for r in feasible_results]
    kappas = [r["kappa"] for r in feasible_results]
    kls = [r["kl"] for r in feasible_results]
    freq_shares = [r["freq_share_pct"] for r in feasible_results]

    n = len(feasible_results)
    x_idx = np.arange(n)

    # Plot 1: Risk multiples time series
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x_idx, multiples, "bo-", lw=2, markersize=6, label="Risk multiple ℓ/(λ·F̄(D))")
    axes[0].set_ylabel("Risk multiple")
    axes[0].set_title("Date-wise Entropy-Optimal Risk Multiples")
    axes[0].legend()
    # Annotate spike
    idx_spike = np.argmax(multiples)
    axes[0].annotate(f"{multiples[idx_spike]:.1f}\n{dates[idx_spike]}",
                     xy=(x_idx[idx_spike], multiples[idx_spike]),
                     xytext=(x_idx[idx_spike]+0.5, multiples[idx_spike]-1),
                     fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"))

    axes[1].plot(x_idx, spreads_pct, "rs-", lw=2, markersize=5, label="Observed spread (%)")
    axes[1].set_ylabel("Spread (%)")
    axes[1].set_xlabel("Quarter-end date")
    axes[1].legend()
    axes[1].set_xticks(x_idx[::2])
    axes[1].set_xticklabels([dates[i] for i in range(0, n, 2)], rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_risk_multiples.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Lambda' and kappa time series
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(x_idx, lambdas, "go-", lw=2, markersize=5, label="λ' (frequency loading)")
    axes[0].set_ylabel("λ'")
    axes[0].legend()
    axes[1].plot(x_idx, kappas, "mo-", lw=2, markersize=5, label="κ (severity loading)")
    axes[1].set_ylabel("κ")
    axes[1].set_xlabel("Quarter-end date")
    axes[1].legend()
    axes[1].set_xticks(x_idx[::2])
    axes[1].set_xticklabels([dates[i] for i in range(0, n, 2)], rotation=30, ha="right")
    fig.suptitle("Date-wise Entropy-Optimal Parameters")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_parameters.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Feasible curve for selected dates
    key_indices = [0, 5, 6, 14, 16]  # 2014-09-30, 2015-12-31, 2016-03-31, 2018-03-31, 2018-09-30
    key_labels = ["2014-09-30", "2015-12-31", "2016-03-31", "2018-03-31", "2018-09-30"]
    colors = ["blue", "red", "green", "orange", "purple"]

    fig, ax = plt.subplots(figsize=(10, 7))
    for fi, (ki, label, col) in enumerate(zip(key_indices, key_labels, colors)):
        # Find in feasible_results by date
        r_list = [r for r in feasible_results if r["date"] == label]
        if not r_list:
            continue
        r = r_list[0]
        ef_feas = r["eta_f_feasible"]
        ka_feas = r["kappa_feasible"]
        kl_feas = r["kl_feasible"]
        ax.plot(ef_feas, ka_feas, "-", color=col, lw=1.5, label=f"Feasible curve {label}")
        ax.plot(r["eta_f"], r["kappa"], "*", color=col, markersize=14)

    ax.set_xlabel("η_f = ln(1+λ')")
    ax.set_ylabel("κ")
    ax.set_title("Exact-Fit Feasible Curves and Entropy-Selected Points\n(★ = entropy maximum)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_feasible_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 4: Frequency share
    fig, ax = plt.subplots(figsize=(10, 4))
    fs_clean = [f for f in freq_shares if np.isfinite(f)]
    x_clean = [x_idx[i] for i, f in enumerate(freq_shares) if np.isfinite(f)]
    d_clean = [dates[i] for i, f in enumerate(freq_shares) if np.isfinite(f)]
    ax.bar(x_clean, fs_clean, color="steelblue", alpha=0.8)
    ax.axhline(50, color="red", ls="--", label="50% frequency share")
    ax.set_xticks(x_clean[::2])
    ax.set_xticklabels(d_clean[::2], rotation=30, ha="right")
    ax.set_ylabel("Frequency share (%)")
    ax.set_title("Date-wise Frequency Share of Total Risk Multiple\nη_f / ln(ℓ/λ·F̄(D)) × 100%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_frequency_share.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("Plots saved to results/")


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(results: list[dict]) -> None:
    """Append experiment 3 results to RESULTS.md."""
    feasible_results = [r for r in results if r.get("feasible", False)]
    n_feasible = len(feasible_results)

    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(md_path, "a") as f:
        f.write("\n\n## Experiment 3: Date-wise Exact-Fit Entropy Calibration\n\n")
        f.write(f"- Feasible dates: {n_feasible} / 17 (excluding maturity date)\n\n")
        f.write("### Date-wise Results Summary\n\n")
        f.write("| Date | λ' | κ | Risk Multiple | K_C | Freq Share% | Price Check |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        # Expected targets for key dates
        targets = {
            "2014-09-30": (2.437, 1.029, 8.867),
            "2015-12-31": (5.366, 1.224, 18.260),
            "2016-03-31": (2.940, 1.076, 10.449),
            "2018-03-31": (1.741, 0.913, 6.576),
            "2018-09-30": (1.788, 0.920, 6.719),
        }

        for r in feasible_results:
            f.write(f"| {r['date']} | {r['lambda_prime']:.3f} | {r['kappa']:.3f} | "
                    f"{r['risk_multiple']:.3f} | {r['kl']:.4f} | "
                    f"{r['freq_share_pct']:.1f}% | {r['price_check']:.3f} |\n")

        f.write("\n### Key Date Comparison vs Paper Targets\n\n")
        f.write("| Date | λ'_est | λ'_target | κ_est | κ_target | Multiple_est | Multiple_target |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in feasible_results:
            if r["date"] in targets:
                tgt = targets[r["date"]]
                f.write(f"| {r['date']} | {r['lambda_prime']:.3f} | {tgt[0]} | "
                        f"{r['kappa']:.3f} | {tgt[1]} | "
                        f"{r['risk_multiple']:.3f} | {tgt[2]} |\n")

    print(f"\nResults appended to {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("EXPERIMENT 3: DATE-WISE EXACT-FIT ENTROPY CALIBRATION")
    print("Golden State Re II Series 2014-1")
    print("=" * 60)

    print("\nLoading Treasury data and building market inputs...")
    treasury_data = get_treasury_data()
    mkt_inputs = build_market_inputs(treasury_data)

    # Expected key-date targets from the paper
    targets = {
        "2014-09-30": {"lambda_prime": 2.437, "kappa": 1.029, "multiple": 8.867},
        "2015-12-31": {"lambda_prime": 5.366, "kappa": 1.224, "multiple": 18.260},
        "2016-03-31": {"lambda_prime": 2.940, "kappa": 1.076, "multiple": 10.449},
        "2018-03-31": {"lambda_prime": 1.741, "kappa": 0.913, "multiple": 6.576},
        "2018-09-30": {"lambda_prime": 1.788, "kappa": 0.920, "multiple": 6.719},
    }

    all_results = []
    for inp in mkt_inputs:
        k = inp["k"]
        date = inp["date"]

        if k == 17:
            print(f"  {date}: Maturity — SKIPPED (uninformative)")
            all_results.append({"feasible": None, "date": date, "k": k})
            continue

        print(f"\n  Processing {date} (k={k})...")
        res = datewise_constrained_entropy_max(inp, n_ef_grid=70)
        all_results.append(res)

        if res["feasible"]:
            lp = res["lambda_prime"]
            ka = res["kappa"]
            mult = res["risk_multiple"]
            kl = res["kl"]
            pcheck = res["price_check"]
            fs = res["freq_share_pct"]
            print(f"    λ'={lp:.4f}, κ={ka:.4f}, multiple={mult:.3f}, "
                  f"K_C={kl:.4f}, freq_share={fs:.1f}%, price={pcheck:.4f}")

            if date in targets:
                tgt = targets[date]
                print(f"    Target: λ'={tgt['lambda_prime']}, κ={tgt['kappa']}, "
                      f"multiple={tgt['multiple']}")
        else:
            print(f"    INFEASIBLE — no (eta_f, kappa) pair achieves exact fit")

    # Summary statistics
    feasible_results = [r for r in all_results if r.get("feasible", False)]
    n_feasible = len(feasible_results)
    multiples = [r["risk_multiple"] for r in feasible_results]
    freq_shares = [r["freq_share_pct"] for r in feasible_results if np.isfinite(r["freq_share_pct"])]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Feasible dates: {n_feasible} / 17 (excl. maturity)")
    if multiples:
        print(f"  Risk multiple range: [{min(multiples):.3f}, {max(multiples):.3f}]")
        print(f"  Risk multiple mean: {np.mean(multiples):.3f}")
    if freq_shares:
        print(f"  Freq share range: [{min(freq_shares):.1f}%, {max(freq_shares):.1f}%]")

    print("\n  Key date results vs targets:")
    for r in feasible_results:
        if r["date"] in targets:
            tgt = targets[r["date"]]
            print(f"    {r['date']}: λ'={r['lambda_prime']:.3f} (tgt={tgt['lambda_prime']}), "
                  f"κ={r['kappa']:.3f} (tgt={tgt['kappa']}), "
                  f"mult={r['risk_multiple']:.3f} (tgt={tgt['multiple']})")

    make_plots(all_results)
    save_results(all_results)

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
