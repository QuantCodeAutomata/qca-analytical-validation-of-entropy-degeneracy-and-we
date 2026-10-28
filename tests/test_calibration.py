import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropy_geometry import kl_divergence, financial_entropy, riemannian_gradient
from measure_change import trigger_intensity, eta_f_to_lambda_prime, wang_distorted_cdf
from physical_model import (
    LAMBDA_PHYS,
    A_SHAPE,
    C_SCALE,
    D_THRESHOLD,
    gpd_cdf,
    gpd_survival,
)
from treasury_curve import (
    get_treasury_data,
    bootstrap_discount_factors,
    compute_par_yield,
    build_curve,
    get_discount_factor,
)
from compound_poisson_fft import aggregate_loss_prob_less_than
import numpy as np
from scipy import optimize
from scipy.stats import norm

T = 4.25
LAMBDA = LAMBDA_PHYS


def test_entropy_maximized_at_physical():
    """Grid search confirms H_V is maximised (H_V=0) at the physical measure (0, 0)."""
    best_h = -np.inf
    best_pt = None
    n = 31  # odd so that 0.0 is exactly on the symmetric grid
    for ef in np.linspace(-3, 3, n):
        for k in np.linspace(-3, 3, n):
            h = financial_entropy(ef, k, LAMBDA, T)
            if h > best_h:
                best_h = h
                best_pt = (ef, k)
    assert abs(best_pt[0]) < 0.3 and abs(best_pt[1]) < 0.3, f'Grid maximum at {best_pt}'
    assert abs(best_h) < 1e-6  # H_V(0,0) = 0


def test_scipy_optimization_finds_entropy_max():
    """Multiple L-BFGS-B starts all converge to (0, 0), the entropy maximum."""
    def neg_entropy(vartheta):
        return -financial_entropy(vartheta[0], vartheta[1], LAMBDA, T)

    results = []
    for start in [(1.0, 1.0), (-1.0, 0.5), (0.5, -0.5), (2.0, -2.0)]:
        res = optimize.minimize(
            neg_entropy,
            start,
            method='L-BFGS-B',
            bounds=[(-5, 5), (-5, 5)],
            options={'ftol': 1e-12, 'gtol': 1e-10},
        )
        results.append(res.x)

    for x in results:
        assert abs(x[0]) < 1e-5 and abs(x[1]) < 1e-5, f'Optimizer found {x} not (0,0)'


def test_trigger_intensity_positive():
    ell = trigger_intensity(1.0, 0.5, LAMBDA, A_SHAPE, C_SCALE, D_THRESHOLD)
    assert ell > 0


def test_trigger_intensity_increases_with_eta_f():
    ells = [trigger_intensity(ef, 0.5, LAMBDA, A_SHAPE, C_SCALE, D_THRESHOLD)
            for ef in [0.0, 0.5, 1.0, 1.5]]
    assert ells[0] < ells[1] < ells[2] < ells[3]


def test_treasury_data_has_18_dates():
    data = get_treasury_data()
    assert len(data) == 18


def test_treasury_data_yields_positive():
    data = get_treasury_data()
    for date, yields in data.items():
        for maturity, rate in yields.items():
            assert rate >= 0, f'Negative yield at {date}, maturity {maturity}'


def test_discount_factors_decreasing():
    data = get_treasury_data()
    date = '2014-09-30'
    interp = build_curve(date, data)
    maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    dfs = [get_discount_factor(t, interp) for t in maturities]
    assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))


def test_discount_factors_in_unit_interval():
    data = get_treasury_data()
    interp = build_curve('2016-06-30', data)
    for t in [0.25, 1.0, 2.0, 4.25]:
        df = get_discount_factor(t, interp)
        assert 0 < df < 1, f'Discount factor {df} not in (0,1) at t={t}'


def test_par_yield_formula():
    """With a flat 2 % discount curve the par yield should be ≈ 2 %."""
    r = 0.02
    Delta = 0.25
    payment_times = np.arange(1, 18) * Delta
    disc_factors = np.exp(-r * payment_times)
    y = compute_par_yield(payment_times, disc_factors, Delta)
    assert abs(y - r) < 0.002, f'Par yield {y} far from {r}'


def test_kl_divergence_formula():
    """KL divergence matches the closed-form exponential-family expression."""
    eta_f, kappa = 1.5, 0.8
    lambda_prime = np.exp(eta_f) - 1
    expected = LAMBDA * T * (
        (1 + lambda_prime) * np.log(1 + lambda_prime)
        - lambda_prime
        + (1 + lambda_prime) * kappa ** 2 / 2
    )
    computed = kl_divergence(eta_f, kappa, LAMBDA, T)
    assert abs(computed - expected) / expected < 1e-10


def test_riemannian_gradient_direction():
    """Riemannian gradient at (1, 0.5) points toward the physical measure (0, 0)."""
    eta_f, kappa = 1.0, 0.5
    rg = riemannian_gradient(eta_f, kappa, LAMBDA, T)
    # rg = -[eta_f + kappa^2/2, kappa]; both components are negative here
    assert rg[0] < 0  # pointing toward decreasing eta_f
    assert rg[1] < 0  # pointing toward decreasing kappa


def test_weak_identification_structure():
    """Two parameter pairs sharing the same trigger intensity produce similar Q-survivals."""
    eta_f_0, kappa_0 = 1.267, 1.037
    ell_0 = trigger_intensity(eta_f_0, kappa_0, LAMBDA, A_SHAPE, C_SCALE, D_THRESHOLD)

    # Probit representation: ell = lambda_hat * Phi(kappa - z_D)
    z_D = norm.ppf(gpd_cdf(D_THRESHOLD))

    # Shift eta_f by 0.3 and back-solve for kappa that preserves ell_0
    eta_f_1 = eta_f_0 + 0.3
    hat_lambda_1 = LAMBDA * np.exp(eta_f_1)
    arg = ell_0 / hat_lambda_1
    if 0 < arg < 1:
        kappa_1 = norm.ppf(arg) + z_D
        if abs(kappa_1) <= 5:
            ell_1 = trigger_intensity(eta_f_1, kappa_1, LAMBDA, A_SHAPE, C_SCALE, D_THRESHOLD)
            assert abs(ell_1 - ell_0) / ell_0 < 0.01

            # Q-survival at T_test=1.0 under both parameter pairs
            T_test = 1.0
            sev_cdf_0 = lambda x: wang_distorted_cdf(x, kappa_0, A_SHAPE, C_SCALE)
            sev_cdf_1 = lambda x: wang_distorted_cdf(x, kappa_1, A_SHAPE, C_SCALE)
            surv_0 = aggregate_loss_prob_less_than(
                D_THRESHOLD, LAMBDA * np.exp(eta_f_0), T_test, sev_cdf_0
            )
            surv_1 = aggregate_loss_prob_less_than(
                D_THRESHOLD, LAMBDA * np.exp(eta_f_1), T_test, sev_cdf_1
            )
            assert abs(surv_0 - surv_1) < 0.05, f'Survivals differ: {surv_0} vs {surv_1}'
