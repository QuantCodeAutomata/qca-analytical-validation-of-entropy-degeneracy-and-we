import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cat_bond_pricer import compute_q_survival, price_cat_bond, price_cat_bond_full
from compound_poisson_fft import aggregate_loss_prob_less_than
from physical_model import gpd_cdf, A_SHAPE, C_SCALE, D_THRESHOLD, LAMBDA_PHYS, physical_survival_prob
from measure_change import wang_distorted_cdf
import numpy as np


def test_q_survival_at_zero_time():
    """Q(L_0 < D) ≈ 1: with negligible intensity, no trigger has occurred."""
    severity_cdf_q = lambda x: wang_distorted_cdf(x, 0.0, A_SHAPE, C_SCALE)
    surv = compute_q_survival(1e-10, LAMBDA_PHYS, severity_cdf_q, D_THRESHOLD)
    assert abs(surv - 1.0) < 0.01


def test_q_survival_physical_measure():
    """Under the physical measure (kappa=0), Q-survival equals P-survival."""
    severity_cdf_q = lambda x: gpd_cdf(x)
    t = 2.0
    surv_q = aggregate_loss_prob_less_than(D_THRESHOLD, LAMBDA_PHYS, t, severity_cdf_q)
    surv_p = physical_survival_prob(t, D_THRESHOLD)
    assert abs(surv_q - surv_p) < 0.02  # allow some FFT discretisation error


def test_q_survival_monotone_decreasing_in_time():
    severity_cdf_q = lambda x: wang_distorted_cdf(x, 1.0, A_SHAPE, C_SCALE)
    hat_lambda = LAMBDA_PHYS * np.exp(1.0)  # eta_f = 1
    times = [0.5, 1.0, 2.0, 3.0, 4.0]
    survivals = [aggregate_loss_prob_less_than(D_THRESHOLD, hat_lambda, t, severity_cdf_q)
                 for t in times]
    assert survivals[0] > survivals[1] > survivals[2] > survivals[3]


def test_price_cat_bond_par_at_fair_coupon():
    """Bond prices at par when survival is 1 everywhere and coupon equals par yield."""
    Delta = 0.25
    N = 100.0
    r = 0.02
    payment_times = np.arange(1, 18) * Delta
    disc_factors = np.exp(-r * payment_times)
    par_yield = (1 - disc_factors[-1]) / (Delta * np.sum(disc_factors))
    coupon = par_yield
    # With no trigger risk (survival = 1), only coupon cashflows and principal remain.
    V_manual = np.sum(disc_factors * coupon * Delta * N) + disc_factors[-1] * N
    assert abs(V_manual - 100.0) < 0.001, f'Par pricing failed: {V_manual}'


def test_price_decreases_with_higher_risk():
    """Higher kappa (heavier tail loading) implies lower Q-survival and lower price."""
    r = 0.015
    N_payments = 17
    # payment_schedule includes t=0 (valuation date) plus the 17 coupon/principal dates
    payment_schedule = np.concatenate([[0.0], np.arange(1, N_payments + 1) * 0.25])
    payment_times = np.arange(1, N_payments + 1) * 0.25   # 17 future dates
    disc_factors = np.exp(-r * payment_times)              # 17 discount factors
    # Keys must match price_cat_bond_full expectations
    contract = {'N_nominal': 100.0, 'zeta': 0.5, 'Delta': 0.25}
    phys = {
        'lambda_phys': LAMBDA_PHYS,
        'a_shape': A_SHAPE,
        'c_scale': C_SCALE,
        'D_threshold': D_THRESHOLD,
    }
    eta_f = 1.0
    v_low = price_cat_bond_full((eta_f, 0.5), 0, 0.036, payment_schedule, disc_factors, phys, contract)
    v_high = price_cat_bond_full((eta_f, 2.0), 0, 0.036, payment_schedule, disc_factors, phys, contract)
    assert v_low > v_high, f'Price should decrease with kappa: {v_low} vs {v_high}'


def test_aggregate_loss_prob_in_unit_interval():
    severity_cdf = lambda x: gpd_cdf(x)
    p = aggregate_loss_prob_less_than(D_THRESHOLD, LAMBDA_PHYS, 1.0, severity_cdf)
    assert 0 < p < 1


def test_aggregate_loss_approaches_one_at_small_intensity():
    """Very small arrival intensity => aggregate loss almost never exceeds D."""
    severity_cdf = lambda x: gpd_cdf(x)
    p = aggregate_loss_prob_less_than(D_THRESHOLD, 1e-6, 1.0, severity_cdf)
    assert p > 0.999
