"""
CAT bond pricing under the risk-neutral Q-measure.

Pricing formula
---------------
Let t_k be the current valuation date and t_1 < t_2 < ... < t_n = T be the
remaining (quarterly) coupon and principal payment dates.

The model price is

  V_k = N * sum_{j=1}^{n} P(t_k, t_j) * [C*Delta*S_k^Q(t_j)
                                           + zeta*(S_k^Q(t_{j-1}) - S_k^Q(t_j))]
        + N * P(t_k, T) * S_k^Q(T)

where
  P(t_k, t_j)     risk-free discount factor (from treasury bootstrapping)
  C                annual coupon rate (e.g. 0.05 for 5%)
  Delta            coupon period length in years (e.g. 0.25 for quarterly)
  N                notional (e.g. 100)
  zeta             recovery fraction on trigger (e.g. 0 for zero-recovery)
  S_k^Q(t_j)      conditional survival probability = Q(L_{t_j} < D) / Q(L_{t_k} < D)
  t_0 = t_k        valuation date (S_k^Q(t_k) = 1 by definition)

The conditional survival S_k^Q(t_j) = Q(L_{t_j} < D) / Q(L_{t_k} < D) is valid because
the aggregate compound Poisson loss L_t is non-decreasing, so

  Q(tau > t_j | tau > t_k) = Q(L_{t_j} < D) / Q(L_{t_k} < D),  tau = trigger time.

Notation
--------
vartheta = (eta_f, kappa)  risk-neutral parameters
  eta_f  = ln(1 + lambda_prime)   log-intensity-loading
  kappa                            Wang severity distortion
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from compound_poisson_fft import aggregate_loss_prob_less_than
from measure_change import q_severity_cdf, q_intensity
from physical_model import A_SHAPE, C_SCALE, D_THRESHOLD, LAMBDA_PHYS


# ---------------------------------------------------------------------------
# Q-measure survival probability
# ---------------------------------------------------------------------------

def compute_q_survival(
    t: float,
    lambda_hat: float,
    severity_cdf_q: Callable[[np.ndarray], np.ndarray],
    D: float = D_THRESHOLD,
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> float:
    """
    Compute the Q-measure survival probability Q(L_t < D).

    L_t is a compound Poisson process under Q with intensity lambda_hat and
    severity CDF severity_cdf_q (the Wang-distorted GPD).

    Parameters
    ----------
    t : float
        Time horizon (years from valuation date).
    lambda_hat : float
        Q-measure Poisson event intensity per year.
    severity_cdf_q : callable
        Q-measure severity CDF, e.g. from :func:`measure_change.q_severity_cdf`.
    D : float
        Trigger threshold (USD).
    x_max : float
        Upper bound of FFT grid (USD).
    n_grid : int
        FFT grid size (power of 2).

    Returns
    -------
    float
        Q(L_t < D) in [0, 1].
    """
    return aggregate_loss_prob_less_than(
        D=D,
        lambda_hat=lambda_hat,
        T=t,
        severity_cdf=severity_cdf_q,
        x_max=x_max,
        n_grid=n_grid,
    )


# ---------------------------------------------------------------------------
# Core bond pricer
# ---------------------------------------------------------------------------

def price_cat_bond(
    coupon: float,
    discount_factors: np.ndarray,
    payment_times: np.ndarray,
    q_survival: np.ndarray,
    N_nominal: float = 100.0,
    zeta: float = 0.0,
    Delta: float = 0.25,
) -> float:
    """
    Model price of a CAT bond at the current valuation date t_k.

    V_k = N * sum_{j=1}^{n} P(t_k, t_j) * [C*Delta*S_k^Q(t_j)
                                             + zeta*(S_k^Q(t_{j-1}) - S_k^Q(t_j))]
          + N * P(t_k, T) * S_k^Q(T)

    Parameters
    ----------
    coupon : float
        Annual coupon rate C (e.g. 0.05 for 5 %).
    discount_factors : np.ndarray, shape (n,)
        Risk-free discount factors P(t_k, t_j) for j = 1, ..., n.
        Last element is P(t_k, T) used for principal discounting.
    payment_times : np.ndarray, shape (n,)
        Remaining payment dates t_1, ..., t_n (years after t_k).
        Not used directly in pricing but required to define the schedule.
    q_survival : np.ndarray, shape (n+1,)
        Q-measure survival probabilities Q(L_{t_j} < D):
          q_survival[0]   = Q(L_{t_k} < D)   (valuation date)
          q_survival[j]   = Q(L_{t_j} < D)   for j = 1, ..., n
    N_nominal : float
        Bond notional (default 100).
    zeta : float
        Recovery fraction on trigger event (default 0, zero recovery).
    Delta : float
        Length of each coupon period in years (default 0.25 for quarterly).

    Returns
    -------
    float
        Model price V_k (same units as N_nominal).

    Raises
    ------
    ValueError
        If q_survival[0] is effectively zero (bond has certainly triggered).
    """
    q_surv_tk = q_survival[0]
    if q_surv_tk < 1e-14:
        raise ValueError(
            "Q(L_{t_k} < D) is essentially zero: bond has already triggered "
            "with near-certainty. Pricing is not meaningful."
        )

    n = len(payment_times)
    if len(discount_factors) != n:
        raise ValueError(
            f"discount_factors length {len(discount_factors)} must match "
            f"payment_times length {n}."
        )
    if len(q_survival) != n + 1:
        raise ValueError(
            f"q_survival length {len(q_survival)} must equal len(payment_times)+1 = {n + 1}."
        )

    # Conditional survival S_k^Q(t_j) = Q(L_{t_j} < D) / Q(L_{t_k} < D)
    # q_survival[0] = Q(L_{t_k} < D), q_survival[1..n] = Q(L_{t_j} < D)
    # S_k^Q(t_0) = 1 (included implicitly for the j=1 recovery term)

    coupon_leg = 0.0
    recovery_leg = 0.0

    for j in range(1, n + 1):
        S_j = q_survival[j] / q_surv_tk       # S_k^Q(t_j)
        S_j_prev = q_survival[j - 1] / q_surv_tk  # S_k^Q(t_{j-1})
        df_j = discount_factors[j - 1]

        coupon_leg += df_j * coupon * Delta * S_j
        recovery_leg += df_j * zeta * (S_j_prev - S_j)

    # Principal repayment at T = t_n
    principal = discount_factors[-1] * (q_survival[-1] / q_surv_tk)

    return N_nominal * (coupon_leg + recovery_leg + principal)


# ---------------------------------------------------------------------------
# Full pricer: vartheta -> model price
# ---------------------------------------------------------------------------

def price_cat_bond_full(
    vartheta: tuple[float, float],
    valuation_idx: int,
    coupon: float,
    payment_schedule: np.ndarray,
    market_df: np.ndarray,
    phys_params: dict,
    contract_params: dict,
) -> float:
    """
    Compute the model CAT bond price for given risk-neutral parameters.

    This function wires together the Q-measure construction, FFT compound
    Poisson survival computation, and the bond pricing formula.

    Parameters
    ----------
    vartheta : tuple (eta_f, kappa)
        Risk-neutral parameters:
          eta_f  = ln(1 + lambda_prime)  log-intensity-loading
          kappa                           Wang distortion parameter
    valuation_idx : int
        Index of the current valuation date within payment_schedule.
        The valuation time is payment_schedule[valuation_idx].
    coupon : float
        Annual coupon rate C.
    payment_schedule : np.ndarray, shape (n_total,)
        All scheduled payment times in years (from bond inception).
        payment_schedule[valuation_idx:] are the remaining payment times.
    market_df : np.ndarray, shape (n_total - valuation_idx,)
        Risk-free discount factors P(t_k, t_j) for each remaining payment
        date t_j > t_k, where t_k = payment_schedule[valuation_idx].
    phys_params : dict
        Physical-measure parameters with keys:
          'lambda_phys'  float  annual Poisson intensity under P
          'a_shape'      float  GPD shape parameter a
          'c_scale'      float  GPD scale parameter c (USD)
          'D_threshold'  float  trigger threshold D (USD)
    contract_params : dict
        Bond contract parameters with keys:
          'N_nominal'  float  notional (e.g. 100)
          'zeta'       float  recovery fraction
          'Delta'      float  coupon period length (years)
          'x_max'      float  FFT upper bound (USD)  [optional, default 5e9]
          'n_grid'     int    FFT grid size           [optional, default 2^16]

    Returns
    -------
    float
        Model price V_k (same units as N_nominal).
    """
    eta_f, kappa = vartheta

    # Physical parameters
    lambda_phys = phys_params['lambda_phys']
    a = phys_params['a_shape']
    c = phys_params['c_scale']
    D = phys_params['D_threshold']

    # Contract parameters
    N_nominal = contract_params['N_nominal']
    zeta = contract_params['zeta']
    Delta = contract_params['Delta']
    x_max = contract_params.get('x_max', 5e9)
    n_grid = contract_params.get('n_grid', 2 ** 16)

    # Q-measure intensity
    lambda_hat = q_intensity(eta_f, lambda_phys=lambda_phys)

    # Q-measure severity CDF (Wang-distorted GPD)
    sev_cdf_q: Callable[[np.ndarray], np.ndarray] = (
        lambda x: q_severity_cdf(x, kappa=kappa, a=a, c=c)
    )

    # Remaining payment times relative to valuation date t_k
    t_k = payment_schedule[valuation_idx]
    remaining_times = payment_schedule[valuation_idx:]   # includes t_k at index 0
    payment_times_rel = remaining_times[1:] - t_k        # t_1 - t_k, ..., T - t_k (positive)

    # Q-measure survival probabilities at all relevant times
    # q_surv[0] = Q(L_{t_k} < D), q_surv[j] = Q(L_{t_{k+j}} < D)
    times_for_surv = remaining_times - t_k               # 0, dt_1, dt_2, ..., T - t_k
    q_surv = np.array([
        aggregate_loss_prob_less_than(
            D=D,
            lambda_hat=lambda_hat,
            T=float(dt),
            severity_cdf=sev_cdf_q,
            x_max=x_max,
            n_grid=n_grid,
        )
        for dt in times_for_surv
    ])

    # At dt = 0 (the valuation date itself): Q(L_0 < D) = 1.0 exactly
    # Correct the first element to avoid FFT discretisation noise
    q_surv[0] = 1.0

    return price_cat_bond(
        coupon=coupon,
        discount_factors=market_df,
        payment_times=payment_times_rel,
        q_survival=q_surv,
        N_nominal=N_nominal,
        zeta=zeta,
        Delta=Delta,
    )


# ---------------------------------------------------------------------------
# Convenience: compute all Q-survival values for a given vartheta
# ---------------------------------------------------------------------------

def compute_all_q_survivals(
    vartheta: tuple[float, float],
    time_grid: np.ndarray,
    phys_params: dict,
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> np.ndarray:
    """
    Compute Q(L_t < D) for each t in time_grid.

    Useful for pre-computing survival curves before pricing multiple bonds
    or performing calibration sweeps.

    Parameters
    ----------
    vartheta : tuple (eta_f, kappa)
        Risk-neutral parameters.
    time_grid : np.ndarray
        Array of time horizons in years (>= 0).
    phys_params : dict
        Physical-measure parameters (same keys as in :func:`price_cat_bond_full`).
    x_max : float
        FFT upper bound (USD).
    n_grid : int
        FFT grid size (power of 2).

    Returns
    -------
    np.ndarray, shape (len(time_grid),)
        Q-measure survival probabilities Q(L_t < D).
    """
    eta_f, kappa = vartheta
    lambda_phys = phys_params['lambda_phys']
    a = phys_params['a_shape']
    c = phys_params['c_scale']
    D = phys_params['D_threshold']

    lambda_hat = q_intensity(eta_f, lambda_phys=lambda_phys)
    sev_cdf_q: Callable[[np.ndarray], np.ndarray] = (
        lambda x: q_severity_cdf(x, kappa=kappa, a=a, c=c)
    )

    result = np.empty(len(time_grid))
    for i, t in enumerate(time_grid):
        if t == 0.0:
            result[i] = 1.0
        else:
            result[i] = aggregate_loss_prob_less_than(
                D=D,
                lambda_hat=lambda_hat,
                T=float(t),
                severity_cdf=sev_cdf_q,
                x_max=x_max,
                n_grid=n_grid,
            )
    return result
