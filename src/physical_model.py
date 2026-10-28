"""
Physical catastrophe model for CAT bond entropy calibration.

Parametrisation note
--------------------
The severity distribution is a Generalised Pareto Distribution (GPD) written
in the *Pareto* parametrisation:

    F(x; a, c) = 1 - (1 + a*x/c)^{-1/a},   x >= 0

where
  a  = shape parameter  (a > 0; tail index alpha = 1/a)
  c  = scale parameter  (c > 0)

This differs from the *standard* GPD parametrisation
  F(x; xi, sigma) = 1 - (1 + xi*x/sigma)^{-1/xi}
only in notation: a = xi, c = sigma.

Calibrated physical-measure constants
--------------------------------------
lambda_phys  = 0.02     annual Poisson event intensity
a_shape      = 2.2      GPD shape
c_scale      = 2.6e7    GPD scale  (USD)
D_threshold  = 2.483e8  attachment / trigger threshold (USD)
"""

from __future__ import annotations

import numpy as np
from scipy import stats as _stats


# ---------------------------------------------------------------------------
# Calibrated constants
# ---------------------------------------------------------------------------

LAMBDA_PHYS: float = 0.02       # annual event intensity under P
A_SHAPE: float = 2.2            # GPD shape parameter  (a)
C_SCALE: float = 2.6e7          # GPD scale parameter  (c, USD)
D_THRESHOLD: float = 2.483e8    # trigger threshold D  (USD)


# ---------------------------------------------------------------------------
# GPD severity distribution (Pareto parametrisation)
# ---------------------------------------------------------------------------

def gpd_cdf(x: np.ndarray | float, a: float = A_SHAPE, c: float = C_SCALE) -> np.ndarray | float:
    """
    Cumulative distribution function of the GPD severity.

    F(x; a, c) = 1 - (1 + a*x/c)^{-1/a},   x >= 0.

    Parameters
    ----------
    x : array_like
        Loss amount(s) in USD, x >= 0.
    a : float
        Shape parameter (a > 0).
    c : float
        Scale parameter (c > 0, USD).

    Returns
    -------
    np.ndarray | float
        CDF value(s) in [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return 1.0 - (1.0 + a * x / c) ** (-1.0 / a)


def gpd_survival(x: np.ndarray | float, a: float = A_SHAPE, c: float = C_SCALE) -> np.ndarray | float:
    """
    Survival function (complementary CDF) of the GPD severity.

    barF(x; a, c) = (1 + a*x/c)^{-1/a},   x >= 0.

    Parameters
    ----------
    x : array_like
        Loss amount(s) in USD, x >= 0.
    a : float
        Shape parameter (a > 0).
    c : float
        Scale parameter (c > 0, USD).

    Returns
    -------
    np.ndarray | float
        Survival probability value(s) in (0, 1].
    """
    x = np.asarray(x, dtype=float)
    return (1.0 + a * x / c) ** (-1.0 / a)


def gpd_quantile(p: np.ndarray | float, a: float = A_SHAPE, c: float = C_SCALE) -> np.ndarray | float:
    """
    Quantile function (inverse CDF) of the GPD severity.

    Q(p; a, c) = c/a * ((1 - p)^{-a} - 1),   p in [0, 1).

    Derived by inverting F(x; a, c) = p:
        (1 + a*x/c)^{-1/a} = 1 - p
        1 + a*x/c = (1-p)^{-a}
        x = c/a * ((1-p)^{-a} - 1)

    Parameters
    ----------
    p : array_like
        Probability value(s) in [0, 1).
    a : float
        Shape parameter (a > 0).
    c : float
        Scale parameter (c > 0, USD).

    Returns
    -------
    np.ndarray | float
        Quantile(s) in USD.
    """
    p = np.asarray(p, dtype=float)
    return (c / a) * ((1.0 - p) ** (-a) - 1.0)


def trigger_probability(
    D: float = D_THRESHOLD,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Physical per-event trigger probability: P(X > D) = barF(D).

    barF(D; a, c) = (1 + a*D/c)^{-1/a}.

    Parameters
    ----------
    D : float
        Trigger threshold (USD).
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        Per-event exceedance probability in (0, 1).
    """
    return float(gpd_survival(D, a=a, c=c))


# ---------------------------------------------------------------------------
# Aggregate loss (compound Poisson) survival probability
# ---------------------------------------------------------------------------

def physical_survival_prob(
    t: float = 1.0,
    D: float = D_THRESHOLD,
    lambda_phys: float = LAMBDA_PHYS,
    a: float = A_SHAPE,
    c: float = C_SCALE,
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> float:
    """
    Physical-measure survival probability: P(L_t < D).

    L_t is a compound Poisson process with annual intensity *lambda_phys*
    and GPD(a, c) severity distribution.  The computation is delegated to
    :mod:`compound_poisson_fft`.

    P(L_t < D) = sum_{n=0}^{inf} Poisson(n; lambda_phys*t) * F_{S_n}(D)

    where S_n = X_1 + ... + X_n (S_0 = 0 with probability 1).

    Parameters
    ----------
    D : float
        Trigger threshold (USD).
    lambda_phys : float
        Annual Poisson event intensity under P.
    t : float
        Time horizon in years.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).
    x_max : float
        Upper bound of the FFT grid (USD).  Should be large enough to
        capture essentially all probability mass.
    n_grid : int
        Number of FFT grid points (should be a power of 2).

    Returns
    -------
    float
        Q(L_t < D) = P(L_t < D) under the physical measure.
    """
    from compound_poisson_fft import aggregate_loss_prob_less_than

    severity_cdf = lambda x: gpd_cdf(x, a=a, c=c)  # noqa: E731
    return aggregate_loss_prob_less_than(
        D=D,
        lambda_hat=lambda_phys,
        T=t,
        severity_cdf=severity_cdf,
        x_max=x_max,
        n_grid=n_grid,
    )


def physical_aggregate_cdf(
    x_query: np.ndarray,
    lambda_phys: float = LAMBDA_PHYS,
    t: float = 1.0,
    a: float = A_SHAPE,
    c: float = C_SCALE,
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> np.ndarray:
    """
    CDF of the aggregate loss L_t at an array of query points under P.

    Parameters
    ----------
    x_query : np.ndarray
        Query points (USD), must be within [0, x_max].
    lambda_phys : float
        Annual Poisson event intensity under P.
    t : float
        Time horizon in years.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).
    x_max : float
        Upper bound of the FFT grid (USD).
    n_grid : int
        Number of FFT grid points (power of 2).

    Returns
    -------
    np.ndarray
        CDF values F_{L_t}(x) for each x in x_query.
    """
    from compound_poisson_fft import aggregate_loss_cdf

    severity_cdf = lambda x: gpd_cdf(x, a=a, c=c)  # noqa: E731
    dx = x_max / n_grid
    x_grid = np.arange(n_grid + 1) * dx
    severity_cdf_vals = severity_cdf(x_grid)

    return aggregate_loss_cdf(
        x_query=x_query,
        lambda_hat=lambda_phys,
        T=t,
        severity_cdf_vals=severity_cdf_vals,
        severity_x_grid=x_grid,
        x_max=x_max,
        n_grid=n_grid,
    )
