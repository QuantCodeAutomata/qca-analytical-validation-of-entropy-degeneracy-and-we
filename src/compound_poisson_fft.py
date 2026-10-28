"""
FFT-based compound Poisson aggregate loss distribution.

Algorithm overview
------------------
Let L_T = X_1 + ... + X_{N(T)} where N(T) ~ Poisson(mu), mu = lambda_hat * T,
and X_i are i.i.d. non-negative random variables with CDF F_X.

**Discretisation**
Partition [0, x_max] into n_grid bins of width dx = x_max / n_grid.
The grid points are x_k = k * dx, k = 0, 1, ..., n_grid.

The severity PMF on this grid is
  p[k] = P(X in [x_k, x_{k+1})) = F_X(x_{k+1}) - F_X(x_k),  k = 0, ..., n_grid-1.

Note:  sum(p) = F_X(x_max) <= 1 (strict if F_X has a non-negligible tail beyond x_max).

**Truncation correctness for P(L_T < D)**
If any realisation X_i > x_max >= D, then L_T >= X_i > D, so that path does NOT
contribute to P(L_T < D).  Therefore leaving p unnormalised (sum < 1) correctly
accounts for paths that are truncated at x_max:

  P(L_T < D) = P(L_T < D and all X_i < x_max)   [exact, since x_max >= D]

**FFT compound step**
In the discrete frequency domain (DFT convention):
  Phi_X[k] = FFT(p)[k] = sum_{j} p[j] exp(-2*pi*i*j*k/n)
  Phi_L[k] = exp(mu * (Phi_X[k] - 1))          [compound Poisson pgf identity]
  f_L       = IFFT(Phi_L)                        [aggregate PMF on the same grid]

f_L[k] = P(L_T in [x_k, x_{k+1})) (including the point mass P(N=0) = exp(-mu) at k=0).

**Panjer recursion (reference)**
For validation / small grids one can use
  f_L[0] = exp(-mu)
  f_L[k] = (mu/k) * sum_{j=1}^{k} j * p[j] * f_L[k-j],  k >= 1.
This O(n^2) algorithm is not implemented here but the FFT result matches it exactly.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_severity_pmf(
    x_grid: np.ndarray,
    severity_cdf: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Discretise a continuous severity CDF onto a uniform grid.

    p[k] = P(X in [x_grid[k], x_grid[k+1]))
          = F_X(x_grid[k+1]) - F_X(x_grid[k]),   k = 0, ..., len(x_grid)-2.

    The result is NOT normalised to 1 when x_grid[-1] < infinity, which is
    intentional: the missing mass represents severity realisations beyond
    x_grid[-1] that always trigger the bond (see module docstring).

    Parameters
    ----------
    x_grid : np.ndarray, shape (n_grid+1,)
        Uniform grid endpoints 0 = x_0 < x_1 < ... < x_{n_grid} = x_max.
    severity_cdf : callable
        Vectorised CDF F_X(x) returning values in [0, 1].

    Returns
    -------
    np.ndarray, shape (n_grid,)
        Severity PMF (bin probabilities).
    """
    cdf_vals = severity_cdf(x_grid)
    return np.diff(cdf_vals)


def compound_poisson_fft(
    mu: float,
    severity_pmf: np.ndarray,
    n_grid: int,
) -> np.ndarray:
    """
    Compute the aggregate-loss PMF for a compound Poisson(mu) distribution via FFT.

    Algorithm
    ---------
    1. Phi_X = FFT(severity_pmf)
    2. Phi_L = exp(mu * (Phi_X - 1))
    3. f_L   = real(IFFT(Phi_L)),  clipped to >= 0.

    The index-0 bin f_L[0] includes the zero-event atom exp(-mu).

    Parameters
    ----------
    mu : float
        Poisson mean number of events: mu = lambda_hat * T.
    severity_pmf : np.ndarray, shape (n_grid,)
        Discretised severity probabilities p[k] on the uniform loss grid.
        Need not sum to 1 (truncation at x_max is handled correctly).
    n_grid : int
        FFT size (should be a power of 2 for efficiency).

    Returns
    -------
    np.ndarray, shape (n_grid,)
        Aggregate loss PMF: f_L[k] = P(L_T in [x_k, x_{k+1})).
    """
    # rfft exploits real-valued input; irfft inverts it
    phi_X = np.fft.rfft(severity_pmf, n=n_grid)
    phi_L = np.exp(mu * (phi_X - 1.0))
    f_L = np.fft.irfft(phi_L, n=n_grid)
    # Clip numerical noise; small negative values can appear near machine epsilon
    return np.maximum(f_L, 0.0)


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------

def aggregate_loss_cdf(
    x_query: np.ndarray,
    lambda_hat: float,
    T: float,
    severity_cdf_vals: np.ndarray,
    severity_x_grid: np.ndarray,
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> np.ndarray:
    """
    CDF of the compound Poisson aggregate loss at arbitrary query points.

    Parameters
    ----------
    x_query : np.ndarray
        Query points x (USD), each in [0, x_max].
    lambda_hat : float
        Poisson event intensity (per year, under P or Q).
    T : float
        Time horizon (years).
    severity_cdf_vals : np.ndarray, shape (n_grid+1,)
        Pre-evaluated CDF values F_X(severity_x_grid[k]).
    severity_x_grid : np.ndarray, shape (n_grid+1,)
        Uniform grid of severity evaluation points 0, dx, ..., x_max.
    x_max : float
        Upper bound of the FFT grid (USD).
    n_grid : int
        FFT grid size (power of 2).

    Returns
    -------
    np.ndarray
        F_{L_T}(x_query[k]) for each query point, values in [0, 1].
    """
    dx = x_max / n_grid
    severity_pmf = np.diff(severity_cdf_vals)         # length n_grid
    mu = lambda_hat * T
    f_L = compound_poisson_fft(mu, severity_pmf, n_grid)
    cdf_L = np.cumsum(f_L)                             # cdf_L[k] ≈ P(L_T <= (k+1)*dx)

    # Map query points to grid indices: floor(x / dx)
    indices = np.floor(x_query / dx).astype(int)
    indices = np.clip(indices, 0, n_grid - 1)
    return cdf_L[indices]


def aggregate_loss_survival(
    D: float,
    lambda_hat: float,
    T: float,
    severity_cdf: Callable[[np.ndarray], np.ndarray],
    x_max: float = 5e9,
    n_grid: int = 2 ** 16,
) -> float:
    """
    Survival probability P(L_T >= D) for the compound Poisson aggregate loss.

    Returns 1 - P(L_T < D).

    Parameters
    ----------
    D : float
        Trigger threshold (USD).  Must satisfy 0 < D <= x_max.
    lambda_hat : float
        Poisson event intensity (per year).
    T : float
        Time horizon (years).
    severity_cdf : callable
        Vectorised CDF F_X(x), e.g. from physical_model or measure_change.
    x_max : float
        Upper bound of the FFT grid (USD).  Must be >= D.
    n_grid : int
        FFT grid size (power of 2).

    Returns
    -------
    float
        P(L_T >= D) in [0, 1].
    """
    return 1.0 - aggregate_loss_prob_less_than(
        D=D,
        lambda_hat=lambda_hat,
        T=T,
        severity_cdf=severity_cdf,
        x_max=x_max,
        n_grid=n_grid,
    )


def aggregate_loss_prob_less_than_from_pmf(
    D: float,
    lambda_hat: float,
    T: float,
    severity_pmf: np.ndarray,
    dx: float,
) -> float:
    """
    P(L_T < D) from a **pre-computed** severity PMF, avoiding redundant CDF evaluations.

    Use this inside tight optimisation loops where the severity distribution
    (kappa, A, C) is fixed but lambda_hat or T varies.

    Parameters
    ----------
    D : float
        Trigger threshold (USD).
    lambda_hat : float
        Poisson intensity per year under Q.
    T : float
        Time horizon (years).
    severity_pmf : np.ndarray, shape (n_grid,)
        Pre-computed discretised severity PMF.  Use :func:`make_severity_pmf`.
    dx : float
        Grid spacing (USD) matching the PMF grid.

    Returns
    -------
    float
        P(L_T < D).
    """
    n_grid = len(severity_pmf)
    mu = lambda_hat * T
    f_L = compound_poisson_fft(mu, severity_pmf, n_grid)
    cdf_L = np.cumsum(f_L)
    idx = min(int(D / dx), n_grid - 1)
    return float(cdf_L[idx])


def make_severity_pmf(
    severity_cdf: Callable[[np.ndarray], np.ndarray],
    x_max: float = 5e9,
    n_grid: int = 2 ** 12,
) -> tuple[np.ndarray, float]:
    """
    Discretise a severity CDF into a PMF on a uniform grid.

    Returns
    -------
    severity_pmf : np.ndarray, shape (n_grid,)
        Bin probabilities.
    dx : float
        Grid spacing (USD).
    """
    dx = x_max / n_grid
    x_grid = np.arange(n_grid + 1, dtype=float) * dx
    severity_pmf = compute_severity_pmf(x_grid, severity_cdf)
    return severity_pmf, dx


def aggregate_loss_prob_less_than(
    D: float,
    lambda_hat: float,
    T: float,
    severity_cdf: Callable[[np.ndarray], np.ndarray],
    x_max: float = 5e9,
    n_grid: int = 2 ** 12,
) -> float:
    """
    Probability P(L_T < D) for the compound Poisson aggregate loss.

    This is the key quantity used in CAT bond pricing and the physical survival
    probability (see :mod:`physical_model`).

    The computation is exact (up to discretisation error) when x_max >= D:
    any severity realisation X_i > x_max automatically pushes L_T > D, so the
    truncated distribution gives the correct conditional probability.

    Parameters
    ----------
    D : float
        Trigger threshold (USD).  Must satisfy 0 < D <= x_max.
    lambda_hat : float
        Poisson event intensity (per year, under P or Q as appropriate).
    T : float
        Time horizon (years).
    severity_cdf : callable
        Vectorised CDF F_X(x), e.g. :func:`physical_model.gpd_cdf` for P-measure
        or :func:`measure_change.q_severity_cdf` for Q-measure pricing.
    x_max : float
        Upper bound of the FFT grid (USD).  Default 5e9 (five billion USD).
        Must be >= D.
    n_grid : int
        Number of FFT grid points (power of 2).  Default 2^16 = 65 536.

    Returns
    -------
    float
        P(L_T < D) in [0, 1].

    Notes
    -----
    P(L_T < D) includes the zero-event atom: if N(T) = 0 then L_T = 0 < D,
    contributing exp(-lambda_hat * T) to the probability.

    Grid resolution: dx = x_max / n_grid.  At the defaults (n_grid=2^12, x_max=5e9),
    dx ≈ 1.22 M USD.  For D = 2.483e8, the index is ≈ 204 — well-resolved.
    For finer accuracy use n_grid=2^14 at the cost of higher memory and runtime.
    """
    if D > x_max:
        raise ValueError(
            f"Threshold D={D:.3e} exceeds x_max={x_max:.3e}. "
            "Increase x_max so that x_max >= D."
        )

    dx = x_max / n_grid
    # Grid with n_grid+1 endpoints: 0, dx, 2*dx, ..., x_max
    x_grid = np.arange(n_grid + 1, dtype=float) * dx
    severity_pmf = compute_severity_pmf(x_grid, severity_cdf)  # length n_grid

    mu = lambda_hat * T
    f_L = compound_poisson_fft(mu, severity_pmf, n_grid)        # length n_grid
    cdf_L = np.cumsum(f_L)                                       # length n_grid

    # Index of the bin that contains D: floor(D / dx)
    # cdf_L[k] = P(L_T in [0, (k+1)*dx)), so P(L_T < D) ≈ cdf_L[floor(D/dx)]
    idx = min(int(D / dx), n_grid - 1)
    return float(cdf_L[idx])
