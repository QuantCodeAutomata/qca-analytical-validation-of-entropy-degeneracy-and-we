"""
Q-measure family via Wang distortion for CAT bond entropy calibration.

Measure-change parametrisation
-------------------------------
The risk-neutral measure Q is characterised by two parameters:

  theta  = (lambda_prime, kappa)
  vartheta = (eta_f, kappa)

with the bijection  eta_f = ln(1 + lambda_prime),  lambda_prime = exp(eta_f) - 1.

**Intensity change**
  hat_lambda = lambda * (1 + lambda_prime) = lambda * exp(eta_f)

**Wang severity distortion**
The Wang transform shifts the standard-normal quantile of F by kappa:

  hat_F(x; kappa) = Phi( Phi^{-1}(F(x)) - kappa )

where Phi is the standard normal CDF.  For kappa > 0 the distorted
distribution stochastically dominates the original (heavier right tail).

**Distorted survival at threshold D**
  barhat_F(D) = 1 - hat_F(D)
              = 1 - Phi( Phi^{-1}(F(D)) - kappa )
              = Phi( kappa - Phi^{-1}(F(D)) )

**Risk-neutral trigger intensity**
  ell(vartheta) = hat_lambda * barhat_F(D)
                = lambda * exp(eta_f) * Phi( kappa - Phi^{-1}(F(D)) )

**Density ratio (Wang Radon-Nikodym kernel)**
The likelihood ratio of the distorted w.r.t. the original severity density:

  g(x; kappa) = dhat_F/dF(x) = exp( kappa * Phi^{-1}(F(x)) - kappa^2/2 )

**Distorted quantile (inverse CDF of hat_F)**
  hat_Q(u) = Q( Phi( Phi^{-1}(u) + kappa ) )

where Q is the physical GPD quantile function.  This enables sampling from
hat_F via the inverse-CDF method.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

from physical_model import (
    A_SHAPE,
    C_SCALE,
    D_THRESHOLD,
    LAMBDA_PHYS,
    gpd_cdf,
    gpd_quantile,
    gpd_survival,
)


# ---------------------------------------------------------------------------
# Wang distortion of the severity distribution
# ---------------------------------------------------------------------------

def wang_distorted_cdf(
    x: np.ndarray | float,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> np.ndarray | float:
    """
    Wang-distorted CDF of the GPD severity.

    hat_F(x; kappa) = Phi( Phi^{-1}(F(x)) - kappa ),   x >= 0.

    Parameters
    ----------
    x : array_like
        Loss amount(s) in USD, x >= 0.
    kappa : float
        Wang distortion parameter.  kappa > 0 loads the tail.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    np.ndarray | float
        Distorted CDF value(s) in (0, 1).
    """
    x = np.asarray(x, dtype=float)
    F_x = gpd_cdf(x, a=a, c=c)
    # Clip to avoid Phi^{-1}(0) or Phi^{-1}(1) blow-ups
    F_x = np.clip(F_x, 1e-15, 1.0 - 1e-15)
    return _norm.cdf(_norm.ppf(F_x) - kappa)


def wang_distorted_survival(
    D: float,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Wang-distorted survival probability at threshold D.

    barhat_F(D; kappa) = Phi( kappa - Phi^{-1}(F(D)) ).

    Parameters
    ----------
    D : float
        Trigger threshold (USD).
    kappa : float
        Wang distortion parameter.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        Distorted exceedance probability Q(X > D) in (0, 1).
    """
    F_D = float(gpd_cdf(D, a=a, c=c))
    F_D = np.clip(F_D, 1e-15, 1.0 - 1e-15)
    return float(_norm.cdf(kappa - _norm.ppf(F_D)))


def wang_density_ratio(
    x: np.ndarray | float,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> np.ndarray | float:
    """
    Radon-Nikodym density ratio of the distorted w.r.t. the physical severity.

    g(x; kappa) = exp( kappa * Phi^{-1}(F(x)) - kappa^2/2 ).

    This equals dhat_F/dF evaluated pointwise, i.e. the likelihood ratio
    of the distorted density to the original density.

    Parameters
    ----------
    x : array_like
        Loss amount(s) in USD, x >= 0.
    kappa : float
        Wang distortion parameter.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    np.ndarray | float
        Density ratio g(x) >= 0.
    """
    x = np.asarray(x, dtype=float)
    F_x = gpd_cdf(x, a=a, c=c)
    F_x = np.clip(F_x, 1e-15, 1.0 - 1e-15)
    z = _norm.ppf(F_x)
    return np.exp(kappa * z - kappa ** 2 / 2.0)


def wang_distorted_quantile(
    u: np.ndarray | float,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> np.ndarray | float:
    """
    Quantile function (inverse CDF) of the Wang-distorted severity distribution.

    hat_Q(u) = Q( Phi( Phi^{-1}(u) + kappa ) )

    where Q is the physical GPD quantile function.  This enables sampling
    from hat_F via the standard inverse-CDF method:
      X ~ hat_F  iff  X = hat_Q(U),  U ~ Uniform(0, 1).

    Derivation:
      hat_F(x) = u
      Phi^{-1}(F(x)) - kappa = Phi^{-1}(u)
      F(x) = Phi( Phi^{-1}(u) + kappa )
      x = Q( Phi( Phi^{-1}(u) + kappa ) )

    Parameters
    ----------
    u : array_like
        Probability value(s) in (0, 1).
    kappa : float
        Wang distortion parameter.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    np.ndarray | float
        Quantile(s) in USD.
    """
    u = np.asarray(u, dtype=float)
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    p = _norm.cdf(_norm.ppf(u) + kappa)
    p = np.clip(p, 0.0, 1.0 - 1e-15)
    return gpd_quantile(p, a=a, c=c)


# ---------------------------------------------------------------------------
# Q-measure intensity and trigger rate
# ---------------------------------------------------------------------------

def q_intensity(
    eta_f: float,
    lambda_phys: float = LAMBDA_PHYS,
) -> float:
    """
    Q-measure Poisson event intensity.

    hat_lambda = lambda * exp(eta_f) = lambda * (1 + lambda_prime).

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading:  eta_f = ln(1 + lambda_prime).
    lambda_phys : float
        Physical annual intensity lambda.

    Returns
    -------
    float
        Risk-neutral Poisson intensity hat_lambda.
    """
    return lambda_phys * np.exp(eta_f)


def trigger_intensity(
    eta_f: float,
    kappa: float,
    D: float = D_THRESHOLD,
    lambda_phys: float = LAMBDA_PHYS,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Risk-neutral trigger intensity ell(vartheta).

    ell(vartheta) = hat_lambda * barhat_F(D)
                  = lambda * exp(eta_f) * Phi( kappa - Phi^{-1}(F(D)) )

    This is the Q-measure expected number of triggered events per unit time.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading eta_f = ln(1 + lambda_prime).
    kappa : float
        Wang distortion parameter.
    D : float
        Trigger threshold (USD).
    lambda_phys : float
        Physical annual intensity lambda.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        Risk-neutral trigger intensity ell >= 0.
    """
    hat_lam = q_intensity(eta_f, lambda_phys=lambda_phys)
    bar_hat_F = wang_distorted_survival(D, kappa=kappa, a=a, c=c)
    return hat_lam * bar_hat_F


def q_severity_cdf(
    x: np.ndarray | float,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> np.ndarray | float:
    """
    CDF of the severity distribution under Q (Wang-distorted GPD).

    This is a convenience alias for :func:`wang_distorted_cdf` for use
    in the compound Poisson FFT pricing routine.

    Parameters
    ----------
    x : array_like
        Loss amount(s) in USD.
    kappa : float
        Wang distortion parameter.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    np.ndarray | float
        Q-measure CDF of individual event severity.
    """
    return wang_distorted_cdf(x, kappa=kappa, a=a, c=c)


# ---------------------------------------------------------------------------
# Conversion utilities between theta and vartheta parametrisations
# ---------------------------------------------------------------------------

def lambda_prime_to_eta_f(lambda_prime: float) -> float:
    """
    Convert relative intensity loading to log-loading.

    eta_f = ln(1 + lambda_prime).

    Parameters
    ----------
    lambda_prime : float
        Relative intensity change (lambda_prime > -1).

    Returns
    -------
    float
        Log-intensity-loading eta_f.
    """
    return float(np.log1p(lambda_prime))


def eta_f_to_lambda_prime(eta_f: float) -> float:
    """
    Convert log-intensity-loading to relative intensity loading.

    lambda_prime = exp(eta_f) - 1.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.

    Returns
    -------
    float
        Relative intensity change lambda_prime.
    """
    return float(np.expm1(eta_f))
