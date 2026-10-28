"""
Entropy and information geometry for the CAT bond calibration problem.

Mathematical summary
--------------------
Parameters:
  vartheta = (eta_f, kappa)   natural coordinates
  theta    = (lambda_prime, kappa)   original coordinates
  lambda_prime = exp(eta_f) - 1,   eta_f = ln(1 + lambda_prime)

**KL divergence (catastrophe component)**
The relative entropy between Q and P for the compound Poisson model over [0,T]:

  K_C(vartheta) = E_Q[ log(dQ/dP) ]
               = lambda*T * [ exp(eta_f)*(eta_f + kappa^2/2) - (exp(eta_f) - 1) ]

Equivalently in theta coordinates:
  K_C(theta) = lambda*T * [ (1+lambda')*ln(1+lambda') - lambda' + (1+lambda')*kappa^2/2 ]

**Financial entropy**
  H_V(Q) = -K_C(vartheta)

**Fisher / Riemannian metric** (information matrix of the log-likelihood w.r.t. Q)

In (lambda', kappa):
  g(theta) = lambda*T * diag( 1/(1+lambda'),  1+lambda' )

In (eta_f, kappa):
  g(vartheta) = lambda*T * exp(eta_f) * I_2

**Entropy gradient** in (eta_f, kappa):
  grad H_V = -lambda*T * exp(eta_f) * [eta_f + kappa^2/2,  kappa]^T

**Riemannian gradient** of H_V (metric-corrected):
  g^{-1} grad H_V = -[eta_f + kappa^2/2,  kappa]^T
  (the factor lambda*T*exp(eta_f) cancels because g = lambda*T*exp(eta_f)*I_2)

**Score functions** (log-likelihood scores under Q):
  G_1(theta) = d log L / d lambda'  = -lambda*T + N(T)/(1+lambda')
  G_2(theta) = d log L / d kappa    = sum_{i=1}^{N(T)} [Phi^{-1}(F(X_i)) - kappa]

These satisfy E_Q[G_j] = 0 and can be used to validate Monte Carlo estimators.

**Log-likelihood ratio** of a single sample path:
  log(dQ/dP) = -lambda*T*(exp(eta_f)-1) + N(T)*eta_f
               + kappa * sum_i Phi^{-1}(F(X_i)) - N(T)*kappa^2/2
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm as _norm

from physical_model import A_SHAPE, C_SCALE, LAMBDA_PHYS, gpd_cdf


# ---------------------------------------------------------------------------
# KL divergence and financial entropy
# ---------------------------------------------------------------------------

def kl_divergence(
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> float:
    """
    Catastrophe-component KL divergence K_C(vartheta) = E_Q[log(dQ/dP)].

    K_C(vartheta) = lambda*T * [ exp(eta_f)*(eta_f + kappa^2/2) - (exp(eta_f) - 1) ]

    Equivalently in theta = (lambda_prime, kappa):
    K_C = lambda*T * [ (1+lambda')*ln(1+lambda') - lambda' + (1+lambda')*kappa^2/2 ]

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading, eta_f = ln(1 + lambda_prime).
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity lambda.
    T : float
        Time horizon in years.

    Returns
    -------
    float
        K_C >= 0 (non-negative by Gibbs' inequality).
    """
    e_eta = np.exp(eta_f)
    return float(lambda_phys * T * (e_eta * (eta_f + kappa ** 2 / 2.0) - (e_eta - 1.0)))


def kl_divergence_theta(
    lambda_prime: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> float:
    """
    Catastrophe-component KL divergence in theta = (lambda_prime, kappa) coordinates.

    K_C(theta) = lambda*T * [ (1+lambda')*ln(1+lambda') - lambda' + (1+lambda')*kappa^2/2 ]

    Parameters
    ----------
    lambda_prime : float
        Relative intensity loading (lambda_prime > -1).
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    float
        K_C >= 0.
    """
    lp1 = 1.0 + lambda_prime
    return float(
        lambda_phys * T * (lp1 * np.log(lp1) - lambda_prime + lp1 * kappa ** 2 / 2.0)
    )


def financial_entropy(
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> float:
    """
    Financial entropy H_V(Q) = -K_C(vartheta).

    H_V is maximised (= 0) at the physical measure (eta_f = 0, kappa = 0)
    and strictly negative elsewhere.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    float
        H_V <= 0.
    """
    return -kl_divergence(eta_f, kappa, lambda_phys=lambda_phys, T=T)


# ---------------------------------------------------------------------------
# Fisher / Riemannian metric
# ---------------------------------------------------------------------------

def fisher_metric_vartheta(
    eta_f: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> np.ndarray:
    """
    Fisher information matrix in natural coordinates (eta_f, kappa).

    g(vartheta) = lambda*T * exp(eta_f) * I_2

    The metric is proportional to the 2x2 identity matrix in (eta_f, kappa)
    coordinates, which makes these *exponential* or *natural* coordinates for
    the information manifold.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    np.ndarray, shape (2, 2)
        2x2 Fisher metric matrix.
    """
    scale = lambda_phys * T * np.exp(eta_f)
    return scale * np.eye(2)


def fisher_metric_theta(
    lambda_prime: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> np.ndarray:
    """
    Fisher information matrix in original coordinates (lambda_prime, kappa).

    g(theta) = lambda*T * diag( 1/(1+lambda'),  1+lambda' )

    Parameters
    ----------
    lambda_prime : float
        Relative intensity loading.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    np.ndarray, shape (2, 2)
        2x2 Fisher metric matrix.
    """
    lp1 = 1.0 + lambda_prime
    scale = lambda_phys * T
    return scale * np.diag([1.0 / lp1, lp1])


# ---------------------------------------------------------------------------
# Entropy gradient and Riemannian gradient
# ---------------------------------------------------------------------------

def entropy_gradient(
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> np.ndarray:
    """
    Euclidean gradient of H_V in (eta_f, kappa) coordinates.

    grad H_V = -lambda*T * exp(eta_f) * [eta_f + kappa^2/2,  kappa]^T

    Derivation:
      H_V = -lambda*T * [exp(eta_f)*(eta_f + kappa^2/2) - (exp(eta_f) - 1)]
      dH_V/d(eta_f) = -lambda*T * exp(eta_f) * (eta_f + kappa^2/2)
      dH_V/d(kappa) = -lambda*T * exp(eta_f) * kappa

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    np.ndarray, shape (2,)
        Gradient vector [dH_V/d(eta_f), dH_V/d(kappa)].
    """
    e_eta = np.exp(eta_f)
    scale = lambda_phys * T * e_eta
    return -scale * np.array([eta_f + kappa ** 2 / 2.0, kappa])


def riemannian_gradient(
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 4.25,
) -> np.ndarray:
    """
    Riemannian gradient of H_V in (eta_f, kappa) coordinates.

    grad_Riem H_V = g^{-1} * grad H_V
                  = -[eta_f + kappa^2/2,  kappa]^T

    Because g(vartheta) = lambda*T*exp(eta_f)*I_2, the metric factor cancels
    exactly, yielding a simple closed form independent of lambda and T.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.

    Returns
    -------
    np.ndarray, shape (2,)
        Riemannian gradient vector.
    """
    return -np.array([eta_f + kappa ** 2 / 2.0, kappa])


# ---------------------------------------------------------------------------
# Score functions for Monte Carlo validation
# ---------------------------------------------------------------------------

def score_lambda_prime(
    N_T: int,
    lambda_prime: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> float:
    """
    Score function for lambda_prime: G_1(theta) = d log L / d lambda'.

    G_1(theta) = -lambda*T + N(T) / (1 + lambda')

    Under Q, E_Q[G_1] = 0 (score property), which provides a Monte Carlo
    consistency check.

    Parameters
    ----------
    N_T : int
        Realised number of events on [0, T].
    lambda_prime : float
        Relative intensity loading parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    float
        Score G_1(theta) for this realisation.
    """
    return -lambda_phys * T + N_T / (1.0 + lambda_prime)


def score_kappa(
    event_sizes: np.ndarray,
    kappa: float,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Score function for kappa: G_2(theta) = d log L / d kappa.

    G_2(theta) = sum_{i=1}^{N(T)} [ Phi^{-1}(F(X_i)) - kappa ]

    Under Q, E_Q[G_2] = 0 (score property).

    Parameters
    ----------
    event_sizes : np.ndarray
        Array of realised event loss amounts X_1, ..., X_{N(T)}.
    kappa : float
        Wang distortion parameter.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        Score G_2(theta) for this realisation.
    """
    if len(event_sizes) == 0:
        return 0.0
    F_vals = gpd_cdf(event_sizes, a=a, c=c)
    F_vals = np.clip(F_vals, 1e-15, 1.0 - 1e-15)
    z_vals = _norm.ppf(F_vals)
    return float(np.sum(z_vals - kappa))


# ---------------------------------------------------------------------------
# Log-likelihood ratio and Monte Carlo KL estimator
# ---------------------------------------------------------------------------

def log_likelihood_ratio(
    N_T: int,
    event_sizes: np.ndarray,
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Log-likelihood ratio log(dQ/dP) for a single compound Poisson sample path.

    log(dQ/dP) = -lambda*T*(exp(eta_f) - 1)
                 + N(T) * eta_f
                 + kappa * sum_i Phi^{-1}(F(X_i))
                 - N(T) * kappa^2 / 2

    Parameters
    ----------
    N_T : int
        Realised number of events on [0, T].
    event_sizes : np.ndarray
        Realised event losses X_1, ..., X_{N(T)}.
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        log(dQ/dP) for this path.
    """
    intensity_term = -lambda_phys * T * (np.exp(eta_f) - 1.0) + N_T * eta_f

    severity_term = 0.0
    if N_T > 0 and len(event_sizes) > 0:
        F_vals = np.clip(gpd_cdf(event_sizes, a=a, c=c), 1e-15, 1.0 - 1e-15)
        z_vals = _norm.ppf(F_vals)
        severity_term = kappa * np.sum(z_vals) - N_T * kappa ** 2 / 2.0

    return float(intensity_term + severity_term)


def mc_kl_estimate(
    sample_paths: list[tuple[int, np.ndarray]],
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
    a: float = A_SHAPE,
    c: float = C_SCALE,
) -> float:
    """
    Monte Carlo estimator of K_C = E_Q[log(dQ/dP)].

    Sample paths should be drawn under Q (i.e. with intensity hat_lambda and
    Wang-distorted severities).  Each path is a tuple (N_T, event_sizes).

    Parameters
    ----------
    sample_paths : list of (N_T, event_sizes) tuples
        Sample paths drawn under Q.
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.
    a : float
        GPD shape parameter.
    c : float
        GPD scale parameter (USD).

    Returns
    -------
    float
        Monte Carlo estimate of K_C.
    """
    log_ratios = [
        log_likelihood_ratio(
            N_T=n, event_sizes=xs,
            eta_f=eta_f, kappa=kappa,
            lambda_phys=lambda_phys, T=T, a=a, c=c,
        )
        for n, xs in sample_paths
    ]
    return float(np.mean(log_ratios))


# ---------------------------------------------------------------------------
# Objective gradient for the entropy-regularised calibration
# ---------------------------------------------------------------------------

def entropy_gradient_J(
    eta_f: float,
    kappa: float,
    lambda_phys: float = LAMBDA_PHYS,
    T: float = 1.0,
) -> np.ndarray:
    """
    Euclidean gradient of the entropy term in the calibration objective.

    The full calibration objective is:
      J_alpha(vartheta) = H_V(vartheta) - 1/(2*alpha) * sum_k (V_k(vartheta) - 100)^2

    This function returns grad H_V (the entropy component only).  The price
    penalty gradient is computed numerically by the outer optimiser.

    Parameters
    ----------
    eta_f : float
        Log-intensity-loading.
    kappa : float
        Wang distortion parameter.
    lambda_phys : float
        Physical annual event intensity.
    T : float
        Time horizon in years.

    Returns
    -------
    np.ndarray, shape (2,)
        Euclidean gradient [dH_V/d(eta_f), dH_V/d(kappa)].
    """
    return entropy_gradient(eta_f, kappa, lambda_phys=lambda_phys, T=T)
