import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from entropy_geometry import (
    kl_divergence,
    financial_entropy,
    fisher_metric_vartheta,
    entropy_gradient,
    riemannian_gradient,
)
from physical_model import LAMBDA_PHYS
import numpy as np

T = 4.25
LAMBDA = LAMBDA_PHYS


def test_kl_zero_at_physical_measure():
    kl = kl_divergence(0.0, 0.0, LAMBDA, T)
    assert abs(kl) < 1e-12


def test_kl_positive_elsewhere():
    for etaf, kappa in [(1.0, 0.0), (0.0, 1.0), (-0.5, 0.5), (1.267, 1.037)]:
        kl = kl_divergence(etaf, kappa, LAMBDA, T)
        assert kl > 0, f'KL should be positive at ({etaf}, {kappa}), got {kl}'


def test_financial_entropy_maximum_at_physical():
    h_phys = financial_entropy(0.0, 0.0, LAMBDA, T)
    h_other = financial_entropy(1.0, 0.5, LAMBDA, T)
    assert h_phys == 0.0        # -KL(0,0) = 0
    assert h_other < h_phys     # entropy is maximised at the physical measure


def test_kl_formula_explicit():
    """KL divergence matches the closed-form exponential-family expression."""
    eta_f, kappa = 1.267, 1.037
    lambda_prime = np.exp(eta_f) - 1
    expected = LAMBDA * T * (
        (1 + lambda_prime) * np.log(1 + lambda_prime)
        - lambda_prime
        + (1 + lambda_prime) * kappa ** 2 / 2
    )
    computed = kl_divergence(eta_f, kappa, LAMBDA, T)
    assert abs(computed - expected) < 1e-10


def test_fisher_metric_positive_definite():
    for eta_f in [-1.0, 0.0, 0.5, 1.267]:
        g = fisher_metric_vartheta(eta_f, LAMBDA, T)
        assert g.shape == (2, 2)
        eigenvalues = np.linalg.eigvalsh(g)
        assert np.all(eigenvalues > 0), f'Metric not positive definite at eta_f={eta_f}'


def test_fisher_metric_diagonal_in_vartheta():
    """g(vartheta) = LAMBDA*T*exp(eta_f)*I_2, so g is diagonal with equal entries."""
    g = fisher_metric_vartheta(1.0, LAMBDA, T)
    assert abs(g[0, 1]) < 1e-12
    assert abs(g[1, 0]) < 1e-12
    assert abs(g[0, 0] - g[1, 1]) < 1e-12  # both diagonal entries equal


def test_entropy_gradient_zero_at_physical():
    grad = entropy_gradient(0.0, 0.0, LAMBDA, T)
    assert np.allclose(grad, [0, 0], atol=1e-12)


def test_riemannian_gradient_zero_at_physical():
    rg = riemannian_gradient(0.0, 0.0, LAMBDA, T)
    assert np.allclose(rg, [0, 0], atol=1e-12)


def test_riemannian_gradient_formula():
    """Riemannian gradient of H_V equals -[eta_f + kappa^2/2, kappa]."""
    eta_f, kappa = 1.0, 0.5
    rg = riemannian_gradient(eta_f, kappa, LAMBDA, T)
    expected = -np.array([eta_f + kappa ** 2 / 2, kappa])
    assert np.allclose(rg, expected, atol=1e-10)


def test_kl_nonnegative_grid():
    eta_f_vals = np.linspace(-3, 3, 20)
    kappa_vals = np.linspace(-3, 3, 20)
    for ef in eta_f_vals:
        for k in kappa_vals:
            kl = kl_divergence(ef, k, LAMBDA, T)
            assert kl >= -1e-12, f'KL negative at ({ef}, {k}): {kl}'


def test_kl_convex_in_eta_f():
    """KL divergence is convex in eta_f (second finite differences are non-negative)."""
    kappa = 0.5
    etaf_vals = np.linspace(-1, 2, 20)
    kls = [kl_divergence(ef, kappa, LAMBDA, T) for ef in etaf_vals]
    second_diffs = np.diff(np.diff(kls))
    assert np.all(second_diffs >= -1e-10)
