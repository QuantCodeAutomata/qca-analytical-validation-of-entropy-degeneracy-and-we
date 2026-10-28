import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physical_model import (
    gpd_cdf,
    gpd_survival,
    gpd_quantile,
    trigger_probability,
    physical_survival_prob,
    LAMBDA_PHYS,
    A_SHAPE,
    C_SCALE,
    D_THRESHOLD,
)
import numpy as np


def test_gpd_cdf_zero():
    assert gpd_cdf(0.0) == 0.0


def test_gpd_cdf_monotone():
    xs = np.linspace(0, 1e8, 100)
    vals = gpd_cdf(xs)
    assert np.all(np.diff(vals) >= 0)


def test_gpd_cdf_range():
    xs = np.linspace(0, 1e9, 1000)
    vals = gpd_cdf(xs)
    assert np.all(vals >= 0) and np.all(vals <= 1)


def test_gpd_survival_complements_cdf():
    xs = np.linspace(0, 1e8, 50)
    assert np.allclose(gpd_cdf(xs) + gpd_survival(xs), 1.0)


def test_gpd_quantile_inverse():
    ps = np.linspace(0.01, 0.99, 50)
    xs = gpd_quantile(ps)
    assert np.allclose(gpd_cdf(xs), ps, atol=1e-10)


def test_gpd_quantile_zero_at_zero():
    assert gpd_quantile(0.0) == 0.0


def test_trigger_probability_positive():
    barF_D = trigger_probability(D_THRESHOLD)
    assert 0 < barF_D < 1
    # For GPD with a=2.2, c=2.6e7, D=2.483e8: barF(D) = (1+2.2*2.483e8/2.6e7)^(-1/2.2)
    expected = (1 + A_SHAPE * D_THRESHOLD / C_SCALE) ** (-1 / A_SHAPE)
    assert abs(barF_D - expected) < 1e-12


def test_trigger_probability_at_zero():
    assert trigger_probability(0.0) == 1.0  # all events exceed 0


def test_physical_survival_at_zero_time():
    """At very short t, N(t)=0 almost surely so aggregate loss < D with probability ~1."""
    surv = physical_survival_prob(0.001, D_THRESHOLD)
    assert 0.99 < surv <= 1.0


def test_physical_survival_monotone_in_time():
    times = [0.5, 1.0, 2.0, 4.25]
    survivals = [physical_survival_prob(t, D_THRESHOLD) for t in times]
    assert survivals[0] > survivals[1] > survivals[2] > survivals[3]


def test_gpd_heavy_tail():
    """GPD with a=2.2 is heavy-tailed: barF(x) ~ (ax/c)^{-1/a} for large x."""
    x_large = 1e10
    barF = gpd_survival(x_large)
    approx = (A_SHAPE * x_large / C_SCALE) ** (-1 / A_SHAPE)
    assert abs(barF - approx) / approx < 0.01
