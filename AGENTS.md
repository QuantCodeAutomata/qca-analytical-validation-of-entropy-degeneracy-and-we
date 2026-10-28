# AGENTS.md — Repository Knowledge for AI Assistants

This file documents repository-specific context that should be loaded at the
start of every AI-assisted session on this project.

---

## Repository Purpose

CAT bond entropy calibration — a Python implementation of the minimum relative
entropy (KL-divergence) framework for pricing catastrophe bonds. Given a set of
observable market spreads, the calibration finds the risk-neutral measure Q that
is closest (in KL sense) to the physical measure P while exactly or approximately
fitting the spread panel.

---

## Mathematical Model

### Physical (P) measure parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `lambda` | 0.02 | Poisson arrival rate of catastrophe events (per year) |
| `D` | 2.483 × 10⁸ | Attachment / trigger loss threshold (USD) |
| `N` | 100 | Number of FFT grid points (loss discretisation) |
| `zeta` | 0.5 | Loss-given-event fraction applied to the tranche |
| `T` | 4.25 | Bond tenor in years |

### Severity distribution

GPD (Generalised Pareto Distribution) for individual catastrophe loss *X*:

```
P(X > x) = (1 + x / c)^{-a}    for x > 0
```

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Shape (tail index) | `a` | 2.2 |
| Scale | `c` | 2.6 × 10⁷ |

### Risk-neutral (Q) measure construction

The Q measure is obtained via a **Wang distortion** applied to the GPD
survival function:

```
S_Q(x) = Φ(Φ^{-1}(S_P(x)) + η_f)
```

where `η_f` is the distortion (market price of frequency/severity risk) and
`Φ` is the standard normal CDF.

The Poisson intensity under Q is scaled by a factor `exp(κ)`:

```
λ_Q = λ · exp(κ)
```

The calibration solves for `(η_f, κ)`.

### KL divergence (catastrophe component)

The objective penalises deviation from the physical measure:

```
KL(Q ‖ P) = λ_Q - λ_P - λ_P · log(λ_Q / λ_P)
           + λ_P · ∫ [q(x) log(q(x)/p(x))] dx
```

where `p(x)` and `q(x)` are the GPD density under P and Q respectively.
Entropy is **maximised** (KL minimised) at `λ′ = 0, κ = 0`, i.e. at the
physical measure.

### FFT compound-Poisson pricing

The aggregate loss distribution is computed by FFT convolution of the
discretised severity distribution with the compound-Poisson pgf. The
survival probability at threshold `D` is used to compute the expected
principal recovery, which feeds directly into the spread formula.

---

## Module Dependency Graph

```
compound_poisson_fft.py   ← core FFT engine (no internal deps)
        ↓
  physical_model.py        ← GPD severity, physical survival P(L_t < D)
        ↓
  measure_change.py        ← Wang distortion, Q-measure intensity & severity
        ↓
  entropy_geometry.py      ← KL divergence, Fisher metric, score functions
        ↓
  cat_bond_pricer.py       ← full pricing pipeline: vartheta → bond price
        ↓
  treasury_curve.py        ← Treasury bootstrapping (no CAT-bond deps)
        ↓
  exp/exp_*.py             ← experiment entry points
```

All modules live in `src/` and are importable when `src/` is on `sys.path`.
There is **no** `__init__.py` in `src/`.

Tests mirror this layering:

```
tests/test_compound_poisson_fft.py
tests/test_cat_bond_pricer.py
tests/test_entropy_calibration.py
```

---

## Key Invariants

- Entropy is maximised (KL = 0) at `(η_f, κ) = (0, 0)` — the physical measure.
- The entropy surface is strictly concave in `(η_f, κ)` over the box constraint.
- A flat ridge exists along the spread-consistent manifold when only one spread
  is observed (weak identification).
- Adding more spread observations shrinks the ridge; the 18-date panel renders
  the joint optimum unique to numerical precision.

---

## Parameter Box Constraints

| Parameter | Lower bound | Upper bound |
|-----------|-------------|-------------|
| `η_f`     | −5          | +5          |
| `κ`       | −5          | +5          |

These bounds prevent overflow in `exp(κ)` and in the normal quantile transform
of the GPD tail. Calibrated values for Golden State Re II typically fall well
inside these limits.

---

## Bond Schedule — Golden State Re II, Series 2014-1

18 quarterly observation dates and market spreads (% p.a., mid-market):

| # | Date | Spread (%) |
|---|------|-----------|
| 1 | 2014-09-30 | 2.16 |
| 2 | 2014-12-31 | 2.23 |
| 3 | 2015-03-31 | 2.26 |
| 4 | 2015-06-30 | 2.32 |
| 5 | 2015-09-30 | 2.18 |
| 6 | 2015-12-31 | 4.56 |
| 7 | 2016-03-31 | 2.60 |
| 8 | 2016-06-30 | 2.50 |
| 9 | 2016-09-30 | 2.36 |
| 10 | 2016-12-31 | 2.27 |
| 11 | 2017-03-31 | 2.33 |
| 12 | 2017-06-30 | 2.38 |
| 13 | 2017-09-30 | 2.41 |
| 14 | 2017-12-31 | 1.83 |
| 15 | 2018-03-31 | 1.65 |
| 16 | 2018-06-30 | 1.96 |
| 17 | 2018-09-30 | 1.69 |
| 18 | 2018-12-31 | 3.11 |

The spike at observation 6 (2015-12-31, 4.56 %) is associated with elevated
wildfire / drought risk perception and is a known structural outlier; exp_4
tests robustness to its inclusion/exclusion.

---

## Testing

Run the full suite from the project root:

```bash
pytest tests/
```

Individual test modules:

```bash
pytest tests/test_compound_poisson_fft.py  # FFT engine unit tests
pytest tests/test_cat_bond_pricer.py       # pricing round-trip tests
pytest tests/test_entropy_calibration.py   # KL objective and optimiser tests
```

Tests do **not** use mocks; they exercise real code paths against known
analytical values (e.g., KL = 0 at the physical measure, spread monotonicity
in η_f).

---

## Implementation Notes for Core Modules

### physical_model.py
- GPD parametrisation: `F(x; a, c) = 1 - (1 + a*x/c)^{-1/a}`, NOT the scipy
  convention. Here `a = xi` (shape), `c = sigma` (scale).
- `trigger_probability(D)` returns `barF(D) ≈ 0.2455` for the calibrated params.
- `physical_survival_prob()` delegates to `compound_poisson_fft`; import chain is
  `physical_model → compound_poisson_fft` (not circular, one direction only).

### measure_change.py
- Wang distortion: `hat_F(x) = Phi(Phi^{-1}(F(x)) - kappa)`.
- Distorted quantile: `hat_Q(u) = Q(Phi(Phi^{-1}(u) + kappa))` — note the **+kappa**
  sign (inverse of hat_F which uses −kappa).
- `q_severity_cdf(x, kappa)` is the function to pass into `aggregate_loss_prob_less_than`
  for Q-measure pricing.
- Parametrisation: `vartheta = (eta_f, kappa)` where `eta_f = ln(1 + lambda_prime)`.

### entropy_geometry.py
- KL formula in eta_f coordinates: `K_C = lambda*T * [exp(eta_f)*(eta_f + kappa^2/2) - (exp(eta_f)-1)]`
- Fisher metric in (eta_f, kappa) is proportional to identity: `g = lambda*T*exp(eta_f)*I_2`.
- Riemannian gradient of H_V = `−[eta_f + kappa^2/2, kappa]^T` (metric factor cancels).
- Score functions: `G_1 = −lambda*T + N(T)/(1+lambda')`, `G_2 = sum_i [Phi^{-1}(F(X_i)) − kappa]`.

### compound_poisson_fft.py
- FFT algorithm: `phi_L = exp(mu * (FFT(severity_pmf) - 1))`, then IFFT.
- Severity PMF is **NOT normalised** to 1; `sum(p) = F(x_max) < 1` for heavy GPD.
- The truncation is **exact** for computing `P(L_T < D)` provided `x_max >= D`:
  any path with `X_i > x_max > D` has `L_T > D` and correctly contributes 0.
- Default `x_max=5e9, n_grid=2^16`: gives `dx ≈ 76 294 USD`, D at index ≈ 3254.
- GPD with `a=2.2` has infinite mean (`1/a ≈ 0.45 < 1`); `F(5e9) ≈ 0.936`.
  This truncation is accounted for by the exactness argument above.
- The FFT is very fast: `aggregate_loss_prob_less_than` with `n_grid=2^16` runs in < 0.1 s.

### cat_bond_pricer.py
- `price_cat_bond_full(vartheta, ...)` is the main entry point; call it with:
  - `vartheta = (eta_f, kappa)`, `payment_schedule` in years from inception
  - `market_df` = discount factors for payments after valuation date
  - `phys_params` dict with `lambda_phys, a_shape, c_scale, D_threshold`
  - `contract_params` dict with `N_nominal, zeta, Delta, x_max, n_grid`
- At `t=0` (valuation date), `Q(L_0 < D) = 1.0` exactly (hardcoded to avoid FFT noise).
- Pricing formula: `V = N * [coupon_leg + recovery_leg + principal]`
  where each leg uses conditional survivals `S_k^Q(t_j) = Q(L_{t_j}<D)/Q(L_{t_k}<D)`.

### treasury_curve.py
- Bills: `P(0,T) = (1 + y/2)^{-2T}` (semiannual BEY).
- Notes: nonlinear bootstrap using `scipy.optimize.brentq`; circular dependency
  for intermediate maturities resolved by including the unknown DF in the interpolation.
- `bootstrap_discount_factors(yields_dict)` returns a `scipy.interpolate.interp1d`
  on **log-discount-factors** (linear interpolation = piecewise constant forward rates).
- `get_treasury_data()` returns 18 quarterly dates 2014-Q3 to 2018-Q4 (approximate
  Federal Reserve H.15 data; values in decimal, not percent).

---

## Common Pitfalls

1. **FFT grid size** — `n_grid` must be a power of 2 for efficient FFT. Default
   `2**16 = 65536`. Smaller grids (e.g., `2**14`) are faster but less accurate.
2. **Overflow in Wang distortion** — `Phi^{-1}(F(x))` diverges as `F(x) → 0` or
   `F(x) → 1`. Always clip the argument to `[1e-15, 1 − 1e-15]`.
3. **Spread units** — all internal computations use decimal fractions; the bond
   schedule table and experiment outputs display percent.
4. **KL sign convention** — the calibration *minimises* KL(Q ‖ P), which is
   equivalent to *maximising* the negative KL (entropy proxy = H_V). Be careful
   not to flip signs when interpreting optimiser output.
5. **Wang distortion sign** — `hat_F(x) = Phi(Phi^{-1}(F(x)) − kappa)` uses
   `−kappa` so that `kappa > 0` loads the tail (stochastic dominance). The inverse
   `hat_Q(u) = Q(Phi(Phi^{-1}(u) + kappa))` uses `+kappa`.
6. **q_survival length** — `price_cat_bond` expects `q_survival` of length
   `len(payment_times) + 1`, with index 0 = valuation date, indices 1..n = payments.
