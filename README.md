# CAT Bond Entropy Calibration: Analytical Validation and Market-Consistent Pricing

A Python implementation of entropy-regularized CAT bond calibration using the
maximum entropy / minimum relative entropy framework for pricing catastrophe
bonds. The codebase reproduces four experiments that validate the calibration
methodology and apply it to real market data.

---

## Overview

Catastrophe (CAT) bond pricing requires constructing a risk-neutral (Q) measure
that is consistent with observable market spreads while remaining as close as
possible to the physical (P) measure. This repository formalises that problem
as a **minimum relative entropy** (KL-divergence) programme on the space of
compound-Poisson severity distortions, with:

- **GPD severity** (shape *a* = 2.2, scale *c* = 2.6 × 10⁷) for catastrophe
  losses.
- **Wang distortion** (parameter *η_f*) as the risk-adjustment map from P to Q.
- **FFT-based compound-Poisson pricing** for fast, accurate tranche evaluation.
- A **KL-divergence** penalty that anchors the calibrated measure to the
  physical baseline.

The four experiments progress from analytical sanity-checking through full
market calibration to robustness analysis.

---

## Experiments

| # | Script | Description |
|---|--------|-------------|
| 1 | `exp/exp_1_analytical_validation.py` | Analytical validation of entropy degeneracy and weak identification — confirms that the entropy surface is maximised at the physical measure (λ′ = 0, κ = 0) and characterises the flat ridge that causes weak identification. |
| 2 | `exp/exp_2_pooled_calibration.py` | Pooled market-consistent entropy calibration using all 18 quarterly spreads from Golden State Re II 2014-1 — fits a single (*η_f*, κ) pair to the full panel. |
| 3 | `exp/exp_3_datewise_calibration.py` | Date-wise exact-fit entropy calibration and minimal-entropy path — calibrates (*η_f*, κ) independently at each observation date and traces the minimal-entropy trajectory through parameter space. |
| 4 | `exp/exp_4_robustness.py` | Robustness and convention-sensitivity analysis — stress-tests the calibrated parameters against variations in the λ prior, GPD tail index, and spread-quoting conventions. |

---

## Installation

**Requirements:** Python 3.9 or later.

```bash
# 1. Clone the repository
git clone <repository-url>
cd cat-bond-entropy-calibration

# 2. (Optional but recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

Each experiment is a standalone script that writes figures to `results/` and
prints a summary table to stdout.

```bash
# Experiment 1 — analytical validation
python exp/exp_1_analytical_validation.py

# Experiment 2 — pooled market calibration (Golden State Re II 2014-1)
python exp/exp_2_pooled_calibration.py

# Experiment 3 — date-wise calibration and minimal-entropy path
python exp/exp_3_datewise_calibration.py

# Experiment 4 — robustness and convention-sensitivity
python exp/exp_4_robustness.py
```

Run the full test suite from the project root:

```bash
pytest tests/
```

---

## Project Structure

```
cat-bond-entropy-calibration/
├── README.md                   # This file
├── AGENTS.md                   # Repository knowledge for AI assistants
├── requirements.txt            # Pinned Python dependencies
├── .gitignore
│
├── src/                        # Core library modules
│   ├── __init__.py
│   ├── compound_poisson_fft.py # FFT pricing engine for compound-Poisson losses
│   ├── cat_bond_pricer.py      # CAT bond valuation (survival probability, spread)
│   └── entropy_calibration.py  # KL-divergence objective and optimisation helpers
│
├── exp/                        # Experiment entry-point scripts
│   ├── exp_1_analytical_validation.py
│   ├── exp_2_pooled_calibration.py
│   ├── exp_3_datewise_calibration.py
│   └── exp_4_robustness.py
│
├── tests/                      # pytest unit and integration tests
│   ├── test_compound_poisson_fft.py
│   ├── test_cat_bond_pricer.py
│   └── test_entropy_calibration.py
│
├── data/                       # Input data (see data/README.md)
│   └── README.md
│
└── results/                    # Generated figures and summary tables
    └── RESULTS.md
```

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python  | 3.9            |
| numpy   | 1.26           |
| pandas  | 2.0            |
| scipy   | 1.10           |
| matplotlib | 3.7         |
| seaborn | 0.12           |
| pytest  | 7.0            |

---

## Key References

- **Wang (2000):** A class of distortion operators for pricing financial and
  insurance risks. *Journal of Risk and Insurance*, 67(1), 15–36.
- **Embrechts & Meister (1997):** Pricing insurance derivatives, the case of
  CAT-futures. *Proceedings of the 1995 Bowles Symposium*.
- **Aase (1999):** An equilibrium model of catastrophe insurance futures and
  spreads. *The Geneva Papers on Risk and Insurance Theory*, 24, 69–96.
- **Cover & Thomas (2006):** *Elements of Information Theory* (2nd ed.).
  Wiley-Interscience.

---

## License

MIT License. See `LICENSE` for details.
