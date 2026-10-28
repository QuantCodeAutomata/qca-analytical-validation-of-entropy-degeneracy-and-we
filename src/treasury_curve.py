"""
U.S. Treasury yield-curve bootstrapping for the CAT bond calibration project.

Discount-factor conventions
----------------------------
Bills (T <= 1 year) are quoted as bond-equivalent yields y with semiannual
compounding:
  P(0, T) = (1 + y/2)^{-2T}

Notes (T = 2, 3, 5, 7 years) are quoted as semiannual par yields y.
The par condition is:
  y/2 * sum_{k=1}^{2T} P(0, k/2) + P(0, T) = 1

which gives P(0, T) = (1 + y/2)^{-1} * [1 - y/2 * sum_{k=1}^{2T-1} P(0, k/2)].

Intermediate semiannual maturities that are not direct bootstrap outputs are
obtained by **log-linear interpolation** (i.e., linear interpolation of
log-discount-factors), which is equivalent to piecewise constant continuously
compounded forward rates.

Historical data
---------------
Approximate U.S. Treasury yields for 18 quarter-end dates 2014-Q3 to 2018-Q4,
sourced from the Federal Reserve H.15 release.  Values are in decimal form
(divide the percent by 100).  These are used for the CAT bond calibration
study covering the 2014-2018 period.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Historical Treasury yield data
# ---------------------------------------------------------------------------

def get_treasury_data() -> dict[str, dict[float, float]]:
    """
    Return approximate U.S. Treasury par/zero yields for 18 quarter-end dates.

    Source: Federal Reserve H.15 Selected Interest Rates (approximate).
    Yields are in *decimal* form (not percent).

    Maturities (inner dict keys, in years):
      1/12  1-month T-bill  (zero yield, semiannual compounding convention)
      3/12  3-month T-bill  (zero yield)
      6/12  6-month T-bill  (zero yield)
      1     1-year T-bill   (zero yield)
      2     2-year T-note   (semiannual par yield)
      3     3-year T-note   (semiannual par yield)
      5     5-year T-note   (semiannual par yield)
      7     7-year T-note   (semiannual par yield)

    Returns
    -------
    dict[str, dict[float, float]]
        Outer key: date string 'YYYY-MM-DD'.
        Inner key: maturity in years.
        Inner value: yield in decimal (e.g. 0.0054 for 0.54 %).
    """
    dates = [
        '2014-09-30', '2014-12-31',
        '2015-03-31', '2015-06-30', '2015-09-30', '2015-12-31',
        '2016-03-31', '2016-06-30', '2016-09-30', '2016-12-31',
        '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31',
        '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31',
    ]

    # Raw yield data in percent — will be divided by 100 below.
    # Source: Federal Reserve H.15, approximate values.
    raw = {
        '1m':  [0.02, 0.03, 0.02, 0.01, 0.00, 0.21, 0.27, 0.22, 0.24, 0.42,
                0.65, 0.89, 0.98, 1.22, 1.57, 1.80, 1.95, 2.35],
        '3m':  [0.02, 0.04, 0.03, 0.02, 0.00, 0.23, 0.27, 0.27, 0.29, 0.51,
                0.75, 1.01, 1.06, 1.39, 1.70, 1.93, 2.16, 2.45],
        '6m':  [0.04, 0.07, 0.06, 0.07, 0.02, 0.42, 0.38, 0.38, 0.46, 0.61,
                0.93, 1.12, 1.18, 1.47, 1.87, 2.12, 2.30, 2.56],
        '1y':  [0.09, 0.12, 0.21, 0.26, 0.29, 0.65, 0.57, 0.51, 0.56, 0.81,
                1.03, 1.24, 1.31, 1.76, 2.09, 2.33, 2.59, 2.72],
        '2y':  [0.58, 0.67, 0.56, 0.64, 0.66, 1.05, 0.73, 0.58, 0.76, 1.19,
                1.27, 1.38, 1.47, 1.89, 2.27, 2.52, 2.81, 2.48],
        '3y':  [1.11, 1.10, 0.93, 1.00, 1.01, 1.31, 0.87, 0.71, 0.87, 1.45,
                1.50, 1.57, 1.62, 2.00, 2.38, 2.62, 2.87, 2.46],
        '5y':  [1.80, 1.65, 1.37, 1.63, 1.37, 1.76, 1.21, 1.00, 1.14, 1.93,
                1.93, 1.89, 1.94, 2.20, 2.56, 2.74, 2.94, 2.51],
        '7y':  [2.21, 2.04, 1.76, 2.03, 1.79, 2.08, 1.53, 1.32, 1.45, 2.22,
                2.19, 2.16, 2.19, 2.37, 2.73, 2.85, 3.00, 2.59],
    }

    # Maturity mapping: label -> years
    mat_map: dict[str, float] = {
        '1m': 1.0 / 12,
        '3m': 3.0 / 12,
        '6m': 6.0 / 12,
        '1y': 1.0,
        '2y': 2.0,
        '3y': 3.0,
        '5y': 5.0,
        '7y': 7.0,
    }

    data: dict[str, dict[float, float]] = {}
    for i, date in enumerate(dates):
        data[date] = {mat_map[key]: vals[i] / 100.0 for key, vals in raw.items()}

    return data


def load_treasury_data(filepath: str | None = None) -> dict[str, dict[float, float]]:
    """
    Load Treasury yield data from a CSV file, falling back to hardcoded values.

    CSV format (if the file exists):
      date,1m,3m,6m,1y,2y,3y,5y,7y
      2014-09-30,0.02,0.02,0.04,0.09,0.58,1.11,1.80,2.21
      ...
    Values in the CSV should be in **percent** (they are divided by 100 internally).

    Parameters
    ----------
    filepath : str or None
        Path to the CSV file.  If None, defaults to 'data/treasury_yields.csv'
        relative to this file's parent directory.

    Returns
    -------
    dict[str, dict[float, float]]
        Same structure as :func:`get_treasury_data`.
    """
    if filepath is None:
        here = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(here, '..', 'data', 'treasury_yields.csv')

    if not os.path.isfile(filepath):
        return get_treasury_data()

    mat_map: dict[str, float] = {
        '1m': 1.0 / 12,
        '3m': 3.0 / 12,
        '6m': 6.0 / 12,
        '1y': 1.0,
        '2y': 2.0,
        '3y': 3.0,
        '5y': 5.0,
        '7y': 7.0,
    }

    data: dict[str, dict[float, float]] = {}
    with open(filepath) as fh:
        headers = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if headers is None:
                headers = parts   # first row: date, 1m, 3m, ...
                continue
            date = parts[0]
            row: dict[float, float] = {}
            for col, val in zip(headers[1:], parts[1:]):
                if col in mat_map:
                    row[mat_map[col]] = float(val) / 100.0
            data[date] = row

    return data


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _bill_log_df(maturity: float, yield_: float) -> float:
    """
    Log-discount-factor for a T-bill (semiannual compounding).

    log P(0, T) = -2*T * log(1 + y/2).
    """
    return -2.0 * maturity * np.log1p(yield_ / 2.0)


def _bootstrap_note_log_df(
    y_par: float,
    n_years: float,
    known_mats: list[float],
    known_log_dfs: list[float],
) -> float:
    """
    Solve for log P(0, n_years) given the par yield of a semiannual coupon note.

    The par condition is:
      y/2 * sum_{k=1}^{2n-1} P(0, k/2) + (1 + y/2) * P(0, n) = 1

    Intermediate semiannual maturities between previously bootstrapped anchors
    and n_years are obtained by log-linear interpolation.  When a coupon date
    falls between the last known anchor and n_years, the interpolation uses
    n_years itself, making this a nonlinear equation.  It is solved via
    ``scipy.optimize.brentq``.

    Parameters
    ----------
    y_par : float
        Semiannual par yield (decimal).
    n_years : float
        Maturity of the note in years (2, 3, 5, or 7).
    known_mats : list[float]
        Sorted list of already-bootstrapped maturities (includes bills and
        previously solved notes).  Must include 0 for robustness.
    known_log_dfs : list[float]
        Corresponding log-discount-factors for known_mats.

    Returns
    -------
    float
        log P(0, n_years).
    """
    n_semi = int(round(2 * n_years))  # number of semiannual periods

    def par_residual(log_df_n: float) -> float:
        # Augment knowns with the candidate (n_years, log_df_n)
        aug_mats = known_mats + [n_years]
        aug_log_dfs = known_log_dfs + [log_df_n]

        # Sum of coupon DFs for k = 1, ..., 2n-1
        coupon_sum = 0.0
        for k in range(1, n_semi):   # t = 0.5, 1.0, ..., n_years - 0.5
            t = k / 2.0
            ldf = float(np.interp(t, aug_mats, aug_log_dfs))
            coupon_sum += np.exp(ldf)

        # LHS of par condition: y/2 * sum + (1 + y/2) * P(0, n)
        lhs = y_par / 2.0 * coupon_sum + (1.0 + y_par / 2.0) * np.exp(log_df_n)
        return lhs - 1.0

    # Bracket: log_df_n in (lower, 0).
    # Lower bound: assume implied zero rate <= 30% pa (extreme upper bound for rates)
    lower = -n_years * 0.30
    upper = 0.0   # zero rate = 0 => P = 1

    return float(brentq(par_residual, lower, upper, xtol=1e-12, rtol=1e-10))


# ---------------------------------------------------------------------------
# Public bootstrap API
# ---------------------------------------------------------------------------

def bootstrap_discount_factors(
    yields_dict: dict[float, float],
) -> interp1d:
    """
    Bootstrap discount factors from Treasury bill and note yields.

    Parameters
    ----------
    yields_dict : dict[float, float]
        Mapping from maturity (years) to yield (decimal).
        Bill maturities ``{1/12, 3/12, 6/12, 1}`` are treated as zero yields
        under semiannual compounding.
        Note maturities ``{2, 3, 5, 7}`` are treated as semiannual par yields.

    Returns
    -------
    scipy.interpolate.interp1d
        Interpolation object (linear in log-DF space) that accepts a scalar or
        array of maturities (years) and returns the log-discount-factor.
        Use :func:`get_discount_factor` to convert to P(0, t).

    Notes
    -----
    The interpolation extrapolates at the boundaries using the boundary
    log-DF slope (fill_value='extrapolate').
    """
    # Anchor at t=0: P(0, 0) = 1, log P = 0.
    known_mats: list[float] = [0.0]
    known_log_dfs: list[float] = [0.0]

    # --- Bills (zero yields under semiannual compounding) ---
    bill_mats = sorted(
        m for m in yields_dict if m in {1.0 / 12, 3.0 / 12, 6.0 / 12, 1.0}
    )
    for mat in bill_mats:
        ldf = _bill_log_df(mat, yields_dict[mat])
        known_mats.append(mat)
        known_log_dfs.append(ldf)

    # --- Notes (semiannual par yields) ---
    note_mats = sorted(m for m in yields_dict if m in {2.0, 3.0, 5.0, 7.0})
    for mat in note_mats:
        ldf = _bootstrap_note_log_df(
            y_par=yields_dict[mat],
            n_years=mat,
            known_mats=known_mats,
            known_log_dfs=known_log_dfs,
        )
        known_mats.append(mat)
        known_log_dfs.append(ldf)

    mats_arr = np.array(known_mats)
    ldfs_arr = np.array(known_log_dfs)

    return interp1d(
        mats_arr, ldfs_arr,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=True,
    )


def get_discount_factor(t: float | np.ndarray, interpolator: interp1d) -> float | np.ndarray:
    """
    Return P(0, t) = exp(log-discount-factor interpolated at t).

    Parameters
    ----------
    t : float or array_like
        Maturity in years.
    interpolator : interp1d
        Output of :func:`bootstrap_discount_factors`.

    Returns
    -------
    float or np.ndarray
        Discount factor P(0, t) in (0, 1].
    """
    log_df = interpolator(t)
    return np.exp(log_df)


def forward_discount_factor(
    t_start: float,
    t_end: float,
    interpolator: interp1d,
) -> float:
    """
    Forward discount factor P(t_start, t_end) = P(0, t_end) / P(0, t_start).

    Parameters
    ----------
    t_start : float
        Start of the forward period (years).
    t_end : float
        End of the forward period (years), t_end > t_start.
    interpolator : interp1d
        Output of :func:`bootstrap_discount_factors`.

    Returns
    -------
    float
        P(t_start, t_end) in (0, 1].
    """
    log_df_start = float(interpolator(t_start))
    log_df_end = float(interpolator(t_end))
    return float(np.exp(log_df_end - log_df_start))


# ---------------------------------------------------------------------------
# Par yield computation (for a given discount curve)
# ---------------------------------------------------------------------------

def compute_par_yield(
    remaining_maturities: np.ndarray,
    discount_factors: np.ndarray,
    Delta: float = 0.25,
) -> float:
    """
    Compute the par yield for a bond with given remaining payment maturities.

    Solves the par condition:
      y * Delta * sum_j P(t_k, t_j) + P(t_k, T) = 1

    for y, where the last element of *remaining_maturities* is the maturity T.

    Parameters
    ----------
    remaining_maturities : np.ndarray, shape (n,)
        Coupon and principal payment maturities in years after valuation date.
        The last element is interpreted as the final maturity T.
    discount_factors : np.ndarray, shape (n,)
        P(t_k, t_j) for each payment maturity, including the terminal one.
    Delta : float
        Coupon period length in years (default 0.25 for quarterly).

    Returns
    -------
    float
        Par yield y such that the bond prices at par (= 1).
    """
    annuity = Delta * np.sum(discount_factors)
    terminal_df = discount_factors[-1]   # P(t_k, T)
    return float((1.0 - terminal_df) / annuity)


# ---------------------------------------------------------------------------
# Convenience: full curve builder for a single valuation date
# ---------------------------------------------------------------------------

def build_curve(date: str, treasury_data: dict[str, dict[float, float]] | None = None) -> interp1d:
    """
    Build a discount-factor interpolation object for a single valuation date.

    Parameters
    ----------
    date : str
        Valuation date string, e.g. '2015-06-30'.
    treasury_data : dict or None
        Full dataset from :func:`get_treasury_data` or :func:`load_treasury_data`.
        If None, calls :func:`get_treasury_data`.

    Returns
    -------
    interp1d
        Log-DF interpolator, usable with :func:`get_discount_factor`.

    Raises
    ------
    KeyError
        If *date* is not present in *treasury_data*.
    """
    if treasury_data is None:
        treasury_data = get_treasury_data()
    yields = treasury_data[date]
    return bootstrap_discount_factors(yields)
