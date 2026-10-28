# Data Directory

This directory contains input data for the CAT bond entropy calibration
experiments.

---

## Market Spread Data

Quarterly market spreads for **Golden State Re II, Series 2014-1** are embedded
directly in the source code (`src/cat_bond_pricer.py`) and require no external
download. The 18 observation dates run from 2014-09-30 to 2018-12-31 at
quarterly intervals, with spreads expressed in percent per annum.

---

## Treasury Yield Data

Risk-free discount rates are approximated from historical **Federal Reserve
H.15 Selected Interest Rates** data. The approximations are hardcoded in the
pricing engine and are sufficient for reproducing all four experiments.

### Optional override: `treasury_yields.csv`

If the file `data/treasury_yields.csv` is present it **overrides** the
hardcoded approximations. The file must conform to the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | `YYYY-MM-DD` string | Observation date (end of quarter) |
| `yield_3m` | float | 3-month Treasury yield (decimal, e.g. 0.0005) |
| `yield_6m` | float | 6-month Treasury yield |
| `yield_1y` | float | 1-year Treasury yield |
| `yield_2y` | float | 2-year Treasury yield |
| `yield_3y` | float | 3-year Treasury yield |
| `yield_5y` | float | 5-year Treasury yield |

The pricing engine performs linear interpolation to the required discount
horizon. Dates must span at least 2014-06-30 to 2018-12-31.

Source for actual data:
<https://www.federalreserve.gov/releases/h15/>

---

## No External Downloads Required

All base experiments run without any files in this directory. The optional
`treasury_yields.csv` is only needed if you wish to validate results against
precise historical rates rather than the built-in approximations.
