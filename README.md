# Boukardagha (2026) — Wasserstein HMM replication

Single-script replication. No cvxpy.

## Install

```bash
pip install -r requirements.txt
```

Tested on Python 3.10–3.12.

## Run

```bash
python replication.py
```

Outputs (all 7 tables as CSV, all 11 figures as PNG, plus raw weights/returns) land in `./outputs/`.
Prices are cached in `./data_cache/` after the first run, so re-runs are fast.

## Knobs

All hyperparameters live in **Section 1** of `replication.py` (lines ~30–80). The
ones the paper doesn't pin numerically — and that you'll most likely tune to match
the paper's Sharpe / MaxDD — are:

| Knob          | Default | What it does                                  |
|---------------|---------|-----------------------------------------------|
| `GAMMA`       | 5.0     | Risk aversion in MVO                          |
| `TAU`         | 0.001   | L1 trading-cost penalty                       |
| `ETA`         | 0.05    | Template smoothing rate                       |
| `LAMBDA_K`    | 5.0     | Complexity penalty in K-selection             |
| `G_TEMPLATES` | 6       | Number of persistent templates                |
| `REFIT_DAILY` | True    | False = refit only on K-selection dates (~5× faster) |

## Tickers

Paper says "S&P 500 proxy", "broad bond proxy", etc. Defaults are ETF proxies with
clean adjusted close (`SPY`, `AGG`, `GLD`, `USO`, `UUP`). Edit `TICKERS` in
Section 1 if you want index/futures proxies (`^GSPC`, `TLT`, `GC=F`, `CL=F`, `DX-Y.NYB`).

UUP inception (Feb 2007) is the binding date constraint.

## Runtime

- `REFIT_DAILY=True`  : ~1–3 hours for the full ~700-day OOS window
- `REFIT_DAILY=False` : ~10–20 min, results within a few % of the daily refit

Start with `REFIT_DAILY=False` to verify the qualitative result, then flip to True
for the headline numbers.
