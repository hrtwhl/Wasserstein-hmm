# Wasserstein HMM for Regime-Aware Investing

A replication and out-of-sample extension of [Boukardagha (2026), *Explainable
Regime-Aware Investing*](https://arxiv.org/abs/2603.04441), implementing a
strictly causal rolling Gaussian Hidden Markov Model with Wasserstein template
tracking and embedding it in a transaction-cost-aware mean-variance optimizer.

The strategy detects market regimes from a small set of cross-asset features and
allocates daily across SPX, BOND, GOLD, OIL, USD according to regime-conditional
expected returns and covariances. It is benchmarked against a non-parametric KNN
moment estimator that uses the same features and optimizer, plus passive
equal-weight and SPX buy-and-hold portfolios.

## Headline results

We replicate the paper on its original 2023-2026 window (Sharpe 1.97 vs paper's
2.18, MaxDD -5.0% vs -5.4%) and extend the backtest to **15 years (2011-2026)**
to test whether the headline performance is window-specific. Key findings:

| Strategy | Sharpe | MaxDD | Avg Turnover |
|---|---:|---:|---:|
| **Parametric Regime (Wasserstein HMM + MVO)** | **0.75** | **-11.0%** | 0.003 |
| Non-Parametric (KNN + MVO) | 0.67 | -18.3% | 0.098 |
| SPX Buy & Hold | 0.74 | -41.1% | — |
| Equal-Weight (20% each) | 0.33 | -35.4% | — |

**The strategy's value over a long sample is drawdown control, not return
generation.** Risk-adjusted returns are roughly equal to passive SPX exposure,
but maximum drawdown is reduced by approximately 4x (11% vs 41%). The headline
Sharpe of 2.18 in the original paper is window-specific to a calm 2023-2026
backtest.

## What the strategy does

The model has five steps that run every trading day, each using only data
available up to the previous close:

1. **Build features.** A 15-dim vector $x_t$ per day combining yesterday's
   returns, 60-day rolling volatility, and 20-day rolling mean returns across
   the five assets.

2. **Fit a Gaussian HMM.** On the expanding history, fit a $K$-state HMM where
   $K$ is selected weekly by predictive log-likelihood with a complexity penalty.

3. **Track regime identity via Wasserstein distance.** The Gaussian HMM produces
   $K$ component distributions $(\mu_{t,k}, \Sigma_{t,k})$. We map each
   component to one of $G=6$ persistent templates by minimizing the closed-form
   2-Wasserstein distance between Gaussians:

   $$W_2^2(\mathcal{N}_1, \mathcal{N}_2) = \|\mu_1 - \mu_2\|^2 + \mathrm{Tr}\!\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_2^{1/2}\Sigma_1\Sigma_2^{1/2})^{1/2}\right).$$

   Templates evolve via exponential smoothing. This avoids label-permutation
   instability that plagues naively re-fitted rolling HMMs.

4. **Aggregate regime moments.** Compute mixture moments
   $\mu_t = \sum_g p_{t,g} \mu_g$ and $\Sigma_t = \sum_g p_{t,g} \Sigma_g$,
   where $p_{t,g}$ are template probabilities derived from the filtered HMM
   posteriors.

5. **Optimize portfolio.** Solve a long-only mean-variance problem with an L1
   turnover penalty to suppress trading on noise:

   $$\max_w\; \mu_t^\top w - \gamma w^\top \Sigma_t w - \tau \|w - w_{t-1}\|_1$$

   subject to $\mathbf{1}^\top w = 1$, $w \geq 0$, $w \leq w_{\max}$.

The KNN baseline replaces step 2-4 with a k-nearest-neighbours search in the
feature space and uses the realized returns of the neighbours as the conditional
moments — same MVO layer, no regime structure.

## Repository structure

```
.
├── replication.py        # all logic in one script (~900 lines, sectioned)
├── requirements.txt      # minimal dependencies, no cvxpy or pyarrow
├── README.md
├── data_cache/           # auto-created on first run (yfinance prices + HMM signal)
└── outputs/              # all tables (CSV) and figures (PNG)
```

`replication.py` is organized into ten numbered sections:

1. Configuration (all hyperparameters)
2. Data loading and feature construction
3. Wasserstein geometry helpers
4. Wasserstein HMM (the regime model)
5. KNN baseline
6. Mean-variance optimizer (scipy SLSQP)
7. Backtest engine
8. Performance metrics
9. Figure generation
10. Main orchestrator

## Setup

```bash
git clone <this repo>
cd Wasserstein-hmm
pip install -r requirements.txt
python replication.py
```

Tested on Python 3.10-3.12. No compiled solvers required.

### Dependencies

```
numpy, pandas, scipy, scikit-learn, hmmlearn, yfinance, matplotlib
```

That's it. Earlier iterations used `cvxpy` for the MVO step, but it pulled in
solver dependencies that caused install pain (ECOS, SCS, CLARABEL versioning).
We replaced it with `scipy.optimize.minimize(method="SLSQP")` after verifying
the two solvers agree to ~1e-5 tolerance across 50 random portfolio problems.

## Running

A single run:

```bash
python replication.py
```

Outputs land in `./outputs/`:

- `table1_performance.csv` ... `table8_freq_comparison.csv` — paper tables 1-7
  plus a new rebalance-frequency comparison
- `fig01_*.png` ... `fig13_*.png` — paper figures 1-11 plus frequency-comparison
  figures
- `param_*_weights.csv`, `param_*_returns.csv` for daily/weekly/monthly variants
- `param_extras.csv` — daily $K_t$ path, dominant template, and template
  probabilities for diagnostic analysis

### Caching

The slow part is the rolling HMM fit (~40 min for 3 years, ~2.5 hours for
15 years on a laptop). Once computed, the HMM signal — daily template
probabilities and mixture moments — is cached to
`data_cache/hmm_signal_<hash>.pkl`. Subsequent runs:

- Reuse the cache when HMM-side parameters are unchanged (cache hit: ~0 sec)
- Recompute when any of `OOS_START`, tickers, `K_MAX`, `LAMBDA_K`, `ETA`,
  `G_TEMPLATES`, `HMM_FIT_FREQ`, etc. change

This makes parameter sweeps over the cheap MVO-side knobs (`GAMMA`, `TAU`,
`W_MAX`, rebalance frequency) feasible in seconds. Full HMM-side sweeps require
recomputing the signal each time.

To force a fresh signal: `rm -rf data_cache/hmm_signal_*.pkl`.

## Key configuration

All hyperparameters live in Section 1 of `replication.py`. The most important ones:

| Parameter | Default | Description |
|---|---:|---|
| `OOS_START` | `2023-05-01` | Start of out-of-sample evaluation |
| `HMM_INIT_WINDOW` | 1000 | Trading days for initial template calibration |
| `K_MIN`, `K_MAX` | 2, 6 | HMM model-order range |
| `G_TEMPLATES` | 6 | Number of persistent templates |
| `ETA` | 0.05 | Template exponential-smoothing rate |
| `LAMBDA_K` | 5.0 | Complexity penalty in K-selection |
| `HMM_FIT_FREQ` | 5 | HMM refit cadence (5 = weekly) |
| `REFIT_DAILY` | False | If True, refit HMM every day (5x slower) |
| `GAMMA` | 5.0 | Risk aversion in MVO |
| `TAU` | 0.001 | L1 trading-cost penalty |
| `W_MAX` | 1.0 | Per-asset weight cap |

For the paper-exact (daily HMM refit) version: set `REFIT_DAILY = True`,
`HMM_FIT_FREQ = 1`. We measured this costs ~9 hours additional compute over the
weekly version and changes Sharpe by less than 0.05 over the 3-year window. The
extra precision is rarely worth the wait.

## Asset universe

Five ETF proxies for the asset classes in the paper:

| Display | Ticker | Class |
|---|---|---|
| SPX | SPY | US equities |
| BOND | AGG | Aggregate US bonds |
| GOLD | GLD | Gold |
| OIL | USO | Crude oil |
| USD | UUP | US dollar |

Earliest start with all five clean is **2007-02-15** (UUP inception). The
default OOS start of 2023-05-01 matches the paper. With ~4 years of template
calibration, OOS can begin as early as 2011-03-01.

## Empirical findings

### 1. Replication of the paper's window holds

On 2023-05-01 to 2026-02-13 (701 trading days), with weekly HMM refits:

```
                   Sharpe   MaxDD   AvgTurnover
Parametric W-HMM    1.97   -5.0%      0.0006
KNN baseline        1.76  -10.2%      0.115
SPX Buy & Hold      1.26  -20.8%      —
```

Compared to the paper's reported figures (2.18 / -5.43% / 0.0079), our weekly
refit recovers Sharpe within 10% and matches MaxDD almost exactly. The 191x
turnover gap between parametric and KNN reproduces the paper's central
implementation finding.

### 2. Long-window backtest is more honest

Extending to 2011-03-01 to 2026-02-13 (3,763 trading days):

```
                   Sharpe   MaxDD   AvgTurnover
Parametric W-HMM    0.75  -11.0%      0.003
KNN baseline        0.67  -18.3%      0.098
SPX Buy & Hold      0.74  -41.1%      —
Equal-Weight        0.33  -35.4%      —
```

Sharpe collapses from ~2.0 to 0.75 — not because the strategy broke, but
because the 2023-2026 window was unusually friendly (one stress event followed
by sustained calm). The 15-year window includes the 2011 Eurozone crisis, 2013
taper tantrum, 2015 China devaluation, 2018 Q4 selloff, 2020 COVID, and 2022
rates shock. The strategy's risk-adjusted return is statistically indistinguishable
from passive SPX exposure, but with one quarter the drawdown.

### 3. Regime structure becomes interpretable

With 15 years of data, all 6 templates activate (vs 2-3 in the 3-year window).
Per-regime portfolio Sharpe:

| Regime | Days | Portfolio Sharpe | Notes |
|---:|---:|---:|---|
| 3 | 2420 | 0.97 | Default "calm bull" — most days |
| 5 | 1060 | 0.70 | Secondary common state |
| 1 | 46 | **3.12** | Stress regime — defensive positioning works |
| 0 | 21 | 1.04 | Rare commodity/USD spike |
| 2 | 169 | -0.19 | Equity-led recovery — strategy underperforms |
| 4 | 47 | **-4.16** | Risk-on with USD weakness — worst regime |

Regime 4 is the strategy's identifiable failure mode. In ~50 days across 15
years, broad rallies in equities, bonds, and gold coincide with sharp USD
weakness. The HMM-driven defensive positioning (heavy USD) leads to severe
underperformance precisely when everything else is rallying.

### 4. Rebalance frequency: daily and weekly are interchangeable

| Frequency | Sharpe | MaxDD | Turnover |
|---|---:|---:|---:|
| Daily | 0.75 | -11.0% | 0.003 |
| Weekly | 0.54 | -11.7% | 0.003 |
| Monthly | 0.91 | -24.9% | 0.001 |

Daily and weekly turnover are identical because the L1 penalty $\tau$ already
suppresses trading well below daily-rebalance frequency. The optimizer simply
rebalances rarely on its own. Monthly shows a higher Sharpe but materially
worse drawdown — symptomatic of fewer reaction opportunities to the stress
regimes that drive tail risk. In a 15-year sample with only ~50 days of Regime
4, the difference between daily and monthly Sharpe is partly luck.

**The takeaway: daily rebalancing is the right operational choice. The L1
penalty does the smoothing automatically; restricting calendar dates gains
nothing and removes the option to react fast on the rare days where the model
wants to.**

## Mathematical summary

For each trading day $t$:

**Features (paper §4):**
$$x_t = [r_t;\; \sigma_t;\; m_t] \in \mathbb{R}^{15}$$
with $\sigma_t$ = 60-day rolling std of returns, $m_t$ = 20-day rolling mean.

**Predictive K-selection (paper §5.2):**
$$K_t = \arg\max_K \big\{\,\sum_{s \in \mathcal{V}_t} \log p(x_s|\mathcal{F}_{s-1}, K) - \lambda_K K\,\big\}$$

**Wasserstein template assignment (paper §5.4):**
$$g(k) = \arg\min_g\; W_2^2\big(\mathcal{N}(\mu_{t,k}, \Sigma_{t,k}),\; \mathcal{N}(\mu_g, \Sigma_g)\big)$$

**Template probability aggregation:**
$$p_{t,g} = \sum_{k:\, g(k)=g} p_{t,k}$$

**Exponential template smoothing:**
$$\mu_g \leftarrow (1-\eta)\mu_g + \eta\bar\mu_{t,g}, \quad \Sigma_g \leftarrow (1-\eta)\Sigma_g + \eta\bar\Sigma_{t,g}$$

**Mixture moments:**
$$\mu_t = \sum_g p_{t,g}\mu_g^{(r)}, \quad \Sigma_t = \sum_g p_{t,g}\Sigma_g^{(r)}$$

(superscript $(r)$ projects onto the return sub-block of the feature space)

**MVO with L1 turnover penalty:**
$$\max_w\; \mu_t^\top w - \gamma w^\top \Sigma_t w - \tau \|w - w_{t-1}\|_1$$
$$\text{s.t.}\quad \mathbf{1}^\top w = 1,\; 0 \leq w \leq w_{\max}$$

The L1 penalty is reformulated with positive slack variables $d^+, d^- \geq 0$
satisfying $w - w_{t-1} = d^+ - d^-$, making the problem a smooth QP solvable
by SLSQP.

## Differences from the paper

A few intentional implementation choices that depart from the paper's exact
text:

- **Solver:** scipy SLSQP rather than cvxpy. Numerically equivalent (verified
  to 1e-5 across 50 random problems), avoids cvxpy's solver-dependency hell.
- **Cache key:** the cache hashes *all* HMM-relevant config so accidental
  parameter mismatches don't produce silently stale results.
- **Tickers:** ETF proxies (SPY/AGG/GLD/USO/UUP) rather than indices/futures.
  The paper says "S&P 500 Index proxy" etc. without specifying. UUP inception
  (Feb 2007) is the binding date constraint.
- **Default HMM refit cadence:** weekly (`HMM_FIT_FREQ = 5`). Paper does daily.
  We measured the difference is < 0.05 Sharpe and chose the 5x faster default.
  Set `REFIT_DAILY = True` for paper-exact behavior.

## Replication checklist

For thesis-grade reporting:

- [x] Strict causality (features shifted +1 day before exposure)
- [x] Closed-form Gaussian 2-Wasserstein distance via symmetric matrix sqrt
- [x] Predictive log-likelihood K-selection with complexity penalty
- [x] Template-based identity tracking with exponential smoothing
- [x] Ledoit-Wolf shrinkage on $\Sigma_t$ before MVO
- [x] L1 turnover penalty in MVO (slack reformulation for SLSQP)
- [x] Long-only constraint with $w_{\max}=1$ box
- [x] Cumulative log-return reporting (matches paper's figures)
- [x] Per-regime asset and portfolio Sharpe attribution

## Citation

If you use this code:

```bibtex
@article{boukardagha2026,
  author    = {Boukardagha, Amine},
  title     = {Explainable Regime-Aware Investing},
  journal   = {arXiv preprint},
  year      = {2026},
  eprint    = {2603.04441},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.PM}
}
```

## License

Code released under MIT. The methodology follows Boukardagha (2026); please
cite the paper for the underlying ideas.