"""
Replication of Boukardagha (2026) — Explainable Regime-Aware Investing
======================================================================

Wasserstein HMM with predictive K-selection and template-based identity
tracking, vs. KNN baseline, both feeding a transaction-cost-aware MVO.

Run:
    python replication.py

Outputs:
    outputs/table*.csv  — all 7 tables from the paper
    outputs/fig*.png    — all 11 figures
    outputs/*_returns.csv, *_weights.csv, param_extras.csv

Dependencies (all standard, no cvxpy):
    numpy pandas scipy scikit-learn hmmlearn yfinance matplotlib
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================
# Display name -> Yahoo ticker. Order defines column order everywhere.
TICKERS = {
    "SPX":  "SPY",
    "BOND": "AGG",
    "GOLD": "GLD",
    "OIL":  "USO",
    "USD":  "UUP",
}
ASSETS   = list(TICKERS.keys())
N_ASSETS = len(ASSETS)

# Dates — UUP inception (Feb 2007) is the binding constraint
DATA_START = "2007-02-15"
DATA_END   = "2026-02-15"
OOS_START  = "2023-05-01"      # matches paper's figures

# Features (paper sec. 4)
VOL_WINDOW = 60
MOM_WINDOW = 20

# Wasserstein HMM (paper sec. 5)
K_MIN, K_MAX     = 2, 6
G_TEMPLATES      = 6           # paper: 6 templates (one stays at near-zero mass)
ETA              = 0.05        # template exponential-smoothing rate
LAMBDA_K         = 5.0         # complexity penalty per state
VAL_SLICE_DAYS   = 252         # validation slice for predictive K-score
ORDER_SELECT_FREQ = 5          # K-selection cadence (weekly)
HMM_FIT_FREQ     = 1           # paper says daily; set higher for speed
REFIT_DAILY      = True        # False = refit only at order-selection dates
HMM_INIT_WINDOW  = 1000        # days for initial template calibration
HMM_COV_TYPE     = "full"
HMM_N_ITER       = 50

# KNN baseline (paper sec. 6)
KNN_NEIGHBOURS   = 100
KNN_LOOKBACK     = 252

# MVO
GAMMA  = 5.0
TAU    = 0.001                 # ~10 bps L1 trading-cost penalty
W_MAX  = 1.0                   # paper figures show USD reaching 1.0
LEDOIT_WOLF = True

# Misc
TRADING_DAYS = 252
SEED         = 0
OUTPUT_DIR   = Path("outputs")
DATA_CACHE   = Path("data_cache")


# ============================================================================
# 2. DATA AND FEATURES
# ============================================================================
def load_prices(use_cache: bool = True) -> pd.DataFrame:
    """Daily adjusted-close panel from yfinance, cached on disk via pickle.

    We use pickle (stdlib, lossless for any pandas dtype) rather than
    parquet to avoid pyarrow extension-type incompatibilities.
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cache = DATA_CACHE / f"prices_{DATA_START}_{DATA_END}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    raw = yf.download(list(TICKERS.values()),
                      start=DATA_START, end=DATA_END,
                      auto_adjust=True, progress=False, group_by="ticker")
    cols = {}
    for name, tic in TICKERS.items():
        if (tic, "Close") in raw.columns:
            cols[name] = raw[(tic, "Close")]
        else:
            cols[name] = raw[tic]["Close"]
    px = pd.concat(cols, axis=1)[ASSETS].dropna(how="any").sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_pickle(cache)
    return px


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def build_features(returns: pd.DataFrame) -> pd.DataFrame:
    """x_t = [r_t; sigma_t; m_t]. Shifted forward one row so feature at t
    only uses information available up to t-1 (strict causality)."""
    vol = returns.rolling(VOL_WINDOW).std().add_prefix("sigma_")
    mom = returns.rolling(MOM_WINDOW).mean().add_prefix("m_")
    raw = returns.add_prefix("r_")
    feats = pd.concat([raw, vol, mom], axis=1).shift(1).dropna()
    return feats


# ============================================================================
# 3. WASSERSTEIN GEOMETRY
# ============================================================================
def sym_sqrt(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.clip(w, eps, None))) @ V.T


def gaussian_w2_sq(m1, S1, m2, S2) -> float:
    """Squared 2-Wasserstein distance between two Gaussian distributions."""
    diff = m1 - m2
    S2_h = sym_sqrt(S2)
    inner = S2_h @ S1 @ S2_h
    inner_sqrt_tr = float(np.sum(np.sqrt(np.clip(np.linalg.eigvalsh(inner), 0.0, None))))
    return float(diff @ diff + np.trace(S1) + np.trace(S2) - 2.0 * inner_sqrt_tr)


# ============================================================================
# 4. WASSERSTEIN HMM
# ============================================================================
@dataclass
class Template:
    mean: np.ndarray
    cov:  np.ndarray


def _fit_hmm(X: np.ndarray, K: int, seed: int = SEED):
    """Fit a K-state Gaussian HMM. Returns None on failure."""
    try:
        m = GaussianHMM(n_components=K, covariance_type=HMM_COV_TYPE,
                        n_iter=HMM_N_ITER, tol=1e-3, random_state=seed)
        m.fit(X)
        return m
    except Exception:
        return None


def select_K(X_hist: np.ndarray) -> int:
    """Predictive model-order selection (paper sec. 5.2).

    For each K, the validation-slice predictive log-score equals
    score(X_hist) - score(X_hist[:-V]) since the HMM joint log-likelihood
    factors as a sum of one-step-ahead conditional log-likelihoods.
    """
    if len(X_hist) <= VAL_SLICE_DAYS + 50:
        return K_MIN
    pre = X_hist[:-VAL_SLICE_DAYS]
    best_K, best_score = K_MIN, -np.inf
    for K in range(K_MIN, K_MAX + 1):
        model = _fit_hmm(X_hist, K)
        if model is None:
            continue
        try:
            pred_ll = model.score(X_hist) - model.score(pre)
        except Exception:
            continue
        crit = pred_ll - LAMBDA_K * K
        if crit > best_score:
            best_score, best_K = crit, K
    return best_K


class WassersteinHMM:
    """Rolling Gaussian HMM with template-based identity tracking."""

    def __init__(self):
        self.templates: list[Template] = []
        self.current_K = K_MIN
        self._last_model = None
        self._steps_since_refit = 0
        self._steps_since_order_sel = 0

    def initialize_templates(self, X_init: np.ndarray) -> None:
        model = _fit_hmm(X_init, G_TEMPLATES, seed=SEED)
        if model is None:
            raise RuntimeError("HMM init failed; try a longer init window or fewer templates")
        self.templates = [Template(mean=model.means_[g].copy(),
                                   cov=model.covars_[g].copy())
                          for g in range(G_TEMPLATES)]
        self._last_model = model
        self.current_K = G_TEMPLATES

    def _assign_to_templates(self, comp_means, comp_covs):
        """For each HMM component k, return index of nearest template."""
        K = comp_means.shape[0]
        g_of_k = np.empty(K, dtype=int)
        for k in range(K):
            dists = [gaussian_w2_sq(comp_means[k], comp_covs[k], t.mean, t.cov)
                     for t in self.templates]
            g_of_k[k] = int(np.argmin(dists))
        return g_of_k

    def _update_templates(self, comp_means, comp_covs, comp_probs, g_of_k):
        for g in range(G_TEMPLATES):
            mask = (g_of_k == g)
            if not mask.any():
                continue
            w = comp_probs[mask]
            tot = w.sum()
            if tot < 1e-12:
                continue
            w = w / tot
            mean_bar = (w[:, None] * comp_means[mask]).sum(axis=0)
            cov_bar  = (w[:, None, None] * comp_covs[mask]).sum(axis=0)
            self.templates[g].mean = (1 - ETA) * self.templates[g].mean + ETA * mean_bar
            self.templates[g].cov  = (1 - ETA) * self.templates[g].cov  + ETA * cov_bar

    def step(self, X_hist: np.ndarray, t_index: int) -> dict:
        # Periodic K selection
        if self._steps_since_order_sel == 0 or t_index == 0:
            self.current_K = select_K(X_hist)
        self._steps_since_order_sel = (self._steps_since_order_sel + 1) % ORDER_SELECT_FREQ

        # Refit HMM (every day if REFIT_DAILY, otherwise on order-sel dates)
        do_refit = (REFIT_DAILY or self._steps_since_refit == 0
                    or self._last_model is None
                    or self._last_model.n_components != self.current_K)
        if do_refit:
            new_model = _fit_hmm(X_hist, self.current_K, seed=SEED)
            if new_model is not None:
                self._last_model = new_model
        self._steps_since_refit = (self._steps_since_refit + 1) % HMM_FIT_FREQ
        model = self._last_model

        # Filtered probability at the last observed step
        try:
            posterior = model.predict_proba(X_hist)[-1]
        except Exception:
            posterior = np.full(model.n_components, 1.0 / model.n_components)

        comp_means, comp_covs = model.means_, model.covars_
        g_of_k = self._assign_to_templates(comp_means, comp_covs)

        # Aggregate component probs into template probs
        p_template = np.zeros(G_TEMPLATES)
        for k, g in enumerate(g_of_k):
            p_template[g] += posterior[k]
        p_template /= max(p_template.sum(), 1e-12)

        # Update templates
        self._update_templates(comp_means, comp_covs, posterior, g_of_k)

        # Mixture moments in full feature space
        D = comp_means.shape[1]
        mu_full, cov_full = np.zeros(D), np.zeros((D, D))
        for g, p in enumerate(p_template):
            mu_full  += p * self.templates[g].mean
            cov_full += p * self.templates[g].cov

        return {"p_template": p_template,
                "mu_full":    mu_full,
                "cov_full":   cov_full,
                "K":          int(self.current_K),
                "dominant_g": int(np.argmax(p_template))}


# ============================================================================
# 5. KNN BASELINE
# ============================================================================
def knn_step(X_hist, R_hist, x_t, k=KNN_NEIGHBOURS):
    n = len(X_hist)
    k_eff = int(min(k, max(10, n // 4)))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X_hist)
    idx = nn.kneighbors(x_t.reshape(1, -1), return_distance=False).ravel()
    sample = R_hist[idx]
    mu = sample.mean(axis=0)
    try:
        cov = LedoitWolf().fit(sample).covariance_
    except Exception:
        cov = np.cov(sample, rowvar=False)
        cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(cov.shape[0])
    return mu, cov


# ============================================================================
# 6. MVO (scipy SLSQP — no cvxpy)
# ============================================================================
def _psd_project(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return (V * np.clip(w, eps, None)) @ V.T


def solve_mvo(mu, sigma, w_prev, gamma=GAMMA, tau=TAU, w_max=W_MAX):
    """Long-only MVO with L1 turnover penalty.

        max  mu @ w - gamma * w @ sigma @ w - tau * ||w - w_prev||_1
        s.t. sum(w) = 1, 0 <= w <= w_max

    Reformulated as a smooth QP via slack vars d_pos, d_neg >= 0 with
    w - w_prev = d_pos - d_neg, then ||w - w_prev||_1 = sum(d_pos + d_neg).
    Variable: x = [w (N); d_pos (N); d_neg (N)] of dim 3N.
    """
    N = len(mu)
    sigma = _psd_project(sigma)

    def neg_obj(x):
        w = x[:N]
        return -mu @ w + gamma * w @ sigma @ w + tau * x[N:].sum()

    def neg_grad(x):
        return np.concatenate([-mu + 2 * gamma * sigma @ x[:N],
                               tau * np.ones(2 * N)])

    def eq_con(x):
        w, d_pos, d_neg = x[:N], x[N:2*N], x[2*N:]
        return np.concatenate([[w.sum() - 1.0], w - w_prev - (d_pos - d_neg)])

    bounds = [(0.0, w_max)] * N + [(0.0, None)] * (2 * N)
    x0 = np.concatenate([w_prev, np.zeros(2 * N)])

    res = minimize(neg_obj, x0, jac=neg_grad, bounds=bounds,
                   constraints=[{"type": "eq", "fun": eq_con}],
                   method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-9})

    if not res.success or np.any(np.isnan(res.x[:N])):
        return w_prev.copy()
    w = np.maximum(res.x[:N], 0.0)
    s = w.sum()
    return (w / s) if s > 0 else w_prev.copy()


# ============================================================================
# 7. BACKTEST ENGINE
# ============================================================================
@dataclass
class BacktestResult:
    weights: pd.DataFrame
    returns: pd.Series
    extras:  pd.DataFrame = field(default_factory=pd.DataFrame)


def _shrink(cov, sample=None):
    if not LEDOIT_WOLF:
        return cov
    if sample is not None and len(sample) >= 30:
        try:
            return LedoitWolf().fit(sample).covariance_
        except Exception:
            pass
    return 0.9 * cov + 0.1 * np.diag(np.diag(cov))


def _progress(name, i, n, t0):
    if i % max(1, n // 20) == 0 or i == n - 1:
        elapsed = time.time() - t0
        eta = elapsed * (n - i - 1) / max(i + 1, 1)
        print(f"  [{name}] {i+1}/{n}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)


def run_parametric(features, returns, oos_start=OOS_START, init_window=HMM_INIT_WINDOW):
    print("Parametric (Wasserstein HMM + MVO) backtest...")
    feat_idx = features.index
    oos_idx = feat_idx[feat_idx >= pd.Timestamp(oos_start)]
    if len(oos_idx) == 0:
        raise ValueError(f"No OOS dates >= {oos_start}")

    # Initialise templates on calibration window
    init_end = feat_idx.get_indexer([oos_idx[0]])[0]
    init_X = features.values[max(0, init_end - init_window):init_end]
    model = WassersteinHMM()
    model.initialize_templates(init_X)

    w_prev = np.full(N_ASSETS, 1.0 / N_ASSETS)
    weights_rec, rets_rec, extras_rec = [], [], []
    t0 = time.time()
    n = len(oos_idx)

    for i, t in enumerate(oos_idx):
        ti = feat_idx.get_loc(t)
        X_hist = features.values[:ti + 1]
        R_hist = returns.loc[features.index[:ti]].values

        out = model.step(X_hist, t_index=ti)
        mu = out["mu_full"][:N_ASSETS]
        cov = out["cov_full"][:N_ASSETS, :N_ASSETS]
        cov = _shrink(cov, sample=R_hist[-min(750, len(R_hist)):] if len(R_hist) > 50 else None)

        w = solve_mvo(mu, cov, w_prev)
        r_t = float(returns.loc[t].values @ w)

        weights_rec.append(pd.Series(w, index=ASSETS, name=t))
        rets_rec.append((t, r_t))
        extras_rec.append({"date": t, "K": out["K"],
                           "dominant_g": out["dominant_g"],
                           **{f"p_template_{g}": p for g, p in enumerate(out["p_template"])}})
        w_prev = w
        _progress("param", i, n, t0)

    return BacktestResult(weights=pd.DataFrame(weights_rec),
                          returns=pd.Series({d: r for d, r in rets_rec}, name="ret"),
                          extras=pd.DataFrame(extras_rec).set_index("date"))


def run_knn(features, returns, oos_start=OOS_START):
    print("Non-parametric (KNN + MVO) backtest...")
    feat_idx = features.index
    oos_idx = feat_idx[feat_idx >= pd.Timestamp(oos_start)]

    w_prev = np.full(N_ASSETS, 1.0 / N_ASSETS)
    weights_rec, rets_rec = [], []
    t0 = time.time()
    n = len(oos_idx)

    for i, t in enumerate(oos_idx):
        ti = feat_idx.get_loc(t)
        if ti < KNN_LOOKBACK:
            continue
        X_hist = features.values[:ti]
        R_hist = returns.loc[features.index[:ti]].values
        x_t    = features.values[ti]

        mu, cov = knn_step(X_hist, R_hist, x_t)
        w = solve_mvo(mu, cov, w_prev)
        r_t = float(returns.loc[t].values @ w)

        weights_rec.append(pd.Series(w, index=ASSETS, name=t))
        rets_rec.append((t, r_t))
        w_prev = w
        _progress("knn", i, n, t0)

    return BacktestResult(weights=pd.DataFrame(weights_rec),
                          returns=pd.Series({d: r for d, r in rets_rec}, name="ret"))


def run_equal_weight(returns, oos_start=OOS_START):
    oos = returns.loc[oos_start:]
    w = np.full(N_ASSETS, 1.0 / N_ASSETS)
    weights = pd.DataFrame(np.tile(w, (len(oos), 1)), index=oos.index, columns=ASSETS)
    return BacktestResult(weights=weights,
                          returns=pd.Series((oos.values * w).sum(axis=1),
                                            index=oos.index, name="ret"))


def run_spx_buy_hold(returns, oos_start=OOS_START):
    oos = returns.loc[oos_start:]
    w = np.zeros(N_ASSETS); w[ASSETS.index("SPX")] = 1.0
    weights = pd.DataFrame(np.tile(w, (len(oos), 1)), index=oos.index, columns=ASSETS)
    return BacktestResult(weights=weights,
                          returns=pd.Series(oos["SPX"].values,
                                            index=oos.index, name="ret"))


# ============================================================================
# 8. METRICS
# ============================================================================
def annualised_sharpe(r):
    return 0.0 if r.std(ddof=0) == 0 else float(np.sqrt(TRADING_DAYS) * r.mean() / r.std(ddof=0))


def annualised_sortino(r):
    d = r[r < 0]
    if len(d) == 0 or d.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * r.mean() / d.std(ddof=0))


def max_drawdown(r):
    cum = r.cumsum()
    return float((cum - cum.cummax()).min())


def turnover(weights):
    return 0.5 * weights.diff().abs().sum(axis=1).fillna(0.0)


def n_effective(weights):
    return 1.0 / (weights ** 2).sum(axis=1)


def perf_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Strategy": name,
         "Sharpe":   annualised_sharpe(res.returns),
         "Sortino":  annualised_sortino(res.returns),
         "MaxDD":    max_drawdown(res.returns)}
        for name, res in results.items()
    ]).set_index("Strategy")


def turnover_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        to = turnover(res.weights)
        rows.append({"Strategy": name,
                     "Avg Daily Turnover": to.mean(),
                     "95% Quantile":       to.quantile(0.95),
                     "Days >1% Turnover (%)": 100 * (to > 0.01).mean(),
                     "Days >5% Turnover (%)": 100 * (to > 0.05).mean()})
    return pd.DataFrame(rows).set_index("Strategy")


def weights_summary(results: dict) -> pd.DataFrame:
    pieces = []
    for name, res in results.items():
        w = res.weights
        df = pd.DataFrame({"Avg Wt":   w.mean(),
                           "Wt Vol":   w.std(),
                           "Time>10%": (w > 0.10).mean(),
                           "Avg |Δw|": w.diff().abs().mean()})
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        pieces.append(df)
    return pd.concat(pieces, axis=1)


def concentration_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"Strategy": name,
         "Avg N_eff":    n_effective(res.weights).mean(),
         "Median N_eff": n_effective(res.weights).median()}
        for name, res in results.items()
    ]).set_index("Strategy")


def portfolio_by_regime(returns, dominant_g):
    df = pd.concat([returns.rename("ret"), dominant_g.rename("g")], axis=1).dropna()
    rows = []
    for g, sub in df.groupby("g"):
        r = sub["ret"]
        if len(r) < 5:
            continue
        rows.append({"Regime":  int(g),
                     "Days":    len(r),
                     "Ann Mean": r.mean() * TRADING_DAYS,
                     "Ann Vol":  r.std(ddof=0) * np.sqrt(TRADING_DAYS),
                     "Sharpe":   annualised_sharpe(r),
                     "Hit Rate": (r > 0).mean(),
                     "Max DD (within)": max_drawdown(r)})
    return pd.DataFrame(rows).set_index("Regime").sort_index()


def asset_by_regime(asset_returns, dominant_g):
    df = asset_returns.join(dominant_g.rename("g"), how="inner")
    out = {}
    for g, sub in df.groupby("g"):
        r = sub.drop(columns="g")
        if len(r) < 5:
            continue
        out[int(g)] = (r.mean() * TRADING_DAYS) / (r.std(ddof=0) * np.sqrt(TRADING_DAYS)).replace(0, np.nan)
    return pd.DataFrame(out).T.sort_index()


# ============================================================================
# 9. PLOTS
# ============================================================================
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_param_cum_pnl(res, dom_g, out):
    cum = res.returns.cumsum()
    df = pd.concat([cum.rename("cum"), dom_g.rename("g")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.viridis
    gs = sorted(df["g"].unique())
    colors = {g: cmap(i / max(1, len(gs) - 1)) for i, g in enumerate(gs)}
    for g in gs:
        sub = df[df["g"] == g]
        ax.scatter(sub.index, sub["cum"], s=8, color=colors[g], label=f"Regime {g}")
    ax.set_title("Cumulative PnL Coloured by Template Regime (Parametric)")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return _save(fig, out / "fig01_param_cum_pnl_by_regime.png")


def fig_knn_cum_pnl(res, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.returns.cumsum(), label="KNN + MVO OOS Cumulative PnL")
    ax.set_title("KNN-Based Allocation (STRICTLY CAUSAL) — Cumulative PnL")
    ax.legend()
    return _save(fig, out / "fig02_knn_cum_pnl.png")


def fig_turnover(res, label, out, fname):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(turnover(res.weights), label="Daily Turnover (0.5*L1)")
    ax.set_title(f"Turnover Over Time — {label}")
    ax.legend(loc="upper right")
    return _save(fig, out / fname)


def fig_stacked_weights(res, label, out, fname):
    w = res.weights[ASSETS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(w.index, w.T.values, labels=ASSETS,
                 colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Portfolio Weights (Stacked) — {label}")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    return _save(fig, out / fname)


def fig_n_eff(res, label, out, fname):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(n_effective(res.weights), label="Effective # Positions (N_eff)")
    ax.set_title(f"Portfolio Concentration Over Time — {label}")
    ax.legend()
    return _save(fig, out / fname)


def fig_asset_sharpe_by_regime(asset_sharpes, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    asset_sharpes.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Asset Sharpe Ratios by Regime")
    ax.set_ylabel("Sharpe Ratio (Annualised)")
    ax.set_xticklabels([f"Regime {chr(65 + i)}" for i in range(len(asset_sharpes))], rotation=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
    return _save(fig, out / "fig09_asset_sharpe_by_regime.png")


def fig_stacked_pnl_by_regime(res, dom_g, out):
    df = pd.concat([res.returns.rename("ret"), dom_g.rename("g")], axis=1).dropna()
    gs = sorted(df["g"].unique())
    cum = pd.DataFrame(0.0, index=df.index, columns=gs)
    for g in gs:
        cum[g] = df.where(df["g"] == g)["ret"].fillna(0.0).cumsum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(cum.index, cum.T.values,
                 labels=[f"Regime {chr(65 + i)}" for i, _ in enumerate(gs)])
    ax.set_title("Stacked Cumulative PnL by Template Regime")
    ax.set_ylabel("Cumulative PnL")
    ax.legend(loc="upper left")
    return _save(fig, out / "fig10_stacked_cum_pnl_by_regime.png")


def fig_benchmark_compare(param, ew, spx, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(param.returns.cumsum(), label="Parametric RD (HMM + MVO)")
    ax.plot(ew.returns.cumsum(),    label="Equal-Weight (20% each)", linestyle="--")
    ax.plot(spx.returns.cumsum(),   label="SPX Buy & Hold",          linestyle=":")
    ax.set_title("Cumulative PnL Comparison (Same Backtest Period)")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend(loc="upper left")
    return _save(fig, out / "fig11_benchmark_compare.png")


# ============================================================================
# 10. MAIN
# ============================================================================
def banner(s):
    print("\n" + "=" * 78 + f"\n  {s}\n" + "=" * 78)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner("1. Load prices and build features (strict causality)")
    prices = load_prices()
    rets   = log_returns(prices)
    feats  = build_features(rets)
    rets_a = rets.loc[feats.index]
    print(f"prices  : {prices.shape}, {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"features: {feats.shape}, OOS start: {OOS_START}")

    banner("2. Parametric backtest (Wasserstein HMM + MVO)")
    t0 = time.time()
    param = run_parametric(feats, rets_a)
    print(f"done in {(time.time()-t0)/60:.1f} min, {len(param.returns)} OOS days")

    banner("3. KNN backtest")
    t0 = time.time()
    knn = run_knn(feats, rets_a)
    print(f"done in {(time.time()-t0)/60:.1f} min, {len(knn.returns)} OOS days")

    # Align onto common dates
    common = param.returns.index.intersection(knn.returns.index)
    param.weights = param.weights.loc[common]
    param.returns = param.returns.loc[common]
    param.extras  = param.extras.loc[common]
    knn.weights   = knn.weights.loc[common]
    knn.returns   = knn.returns.loc[common]

    banner("4. Passive benchmarks")
    ew  = run_equal_weight(rets_a, oos_start=str(common.min().date()))
    spx = run_spx_buy_hold(rets_a, oos_start=str(common.min().date()))

    banner("5. Tables")
    out = OUTPUT_DIR
    t1 = perf_table({"Non-Parametric (KNN+MVO)": knn,
                     "Parametric (W-HMM+MVO)":   param})
    print("\nTable 1 — performance comparison\n", t1.round(4))
    t1.to_csv(out / "table1_performance.csv")

    t2 = turnover_table({"KNN+MVO": knn, "Parametric+MVO": param})
    print("\nTable 2 — turnover\n", t2.round(4))
    t2.to_csv(out / "table2_turnover.csv")

    t3 = weights_summary({"KNN+MVO": knn, "Parametric+MVO": param})
    print("\nTable 3 — weights\n", t3.round(4))
    t3.to_csv(out / "table3_weights.csv")

    t4 = concentration_table({"KNN+MVO": knn, "Parametric+MVO": param})
    print("\nTable 4 — concentration\n", t4.round(4))
    t4.to_csv(out / "table4_concentration.csv")

    dom_g = param.extras["dominant_g"]
    t5 = portfolio_by_regime(param.returns, dom_g)
    print("\nTable 5 — portfolio by regime\n", t5.round(4))
    t5.to_csv(out / "table5_portfolio_by_regime.csv")

    t6 = asset_by_regime(rets_a.loc[param.returns.index], dom_g)
    print("\nTable 6 — asset Sharpe by regime\n", t6.round(4))
    t6.to_csv(out / "table6_asset_sharpe_by_regime.csv")

    t7 = perf_table({"Parametric Regime Investing": param,
                     "Equal-Weight (20%)":          ew,
                     "SPX Buy & Hold":              spx})
    print("\nTable 7 — vs benchmarks\n", t7.round(4))
    t7.to_csv(out / "table7_vs_benchmarks.csv")

    banner("6. Figures")
    fig_param_cum_pnl(param, dom_g, out)
    fig_knn_cum_pnl(knn, out)
    fig_turnover(knn, "Non-Parametric (KNN)",   out, "fig03_knn_turnover.png")
    fig_turnover(param, "Parametric (W-HMM)",   out, "fig04_param_turnover.png")
    fig_stacked_weights(knn, "Non-Parametric (KNN)", out, "fig05_knn_weights.png")
    fig_stacked_weights(param, "Parametric (W-HMM)", out, "fig06_param_weights.png")
    fig_n_eff(knn, "Non-Parametric (KNN)",  out, "fig07_knn_neff.png")
    fig_n_eff(param, "Parametric (W-HMM)",  out, "fig08_param_neff.png")
    fig_asset_sharpe_by_regime(t6, out)
    fig_stacked_pnl_by_regime(param, dom_g, out)
    fig_benchmark_compare(param, ew, spx, out)
    print("Wrote figures to", out)

    banner("7. Saving raw outputs")
    param.weights.to_csv(out / "param_weights.csv")
    param.returns.to_csv(out / "param_returns.csv")
    param.extras.to_csv (out / "param_extras.csv")
    knn.weights.to_csv  (out / "knn_weights.csv")
    knn.returns.to_csv  (out / "knn_returns.csv")
    print("All artefacts written to", out.resolve())


if __name__ == "__main__":
    main()
