# pomdp_agentic_ai_portfolio_validation.py
# Run:
#   pip install yfinance openai pandas numpy scipy matplotlib tabulate
#   export OPENAI_API_KEY="YOUR_KEY"
#   python pomdp_agentic_ai_portfolio_validation.py
#
# Optional OpenAI calls:
#   RUN_OPENAI=1 python pomdp_agentic_ai_portfolio_validation.py

import os, json, time
from dataclasses import dataclass, replace 
import numpy as np
import pandas as pd
#from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import textwrap
from scipy.optimize import minimize
from openai import OpenAI
import requests
from scipy.stats import chi2

OUT = "pomdp_results"
os.makedirs(OUT, exist_ok=True)

MASSIVE_BASE_URL = "https://api.massive.com"
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "JPM", "IBM", "GLD", "TLT", "SPY"]
RISKY = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "JPM", "IBM", "GLD", "TLT"]
BENCHMARK = "SPY"

BELIEF_EVENTS = [
    #("2024-08-05", "Yen carry unwind / VIX shock"),
    ("2025-01-27", "DeepSeek AI selloff"),
    ("2025-04-02", "Tariff uncertainty"),
    ("2026-04-29", "Fed / delayed cuts"),
    ("2026-06-08", "Geopolitical / oil risk"),
]

MACRO_TICKERS = {
    "VIX":  "VIXY",  # VIX futures ETF proxy
    "OIL":  "USO",   # crude oil ETF proxy
    "GOLD": "GLD",   # gold ETF proxy
    "HYG":  "HYG",   # high-yield corporate credit ETF
    "LQD":  "LQD",   # investment-grade corporate credit ETF
}

ABLATION_MODES = [
    "Historical_Only",
    "Market_Only",
    "Market_Plus_Direct_Macro",
    "Market_Plus_Beliefs",
    "Full_POMDP_Macro_Inferred_Beliefs",
]

LLM_STATES = [
    "AI_Boom",
    "Soft_Landing",
    "Inflation_Shock",
    "Recession",
    "Crisis",
]

BELIEF_STATE_COLS = [
    "AI_Boom",
    "Soft_Landing",
    "Inflation_Shock",
    "Recession",
    "Crisis",
]


MACRO_EVENTS = [
    ("2024-06-12", "Fed higher-for-longer"),
    ("2024-08-05", "Volatility spike"),
    ("2024-11-06", "US election repricing"),
    ("2025-01-27", "AI / tech volatility"),
    ("2025-04-02", "Tariff uncertainty"),
    ("2025-09-18", "Fed cut expectations"),
    ("2026-04-29", "Delayed cuts"),
    ("2026-06-08", "Geopolitical / oil risk"),
]


SENSITIVITY_GRID = {
    "risk_aversion": [2.0, 3.5, 5.0],
    "prior_shrinkage": [0.10, 0.25, 0.50],
    "view_weight": [0.40, 0.65, 0.80],
}

START = "2024-06-10"
#END = None
END = "2026-06-14"
#END_DATE = pd.Timestamp(date.today()).normalize()
#FORWARD_DAYS = 21
FORWARD_DAYS = 21
LOOKBACK_DAYS = 100
REBALANCE_FREQ = "ME"
RISK_AVERSION = 3.5
MODEL = "gpt-5"
SLEEP_BETWEEN_OPENAI_CALLS = 20.0


# -----------------------------
# LaTeX tables
# -----------------------------

def save_latex_table(df, path, caption, label, float_format="%.4f"):
    latex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: float_format % x,
        caption=caption,
        label=label,
        position="htbp",
    )
    with open(path, "w") as f:
        f.write(latex)


def set_quant_journal_style():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.22,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
# -----------------------------
# Data
# -----------------------------


CACHE_FILE = f"{OUT}/cached_prices.csv"

# -----------------------------
# Massive.com market data
# -----------------------------

def massive_get_json(url, params):
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MASSIVE_API_KEY")

    params = dict(params)
    params["apiKey"] = api_key

    r = requests.get(url, params=params, timeout=60)

    if r.status_code == 403:
        raise RuntimeError(
            f"Massive 403 Forbidden for URL {url}. "
            "This usually means your plan does not include this asset class "
            "or ticker. Use ETF proxies like VIXY instead of index tickers like I:VIX."
        )

    r.raise_for_status()
    return r.json()


def download_massive_agg_series(ticker, name=None, adjusted=True):
    start = pd.to_datetime(START).strftime("%Y-%m-%d")
    end = pd.Timestamp.today().strftime("%Y-%m-%d") if END is None else pd.to_datetime(END).strftime("%Y-%m-%d")

    url = f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"

    payload = massive_get_json(
        url,
        {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 50000,
        },
    )

    results = payload.get("results", [])

    if not results:
        raise RuntimeError(f"No Massive aggregate results for {ticker}: {payload}")

    df = pd.DataFrame(results)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    return df["c"].rename(name or ticker)


def download_treasury_10y():
    cache_file = f"{OUT}/cached_10y_treasury.csv"

    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)["UST10Y"]

    url = f"{MASSIVE_BASE_URL}/fed/v1/treasury-yields"

    payload = massive_get_json(
        url,
        {
            "limit": 50000,
            "sort": "date.asc",
        },
    )

    rows = payload.get("results", [])

    if not rows:
        raise RuntimeError(f"No treasury yield results returned: {payload}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    y10 = df["yield_10_year"].rename("UST10Y")
    y10 = y10.loc[y10.index >= pd.to_datetime(START)]

    if END is not None:
        y10 = y10.loc[y10.index <= pd.to_datetime(END)]

    y10.to_csv(cache_file)
    return y10

def download_massive_ticker(ticker):
    api_key = os.getenv("MASSIVE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing MASSIVE_API_KEY. Set it with:\n"
            "export MASSIVE_API_KEY='YOUR_KEY'"
        )

    start = pd.to_datetime(START).strftime("%Y-%m-%d")

    if END is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    else:
        end = pd.to_datetime(END).strftime("%Y-%m-%d")

    url = (
        f"{MASSIVE_BASE_URL}/v2/aggs/ticker/"
        f"{ticker}/range/1/day/{start}/{end}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    payload = r.json()

    if payload.get("status") not in ["OK", "DELAYED"]:
        raise RuntimeError(
            f"Massive API error for {ticker}: {payload}"
        )

    results = payload.get("results", [])

    if not results:
        raise RuntimeError(
            f"No Massive results returned for {ticker}: {payload}"
        )

    df = pd.DataFrame(results)

    # Massive aggregate fields:
    # t = Unix timestamp in milliseconds
    # o = open
    # h = high
    # l = low
    # c = close
    # v = volume
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize(None)
    df = df.set_index("date").sort_index()

    close = df["c"].rename(ticker)

    return close


def download_prices():
    cache_file = f"{OUT}/cached_prices_massive.csv"

    if os.path.exists(cache_file):
        print(f"Loading cached Massive data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print("Downloading portfolio prices from Massive...")

    series_list = []

    for ticker in TICKERS:
        print(f"Downloading {ticker}")
        close = download_massive_agg_series(ticker, name=ticker, adjusted=True)
        series_list.append(close)
        time.sleep(SLEEP_BETWEEN_OPENAI_CALLS)

    prices = pd.concat(series_list, axis=1).sort_index().ffill().dropna()
    prices.to_csv(cache_file)

    return prices

def download_macro_data():
    cache_file = f"{OUT}/cached_macro_massive.csv"

    if os.path.exists(cache_file):
        print(f"Loading cached macro data from {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    print("Downloading macro data from Massive...")

    series = []

    for name, ticker in MACRO_TICKERS.items():
        print(f"Downloading macro {name}: {ticker}")
        s = download_massive_agg_series(ticker, name=name, adjusted=True)
        series.append(s)
        time.sleep(0.25)

    y10 = download_treasury_10y()
    series.append(y10)

    macro = pd.concat(series, axis=1).sort_index().ffill()

    # Credit spread proxy: high-yield underperformance vs investment grade.
    # Higher value = credit stress proxy.
    macro["CREDIT_SPREAD_PROXY"] = (
        macro["LQD"].pct_change(21) - macro["HYG"].pct_change(21)
    )

    macro = macro.ffill().dropna()
    macro.to_csv(cache_file)

    return macro

def daily_returns(prices):
    return prices.pct_change().dropna()


# -----------------------------
# POMDP belief state
# Hidden states: bull, neutral, bear, crisis
# -----------------------------


def build_macro_features(date, macro):
    m = macro.loc[:date].tail(63)

    if len(m) < 22:
        return None

    latest = m.iloc[-1]
    prev_21 = m.iloc[-22]

    features = {
        "date": str(date.date()),
        "vix_level": float(latest.get("VIX", np.nan)),
        "vix_21d_change": float(latest.get("VIX", np.nan) - prev_21.get("VIX", np.nan)),
        "ust10y_level": float(latest.get("UST10Y", np.nan)),
        "ust10y_21d_change": float(latest.get("UST10Y", np.nan) - prev_21.get("UST10Y", np.nan)),
        "oil_21d_return": float(latest.get("OIL", np.nan) / prev_21.get("OIL", np.nan) - 1),
        "gold_21d_return": float(latest.get("GOLD", np.nan) / prev_21.get("GOLD", np.nan) - 1),
        "credit_spread_proxy": float(latest.get("CREDIT_SPREAD_PROXY", np.nan)),
        "hyg_21d_return": float(latest.get("HYG", np.nan) / prev_21.get("HYG", np.nan) - 1),
        "lqd_21d_return": float(latest.get("LQD", np.nan) / prev_21.get("LQD", np.nan) - 1),
    }

    return features


    # -----------------------------
# LLM expected return forecast
# -----------------------------

def infer_expected_returns_with_llm(date, names, window, macro_features, belief):
    client = OpenAI()

    asset_features = {}

    for ticker in names:
        r = window[ticker]

        asset_features[ticker] = {
            "return_21d": float((1 + r.tail(21)).prod() - 1),
            "return_63d": float((1 + r.tail(63)).prod() - 1),
            "vol_63d_annualized": float(r.tail(63).std() * np.sqrt(252)),
            "max_drawdown_63d": float(
                ((1 + r.tail(63)).cumprod() /
                 (1 + r.tail(63)).cumprod().cummax() - 1).min()
            ),
        }

    prompt = f"""
    You are an institutional portfolio strategist.

    Estimate annualized expected returns for the following assets over the next month,
    conditional on the inferred macro regime.

    Return JSON only.

    Required schema:
    {{
    "expected_returns": {{
        "AAPL": 0.10,
        "MSFT": 0.10
    }},
    "rationale": "brief explanation"
    }}

    Rules:
    - Values are annualized expected arithmetic returns.
    - Use decimal form: 0.10 means 10%.
    - Keep forecasts conservative.
    - Do not output markdown.

    Date:
    {date.date()}

    Assets:
    {names}

    POMDP hidden-state belief:
    {json.dumps(dict(zip(LLM_STATES, belief)), indent=2)}

    Macro features:
    {json.dumps(macro_features, indent=2)}

    Asset features:
    {json.dumps(asset_features, indent=2)}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    txt = response.output_text.strip()
    data = json.loads(txt)

    mu = pd.Series(data["expected_returns"], dtype=float)
    mu = mu.reindex(names).fillna(0.0)

    # Convert annualized forecast to daily expected return
    mu_daily = mu / 252.0

    return mu_daily.values, data

def black_litterman_blend(mu_hist, mu_view, cov, names, tau=0.25, view_weight=0.60):
    """
    Simple robust Black-Litterman-style blend.

    mu_hist: daily historical mean
    mu_view: daily LLM/regime/momentum view
    cov: daily covariance
    """

    mu_hist = np.asarray(mu_hist)
    mu_view = np.asarray(mu_view)

    # Shrink noisy historical means
    mu_prior = 0.25 * mu_hist

    # Blend prior and views
    mu_post = (1.0 - view_weight) * mu_prior + view_weight * mu_view

    # Volatility-aware damping
    vols = np.sqrt(np.diag(cov))
    penalty = 0.10 * vols / np.sqrt(252)

    mu_post = mu_post - penalty

    return mu_post    


def infer_hidden_state_with_llm(date, window, macro_features, previous_belief=None):
    client = OpenAI()

    asset_summary = {
        "spy_21d_return": float((1 + window["SPY"].tail(21)).prod() - 1),
        "spy_63d_return": float((1 + window["SPY"].tail(63)).prod() - 1),
        "spy_21d_vol_annualized": float(window["SPY"].tail(21).std() * np.sqrt(252)),
        "spy_63d_vol_annualized": float(window["SPY"].tail(63).std() * np.sqrt(252)),
    }

    prompt = f"""
You are estimating the hidden macro-market state for a POMDP portfolio agent.

Return JSON only.

States:
{LLM_STATES}

Definitions:
AI_Boom: growth led by technology/AI, risk-on, strong equities.
Soft_Landing: moderate growth, contained inflation, benign risk.
Inflation_Shock: rates/oil/inflation pressure dominating markets.
Recession: deteriorating growth, earnings risk, defensive rotation.
Crisis: acute market stress, volatility spike, credit stress, liquidity shock.

The probabilities must be nonnegative and sum to 1.

Required JSON schema:
{{
  "belief": {{
    "AI_Boom": 0.0,
    "Soft_Landing": 0.0,
    "Inflation_Shock": 0.0,
    "Recession": 0.0,
    "Crisis": 0.0
  }},
  "rationale": "brief explanation"
}}

Date:
{date.date()}

Previous belief:
{json.dumps(previous_belief, indent=2) if previous_belief else "None"}

Market summary:
{json.dumps(asset_summary, indent=2)}

Macro features:
{json.dumps(macro_features, indent=2)}
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )

    txt = response.output_text.strip()
    data = json.loads(txt)

    b = pd.Series(data["belief"], dtype=float).reindex(LLM_STATES).fillna(0.0)
    b = b.clip(lower=0.0)

    if b.sum() <= 0:
        b[:] = 1.0 / len(b)
    else:
        b = b / b.sum()

    data["belief"] = b.to_dict()

    return b.values, data

STATES = ["bull", "neutral", "bear", "crisis"]

TRANSITION = np.array([
    [0.82, 0.13, 0.04, 0.01],
    [0.20, 0.60, 0.17, 0.03],
    [0.05, 0.20, 0.62, 0.13],
    [0.03, 0.07, 0.25, 0.65],
])

STATE_MEAN = np.array([0.0008, 0.00025, -0.00035, -0.0015])
STATE_VOL = np.array([0.008, 0.010, 0.016, 0.028])


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def bayes_filter_update(belief, market_return):
    pred = belief @ TRANSITION
    likelihood = normal_pdf(market_return, STATE_MEAN, STATE_VOL)
    posterior = pred * likelihood
    return posterior / posterior.sum()


def belief_risk_score(belief):
    return float(belief[2] + 2.0 * belief[3])


# -----------------------------
# Portfolio construction
# -----------------------------

def normalize_weights(w, names):
    w = pd.Series(w, index=names).astype(float)
    w[w < 0] = 0.0
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= w.sum()
    return w


def equal_weight(names):
    return pd.Series(1.0 / len(names), index=names)


def max_sharpe_weights(mu, cov, names):
    n = len(names)

    def objective(w):
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        return -ret / (vol + 1e-12)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n

    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    if not res.success:
        return equal_weight(names)
    return normalize_weights(res.x, names)


def risk_parity_weights(cov, names):
    n = len(names)

    def risk_contribution(w):
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / (sigma + 1e-12)
        return w * mrc

    def objective(w):
        rc = risk_contribution(w)
        return np.sum((rc - rc.mean()) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.001, 1)] * n
    x0 = np.ones(n) / n

    res = minimize(objective, x0, bounds=bounds, constraints=cons)
    if not res.success:
        return equal_weight(names)
    return normalize_weights(res.x, names)


def forecasting_pomdp_policy_weights(
    mu_hist,
    mu_view,
    cov,
    belief,
    names,
    config,
):
    """
    Forecasting POMDP policy:
    hidden state -> return views -> BL posterior -> constrained utility optimization.
    """

    cov = np.asarray(cov)

    mu_hist_shrunk = config.prior_shrinkage * mu_hist

    mu_post = black_litterman_blend(
        mu_hist=mu_hist_shrunk,
        mu_view=mu_view,
        cov=cov,
        names=names,
        tau=0.25,
        view_weight=config.view_weight,
    )

    b = pd.Series(belief, index=LLM_STATES)

    p_stress = b["Recession"] + 1.5 * b["Crisis"]
    p_growth = b["AI_Boom"] + 0.5 * b["Soft_Landing"]

    # Risk aversion rises in stress regimes

    
    lam = (
        config.risk_aversion
        + 8.0 * p_stress
        - 1.0 * p_growth
        )
    lam = float(np.clip(lam, 1.5, 10.0))

    n = len(names)

    def objective(w):
        ret = w @ mu_post * 252
        var = w @ cov @ w * 252
        return -(ret - lam * var)

    max_weight = 0.35

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]

    bounds = [(0.0, max_weight)] * n

    # Start from equal weight
    x0 = np.ones(n) / n

    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500},
    )

    if not res.success:
        w = equal_weight(names)
    else:
        w = normalize_weights(res.x, names)

    # Defensive regime overlay
    if p_stress > 0.30:
        defensive_assets = [x for x in ["GLD", "TLT"] if x in names]
        stress_tilt = min(0.25, 0.35 * p_stress)

        if defensive_assets:
            w = (1.0 - stress_tilt) * w
            for x in defensive_assets:
                w[x] += stress_tilt / len(defensive_assets)

            w = normalize_weights(w, names)

    # Growth regime overlay
    if p_growth > 0.45:
        growth_assets = [x for x in ["NVDA", "MSFT", "GOOGL", "AAPL", "AMZN"] if x in names]
        growth_tilt = min(0.20, 0.20 * p_growth)

        if growth_assets:
            w = (1.0 - growth_tilt) * w
            for x in growth_assets:
                w[x] += growth_tilt / len(growth_assets)

            w = normalize_weights(w, names)

    return w    


def pomdp_policy_weights(mu, cov, belief, names):
    """
    LLM-regime-aware POMDP policy.
    belief order:
    AI_Boom, Soft_Landing, Inflation_Shock, Recession, Crisis
    """

    mu = np.asarray(mu)
    cov = np.asarray(cov)

    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / (vols + 1e-12)
    w = pd.Series(inv_vol / inv_vol.sum(), index=names)

    b = pd.Series(belief, index=LLM_STATES)

    p_growth = b["AI_Boom"] + 0.5 * b["Soft_Landing"]
    p_stress = b["Recession"] + 1.5 * b["Crisis"]
    p_inflation = b["Inflation_Shock"]

    growth_assets = [x for x in ["NVDA", "MSFT", "GOOGL", "AAPL", "AMZN"] if x in names]
    defensive_assets = [x for x in ["GLD", "TLT"] if x in names]
    value_assets = [x for x in ["JPM", "XOM", "IBM"] if x in names]

    # Growth tilt when AI boom / soft landing probability is high.
    growth_tilt = min(0.35, 0.30 * p_growth)

    if growth_assets:
        w *= 1.0 - growth_tilt
        for x in growth_assets:
            w[x] += growth_tilt / len(growth_assets)

    # Inflation tilt: oil/value/gold, reduce long-duration bonds.
    inflation_tilt = min(0.25, 0.30 * p_inflation)

    if inflation_tilt > 0:
        recipients = [x for x in value_assets + ["GLD"] if x in names]

        if recipients:
            w *= 1.0 - inflation_tilt
            for x in recipients:
                w[x] += inflation_tilt / len(recipients)

        if "TLT" in names:
            w["TLT"] *= max(0.25, 1.0 - 1.5 * inflation_tilt)

    # Stress tilt: gold and Treasuries.
    stress_tilt = min(0.45, 0.35 * p_stress)

    if defensive_assets:
        w *= 1.0 - stress_tilt
        for x in defensive_assets:
            w[x] += stress_tilt / len(defensive_assets)

    # Mild momentum / Sharpe tilt.
    signal = pd.Series(mu / (vols + 1e-12), index=names)
    signal = signal - signal.mean()
    signal = signal / (signal.abs().sum() + 1e-12)

    signal_strength = 0.10 + 0.15 * p_growth - 0.10 * p_stress
    signal_strength = max(0.0, min(0.25, signal_strength))

    w = w + signal_strength * signal
    w = w.clip(lower=0.0, upper=0.30)

    # Normalize and blend lightly with equal weight for robustness.
    w = normalize_weights(w, names)

    entropy = -np.sum(b.values * np.log(b.values + 1e-12)) / np.log(len(b))
    blend = 0.10 + 0.15 * entropy

    w = (1.0 - blend) * w + blend * equal_weight(names)
    w = normalize_weights(w, names)

    return w


# -----------------------------
# OpenAI agent
# -----------------------------

def make_agent_prompt(date, names, mu, vol, corr_to_spy, belief, disclosure_mode):
    payload = {
        "date": str(date.date()),
        "assets": names,
        "disclosure_mode": disclosure_mode,
        "annualized_expected_returns": dict(zip(names, np.round(mu * 252, 4))),
        "annualized_volatility": dict(zip(names, np.round(vol * np.sqrt(252), 4))),
        "correlation_to_spy": dict(zip(names, np.round(corr_to_spy, 4))),
        "pomdp_belief_state": dict(zip(STATES, np.round(belief, 4))),
        "constraints": {
            "long_only": True,
            "weights_sum_to_one": True,
            "max_single_asset_weight": 0.35,
            "objective": "maximize risk-adjusted utility under partial observability",
        },
    }

    if disclosure_mode == "risk_only":
        payload = {
            "date": str(date.date()),
            "assets": names,
            "disclosure_mode": disclosure_mode,
            "average_annualized_volatility": float(np.mean(vol) * np.sqrt(252)),
            "pomdp_belief_state": dict(zip(STATES, np.round(belief, 4))),
            "constraints": payload["constraints"],
        }

    return f"""
You are a cautious institutional portfolio risk agent.

Given the following observation from a partially observable market environment,
recommend long-only portfolio weights.

Return JSON only, no markdown.

Required JSON schema:
{{
  "weights": {{"AAPL": 0.0, "MSFT": 0.0}},
  "rationale": "brief rationale",
  "belief_interpretation": "brief interpretation",
  "risk_controls": ["control 1", "control 2"]
}}

Observation:
{json.dumps(payload, indent=2)}
"""


def call_openai_agent(prompt, names, model=MODEL):
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    txt = response.output_text.strip()

    try:
        data = json.loads(txt)
        w = pd.Series(data.get("weights", {}), dtype=float)
        w = w.reindex(names).fillna(0.0)
        w = w.clip(lower=0.0, upper=0.35)
        return normalize_weights(w, names), data
    except Exception:
        print("Could not parse OpenAI response:")
        print(txt)
        return equal_weight(names), {"raw": txt}


# -----------------------------
# Backtest
# -----------------------------

@dataclass
class BacktestConfig:
    use_openai: bool = False
    openai_every_n_rebalances: int = 1
    disclosure_mode: str = "full"
    prior_shrinkage: float = 0.25
    risk_aversion: float = 3.5
    view_weight: float = 0.65


def get_rebalance_dates(rets):
    dates = []

    for _, group in rets.groupby(pd.Grouper(freq="ME")):
        if len(group) > 0:
            dates.append(group.index[-1])

    return dates


def macro_only_expected_returns(features, names, config):
    mu = pd.Series(0.0, index=names)

    risk_score = float(features.get("risk_score", 0.0))

    for name in names:
        if name in ["GLD", "TLT"]:
            mu[name] += 0.03 * risk_score
        else:
            mu[name] -= 0.02 * risk_score

    return mu


def realized_proxy_state_label(fwd_returns, names):
    """
    Creates an ex post proxy label for the latent state using realized
    forward returns.

    This is not the true hidden state. It is a validation proxy used to
    evaluate whether inferred beliefs are directionally aligned with
    realized market conditions.
    """

    r = fwd_returns[names].mean()

    growth_assets = [x for x in ["NVDA", "MSFT", "GOOGL", "AAPL", "AMZN"] if x in names]
    defensive_assets = [x for x in ["GLD", "TLT"] if x in names]
    cyclical_assets = [x for x in ["JPM", "IBM", "XOM"] if x in names]

    growth_ret = fwd_returns[growth_assets].mean(axis=1).add(1).prod() - 1 if growth_assets else 0.0
    defensive_ret = fwd_returns[defensive_assets].mean(axis=1).add(1).prod() - 1 if defensive_assets else 0.0
    cyclical_ret = fwd_returns[cyclical_assets].mean(axis=1).add(1).prod() - 1 if cyclical_assets else 0.0
    all_ret = fwd_returns[names].mean(axis=1).add(1).prod() - 1

    scores = {
        "AI_Boom": growth_ret,
        "Soft_Landing": all_ret + 0.5 * cyclical_ret,
        "Inflation_Shock": defensive_ret + 0.5 * cyclical_ret,
        "Recession": defensive_ret - all_ret,
        "Crisis": defensive_ret - growth_ret,
    }

    return max(scores, key=scores.get)


def daily_strategy_returns_from_weights(rets, weights_df, strategy, names):
    rows = weights_df[weights_df["strategy"] == strategy].copy()
    rows["date"] = pd.to_datetime(rows["date"])
    rows = rows.sort_values("date")

    daily_parts = []

    for i in range(len(rows)):
        start = rows.iloc[i]["date"]

        if i + 1 < len(rows):
            end = rows.iloc[i + 1]["date"]
            daily = rets.loc[(rets.index > start) & (rets.index <= end), names]
        else:
            daily = rets.loc[rets.index > start, names]

        if daily.empty:
            continue

        w = rows.iloc[i][names].astype(float).values
        strat_ret = daily.values @ w

        daily_parts.append(
            pd.Series(strat_ret, index=daily.index)
        )

    if not daily_parts:
        return pd.Series(dtype=float)

    return pd.concat(daily_parts).sort_index()    

def belief_only_expected_returns(belief, names, config):
    mu = pd.Series(0.0, index=names)

    ai_boom = float(belief.get("AI_Boom", 0.0))
    soft = float(belief.get("Soft_Landing", 0.0))
    inflation = float(belief.get("Inflation_Shock", 0.0))
    recession = float(belief.get("Recession", 0.0))
    crisis = float(belief.get("Crisis", 0.0))

    for name in names:
        if name in ["NVDA", "MSFT", "GOOGL", "AMZN"]:
            mu[name] += 0.08 * ai_boom + 0.03 * soft
            mu[name] -= 0.04 * recession + 0.05 * crisis

        elif name in ["GLD"]:
            mu[name] += 0.05 * inflation + 0.04 * crisis

        elif name in ["TLT"]:
            mu[name] += 0.05 * recession + 0.03 * crisis
            mu[name] -= 0.03 * inflation

        else:
            mu[name] += 0.02 * soft
            mu[name] -= 0.03 * recession + 0.03 * crisis

    return mu

def heuristic_expected_returns(
    names,
    window,
    belief,
    macro_features=None,
    use_market=True,
    use_macro=True,
    use_beliefs=True,
):
    """
    Regime-aware expected returns.

    Ablation controls:
        use_market  : momentum and volatility terms
        use_macro   : VIX, oil, credit, rates terms
        use_beliefs : POMDP hidden-state belief terms
    """

    if isinstance(belief, dict):
        b = pd.Series(belief).reindex(LLM_STATES).fillna(0.0)
    else:
        b = pd.Series(belief, index=LLM_STATES)

    if macro_features is None:
        macro_features = {}

    vix = float(macro_features.get("vix_level", 0.0))
    vix_chg = float(macro_features.get("vix_21d_change", 0.0))
    oil_ret = float(macro_features.get("oil_21d_return", 0.0))
    gold_ret = float(macro_features.get("gold_21d_return", 0.0))
    credit = float(macro_features.get("credit_spread_proxy", 0.0))
    y10_chg = float(macro_features.get("ust10y_21d_change", 0.0))

    forecasts = {}

    for ticker in names:

        forecast = 0.03

        r = window[ticker]

        if use_market:
            mom_21 = (1 + r.tail(21)).prod() - 1
            mom_63 = (1 + r.tail(63)).prod() - 1
            vol = r.tail(63).std() * np.sqrt(252)

            annual_momentum = 0.5 * mom_21 * 12 + 0.5 * mom_63 * 4

            forecast += 0.40 * annual_momentum
            forecast -= 0.08 * vol

        if use_macro:
            risk_pressure = (
                0.02 * max(vix - 20.0, 0.0)
                + 0.03 * max(vix_chg, 0.0)
                + 0.50 * max(credit, 0.0)
            )

            inflation_pressure = (
                0.50 * max(oil_ret, 0.0)
                + 0.50 * max(y10_chg, 0.0)
            )

            if ticker in ["NVDA", "MSFT", "GOOGL", "AAPL", "AMZN"]:
                forecast -= risk_pressure
                forecast -= 0.30 * inflation_pressure

            elif ticker in ["GLD"]:
                forecast += 0.40 * inflation_pressure
                forecast += 0.30 * risk_pressure
                forecast += 0.20 * gold_ret

            elif ticker in ["TLT"]:
                forecast += 0.20 * risk_pressure
                forecast -= 0.50 * inflation_pressure

            elif ticker in ["JPM", "IBM", "XOM"]:
                forecast -= 0.50 * risk_pressure
                forecast += 0.20 * inflation_pressure

        if use_beliefs:

            if ticker in ["NVDA", "MSFT", "GOOGL", "AAPL", "AMZN"]:
                forecast += (
                    0.10 * b["AI_Boom"]
                    + 0.05 * b["Soft_Landing"]
                    - 0.08 * b["Recession"]
                    - 0.15 * b["Crisis"]
                )

            if ticker in ["GLD", "TLT"]:
                forecast += (
                    0.08 * b["Recession"]
                    + 0.12 * b["Crisis"]
                )

            if ticker == "GLD":
                forecast += 0.10 * b["Inflation_Shock"]

            if ticker == "TLT":
                forecast -= 0.10 * b["Inflation_Shock"]

            if ticker in ["JPM", "IBM", "XOM"]:
                forecast += (
                    0.05 * b["Soft_Landing"]
                    + 0.05 * b["Inflation_Shock"]
                    - 0.08 * b["Crisis"]
                )

        forecasts[ticker] = forecast / 252.0

    return pd.Series(forecasts).reindex(names).values

    

def run_backtest(rets, macro, config, ablation_mode="Full_POMDP_Macro_Inferred_Beliefs"):
    names = RISKY
    spy = BENCHMARK

    belief = np.ones(len(LLM_STATES)) / len(LLM_STATES)

    wealth = {
        "60_40_SPY_TLT": [1.0],
        "EqualWeight": [1.0],
        "RiskParity": [1.0],
        "MaxSharpe": [1.0],
        "POMDP_Utility": [1.0],
        "Forecasting_POMDP": [1.0],
    }

    if config.use_openai:
        wealth["OpenAI_POMDP_Agent"] = [1.0]

    weights_history = []
    belief_history = []
    agent_logs = []
    dates = []

    rebal_dates = get_rebalance_dates(rets)

    for i, d in enumerate(rebal_dates):
        loc = rets.index.get_loc(d)

        if loc < LOOKBACK_DAYS or loc + FORWARD_DAYS >= len(rets):
            continue

        window = rets.iloc[loc - LOOKBACK_DAYS:loc]
        fwd = rets.iloc[loc + 1: loc + 1 + FORWARD_DAYS]

        macro_features = build_macro_features(d, macro)
        if macro_features is None:
            continue

        previous_belief_dict = dict(zip(LLM_STATES, belief))

        if config.use_openai:
            llm_belief, belief_log = infer_hidden_state_with_llm(
                d,
                window,
                macro_features,
                previous_belief=previous_belief_dict,
            )

            belief = llm_belief
            belief_log["date"] = str(d.date())

            agent_logs.append({
                "type": "belief",
                **belief_log,
            })

        else:
            vix = macro_features["vix_level"]
            vix_change = macro_features["vix_21d_change"]
            y10_change = macro_features["ust10y_21d_change"]
            oil_ret = macro_features["oil_21d_return"]
            credit = macro_features["credit_spread_proxy"]
            spy_ret = float((1 + window[spy].tail(21)).prod() - 1)

            scores = pd.Series({
                "AI_Boom": 1.0 + 8.0 * max(spy_ret, 0.0) - 0.05 * max(vix - 18, 0.0),
                "Soft_Landing": 1.0 - 0.03 * abs(vix - 16),
                "Inflation_Shock": 1.0 + 3.0 * max(oil_ret, 0.0) + 2.0 * max(y10_change, 0.0),
                "Recession": 1.0 + 5.0 * max(-spy_ret, 0.0) + 2.0 * max(credit, 0.0),
                "Crisis": 1.0 + 0.10 * max(vix - 25, 0.0) + 0.10 * max(vix_change, 0.0),
            })

            scores = np.exp(scores - scores.max())
            belief = (scores / scores.sum()).values

        asset_window = window[names]

        mu = asset_window.mean().values
        cov = asset_window.cov().values + 1e-8 * np.eye(len(names))
        vol = asset_window.std().values
        corr_to_spy = [
            asset_window[x].corr(window[spy])
            for x in names
        ]

        weights = {}

        weights["60_40_SPY_TLT"] = pd.Series(
            {x: 0.0 for x in names}
        )
        weights["60_40_SPY_TLT"]["TLT"] = 0.40

        equity_names = [
            x for x in names
            if x not in ["TLT", "GLD"]
        ]

        for x in equity_names:
            weights["60_40_SPY_TLT"][x] = 0.60 / len(equity_names)

        weights["EqualWeight"] = equal_weight(names)

        weights["RiskParity"] = risk_parity_weights(
            cov,
            names,
        )

        weights["MaxSharpe"] = max_sharpe_weights(
            mu,
            cov,
            names,
        )

        weights["POMDP_Utility"] = pomdp_policy_weights(
            mu,
            cov,
            belief,
            names,
        )

        if config.use_openai:
            mu_view, forecast_log = infer_expected_returns_with_llm(
                d,
                names,
                window,
                macro_features,
                belief,
            )

            forecast_log["date"] = str(d.date())

            agent_logs.append({
                "type": "forecast",
                **forecast_log,
            })

        else:
            neutral_belief = np.ones(len(LLM_STATES)) / len(LLM_STATES)

            if ablation_mode == "Historical_Only":

                mu_view = config.prior_shrinkage * mu

            elif ablation_mode == "Market_Only":

                mu_view = heuristic_expected_returns(
                    names=names,
                    window=window,
                    belief=neutral_belief,
                    macro_features=None,
                    use_market=True,
                    use_macro=False,
                    use_beliefs=False,
                )

            elif ablation_mode == "Market_Plus_Direct_Macro":

                mu_view = heuristic_expected_returns(
                    names=names,
                    window=window,
                    belief=neutral_belief,
                    macro_features=macro_features,
                    use_market=True,
                    use_macro=True,
                    use_beliefs=False,
                )

            elif ablation_mode == "Market_Plus_Beliefs":

                mu_view = heuristic_expected_returns(
                    names=names,
                    window=window,
                    belief=belief,
                    macro_features=None,
                    use_market=True,
                    use_macro=False,
                    use_beliefs=True,
                )

            elif ablation_mode == "Full_POMDP_Macro_Inferred_Beliefs":

                # Full model: macro variables are used upstream to infer the belief state.
                # They are NOT added again as a direct expected-return adjustment.
                mu_view = heuristic_expected_returns(
                    names=names,
                    window=window,
                    belief=belief,
                    macro_features=None,
                    use_market=True,
                    use_macro=False,
                    use_beliefs=True,
                )

            else:
                raise ValueError(f"Unknown ablation_mode: {ablation_mode}")
        mu_view = np.asarray(mu_view, dtype=float)   


        weights["Forecasting_POMDP"] = forecasting_pomdp_policy_weights(
            mu_hist=mu,
            mu_view=mu_view,
            cov=cov,
            belief=belief,
            names=names,
            config=config,
        )

        if config.use_openai and i % config.openai_every_n_rebalances == 0:
            prompt = make_agent_prompt(
                d,
                names,
                mu,
                vol,
                corr_to_spy,
                belief,
                config.disclosure_mode,
            )

            w_agent, log = call_openai_agent(
                prompt,
                names,
            )

            weights["OpenAI_POMDP_Agent"] = w_agent
            log["date"] = str(d.date())

            agent_logs.append({
                "type": "portfolio",
                **log,
            })

            time.sleep(0.5)

        elif config.use_openai:
            weights["OpenAI_POMDP_Agent"] = weights["Forecasting_POMDP"]

        next_returns = fwd[names]

        for strategy, w in weights.items():
            month_ret = float(
                (next_returns @ w).add(1).prod() - 1
            )

            wealth[strategy].append(
                wealth[strategy][-1] * (1 + month_ret)
            )

            row = {
                "date": d,
                "strategy": strategy,
            }

            row.update(w.to_dict())
            weights_history.append(row)

        belief_row = {
            "date": d,
            **dict(zip(LLM_STATES, belief)),
            "risk_score": float(
                belief[LLM_STATES.index("Recession")]
                + 2.0 * belief[LLM_STATES.index("Crisis")]
            ),
        }

        belief_row["proxy_state"] = realized_proxy_state_label(
            fwd_returns=fwd,
            names=names,
        )

        belief_row.update(macro_features)
        belief_history.append(belief_row)

        dates.append(d)

    n_obs = len(next(iter(wealth.values()))) - 1

    wealth_df = pd.DataFrame(
        {
            k: v[1:]
            for k, v in wealth.items()
        },
        index=dates[:n_obs],
    )

    weights_df = pd.DataFrame(weights_history)
    belief_df = pd.DataFrame(belief_history)
    if "date" in belief_df.columns:
        belief_df["date"] = pd.to_datetime(belief_df["date"])
        belief_df = belief_df.set_index("date")

    return wealth_df, weights_df, belief_df, agent_logs
# -----------------------------
# Metrics
# -----------------------------

import numpy as np
import pandas as pd


def annualized_sharpe(r):
    r = np.asarray(r)
    if np.std(r) < 1e-12:
        return np.nan

    return (
        np.mean(r)
        /
        np.std(r)
        *
        np.sqrt(252)
    )


def stationary_bootstrap_indices(
    n,
    avg_block=20,
):
    """
    Politis-Romano stationary bootstrap.
    """

    p = 1.0 / avg_block

    idx = np.empty(n, dtype=int)

    idx[0] = np.random.randint(0, n)

    for t in range(1, n):

        if np.random.rand() < p:
            idx[t] = np.random.randint(0, n)

        else:
            idx[t] = (idx[t - 1] + 1) % n

    return idx

def bootstrap_sharpe_difference(
    returns_a,
    returns_b,
    n_boot=2000,
    avg_block=20,
):
    """
    Difference:
        Sharpe(A) - Sharpe(B)
    """

    returns_a = np.asarray(returns_a)
    returns_b = np.asarray(returns_b)

    n = len(returns_a)

    diffs = []

    for _ in range(n_boot):

        idx = stationary_bootstrap_indices(
            n,
            avg_block=avg_block,
        )

        sa = annualized_sharpe(
            returns_a[idx]
        )

        sb = annualized_sharpe(
            returns_b[idx]
        )

        diffs.append(sa - sb)

    diffs = np.asarray(diffs)

    diff_hat = (
        annualized_sharpe(returns_a)
        -
        annualized_sharpe(returns_b)
    )

    ci_low = np.percentile(diffs, 2.5)
    ci_high = np.percentile(diffs, 97.5)

    return {
        "Difference": diff_hat,
        "CI Low": ci_low,
        "CI High": ci_high,
    } 

def max_drawdown(wealth):
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    return dd.min()


def belief_coverage_test_table(
    belief_df,
    outdir=OUT,
):
    """
    Kupiec-style unconditional coverage test for belief states.

    For each latent state s, test whether:

        mean_t p_t(s) = mean_t 1{proxy_state_t = s}

    where p_t(s) is the inferred belief probability.

    This is an approximate coverage test because the true latent state is
    unobserved and proxy_state is constructed ex post.
    """

    df = belief_df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    if "proxy_state" not in df.columns:
        raise ValueError("belief_df must contain proxy_state.")

    rows = []
    eps = 1e-12

    for state in LLM_STATES:
        p = pd.to_numeric(df[state], errors="coerce").clip(eps, 1 - eps)
        y = (df["proxy_state"] == state).astype(float)

        valid = p.notna() & y.notna()
        p = p.loc[valid]
        y = y.loc[valid]

        n = len(y)
        observed = int(y.sum())
        expected = float(p.sum())

        observed_rate = observed / n
        expected_rate = expected / n

        # Pearson chi-square coverage statistic:
        # compares observed vs expected event counts.
        denom1 = max(expected, eps)
        denom0 = max(n - expected, eps)

        coverage_stat = (
            ((observed - expected) ** 2) / denom1
            +
            (((n - observed) - (n - expected)) ** 2) / denom0
        )

        p_value = 1.0 - chi2.cdf(coverage_stat, df=1)

        rows.append({
            "State": state.replace("_", " "),
            "Observations": n,
            "Observed Count": observed,
            "Expected Count": expected,
            "Observed Rate": observed_rate,
            "Expected Rate": expected_rate,
            "Coverage Statistic": coverage_stat,
            "p-value": p_value,
            "Reject 5%": "Yes" if p_value < 0.05 else "No",
        })

    out = pd.DataFrame(rows)

    csv_path = f"{outdir}/table_8_belief_coverage.csv"
    tex_path = f"{outdir}/table_8_belief_coverage.tex"

    out.to_csv(csv_path, index=False)

    latex = out.to_latex(
        index=False,
        float_format="%.3f",
        caption=(
            "Kupiec-style unconditional coverage test for inferred belief states "
            "using ex post proxy latent-state labels. The null hypothesis is that "
            "the average inferred belief probability equals the empirical proxy-state "
            "frequency."
        ),
        label="tab:belief_coverage",
        escape=False,
    )

    with open(tex_path, "w") as f:
        f.write(latex)

    print("Saved belief coverage table:", tex_path)

    return out    


def performance_table(wealth_df):
    monthly = wealth_df.pct_change().dropna()

    rows = []

    for col in wealth_df.columns:
        # handle edge cases where there are no monthly returns (insufficient data)
        if col not in monthly.columns or len(monthly[col]) == 0 or len(wealth_df[col]) == 0:
            cagr = np.nan
            vol = np.nan
            sharpe = np.nan
            sortino = np.nan
        else:
            r = monthly[col]
            # CAGR computed from terminal wealth and number of monthly return periods
            cagr = float(wealth_df[col].iloc[-1]) ** (12.0 / len(r)) - 1.0
            vol = r.std() * np.sqrt(12)
            sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(12)
            sortino = r.mean() / (r[r < 0].std() + 1e-12) * np.sqrt(12)
        mdd = max_drawdown(wealth_df[col]) if len(wealth_df[col]) > 0 else np.nan
        calmar = cagr / abs(mdd) if (not np.isnan(cagr) and mdd < 0) else np.nan

        terminal_wealth = wealth_df[col].iloc[-1] if len(wealth_df[col]) > 0 else np.nan
        rows.append({
            "strategy": col,
            "CAGR": cagr,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "MaxDrawdown": mdd,
            "Calmar": calmar,
            "TerminalWealth": terminal_wealth,
        })

    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False)


def utility_table(wealth_df):
    monthly = wealth_df.pct_change().dropna()

    rows = []

    for col in monthly.columns:
        r = monthly[col]
        utility = r.mean() * 12 - RISK_AVERSION * (r.var() * 12)
        rows.append({
            "strategy": col,
            "mean_variance_utility": utility,
        })

    return pd.DataFrame(rows).sort_values(
        "mean_variance_utility",
        ascending=False,
    )


def generate_ablation_table_5(results, outdir="pomdp_results"):
    """
    results should be a dict:
    {
        "Historical_Only": {"Sharpe": ..., "mean_variance_utility": ...},
        ...
    }
    """

    rows = []

    labels = {
        "Historical_Only": "Historical Only",
        "Market_Only": "Market Only",
        "Market_Plus_Direct_Macro": "Market + Direct Macro",
        "Market_Plus_Beliefs": "Market + Beliefs",
        "Full_POMDP_Macro_Inferred_Beliefs": "Full POMDP: Macro-Inferred Beliefs",
    }

    for key in ABLATION_MODES:
        rows.append({
            "Model": labels[key],
            "Sharpe": results[key]["Sharpe"],
            "Utility": results[key]["Utility"],
        })

    df = pd.DataFrame(rows)

    path_csv = f"{outdir}/table_5_ablation_study.csv"
    path_tex = f"{outdir}/table_5_ablation_study.tex"

    df.to_csv(path_csv, index=False)

    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Ablation study showing the contribution of macro variables, LLM-inferred beliefs, and the full Forecasting POMDP framework.",
        label="tab:ablation",
        escape=False,
    )

    with open(path_tex, "w") as f:
        f.write(latex)

    print("Saved Table 5:", path_tex)

    return df

def generate_belief_calibration_table(
    belief_df,
    outdir=OUT,
):
    """
    Generates belief calibration diagnostics using proxy realized state labels.

    Columns required:
        LLM_STATES probabilities
        proxy_state
    """

    df = belief_df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

    missing = [s for s in LLM_STATES if s not in df.columns]
    if missing:
        raise ValueError(f"Missing belief columns: {missing}")

    if "proxy_state" not in df.columns:
        raise ValueError("belief_df must contain proxy_state column.")

    rows = []

    eps = 1e-12

    for state in LLM_STATES:
        p = pd.to_numeric(df[state], errors="coerce").clip(eps, 1.0)
        y = (df["proxy_state"] == state).astype(float)

        brier = float(np.mean((p - y) ** 2))
        log_score = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p.clip(eps, 1.0 - eps))))

        avg_prob = float(p.mean())
        empirical_freq = float(y.mean())
        calibration_gap = avg_prob - empirical_freq

        rows.append({
            "State": state.replace("_", " "),
            "Average Belief": avg_prob,
            "Empirical Frequency": empirical_freq,
            "Calibration Gap": calibration_gap,
            "Brier Score": brier,
            "Log Score": log_score,
        })

    out = pd.DataFrame(rows)

    csv_path = f"{outdir}/table_7_belief_calibration.csv"
    tex_path = f"{outdir}/table_7_belief_calibration.tex"

    out.to_csv(csv_path, index=False)

    latex = out.to_latex(
        index=False,
        float_format="%.3f",
        caption=(
            "Belief calibration diagnostics using ex post proxy latent-state labels. "
            "The proxy labels are constructed from realized forward returns and are used "
            "only for validation diagnostics; they are not observed by the agent."
        ),
        label="tab:belief_calibration",
        escape=False,
    )

    with open(tex_path, "w") as f:
        f.write(latex)

    print("Saved Table 7:", tex_path)

    return out

def generate_table_8_significance_daily(
    ablation_daily_returns,
    outdir=OUT,
    n_boot=5000,
    avg_block=21,
):
    comparisons = [
        (
            "Market + Beliefs vs Market Only",
            "Market_Plus_Beliefs",
            "Market_Only",
        ),
        (
            "Market + Beliefs vs Historical Only",
            "Market_Plus_Beliefs",
            "Historical_Only",
        ),
        (
            "Full POMDP vs Market Only",
            "Full_POMDP_Macro_Inferred_Beliefs",
            "Market_Only",
        ),
    ]

    rows = []

    for label, a, b in comparisons:
        ra = ablation_daily_returns[a]
        rb = ablation_daily_returns[b]

        common = ra.index.intersection(rb.index)
        ra = ra.loc[common].values
        rb = rb.loc[common].values

        stats = bootstrap_sharpe_difference(
            ra,
            rb,
            n_boot=n_boot,
            avg_block=avg_block,
        )

        rows.append({
            "Comparison": label,
            "Sharpe Difference": stats["Difference"],
            "95% CI Lower": stats["CI Low"],
            "95% CI Upper": stats["CI High"],
            "Significant": "Yes" if stats["CI Low"] > 0 else "No",
        })

    df = pd.DataFrame(rows)

    csv_path = f"{outdir}/table_8_significance_daily.csv"
    tex_path = f"{outdir}/table_8_significance_daily.tex"

    df.to_csv(csv_path, index=False)

    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption=(
            "Stationary-bootstrap confidence intervals for Sharpe-ratio "
            "differences using daily portfolio returns from the ablation study."
        ),
        label="tab:significance_daily",
        escape=False,
    )

    with open(tex_path, "w") as f:
        f.write(latex)

    print("Saved Table 8:", tex_path)

    return df
# -----------------------------
# Figures
# -----------------------------


def truncate_to_today_or_last_trading_day(df, end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    else:
        end_date = pd.Timestamp(end_date).normalize()

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    df = df[df.index <= end_date]

    if df.empty:
        raise ValueError("No data available up to end_date.")

    last_trading_day = df.index.max()
    print(f"Plotting through nearest trading day: {last_trading_day.date()}")

    return df

def clean_time_index(df):
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"]).set_index("date")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[~out.index.isna()]
        out.index = out.index.tz_localize(None)

    return out.sort_index()


def restrict_to_belief_columns(belief_df):
    belief_df = clean_time_index(belief_df)

    available = [c for c in BELIEF_STATE_COLS if c in belief_df.columns]

    # fallback if your labels use spaces rather than underscores
    if not available:
        rename_map = {
            "AI Boom": "AI_Boom",
            "Soft Landing": "Soft_Landing",
            "Inflation Shock": "Inflation_Shock",
        }
        belief_df = belief_df.rename(columns=rename_map)
        available = [c for c in BELIEF_STATE_COLS if c in belief_df.columns]

    if not available:
        raise ValueError(
            f"No belief-state columns found. Columns are: {list(belief_df.columns)}"
        )

    out = belief_df[available].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(how="all")

    return out

def add_belief_event_band(event_ax, events=BELIEF_EVENTS):
    event_ax.set_ylim(0, 1)
    event_ax.set_yticks([])
    event_ax.grid(False)

    for spine in event_ax.spines.values():
        spine.set_visible(False)

    for i, (date_str, label) in enumerate(events):
        d = pd.to_datetime(date_str)
        y = 0.22 if i % 2 == 0 else 0.62

        event_ax.axvline(d, color="0.20", lw=0.8, alpha=0.45)

        event_ax.text(
            d,
            y,
            "\n".join(textwrap.wrap(label, 18)),
            ha="center",
            va="center",
            fontsize=7,
            color="0.15",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.75",
                linewidth=0.5,
                alpha=0.95,
            ),
        )

    event_ax.set_xlabel(
        "Selected events used only for ex post interpretation of inferred beliefs",
        fontsize=8,
        color="0.35",
    )

def clean_index(df):
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def clean_label(x):
    return str(x).replace("_", " ")


def save_figure(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)

    png_path = os.path.join(outdir, f"{name}.png")
    pdf_path = os.path.join(outdir, f"{name}.pdf")

    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", length=0)


def add_event_band(event_ax, events=MACRO_EVENTS):
    """
    Adds a clean separate event band underneath the main plot.
    This avoids label collisions with data and legends.
    """
    event_ax.set_ylim(0, 1)
    event_ax.set_yticks([])
    event_ax.grid(False)

    for spine in event_ax.spines.values():
        spine.set_visible(False)

    for i, (date_str, label) in enumerate(events):
        d = pd.to_datetime(date_str)
        y = 0.15 if i % 2 == 0 else 0.55

        event_ax.axvline(d, color="0.25", lw=0.7, alpha=0.45)
        event_ax.text(
            d,
            y,
            "\n".join(textwrap.wrap(label, 16)),
            rotation=0,
            ha="center",
            va="center",
            fontsize=7,
            color="0.20",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.80",
                linewidth=0.4,
                alpha=0.92,
            ),
        )

    event_ax.set_xlabel("Macroeconomic and market event annotations", fontsize=8, color="0.35")


def make_event_figure(figsize=(7.4, 4.8)):
    """
    Creates a main plot plus a lower event annotation band.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2,
        1,
        height_ratios=[5.0, 1.15],
        hspace=0.05,
        figure=fig,
    )

    ax = fig.add_subplot(gs[0])
    event_ax = fig.add_subplot(gs[1], sharex=ax)

    return fig, ax, event_ax


def generate_parameter_sensitivity_table(
    rets,
    macro,
    base_config,
    outdir=OUT,
):
    rows = []

    wealth_base, weights_base, belief_base, logs_base = run_backtest(
        rets,
        macro,
        base_config,
        ablation_mode="Full_POMDP_Macro_Inferred_Beliefs",
    )

    perf_base = performance_table(wealth_base)
    util_base = utility_table(wealth_base)

    perf_row = perf_base.loc[
        perf_base["strategy"] == "Forecasting_POMDP"
    ].iloc[0]

    util_row = util_base.loc[
        util_base["strategy"] == "Forecasting_POMDP"
    ].iloc[0]

    rows.append({
        "Parameter": "BASE MODEL",
        "Value": np.nan,
        "Base Value": np.nan,
        "Sharpe": perf_row["Sharpe"],
        "Utility": util_row["mean_variance_utility"],
        "MaxDrawdown": perf_row["MaxDrawdown"],
    })

    base_values = {
        "risk_aversion": base_config.risk_aversion,
        "prior_shrinkage": base_config.prior_shrinkage,
        "view_weight": base_config.view_weight,
    }

    for param, values in SENSITIVITY_GRID.items():
        for value in values:
            cfg = replace(base_config, **{param: value})

            wealth_s, weights_s, belief_s, logs_s = run_backtest(
                rets,
                macro,
                cfg,
                ablation_mode="Full_POMDP_Macro_Inferred_Beliefs",
            )

            perf_s = performance_table(wealth_s)
            util_s = utility_table(wealth_s)

            row_perf = perf_s.loc[
                perf_s["strategy"] == "Forecasting_POMDP"
            ].iloc[0]

            row_util = util_s.loc[
                util_s["strategy"] == "Forecasting_POMDP"
            ].iloc[0]

            rows.append({
                "Parameter": param.replace("_", " "),
                "Value": value,
                "Base Value": base_values[param],
                "Sharpe": row_perf["Sharpe"],
                "Utility": row_util["mean_variance_utility"],
                "MaxDrawdown": row_perf["MaxDrawdown"],
            })

    df = pd.DataFrame(rows)

    csv_path = f"{outdir}/table_6_parameter_sensitivity.csv"
    tex_path = f"{outdir}/table_6_parameter_sensitivity.tex"

    df.to_csv(csv_path, index=False)


    df["Value"] = df["Value"].fillna("")
    df["Base Value"] = df["Base Value"].fillna("")

    latex = df.to_latex(
        index=False,
        float_format="%.3f",
        caption=(
            "Parameter sensitivity analysis for the Forecasting POMDP. "
            "Each row varies one parameter while holding all other parameters "
            "fixed at their base values."
        ),
        label="tab:sensitivity",
        escape=False,
    )

    with open(tex_path, "w") as f:
        f.write(latex)

    print("Saved Table 6:", tex_path)

    return df

# ============================================================
# Figure 1: cumulative wealth
# ============================================================

def plot_journal_cumulative_wealth(wealth_df, outdir="pomdp_results", plot_end_date=None):
    set_quant_journal_style()
    wealth_df = clean_time_index(wealth_df)

    xmin = wealth_df.index.min()
    #xmax = wealth_df.index.max()

    if plot_end_date is None:
        xmax = wealth_df.index.max()
    else:
        xmax = pd.to_datetime(plot_end_date)

    fig, ax, event_ax = make_event_figure()

    for col in wealth_df.columns:
        lw = 2.2 if "Forecasting" in col else 1.35
        ax.plot(wealth_df.index, wealth_df[col], lw=lw, label=clean_label(col))

    ax.set_title("Cumulative Wealth by Strategy")
    ax.set_ylabel("Wealth Index")
    ax.grid(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        handlelength=2.0,
    )

    add_belief_event_band(event_ax)

    ax.set_xlim(xmin, xmax)
    event_ax.set_xlim(xmin, xmax)

    plt.setp(ax.get_xticklabels(), visible=False)
    format_date_axis(event_ax)

    return save_figure(fig, outdir, "figure_1_cumulative_wealth_journal")
# ============================================================
# Figure 2: drawdowns
# ============================================================

def compute_drawdowns(wealth_df):
    return wealth_df / wealth_df.cummax() - 1.0


def plot_journal_drawdowns(wealth_df, outdir="pomdp_results", plot_end_date=None):
    set_quant_journal_style()
    wealth_df = clean_time_index(wealth_df)
    dd = compute_drawdowns(wealth_df)

    xmin = dd.index.min()
    #xmax = dd.index.max()
    if plot_end_date is None:
        xmax = dd.index.max()
    else:
        xmax = pd.to_datetime(plot_end_date)

    fig, ax, event_ax = make_event_figure()

    for col in dd.columns:
        lw = 2.2 if "Forecasting" in col else 1.35
        ax.plot(dd.index, dd[col], lw=lw, label=clean_label(col))

    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_title("Drawdowns by Strategy")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{100*x:.0f}%")
    ax.grid(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        handlelength=2.0,
    )

    add_belief_event_band(event_ax)

    ax.set_xlim(xmin, xmax)
    event_ax.set_xlim(xmin, xmax)

    plt.setp(ax.get_xticklabels(), visible=False)
    format_date_axis(event_ax)

    return save_figure(fig, outdir, "figure_2_drawdowns_journal")

# ============================================================
# Figure 3: LLM belief states
# ============================================================

def plot_journal_llm_beliefs(belief_df, outdir="pomdp_results", plot_end_date=None):
    set_quant_journal_style()

    belief_df = restrict_to_belief_columns(belief_df)

    fig, ax, event_ax = make_event_figure(figsize=(7.4, 5.0))

    for col in belief_df.columns:
        lw = 2.2 if col == "Crisis" else 1.55
        ax.plot(
            belief_df.index,
            belief_df[col],
            lw=lw,
            label=col.replace("_", " "),
        )

    xmin = belief_df.index.min()
    #xmax = belief_df.index.max()

    if plot_end_date is None:
        xmax = belief_df.index.max()
    else:
        xmax = pd.to_datetime(plot_end_date)

    ax.set_xlim(xmin, xmax)
    event_ax.set_xlim(xmin, xmax)

    ax.set_title("LLM-Inferred Posterior Beliefs over Latent Market States")
    ax.set_ylabel("Posterior Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)

    plt.setp(ax.get_xticklabels(), visible=False)
    format_date_axis(event_ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=3,
        frameon=False,
        handlelength=2.0,
    )

    add_belief_event_band(event_ax)

    return save_figure(fig, outdir, "figure_3_llm_belief_state_journal")

# ============================================================
# Figure 4: Sharpe ratios
# ============================================================

def plot_journal_sharpe(perf_df, outdir="pomdp_results"):
    set_quant_journal_style()

    df = perf_df.copy()
    if "strategy" in df.columns:
        df = df.set_index("strategy")

    df = df.sort_values("Sharpe", ascending=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    labels = [clean_label(x) for x in df.index]
    values = df["Sharpe"].astype(float).values

    bars = ax.barh(labels, values, height=0.58)

    ax.set_title("Sharpe Ratio by Strategy")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")

    xmax = max(values) * 1.15
    ax.set_xlim(0, xmax)

    for bar, value in zip(bars, values):
        ax.text(
            value + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    return save_figure(fig, outdir, "figure_4_sharpe_by_strategy_journal")


# ============================================================
# Figure 5: normalized macro signals
# ============================================================

def plot_journal_macro_panel(macro_df, outdir="pomdp_results"):
    set_quant_journal_style()
    macro_df = clean_index(macro_df)

    candidate_cols = [
        "VIXY",
        "tenY",
        "USO",
        "GLD",
        "HYG",
        "LQD",
        "credit_proxy",
        "Credit",
        "credit_spread",
    ]

    cols = [c for c in candidate_cols if c in macro_df.columns]

    if not cols:
        print("No recognized macro columns found. Skipping macro panel.")
        return None

    normed = macro_df[cols].dropna().copy()
    normed = normed / normed.iloc[0]

    fig, ax, event_ax = make_event_figure()

    for col in normed.columns:
        ax.plot(normed.index, normed[col], lw=1.55, label=clean_label(col))

    ax.set_title("Normalized Macro and Market Signals Used by the Filter")
    ax.set_ylabel("Index, first observation = 1")
    ax.grid(True)

    format_date_axis(event_ax)
    plt.setp(ax.get_xticklabels(), visible=False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=3,
        frameon=False,
        handlelength=2.0,
    )

    add_belief_event_band(event_ax)

    return save_figure(fig, outdir, "figure_5_macro_signals_journal")


# ============================================================
# Generate all figures
# ============================================================

def generate_all_journal_figures(
    wealth_df,
    belief_df,
    perf_df,
    macro_df=None,
    outdir="pomdp_results",
    plot_end_date=None,
):
    paths = []

    paths.append(plot_journal_cumulative_wealth(wealth_df, outdir, plot_end_date=plot_end_date))
    paths.append(plot_journal_drawdowns(wealth_df, outdir, plot_end_date=plot_end_date))

    if belief_df is not None and len(belief_df) > 0:
        paths.append(plot_journal_llm_beliefs(belief_df, outdir, plot_end_date=plot_end_date))

    paths.append(plot_journal_sharpe(perf_df, outdir))

    if macro_df is not None:
        macro_path = plot_journal_macro_panel(macro_df, outdir)
        if macro_path is not None:
            paths.append(macro_path)

    print("\nJournal-grade figures saved:")
    for p in paths:
        print("  ", p)

    return paths

# -----------------------------
# Main
# -----------------------------

def main():
    prices = download_prices()
    print("Prices shape:", prices.shape)
    print("Start:", prices.index.min())
    print("End:", prices.index.max())
    print(prices.tail())
    rets = daily_returns(prices)

    macro = download_macro_data()

    wealth_df, weights_df, belief_df, logs = run_backtest(
        rets,
        macro,
        BacktestConfig(use_openai=False),
    )

    perf = performance_table(wealth_df)
    util = utility_table(wealth_df)


    # -------------------------------------------------
    # TABLE 5: ABLATION STUDY
    # -------------------------------------------------

    ablation_results = {}
    ablation_daily_returns = {}
    ablation_wealth = {}

    for mode in ABLATION_MODES:
        print(f"\nRunning ablation: {mode}")

        wealth_a, weights_a, belief_a, logs_a = run_backtest(
            rets,
            macro,
            BacktestConfig(use_openai=False),
            ablation_mode=mode,
        )
        ablation_daily_returns[mode] = daily_strategy_returns_from_weights(
            rets=rets,
            weights_df=weights_a,
            strategy="Forecasting_POMDP",
            names=RISKY,
        )

        ablation_wealth[mode] = wealth_a["Forecasting_POMDP"].copy()

        perf_a = performance_table(wealth_a)
        util_a = utility_table(wealth_a)

        row = perf_a.loc[
            perf_a["strategy"] == "Forecasting_POMDP"
        ].iloc[0]

        util_row = util_a.loc[
            util_a["strategy"] == "Forecasting_POMDP"
        ].iloc[0]

        ablation_results[mode] = {
            "Sharpe": row["Sharpe"],
            "Utility": util_row["mean_variance_utility"],
        }

    table5 = generate_ablation_table_5(
        ablation_results,
        outdir=OUT,
    )

    table6 = generate_parameter_sensitivity_table(
        rets=rets,
        macro=macro,
        base_config=BacktestConfig(use_openai=False),
        outdir=OUT,
    )

    table7 = generate_belief_calibration_table(
        belief_df=belief_df,
        outdir=OUT,
    )


    table8 = belief_coverage_test_table(
        belief_df=belief_df,
        outdir=OUT,
    )
    #table8 = generate_table_8_significance_daily(
    #    ablation_daily_returns=ablation_daily_returns,
    #    outdir=OUT,
    #)

    wealth_df.to_csv(f"{OUT}/wealth_paths.csv")
    weights_df.to_csv(f"{OUT}/weights_history.csv", index=False)
    belief_df.to_csv(f"{OUT}/belief_history.csv", index=False)

    perf.to_csv(f"{OUT}/table_1_performance.csv", index=False)
    util.to_csv(f"{OUT}/table_2_utility.csv", index=False)

    save_latex_table(
        perf,
        f"{OUT}/table_1_performance.tex",
        "Performance comparison of POMDP portfolio agent and benchmarks.",
        "tab:performance",
    )

    save_latex_table(
        util,
        f"{OUT}/table_2_utility.tex",
        "Mean-variance utility comparison of POMDP portfolio agent and benchmarks.",
        "tab:utility",
    )

    

    end_date = pd.Timestamp("2026-06-14").normalize()

    wealth_df = truncate_to_today_or_last_trading_day(wealth_df, end_date=end_date)
    belief_df = truncate_to_today_or_last_trading_day(belief_df, end_date=end_date)
    macro = truncate_to_today_or_last_trading_day(macro, end_date=end_date)


    print("prices max:", prices.index.max())
    print("rets max:", rets.index.max())
    print("wealth_df max:", wealth_df.index.max())
    print("belief_df max:", belief_df.index.max())
    print("macro max:", macro.index.max())  

    generate_all_journal_figures(
        wealth_df=wealth_df,
        belief_df=belief_df,
        perf_df=perf,
        macro_df=macro,
        outdir="pomdp_results",
        plot_end_date=rets.index.max()
    )

    #plot_wealth(wealth_df)
    #plot_drawdowns(wealth_df)
    #plot_beliefs(belief_df)
    #plot_sharpe_bar(perf)

    print("\nTABLE 1: PERFORMANCE")
    print(perf.to_string(index=False))

    print("\nTABLE 2: MEAN-VARIANCE UTILITY")
    print(util.to_string(index=False))

    RUN_OPENAI = os.getenv("RUN_OPENAI", "0") == "1"

    if RUN_OPENAI:
        results_by_mode = {}

        for mode in ["risk_only", "full"]:
            print(f"\nRunning OpenAI agent with disclosure mode: {mode}")

            w, wh, bh, agent_logs = run_backtest(
                rets,
                macro,
                BacktestConfig(
                    use_openai=True,
                    openai_every_n_rebalances=1,
                    disclosure_mode=mode,
                ),
            )

            p = performance_table(w)
            u = utility_table(w)

            w.to_csv(f"{OUT}/wealth_openai_{mode}.csv")
            wh.to_csv(f"{OUT}/weights_openai_{mode}.csv", index=False)
            bh.to_csv(f"{OUT}/belief_openai_{mode}.csv", index=False)

            p.to_csv(f"{OUT}/performance_openai_{mode}.csv", index=False)
            u.to_csv(f"{OUT}/utility_openai_{mode}.csv", index=False)

            save_latex_table(
                p,
                f"{OUT}/performance_openai_{mode}.tex",
                f"Performance comparison for OpenAI POMDP agent under {mode} disclosure.",
                f"tab:performance-openai-{mode}",
            )

            save_latex_table(
                u,
                f"{OUT}/utility_openai_{mode}.tex",
                f"Mean-variance utility comparison for OpenAI POMDP agent under {mode} disclosure.",
                f"tab:utility-openai-{mode}",
            )

            with open(f"{OUT}/agent_logs_{mode}.json", "w") as f:
                json.dump(agent_logs, f, indent=2)

            results_by_mode[mode] = p

        frontier = plot_privacy_frontier(results_by_mode)
        frontier.to_csv(f"{OUT}/table_3_privacy_frontier.csv", index=False)

        save_latex_table(
            frontier,
            f"{OUT}/table_3_privacy_frontier.tex",
            "Privacy-performance frontier for OpenAI POMDP agent.",
            "tab:privacy-frontier",
        )

        print("\nTABLE 3: PRIVACY FRONTIER")
        print(frontier.to_string(index=False))

    print(f"\nDone. Results saved to: {OUT}/")


if __name__ == "__main__":
    main()
