# pomdp_agentic_ai_portfolio_validation.py
# Run:
#   pip install yfinance openai pandas numpy scipy matplotlib tabulate
#   export OPENAI_API_KEY="YOUR_KEY"
#   python pomdp_agentic_ai_portfolio_validation.py
#
# Optional OpenAI calls:
#   RUN_OPENAI=1 python pomdp_agentic_ai_portfolio_validation.py

import os, json, time
from dataclasses import dataclass
import numpy as np
import pandas as pd
#from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from openai import OpenAI
import requests

OUT = "pomdp_results"
os.makedirs(OUT, exist_ok=True)

MASSIVE_BASE_URL = "https://api.massive.com"
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "JPM", "IBM", "GLD", "TLT", "SPY"]
RISKY = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "JPM", "IBM", "GLD", "TLT"]
BENCHMARK = "SPY"

MACRO_TICKERS = {
    "VIX": "VIXY",      # VIX futures ETF proxy; avoids index-plan 403
    "OIL": "USO",      # crude oil ETF proxy
    "GOLD": "GLD",     # gold ETF proxy
    "HYG": "HYG",      # high-yield credit ETF
    "LQD": "LQD",      # investment-grade credit ETF
}

LLM_STATES = [
    "AI_Boom",
    "Soft_Landing",
    "Inflation_Shock",
    "Recession",
    "Crisis",
]

START = "2024-06-10"
END = None
FORWARD_DAYS = 21
LOOKBACK_DAYS = 100
REBALANCE_FREQ = "ME"
RISK_AVERSION = 3.5
MODEL = "gpt-5"
SLEEP_BETWEEN_OPENAI_CALLS = 10


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
        time.sleep({SLEEP_BETWEEN_OPENAI_CALLS})

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


def forecasting_pomdp_policy_weights(mu_hist, mu_view, cov, belief, names):
    """
    Forecasting POMDP policy:
    hidden state -> return views -> BL posterior -> constrained utility optimization.
    """

    cov = np.asarray(cov)

    mu_post = black_litterman_blend(
        mu_hist=mu_hist,
        mu_view=mu_view,
        cov=cov,
        names=names,
        tau=0.25,
        view_weight=0.65,
    )

    b = pd.Series(belief, index=LLM_STATES)

    p_stress = b["Recession"] + 1.5 * b["Crisis"]
    p_growth = b["AI_Boom"] + 0.5 * b["Soft_Landing"]

    # Risk aversion rises in stress regimes
    lam = 2.0 + 8.0 * p_stress - 1.0 * p_growth
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


def get_rebalance_dates(rets):
    dates = []

    for _, group in rets.groupby(pd.Grouper(freq="ME")):
        if len(group) > 0:
            dates.append(group.index[-1])

    return dates


def heuristic_expected_returns(names, window, belief):
    """
    Deterministic fallback when config.use_openai=False.

    Produces regime-aware expected returns using:
        - momentum
        - volatility penalty
        - POMDP hidden state
    """

    b = pd.Series(
        belief,
        index=LLM_STATES,
    )

    forecasts = {}

    for ticker in names:

        r = window[ticker]

        mom_21 = (
            (1 + r.tail(21)).prod() - 1
        )

        mom_63 = (
            (1 + r.tail(63)).prod() - 1
        )

        vol = (
            r.tail(63).std()
            * np.sqrt(252)
        )

        annual_momentum = (
            0.5 * mom_21 * 12
            +
            0.5 * mom_63 * 4
        )

        forecast = (
            0.03
            +
            0.40 * annual_momentum
        )

        # Growth names
        if ticker in [
            "NVDA",
            "MSFT",
            "GOOGL",
            "AAPL",
            "AMZN",
        ]:

            forecast += (
                0.10 * b["AI_Boom"]
                +
                0.05 * b["Soft_Landing"]
                -
                0.08 * b["Recession"]
                -
                0.15 * b["Crisis"]
            )

        # Defensive names
        if ticker in [
            "GLD",
            "TLT",
        ]:

            forecast += (
                0.08 * b["Recession"]
                +
                0.12 * b["Crisis"]
            )

        # Inflation beneficiaries
        if ticker == "GLD":

            forecast += (
                0.10
                * b["Inflation_Shock"]
            )

        if ticker == "TLT":

            forecast -= (
                0.10
                * b["Inflation_Shock"]
            )

        # Value / cyclical names
        if ticker in [
            "JPM",
            "IBM",
            "XOM",
        ]:

            forecast += (
                0.05 * b["Soft_Landing"]
                +
                0.05 * b["Inflation_Shock"]
                -
                0.08 * b["Crisis"]
            )

        # Risk penalty
        forecast -= (
            0.08 * vol
        )

        forecasts[ticker] = (
            forecast / 252.0
        )

    return (
        pd.Series(forecasts)
        .reindex(names)
        .values
    )

def run_backtest(rets, macro, config):
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
            mu_view = heuristic_expected_returns(
                names,
                window,
                belief,
            )

        weights["Forecasting_POMDP"] = forecasting_pomdp_policy_weights(
            mu_hist=mu,
            mu_view=mu_view,
            cov=cov,
            belief=belief,
            names=names,
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

    return wealth_df, weights_df, belief_df, agent_logs
# -----------------------------
# Metrics
# -----------------------------

def max_drawdown(wealth):
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    return dd.min()


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


# -----------------------------
# Figures
# -----------------------------

def plot_wealth(wealth_df):
    plt.figure(figsize=(11, 6))
    for col in wealth_df.columns:
        plt.plot(wealth_df.index, wealth_df[col], label=col)
    plt.title("Cumulative wealth: POMDP agent vs benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/figure_1_cumulative_wealth.png", dpi=200)
    plt.close()


def plot_drawdowns(wealth_df):
    plt.figure(figsize=(11, 6))
    for col in wealth_df.columns:
        dd = wealth_df[col] / wealth_df[col].cummax() - 1
        plt.plot(wealth_df.index, dd, label=col)
    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/figure_2_drawdowns.png", dpi=200)
    plt.close()


def plot_beliefs(belief_df):
    if belief_df.empty:
        print("No belief history to plot.")
        return

    plt.figure(figsize=(11, 6))

    for s in LLM_STATES:
        if s in belief_df.columns:
            plt.plot(pd.to_datetime(belief_df["date"]), belief_df[s], label=s)

    plt.title("LLM-inferred POMDP hidden-state beliefs")
    plt.xlabel("Date")
    plt.ylabel("Posterior probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/figure_3_llm_belief_state.png", dpi=200)
    plt.close()


def plot_sharpe_bar(perf):
    plt.figure(figsize=(10, 5))
    plt.bar(perf["strategy"], perf["Sharpe"])
    plt.title("Sharpe ratio by strategy")
    plt.xlabel("Strategy")
    plt.ylabel("Sharpe")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figure_4_sharpe_by_strategy.png", dpi=200)
    plt.close()


def plot_privacy_frontier(results_by_mode):
    rows = []
    info_score = {"risk_only": 1, "full": 5}

    for mode, perf in results_by_mode.items():
        row = perf[perf["strategy"] == "OpenAI_POMDP_Agent"].iloc[0]
        rows.append({
            "mode": mode,
            "information_disclosure": info_score.get(mode, 3),
            "sharpe": row["Sharpe"],
            "terminal_wealth": row["TerminalWealth"],
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5))
    plt.scatter(df["information_disclosure"], df["sharpe"])
    for _, row in df.iterrows():
        plt.text(row["information_disclosure"], row["sharpe"], row["mode"])
    plt.title("Privacy-performance frontier")
    plt.xlabel("Information disclosure score")
    plt.ylabel("Agent Sharpe")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figure_5_privacy_performance_frontier.png", dpi=200)
    plt.close()

    return df


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

    plot_wealth(wealth_df)
    plot_drawdowns(wealth_df)
    plot_beliefs(belief_df)
    plot_sharpe_bar(perf)

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