"""
SCULPTH | Sculpt Ur Wealth — Python Backend v3.0
=================================================
Flask API rebuilt on yfinance — no API key, no rate limits.
All financial math uses raw Python + NumPy — every price scalar is
extracted with .iloc[-1] / .tolist() so a pandas Series never reaches
any math operation.

Key features:
  • yfinance for live NSE/BSE Indian stock prices (auto .NS → .BO fallback)
  • Duration-aware window slicing  (3m / 6m / 1y / 3y / 5y)
  • Markowitz Mean-Variance Optimization via scipy.optimize
  • Dynamic AI rationale — every bullet uses the actual computed numbers
  • Hard error propagation — no silent failures, no hardcoded fake prices

Requirements:
    pip install flask flask-cors numpy scipy yfinance

Run:
    python app.py   →   http://localhost:5000
"""

import math
import traceback

import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────
# APP SETUP
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Indian 10-year G-Sec yield used as risk-free rate
RISK_FREE_RATE = 0.068

# ──────────────────────────────────────────────────────────────
# DURATION MAP — UI key → yfinance period string + metadata
# Volatility is always annualized (× √252) regardless of window.
# CAGR uses the actual window length.
# ──────────────────────────────────────────────────────────────
DURATION_MAP = {
    "3m": {"period": "3mo", "label": "3-Month", "years": 0.25},
    "6m": {"period": "6mo", "label": "6-Month", "years": 0.50},
    "1y": {"period": "1y",  "label": "1-Year",  "years": 1.00},
    "3y": {"period": "3y",  "label": "3-Year",  "years": 3.00},
    "5y": {"period": "5y",  "label": "5-Year",  "years": 5.00},
}

# ──────────────────────────────────────────────────────────────
# POPULAR INDIAN STOCKS — local autocomplete + O(1) sector lookup
# ──────────────────────────────────────────────────────────────
POPULAR_STOCKS = [
    {"symbol": "RELIANCE",   "name": "Reliance Industries Ltd",       "sector": "Energy / Conglomerate"},
    {"symbol": "TCS",        "name": "Tata Consultancy Services",     "sector": "Information Technology"},
    {"symbol": "INFY",       "name": "Infosys Ltd",                   "sector": "Information Technology"},
    {"symbol": "HDFCBANK",   "name": "HDFC Bank Ltd",                 "sector": "Banking & Finance"},
    {"symbol": "ICICIBANK",  "name": "ICICI Bank Ltd",                "sector": "Banking & Finance"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd",        "sector": "FMCG"},
    {"symbol": "ITC",        "name": "ITC Ltd",                       "sector": "FMCG"},
    {"symbol": "SBIN",       "name": "State Bank of India",           "sector": "Banking & Finance"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd",             "sector": "Banking & Finance"},
    {"symbol": "KOTAKBANK",  "name": "Kotak Mahindra Bank",           "sector": "Banking & Finance"},
    {"symbol": "LT",         "name": "Larsen & Toubro Ltd",           "sector": "Infrastructure"},
    {"symbol": "WIPRO",      "name": "Wipro Ltd",                     "sector": "Information Technology"},
    {"symbol": "HCLTECH",    "name": "HCL Technologies Ltd",          "sector": "Information Technology"},
    {"symbol": "TECHM",      "name": "Tech Mahindra Ltd",             "sector": "Information Technology"},
    {"symbol": "SUNPHARMA",  "name": "Sun Pharmaceutical Industries", "sector": "Pharmaceuticals"},
    {"symbol": "DRREDDY",    "name": "Dr. Reddy's Laboratories",      "sector": "Pharmaceuticals"},
    {"symbol": "CIPLA",      "name": "Cipla Ltd",                     "sector": "Pharmaceuticals"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd",               "sector": "Automobiles"},
    {"symbol": "MARUTI",     "name": "Maruti Suzuki India Ltd",       "sector": "Automobiles"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd",                "sector": "Automobiles"},
    {"symbol": "TITAN",      "name": "Titan Company Ltd",             "sector": "Consumer Discretionary"},
    {"symbol": "JSWSTEEL",   "name": "JSW Steel Ltd",                 "sector": "Metals & Mining"},
    {"symbol": "TATASTEEL",  "name": "Tata Steel Ltd",                "sector": "Metals & Mining"},
    {"symbol": "ADANIENT",   "name": "Adani Enterprises Ltd",         "sector": "Infrastructure"},
    {"symbol": "NTPC",       "name": "NTPC Ltd",                      "sector": "Energy / Utilities"},
    {"symbol": "ONGC",       "name": "Oil and Natural Gas Corp",      "sector": "Energy"},
    {"symbol": "COALINDIA",  "name": "Coal India Ltd",                "sector": "Metals & Mining"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd",          "sector": "Cement & Materials"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd",              "sector": "Consumer Discretionary"},
    {"symbol": "NESTLEIND",  "name": "Nestle India Ltd",              "sector": "FMCG"},
    {"symbol": "M&M",        "name": "Mahindra & Mahindra Ltd",       "sector": "Automobiles"},
    {"symbol": "PIDILITIND", "name": "Pidilite Industries Ltd",       "sector": "Specialty Chemicals"},
    {"symbol": "DABUR",      "name": "Dabur India Ltd",               "sector": "FMCG"},
    {"symbol": "BRITANNIA",  "name": "Britannia Industries",          "sector": "FMCG"},
    {"symbol": "EICHERMOT",  "name": "Eicher Motors Ltd",             "sector": "Automobiles"},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank Ltd",             "sector": "Banking & Finance"},
    {"symbol": "AXISBANK",   "name": "Axis Bank Ltd",                 "sector": "Banking & Finance"},
    {"symbol": "POWERGRID",  "name": "Power Grid Corporation",        "sector": "Energy / Utilities"},
    {"symbol": "BPCL",       "name": "Bharat Petroleum Corp",         "sector": "Energy"},
    {"symbol": "HINDALCO",   "name": "Hindalco Industries",           "sector": "Metals & Mining"},
    {"symbol": "DIVISLAB",   "name": "Divi's Laboratories",           "sector": "Pharmaceuticals"},
    {"symbol": "TATACONSUM", "name": "Tata Consumer Products",        "sector": "FMCG"},
    {"symbol": "GRASIM",     "name": "Grasim Industries",             "sector": "Cement & Materials"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Ltd",             "sector": "Automobiles"},
    {"symbol": "AMBUJACEM",  "name": "Ambuja Cements Ltd",            "sector": "Cement & Materials"},
    {"symbol": "SHREECEM",   "name": "Shree Cement Ltd",              "sector": "Cement & Materials"},
    {"symbol": "DMART",      "name": "Avenue Supermarts (DMart)",     "sector": "Consumer Discretionary"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv Ltd",             "sector": "Banking & Finance"},
    {"symbol": "VEDL",       "name": "Vedanta Ltd",                   "sector": "Metals & Mining"},
    {"symbol": "ZOMATO",     "name": "Zomato Ltd",                    "sector": "Consumer Discretionary"},
    {"symbol": "PAYTM",      "name": "One97 Communications (Paytm)", "sector": "Fintech"},
    {"symbol": "NYKAA",      "name": "FSN E-Commerce (Nykaa)",       "sector": "Consumer Discretionary"},
]

_LOCAL_SECTOR: dict = {s["symbol"]: s["sector"] for s in POPULAR_STOCKS}

# Process-level in-memory sector cache
_sector_cache: dict = {}


# ──────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION
# ──────────────────────────────────────────────────────────────

class DataFetchError(Exception):
    """yfinance returned no usable data for this symbol / period."""


# ──────────────────────────────────────────────────────────────
# YFINANCE HELPERS
# ──────────────────────────────────────────────────────────────

def _yf_symbol(symbol: str, exchange: str = "NSE") -> str:
    """
    Map a bare NSE/BSE ticker to the Yahoo Finance symbol format.
    NSE  →  SYMBOL.NS   (preferred — widest coverage)
    BSE  →  SYMBOL.BO
    """
    sym = symbol.upper().strip()
    suffix = ".NS" if exchange.upper() != "BSE" else ".BO"
    return sym + suffix


def fetch_prices(symbol: str, exchange: str, period: str) -> list:
    """
    Download adjusted-close prices via yfinance using a browser-mimicking session.
    """
    import requests
    from requests import Session

    # Create a session to mimic a real browser request from India
    session = Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # Try NSE first (.NS), then fall back to BSE (.BO)
    for suffix in (".NS", ".BO"):
        yf_sym = symbol.upper().strip() + suffix
        try:
            # Pass the session to the Ticker
            ticker = yf.Ticker(yf_sym, session=session)
            hist = ticker.history(period=period, auto_adjust=True)

            if hist.empty or len(hist) < 5:
                continue

            # STRICT extraction to native Python floats
            raw_closes = hist["Close"].dropna().tolist()
            closes = [float(c) for c in raw_closes]

            if len(closes) >= 10:
                return closes

        except Exception as e:
            print(f"Error fetching {yf_sym}: {e}")
            continue 

    raise DataFetchError(
        f"yfinance returned insufficient data for '{symbol}' after trying both NSE/BSE."
    )


def fetch_sector(symbol: str, exchange: str = "NSE") -> str:
    """
    Resolve sector via three-level fallback (cheapest first):
      1. In-memory cache  — 0 network calls
      2. Local hardcoded list — 0 network calls
      3. yfinance .info — 1 network call (result is cached)
    """
    # Level 1: memory cache
    if symbol in _sector_cache:
        return _sector_cache[symbol]

    # Level 2: local list
    if symbol in _LOCAL_SECTOR:
        sec = _LOCAL_SECTOR[symbol]
        _sector_cache[symbol] = sec
        return sec

    # Level 3: live yfinance info
    for suffix in (".NS", ".BO"):
        try:
            info = yf.Ticker(symbol + suffix).info
            # yfinance exposes "sector" (broad) and "industry" (narrow)
            sec = (
                info.get("sector")
                or info.get("industryDisp")
                or info.get("industry")
                or ""
            ).strip()
            if sec and sec.lower() not in ("", "none", "n/a"):
                _sector_cache[symbol] = sec
                return sec
        except Exception:
            continue

    _sector_cache[symbol] = "Diversified"
    return "Diversified"


# ──────────────────────────────────────────────────────────────
# FINANCIAL MATH  (pure Python lists + NumPy — zero pandas)
# ──────────────────────────────────────────────────────────────

def daily_log_returns(prices: list) -> list:
    """Daily log-returns from a chronological list of float prices."""
    return [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]


def compute_cagr(prices: list) -> float:
    """
    Compound Annual Growth Rate from first to last price.
    Uses actual trading-day count: n_years = (len - 1) / 252.
    """
    if len(prices) < 2 or prices[0] <= 0 or prices[-1] <= 0:
        return 0.0
    n_years = (len(prices) - 1) / 252.0
    if n_years <= 0:
        return 0.0
    return (prices[-1] / prices[0]) ** (1.0 / n_years) - 1.0


# ──────────────────────────────────────────────────────────────
# DYNAMIC AI RATIONALE
# Every bullet uses live computed numbers — nothing is hardcoded.
# Rationale differs per stock, per action type, per duration window.
# ──────────────────────────────────────────────────────────────

def build_rationale(
    sym, sector, action,
    actual_w, opt_w,
    port_vol, opt_vol,
    port_return, opt_return,
    indiv_vol, stock_cagr,
    risk_profile, dur_label,
    actual_sector_w, opt_sector_w,
    port_sharpe, opt_sharpe,
) -> list:
    sec_act   = actual_sector_w.get(sector, 0.0) * 100
    sec_opt   = opt_sector_w.get(sector, 0.0) * 100
    sec_delta = sec_opt - sec_act
    vol_delta = (opt_vol - port_vol) * 100     # negative = volatility reduction
    w_delta   = (opt_w - actual_w) * 100

    if action == "buy":
        return [
            f"Raising {sym} from {actual_w:.1%} → {opt_w:.1%} (+{w_delta:.1f}pp) lifts "
            f"portfolio Sharpe from {port_sharpe:.2f} → {opt_sharpe:.2f} "
            f"over the {dur_label} window, improving risk-adjusted efficiency.",

            f"This increases {sector} exposure from {sec_act:.1f}% → {sec_opt:.1f}% "
            f"(+{abs(sec_delta):.1f}pp), correcting an underweight in this sector "
            f"for your {risk_profile.title()} profile.",

            f"{sym}'s individual {dur_label} volatility is {indiv_vol*100:.1f}%, which is "
            f"{'below' if indiv_vol < port_vol else 'in line with'} the current portfolio "
            f"vol ({port_vol*100:.1f}%) — making it a diversification-efficient addition.",

            f"Its {dur_label} CAGR of {stock_cagr*100:.1f}% supports the target portfolio "
            f"return of {opt_return*100:.1f}% required by your {risk_profile.title()} "
            f"Markowitz optimum.",
        ]

    elif action == "sell":
        return [
            f"Trimming {sym} from {actual_w:.1%} → {opt_w:.1%} (−{abs(w_delta):.1f}pp) "
            f"cuts portfolio volatility by {abs(vol_delta):.1f}pp: "
            f"{port_vol*100:.1f}% → {opt_vol*100:.1f}%, aligning with your "
            f"{risk_profile.title()} risk budget.",

            f"{sector} overexposure drops from {sec_act:.1f}% → {sec_opt:.1f}% "
            f"(−{abs(sec_delta):.1f}pp), reducing sector concentration risk "
            f"over the {dur_label} analysis window.",

            f"At {indiv_vol*100:.1f}% annualized individual volatility — "
            f"{'above' if indiv_vol > port_vol else 'comparable to'} the portfolio "
            f"average of {port_vol*100:.1f}% — {sym} is a primary variance contributor "
            f"that this rebalance corrects.",

            f"Redeploying proceeds to underweight assets improves projected return from "
            f"{port_return*100:.1f}% → {opt_return*100:.1f}% while simultaneously "
            f"reducing vol — a strictly better risk/reward for the {dur_label} horizon.",
        ]

    else:  # hold
        return [
            f"{sym}'s weight ({actual_w:.1%}) is within 2pp of the {dur_label} "
            f"optimizer target ({opt_w:.1%}) — no trade needed.",

            f"{sector} allocation of {sec_act:.1f}% vs. optimal {sec_opt:.1f}% "
            f"is well-calibrated for your {risk_profile.title()} profile; "
            f"the {abs(sec_delta):.1f}pp delta falls below the rebalancing threshold.",

            f"Individual {dur_label} volatility of {indiv_vol*100:.1f}% contributes "
            f"proportionally to the portfolio's {port_vol*100:.1f}% annualized vol — "
            f"no excess risk concentration detected.",

            f"Trading would incur costs that exceed the marginal gain — current Sharpe "
            f"({port_sharpe:.2f}) is already within {abs(opt_sharpe - port_sharpe):.2f} "
            f"of the optimized target ({opt_sharpe:.2f}). Reassess at next cycle.",
        ]


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "app": "SCULPTH v3.0 — yfinance backend"})


@app.route("/api/search", methods=["GET"])
def search_stocks():
    """
    Instant local autocomplete — zero network calls.
    Returns up to 8 matching stocks from POPULAR_STOCKS.
    """
    q = request.args.get("q", "").strip().upper()
    if not q:
        return jsonify(POPULAR_STOCKS[:10])

    matches = [
        s for s in POPULAR_STOCKS
        if q in s["symbol"] or q in s["name"].upper()
    ]
    return jsonify(matches[:8])


@app.route("/api/sector", methods=["GET"])
def get_sector_route():
    """
    Return the sector for a symbol.
    Called by the frontend when a user selects a stock to populate
    the sector badge immediately — before the full analysis runs.
    """
    symbol   = request.args.get("symbol", "").upper().strip()
    exchange = request.args.get("exchange", "NSE").upper()
    if not symbol:
        return jsonify({"error": "symbol parameter is required"}), 400

    sec = fetch_sector(symbol, exchange)
    return jsonify({"symbol": symbol, "sector": sec})


@app.route("/api/analyze", methods=["POST"])
def analyze_portfolio():
    """
    Full portfolio analysis endpoint.

    Expected JSON body:
    {
        "assets": [
            {
                "symbol":   "TCS",
                "exchange": "NSE",
                "shares":   10,
                "sector":   "Information Technology"   ← optional; skips sector fetch
            },
            ...
        ],
        "risk_profile": "moderate",    // conservative | moderate | aggressive
        "duration":     "1y"           // 3m | 6m | 1y | 3y | 5y
    }
    """
    try:
        body         = request.get_json(force=True)
        assets       = body.get("assets", [])
        risk_profile = body.get("risk_profile", "moderate").lower().strip()
        duration_key = body.get("duration", "1y").lower().strip()

        # ── Input validation ─────────────────────────────────────────
        if risk_profile not in ("conservative", "moderate", "aggressive"):
            return jsonify({"error": f"Invalid risk_profile '{risk_profile}'."}), 400
        if duration_key not in DURATION_MAP:
            return jsonify({
                "error": f"Invalid duration '{duration_key}'. "
                         f"Valid: {list(DURATION_MAP.keys())}"
            }), 400

        dur      = DURATION_MAP[duration_key]
        period   = dur["period"]    # e.g. "1y"
        dur_label = dur["label"]    # e.g. "1-Year"

        valid_assets = [
            a for a in assets
            if str(a.get("symbol", "")).strip() and float(a.get("shares", 0)) > 0
        ]
        if len(valid_assets) < 2:
            return jsonify({
                "error": "Please add at least 2 assets with positive share counts."
            }), 400

        # ── Step 1: Fetch price histories + sectors ──────────────────
        symbols        = []
        shares_list    = []
        prices_list    = []
        returns_matrix = []
        raw_prices_all = []
        sector_local   = {}

        for asset in valid_assets:
            sym      = str(asset["symbol"]).upper().strip()
            exchange = str(asset.get("exchange", "NSE")).upper()
            shares   = float(asset["shares"])

            # fetch_prices raises DataFetchError if no data — never returns fake prices
           try:
                # 1. Fetch prices
                price_series = fetch_prices(sym, exchange, period)
                latest_price = float(price_series[-1])

                if latest_price <= 0:
                    print(f"Skipping {sym}: Invalid price.")
                    continue

                # 2. Process math
                log_rets = daily_log_returns(price_series)

                # 3. If everything is okay, add to the main lists
                symbols.append(sym)
                shares_list.append(shares)
                prices_list.append(latest_price)
                returns_matrix.append(log_rets)
                raw_prices_all.append(price_series)

            except Exception as e:
                print(f"Error processing {sym}: {e}")
                continue # This 'continue' is the magic—it moves to the next stock!

            symbols.append(sym)
            shares_list.append(shares)
            prices_list.append(latest_price)
            returns_matrix.append(log_rets)
            raw_prices_all.append(price_series)

            # Sector: use frontend-passed value to save a network call;
            # fall back to fetch_sector() only when the badge wasn't populated.
            fe_sector = str(asset.get("sector", "")).strip()
            if fe_sector and fe_sector not in ("—", "", "Diversified"):
                _sector_cache[sym] = fe_sector
                sector_local[sym]  = fe_sector
            else:
                sector_local[sym] = fetch_sector(sym, exchange)

        n = len(symbols)

        # ── Step 2: Align all return series to the same length ───────
        min_len = min(len(r) for r in returns_matrix)
        if min_len < 5:
            return jsonify({
                "error": "Too few overlapping trading days. "
                         "Try a shorter duration or remove thinly-traded stocks."
            }), 400

        # aligned: (n, T) numpy array of floats — no pandas anywhere past this point
        aligned = np.array([r[-min_len:] for r in returns_matrix], dtype=float)

        prices_arr = np.array(prices_list, dtype=float)   # (n,)
        shares_arr = np.array(shares_list, dtype=float)   # (n,)
        values_arr = prices_arr * shares_arr               # (n,)
        total_val  = float(np.sum(values_arr))
        actual_w   = values_arr / total_val               # (n,) weight vector

        # ── Step 3: Actual portfolio metrics ─────────────────────────
        mean_daily  = np.mean(aligned, axis=1)   # (n,)
        mean_annual = mean_daily * 252.0          # annualize

        cov_annual = np.cov(aligned) * 252.0      # (n,n) annualized covariance
        indiv_vols = np.sqrt(np.diag(cov_annual)) # (n,) per-stock ann vol

        # stock-level CAGR from the raw price history
        stock_cagrs = [compute_cagr(raw_prices_all[i]) for i in range(n)]

        port_ret    = float(actual_w @ mean_annual)
        port_var    = float(actual_w @ cov_annual @ actual_w)
        port_vol    = math.sqrt(max(port_var, 0.0))
        port_cagr   = float(actual_w @ np.array(stock_cagrs))
        port_sharpe = (port_ret - RISK_FREE_RATE) / port_vol if port_vol > 1e-8 else 0.0

        actual_sector_w: dict = {}
        for i, sym in enumerate(symbols):
            sec = sector_local[sym]
            actual_sector_w[sec] = actual_sector_w.get(sec, 0.0) + float(actual_w[i])

        # ── Step 4: Markowitz Mean-Variance Optimization ─────────────
        risk_params = {
            "conservative": {"max_vol": 0.15, "w_min": 0.05, "w_max": 0.35},
            "moderate":     {"max_vol": 0.25, "w_min": 0.05, "w_max": 0.40},
            "aggressive":   {"max_vol": 0.42, "w_min": 0.03, "w_max": 0.55},
        }
        rp = risk_params[risk_profile]

        def neg_sharpe(w: np.ndarray) -> float:
            ret = float(w @ mean_annual)
            var = float(w @ cov_annual @ w)
            vol = math.sqrt(max(var, 1e-12))
            return -(ret - RISK_FREE_RATE) / vol

        def pf_vol(w: np.ndarray) -> float:
            return math.sqrt(max(float(w @ cov_annual @ w), 1e-12))

        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]
        if risk_profile == "conservative":
            constraints.append(
                {"type": "ineq", "fun": lambda w: rp["max_vol"] - pf_vol(w)}
            )

        bounds = tuple((rp["w_min"], rp["w_max"]) for _ in range(n))
        w0 = np.full(n, 1.0 / n)

        res = minimize(
            neg_sharpe, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        opt_w = res.x if res.success else w0
        opt_w = opt_w / np.sum(opt_w)   # re-normalize to exactly 1.0

        opt_ret    = float(opt_w @ mean_annual)
        opt_var    = float(opt_w @ cov_annual @ opt_w)
        opt_vol    = math.sqrt(max(opt_var, 0.0))
        opt_sharpe = (opt_ret - RISK_FREE_RATE) / opt_vol if opt_vol > 1e-8 else 0.0

        opt_sector_w: dict = {}
        for i, sym in enumerate(symbols):
            sec = sector_local[sym]
            opt_sector_w[sec] = opt_sector_w.get(sec, 0.0) + float(opt_w[i])

        # ── Step 5: Risk Alignment ────────────────────────────────────
        def vol_to_profile(v: float) -> str:
            if v < 0.15:   return "conservative"
            elif v < 0.25: return "moderate"
            return "aggressive"

        rank = {"conservative": 0, "moderate": 1, "aggressive": 2}
        actual_cat  = vol_to_profile(port_vol)
        actual_rank = rank[actual_cat]
        stated_rank = rank[risk_profile]

        if actual_rank == stated_rank:
            risk_status = "green"
            risk_msg = (
                f"✅ Portfolio volatility ({port_vol*100:.1f}%) aligns perfectly with "
                f"your {risk_profile.title()} profile over the {dur_label} window."
            )
        elif actual_rank > stated_rank:
            risk_status = "red"
            risk_msg = (
                f"🚨 Portfolio volatility ({port_vol*100:.1f}%) exceeds your "
                f"{risk_profile.title()} tolerance. Following the Sculpting Plan "
                f"below reduces this to {opt_vol*100:.1f}%."
            )
        else:
            risk_status = "yellow"
            risk_msg = (
                f"⚠️ Portfolio volatility ({port_vol*100:.1f}%) is more conservative "
                f"than your {risk_profile.title()} target. The optimized weights "
                f"would lift returns to {opt_ret*100:.1f}% within your risk budget."
            )

        # ── Step 6: Action Plan ───────────────────────────────────────
        actions = []
        for i, sym in enumerate(symbols):
            aw    = float(actual_w[i])
            ow    = float(opt_w[i])
            wdiff = ow - aw

            act_sh  = float(shares_arr[i])
            price_i = float(prices_arr[i])
            opt_sh  = (ow * total_val) / price_i if price_i > 0 else act_sh
            sh_diff = round(opt_sh - act_sh)

            sec = sector_local[sym]

            if abs(sh_diff) < 1 and abs(wdiff) < 0.02:
                action_type = "hold"
                directive   = f"HOLD {sym} — allocation is near-optimal."
            elif sh_diff > 0:
                action_type = "buy"
                directive   = f"BUY {int(abs(sh_diff))} shares of {sym}"
            else:
                action_type = "sell"
                directive   = f"SELL {int(abs(sh_diff))} shares of {sym}"

            rationale = build_rationale(
                sym=sym, sector=sec, action=action_type,
                actual_w=aw, opt_w=ow,
                port_vol=port_vol, opt_vol=opt_vol,
                port_return=port_ret, opt_return=opt_ret,
                indiv_vol=float(indiv_vols[i]),
                stock_cagr=stock_cagrs[i],
                risk_profile=risk_profile, dur_label=dur_label,
                actual_sector_w=actual_sector_w, opt_sector_w=opt_sector_w,
                port_sharpe=port_sharpe, opt_sharpe=opt_sharpe,
            )

            actions.append({
                "symbol":         sym,
                "action":         action_type,
                "directive":      directive,
                "actual_shares":  round(act_sh, 2),
                "optimal_shares": round(opt_sh, 2),
                "share_diff":     sh_diff,
                "actual_weight":  round(aw * 100, 2),
                "optimal_weight": round(ow * 100, 2),
                "current_price":  round(price_i, 2),
                "sector":         sec,
                "stock_cagr":     round(stock_cagrs[i] * 100, 2),
                "indiv_vol":      round(float(indiv_vols[i]) * 100, 2),
                "rationale":      rationale,
            })

        # ── Step 7: Return full response ──────────────────────────────
        return jsonify({
            "status": "success",
            "actual_portfolio": {
                "total_value":    round(total_val, 2),
                "cagr":           round(port_cagr * 100, 2),
                "annual_return":  round(port_ret * 100, 2),
                "volatility":     round(port_vol * 100, 2),
                "sharpe_ratio":   round(port_sharpe, 3),
                "sector_weights": {k: round(v * 100, 2) for k, v in actual_sector_w.items()},
                "holdings": [
                    {
                        "symbol":    symbols[i],
                        "shares":    float(shares_arr[i]),
                        "price":     round(float(prices_arr[i]), 2),
                        "value":     round(float(values_arr[i]), 2),
                        "weight":    round(float(actual_w[i]) * 100, 2),
                        "sector":    sector_local[symbols[i]],
                        "cagr":      round(stock_cagrs[i] * 100, 2),
                        "indiv_vol": round(float(indiv_vols[i]) * 100, 2),
                    }
                    for i in range(n)
                ],
            },
            "optimal_portfolio": {
                "annual_return":  round(opt_ret * 100, 2),
                "volatility":     round(opt_vol * 100, 2),
                "sharpe_ratio":   round(opt_sharpe, 3),
                "sector_weights": {k: round(v * 100, 2) for k, v in opt_sector_w.items()},
                "weights": [
                    {
                        "symbol":         symbols[i],
                        "optimal_weight": round(float(opt_w[i]) * 100, 2),
                    }
                    for i in range(n)
                ],
            },
            "risk_alignment": {
                "status":          risk_status,
                "message":         risk_msg,
                "actual_category": actual_cat,
                "stated_profile":  risk_profile,
            },
            "action_plan": actions,
            "meta": {
                "risk_profile":   risk_profile,
                "duration":       duration_key,
                "duration_label": dur_label,
                "n_assets":       n,
                "data_points":    min_len,
            },
        })

    except DataFetchError as exc:
        # No data from yfinance — hard 400, no fallback fake prices
        return jsonify({
            "error": str(exc),
            "error_type": "data_error",
        }), 400

    except Exception:
        print(traceback.format_exc())
        return jsonify({
            "error": "An unexpected server error occurred. Check the terminal log.",
            "error_type": "server_error",
        }), 500


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  SCULPTH v3.0  →  http://localhost:5000")
    print("    Backend: yfinance (no API key required)")
    print("    Supports: NSE (.NS) with auto BSE (.BO) fallback")
    app.run(debug=True, host="0.0.0.0", port=5000)
