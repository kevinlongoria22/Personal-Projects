"""SignalStack local data server.

Fetches real daily price data from Yahoo Finance for a tracked watchlist,
computes simple momentum/trend/volatility signals, and serves the result
as JSON to the SignalStack dashboard. No CSV export — this is purely an
internal cache the frontend reads from automatically.

Run:
    python server.py

Then the dashboard's app.js calls GET /api/watchlist to get live data.
"""

import json
import math
import time
import urllib.request
import statistics
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Watchlist broken into industry/theme groups instead of one flat "core" bucket.
# "Volatile" is still its own bucket for names with outsized swing potential
# that don't cleanly fit an industry group (not obscure micro-cap biotech —
# these all have real liquidity and news flow).
GROUPS = {
    "big_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "semiconductors": ["NVDA", "AMD", "AVGO", "ARM"],
    "pc_makers": ["DELL", "HPQ", "LNVGY"],
    "memory_storage": ["MU", "WDC", "STX"],
    "volatile": ["TSLA", "NFLX", "PLTR", "MSTR", "COIN", "SMCI"],
}
WATCHLIST = [symbol for symbols in GROUPS.values() for symbol in symbols]

# Full company names for display (Yahoo's chart API doesn't reliably return one).
COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "AVGO": "Broadcom Inc.",
    "ARM": "Arm Holdings plc",
    "DELL": "Dell Technologies Inc.",
    "HPQ": "HP Inc.",
    "LNVGY": "Lenovo Group Limited",
    "MU": "Micron Technology, Inc.",
    "WDC": "Western Digital Corporation",
    "STX": "Seagate Technology Holdings plc",
    "TSLA": "Tesla, Inc.",
    "NFLX": "Netflix, Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "MSTR": "MicroStrategy Incorporated",
    "COIN": "Coinbase Global, Inc.",
    "SMCI": "Super Micro Computer, Inc.",
}

GROUP_LABELS = {
    "big_tech": "Big Tech",
    "semiconductors": "Semiconductors",
    "pc_makers": "PC Makers",
    "memory_storage": "Memory & Storage",
    "volatile": "Volatile",
}

CACHE_FILE = Path(__file__).parent / "cache.json"
CACHE_TTL_SECONDS = 15 * 60  # refresh at most every 15 minutes

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_chart(symbol, range_="1y", interval="1d"):
    """Pull daily OHLC closes for a symbol from Yahoo Finance's public chart API.

    1y of history is enough to compute daily/weekly/monthly/6-month/YTD moves
    plus 20d/50d trend and volatility, without needing a second request.
    """
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?range={range_}&interval={interval}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.load(resp)

    result = payload["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    timestamps = result["timestamp"]
    meta = result["meta"]

    # Filter out None values (non-trading gaps)
    series = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
    return series, meta


def pct_change(a, b):
    if a in (0, None) or b is None:
        return 0.0
    return ((b - a) / a) * 100


def find_ytd_start_index(series):
    """Index of the first trading day in the current calendar year (UTC)."""
    from datetime import datetime, timezone

    current_year = datetime.now(timezone.utc).year
    for i, (ts, _) in enumerate(series):
        if datetime.fromtimestamp(ts, tz=timezone.utc).year == current_year:
            return i
    return 0  # whole series is within this year (unlikely) or fallback


def cusum_detect(returns, k_frac=0.5, h_frac=4.0, monitor_window=20):
    """Two-sided CUSUM change-point test on a daily-return series.

    This answers the actual question from the design thread: "has this
    stock's return behavior persistently shifted away from its own recent
    baseline, rather than just having one noisy day?"

    Method (textbook CUSUM):
      1. Split returns into an older `baseline` period and a recent
         `monitor` period (the last `monitor_window` days).
      2. Estimate baseline mean (mu0) and stdev (sigma0) from the baseline
         period only — this defines "normal" behavior for this stock.
      3. Standardize an allowance k = k_frac * sigma0 (how much drift to
         tolerate before it counts as suspicious) and a decision
         threshold h = h_frac * sigma0.
      4. Walk the monitor period accumulating:
           S+ = max(0, S+ + (r - mu0) - k)   -> detects sustained upward shift
           S- = min(0, S- + (r - mu0) + k)   -> detects sustained downward shift
      5. If S+ ever exceeds h, or S- ever drops below -h, that's a
         statistically flagged regime change (not just a single good/bad day).
    """
    if len(returns) < monitor_window + 15:
        return {"status": "insufficient_data", "upperStat": 0.0, "lowerStat": 0.0, "threshold": 0.0}

    baseline = returns[:-monitor_window]
    monitor = returns[-monitor_window:]

    mu0 = statistics.fmean(baseline)
    sigma0 = statistics.pstdev(baseline) or 0.01  # floor to avoid div-by-zero on dead-flat series

    k = k_frac * sigma0
    h = h_frac * sigma0

    s_pos, s_neg = 0.0, 0.0
    max_pos, max_neg = 0.0, 0.0
    for r in monitor:
        s_pos = max(0.0, s_pos + (r - mu0) - k)
        s_neg = min(0.0, s_neg + (r - mu0) + k)
        max_pos = max(max_pos, s_pos)
        max_neg = min(max_neg, s_neg)

    if max_pos > h:
        status = "upside_shift"
    elif max_neg < -h:
        status = "downside_shift"
    else:
        status = "no_change"

    return {
        "status": status,
        "upperStat": round(max_pos, 3),
        "lowerStat": round(max_neg, 3),
        "threshold": round(h, 3),
        "baselineMean": round(mu0, 3),
        "baselineStdev": round(sigma0, 3),
    }


def bayesian_regime_probabilities(returns, baseline_mean, baseline_stdev, recent_n=10):
    """Posterior probability of {bull, neutral, bear} regimes given recent returns.

    Three competing Gaussian hypotheses are calibrated to THIS stock's own
    baseline behavior (the same older-history window CUSUM uses), not a
    generic market-wide assumption:
        bull:    mean = baseline_mean + 0.5 * baseline_stdev
        neutral: mean = baseline_mean
        bear:    mean = baseline_mean - 0.5 * baseline_stdev

    A chronically volatile/high-drift name (e.g. MSTR) ends up with wider,
    differently-centered regime bands than a steady one (e.g. MSFT) because
    baseline_mean/baseline_stdev are computed per-symbol from its own recent
    history, not a shared constant.

    We compute the log-likelihood of the last `recent_n` observed daily
    returns under each hypothesis, then apply Bayes' rule with a uniform
    prior (all three equally likely beforehand) to get the posterior. This
    is a genuine Bayesian update: "given everything we know today (the
    recent return sequence), what is the probability of each scenario?"
    """
    if len(returns) < recent_n:
        recent_n = max(3, len(returns))

    recent = returns[-recent_n:]
    window_for_sigma = returns[-30:] if len(returns) >= 30 else returns
    sigma = statistics.pstdev(window_for_sigma) or 0.05
    sigma = max(sigma, 0.05)

    hypotheses = {
        "bull": baseline_mean + 0.5 * baseline_stdev,
        "neutral": baseline_mean,
        "bear": baseline_mean - 0.5 * baseline_stdev,
    }

    log_likelihoods = {}
    for name, mu in hypotheses.items():
        ll = 0.0
        for r in recent:
            ll += -0.5 * math.log(2 * math.pi * sigma ** 2) - ((r - mu) ** 2) / (2 * sigma ** 2)
        log_likelihoods[name] = ll

    # Log-sum-exp normalization (numerically stable posterior with a uniform prior)
    max_ll = max(log_likelihoods.values())
    exp_vals = {name: math.exp(ll - max_ll) for name, ll in log_likelihoods.items()}
    total = sum(exp_vals.values())
    return {name: round(v / total, 4) for name, v in exp_vals.items()}


def compute_signal(series, meta, group):
    """Turn a closing-price series into the metrics the dashboard displays.

    All "move" fields are simple point-to-point percent changes:
      dailyMove   = last close vs. previous close (1 trading day)
      weeklyMove  = last close vs. close 5 trading days ago (~1 calendar week)
      monthlyMove = last close vs. close ~22 trading days ago (~1 calendar month)
      sixMonthMove = last close vs. close ~126 trading days ago (~6 calendar months)
      ytdMove     = last close vs. first close of the current calendar year
    """
    closes = [c for _, c in series]
    if len(closes) < 22:
        raise ValueError("not enough history")

    price = closes[-1]
    prev_day = closes[-2]
    week_ago = closes[-6] if len(closes) >= 6 else closes[0]
    month_ago = closes[-22] if len(closes) >= 22 else closes[0]
    six_month_ago = closes[-126] if len(closes) >= 126 else closes[0]
    ytd_index = find_ytd_start_index(series)
    ytd_start_price = closes[ytd_index]

    daily_move = pct_change(prev_day, price)
    weekly_move = pct_change(week_ago, price)
    monthly_move = pct_change(month_ago, price)
    six_month_move = pct_change(six_month_ago, price)
    ytd_move = pct_change(ytd_start_price, price)

    # --- Trend: 20-day vs 50-day (or shorter if not enough history) ---
    short_window = closes[-20:]
    long_window = closes[-50:] if len(closes) >= 50 else closes
    sma_short = statistics.fmean(short_window)
    sma_long = statistics.fmean(long_window)
    trend_strength = pct_change(sma_long, sma_short)  # positive = short MA above long MA

    # --- Full daily-return series (used by CUSUM + Bayesian regime model) ---
    all_returns = [pct_change(closes[i - 1], closes[i]) for i in range(1, len(closes))]

    # --- Volatility: stdev of daily returns over the last 20 days ---
    returns_20d = all_returns[-20:] if len(all_returns) >= 20 else all_returns
    volatility = statistics.pstdev(returns_20d) if len(returns_20d) > 1 else 0.0

    # --- Distance from 3-month high/low ---
    recent_window = closes[-63:] if len(closes) >= 63 else closes
    period_high = max(recent_window)
    period_low = min(recent_window)
    from_high = pct_change(period_high, price)  # negative or zero
    from_low = pct_change(period_low, price)  # positive or zero

    # --- CUSUM regime-change detection (real statistical test, not a proxy) ---
    cusum = cusum_detect(all_returns)

    # --- Bayesian posterior over bull/neutral/bear regimes ---
    # Calibrate the hypothesis means/spread to this stock's own baseline
    # history (same window CUSUM uses) instead of a fixed generic assumption.
    if cusum["status"] != "insufficient_data":
        baseline_mean, baseline_stdev = cusum["baselineMean"], cusum["baselineStdev"]
    else:
        baseline_mean = statistics.fmean(all_returns)
        baseline_stdev = statistics.pstdev(all_returns) or 0.05
    regime_probs = bayesian_regime_probabilities(all_returns, baseline_mean, baseline_stdev)
    bull_probability = regime_probs["bull"]

    # --- Composite score (0-100): momentum + trend + drawdown + Bayesian regime ---
    momentum_score = max(0, min(100, 50 + weekly_move * 2.2))
    trend_score = max(0, min(100, 50 + trend_strength * 6))
    drawdown_score = max(0, min(100, 100 + from_high * 2.5))
    regime_score = bull_probability * 100
    # Volatility is NOT one of the weighted factors above — it's applied as a
    # direct penalty subtracted after the weighted blend (capped at -30).
    vol_penalty = max(0, min(30, volatility * 3))

    score = round(
        momentum_score * 0.25
        + trend_score * 0.20
        + drawdown_score * 0.15
        + regime_score * 0.40
        - vol_penalty
    )

    # CUSUM acts as a confirming/disconfirming adjustment on top of the Bayesian score,
    # since it's a different (frequentist) test of the same underlying question.
    if cusum["status"] == "upside_shift":
        score += 6
    elif cusum["status"] == "downside_shift":
        score -= 8
    score = max(0, min(100, score))

    # --- Expected range: a real statistical projection, not a score-derived guess ---
    # Random-walk approximation: over `horizon` trading days, cumulative mean drift
    # scales linearly (mean * horizon) and cumulative stdev scales with sqrt(horizon)
    # (variance of independent daily returns is additive). This gives an explicit,
    # ~1-month-ahead, ±1 standard-deviation range built from this stock's own actual
    # daily-return mean and its recent (20d) volatility — not from the composite score.
    horizon_days = 21  # ~1 trading month
    mean_daily_return = statistics.fmean(all_returns)
    projected_mean = mean_daily_return * horizon_days
    projected_stdev = volatility * math.sqrt(horizon_days)
    expected_range = {
        "horizonTradingDays": horizon_days,
        "low": round(projected_mean - projected_stdev, 1),
        "high": round(projected_mean + projected_stdev, 1),
        "projectedMean": round(projected_mean, 1),
    }

    if score >= 78:
        action = "BUY"
    elif score >= 58:
        action = "WAIT"
    else:
        action = "AVOID"

    if volatility >= 3.2:
        risk = "High"
    elif volatility >= 1.8:
        risk = "Moderate"
    else:
        risk = "Low"

    top_regime = max(regime_probs, key=regime_probs.get)
    top_probability = regime_probs[top_regime]
    if top_probability >= 0.55 and cusum["status"] != "no_change":
        confidence = "High"
    elif top_probability >= 0.42 or cusum["status"] != "no_change":
        confidence = "Medium"
    else:
        confidence = "Low"

    # Trailing 28-trading-day window, sliced into 5-trading-day "weeks."
    # These are rolling 5-day blocks within the last 28 trading days — a
    # relative Week 1..5 inside that trailing window, NOT calendar weeks of
    # the current month (the month may not be over; this is just "how did
    # the last ~6 weeks of trading break down").
    trailing_blocks = []
    chunk = closes[-28:] if len(closes) >= 28 else closes
    for i in range(0, len(chunk) - 5, 5):
        start, end = chunk[i], chunk[min(i + 5, len(chunk) - 1)]
        trailing_blocks.append(round(pct_change(start, end), 2))

    return {
        "symbol": meta["symbol"],
        "companyName": COMPANY_NAMES.get(meta["symbol"], meta["symbol"]),
        "group": group,
        "price": round(price, 2),
        "currency": meta.get("currency", "USD"),
        "dailyMove": round(daily_move, 2),
        "weeklyMove": round(weekly_move, 2),
        "monthlyMove": round(monthly_move, 2),
        "sixMonthMove": round(six_month_move, 2),
        "ytdMove": round(ytd_move, 2),
        "trendStrength": round(trend_strength, 2),
        "volatility": round(volatility, 2),
        "volPenalty": round(vol_penalty, 1),
        "distanceFromHigh": round(from_high, 2),
        "distanceFromLow": round(from_low, 2),
        "cusum": cusum,
        "expectedRange": expected_range,
        "regimeProbabilities": regime_probs,
        "score": score,
        "action": action,
        "risk": risk,
        "confidence": confidence,
        "trailingBlocks": trailing_blocks,
        "updatedAt": int(time.time()),
    }


def build_watchlist_payload():
    data = {}
    for group, symbols in GROUPS.items():
        for symbol in symbols:
            try:
                series, meta = fetch_chart(symbol)
                data[symbol] = compute_signal(series, meta, group)
            except Exception as exc:  # keep going even if one symbol fails
                data[symbol] = {"symbol": symbol, "group": group, "error": str(exc)}
    return {
        "generatedAt": int(time.time()),
        "groups": GROUPS,
        "groupLabels": GROUP_LABELS,
        "stocks": data,
    }


def get_cached_payload():
    if CACHE_FILE.exists():
        try:
            cached = json.loads(CACHE_FILE.read_text())
            if time.time() - cached.get("generatedAt", 0) < CACHE_TTL_SECONDS:
                return cached
        except Exception:
            pass

    payload = build_watchlist_payload()
    CACHE_FILE.write_text(json.dumps(payload, indent=2))
    return payload


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/watchlist"):
            force = "refresh=1" in self.path
            if force and CACHE_FILE.exists():
                CACHE_FILE.unlink()
            try:
                payload = get_cached_payload()
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, format, *args):
        pass  # keep console quiet


if __name__ == "__main__":
    port = 8787
    server = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"SignalStack data server running at http://localhost:{port}/api/watchlist")
    server.serve_forever()
