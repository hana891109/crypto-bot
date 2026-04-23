"""
analysis_engine.py v8.0
========================
專業級交易系統

新增四大系統：
  1. 歷史回測系統（驗證過去訊號實際勝率）
  2. 自適應參數（依市場環境動態調整門檻）
  3. 機器學習評分（Logistic Regression，越用越準）
  4. 風控系統（連續虧損自動暫停）
  5. 進場信心度改為 0~100% 實際勝率預測

目標勝率：75-85%（含回測驗證）
"""

import time
import math
import json
import os
import requests
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────
TOP30_COINS = [
    "BTC", "ETH", "BNB", "XRP", "SOL",
    "ADA", "DOGE", "TRX", "AVAX", "SHIB",
    "DOT", "LINK", "MATIC", "UNI", "ICP",
    "LTC", "APT", "NEAR", "ATOM", "XLM",
    "FIL", "HBAR", "ARB", "OP", "INJ",
    "SUI", "VET", "GRT", "AAVE", "MKR",
]

COIN_TIMEFRAME = {
    "BTC": "4h", "ETH": "4h", "BNB": "4h", "XRP": "4h", "SOL": "4h",
    "ADA": "4h", "DOGE": "4h", "TRX": "4h", "AVAX": "4h", "SHIB": "1h",
    "DOT": "4h", "LINK": "4h", "MATIC": "1h", "UNI": "1h", "ICP": "1h",
    "LTC": "4h", "APT": "1h", "NEAR": "1h", "ATOM": "1h", "XLM": "1h",
    "FIL": "1h", "HBAR": "1h", "ARB": "1h", "OP": "1h", "INJ": "1h",
    "SUI": "1h", "VET": "1h", "GRT": "1h", "AAVE": "4h", "MKR": "4h",
}

UPPER_TIMEFRAME = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION   = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
EPS             = 1e-10

# Binance 備用域名（依序嘗試，解決 451 地理限制）
BINANCE_KLINE_URLS = [
    "https://api.binance.com/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
    "https://api2.binance.com/api/v3/klines",
    "https://api3.binance.com/api/v3/klines",
    "https://api4.binance.com/api/v3/klines",
]
FUTURES_BASE_URLS = [
    "https://fapi.binance.com/fapi/v1",
    "https://fapi1.binance.com/fapi/v1",
    "https://fapi2.binance.com/fapi/v1",
]
# 保留舊變數名相容性
BINANCE_URL = BINANCE_KLINE_URLS[0]
FUTURES_URL = FUTURES_BASE_URLS[0]

# 動態門檻（會被自適應系統調整）
_adaptive_params = {
    "min_score_diff":  4,
    "min_score_total": 7,
    "min_adx":         15,
    "min_rr":          1.5,
    "vol_threshold":   1.5,
}

# 防重複
_signal_cache: dict = {}

# ──────────────────────────────────────────────
# 系統4：風控系統
# ──────────────────────────────────────────────
_risk_control = {
    "consecutive_losses": 0,   # 連續虧損次數
    "max_consecutive_losses": 3,  # 超過此數自動暫停
    "paused": False,
    "total_signals": 0,
    "winning_signals": 0,
    "total_pnl_pct": 0.0,
}

RISK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_control.json")
ML_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_weights.json")


def load_risk_control():
    """從檔案載入風控狀態（跨重啟保存）"""
    global _risk_control
    try:
        if os.path.exists(RISK_FILE):
            with open(RISK_FILE, "r") as f:
                data = json.load(f)
                _risk_control.update(data)
    except Exception:
        pass


def save_risk_control():
    """儲存風控狀態"""
    try:
        with open(RISK_FILE, "w") as f:
            json.dump(_risk_control, f)
    except Exception:
        pass


def record_trade_result(win: bool, pnl_pct: float = 0.0):
    """記錄交易結果，更新風控狀態"""
    global _risk_control
    _risk_control["total_signals"] += 1
    _risk_control["total_pnl_pct"] += pnl_pct

    if win:
        _risk_control["winning_signals"] += 1
        _risk_control["consecutive_losses"] = 0
    else:
        _risk_control["consecutive_losses"] += 1

    # 連續虧損超過閾值 → 暫停
    if _risk_control["consecutive_losses"] >= _risk_control["max_consecutive_losses"]:
        _risk_control["paused"] = True
        print(f"⚠️ 風控：連續虧損{_risk_control['consecutive_losses']}次，系統自動暫停！")

    save_risk_control()


def reset_risk_pause():
    """手動恢復（由 bot 指令觸發）"""
    _risk_control["paused"] = False
    _risk_control["consecutive_losses"] = 0
    save_risk_control()


def get_risk_status() -> dict:
    total = _risk_control["total_signals"]
    wins  = _risk_control["winning_signals"]
    wr    = round(wins / max(total, 1) * 100, 1)
    return {
        "paused":             _risk_control["paused"],
        "consecutive_losses": _risk_control["consecutive_losses"],
        "total_signals":      total,
        "winning_signals":    wins,
        "actual_winrate":     wr,
        "total_pnl_pct":      round(_risk_control["total_pnl_pct"], 2),
    }


# ──────────────────────────────────────────────
# 系統3：機器學習評分（Logistic Regression）
# ──────────────────────────────────────────────
# 特徵向量：[score_diff, adx, rr, has_vol, has_candle,
#            upper_agree, has_breakout, is_trending,
#            rsi_ok, macd_ok, divergence]
# 權重由歷史結果更新（梯度下降）

_ml_weights = {
    "w": [0.15, 0.12, 0.18, 0.08, 0.08, 0.10, 0.12, 0.10, 0.05, 0.07, 0.08],
    "b": -3.5,
    "lr": 0.01,
    "samples": 0,
}


def load_ml_weights():
    global _ml_weights
    try:
        if os.path.exists(ML_FILE):
            with open(ML_FILE, "r") as f:
                data = json.load(f)
                _ml_weights.update(data)
    except Exception:
        pass


def save_ml_weights():
    try:
        with open(ML_FILE, "w") as f:
            json.dump(_ml_weights, f)
    except Exception:
        pass


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    except Exception:
        return 0.5


def ml_predict_winrate(features: list) -> float:
    """
    Logistic Regression 預測勝率
    回傳 0~100 的勝率百分比
    """
    w   = _ml_weights["w"]
    b   = _ml_weights["b"]
    n   = min(len(w), len(features))
    dot = sum(w[i] * features[i] for i in range(n)) + b
    prob = _sigmoid(dot)

    # 限制在合理範圍（不能太極端）
    prob = max(0.35, min(0.92, prob))
    return round(prob * 100, 1)


def ml_update(features: list, win: bool):
    """
    用單筆交易結果更新權重（線上學習）
    梯度下降更新 Logistic Regression
    """
    y   = 1.0 if win else 0.0
    w   = _ml_weights["w"]
    b   = _ml_weights["b"]
    lr  = _ml_weights["lr"]
    n   = min(len(w), len(features))

    dot  = sum(w[i] * features[i] for i in range(n)) + b
    pred = _sigmoid(dot)
    err  = pred - y   # 預測誤差

    # 更新權重
    for i in range(n):
        w[i] -= lr * err * features[i]
    _ml_weights["b"] -= lr * err
    _ml_weights["samples"] += 1

    # 學習率衰減（樣本越多，學習率越小，越穩定）
    samples = _ml_weights["samples"]
    _ml_weights["lr"] = max(0.001, 0.01 / (1 + samples * 0.01))

    save_ml_weights()


def build_ml_features(score_diff: int, adx: float, rr: float,
                      has_vol: bool, has_candle: bool, upper_agree: bool,
                      has_breakout: bool, is_trending: bool,
                      rsi: float, direction: str,
                      hist: float, macd: float, macd_sig: float,
                      has_divergence: bool) -> list:
    """建立機器學習特徵向量（正規化到 0~1）"""
    rsi_ok  = float((rsi < 45 and direction == "long") or
                    (rsi > 55 and direction == "short"))
    macd_ok = float((hist > 0 and macd > macd_sig and direction == "long") or
                    (hist < 0 and macd < macd_sig and direction == "short"))
    return [
        min(score_diff / 20.0, 1.0),   # score_diff 正規化
        min(adx / 60.0, 1.0),           # ADX 正規化
        min(rr / 5.0, 1.0),             # 風報比正規化
        float(has_vol),
        float(has_candle),
        float(upper_agree),
        float(has_breakout),
        float(is_trending),
        rsi_ok,
        macd_ok,
        float(has_divergence),
    ]


# ──────────────────────────────────────────────
# 系統2：自適應參數
# ──────────────────────────────────────────────

def update_adaptive_params(recent_winrate: float, market_volatility: float):
    """
    根據近期勝率和市場波動動態調整門檻
    勝率高 → 可以稍微放寬，捕捉更多機會
    勝率低 → 收緊門檻，只做最強的訊號
    波動高 → 提高 ADX 門檻，避免假訊號
    """
    global _adaptive_params

    # 根據勝率調整評分門檻
    if recent_winrate >= 75:
        _adaptive_params["min_score_diff"]  = 3    # 稍微放寬
        _adaptive_params["min_score_total"] = 6
    elif recent_winrate >= 60:
        _adaptive_params["min_score_diff"]  = 4    # 標準
        _adaptive_params["min_score_total"] = 7
    else:
        _adaptive_params["min_score_diff"]  = 5    # 嚴格
        _adaptive_params["min_score_total"] = 9

    # 根據波動率調整 ADX 門檻
    if market_volatility > 0.05:   # 高波動
        _adaptive_params["min_adx"]         = 20
        _adaptive_params["vol_threshold"]   = 2.0
    elif market_volatility > 0.02:  # 中波動
        _adaptive_params["min_adx"]         = 15
        _adaptive_params["vol_threshold"]   = 1.5
    else:                            # 低波動
        _adaptive_params["min_adx"]         = 12
        _adaptive_params["vol_threshold"]   = 1.3


def get_market_volatility(symbol: str = "BTC") -> float:
    """計算市場波動率（BTC 的 ATR/Price）"""
    try:
        data = fetch_klines(symbol, "1d", 14)
        if data is None or len(data) < 10:
            return 0.03
        highs  = data[:, 2]
        lows   = data[:, 3]
        closes = data[:, 4]
        atr    = calc_atr(highs, lows, closes, 14)
        price  = float(closes[-1])
        return float(atr / max(price, EPS))
    except Exception:
        return 0.03


# ──────────────────────────────────────────────
# 系統1：歷史回測
# ──────────────────────────────────────────────

def backtest_symbol(symbol: str, periods: int = 100) -> dict:
    """
    對單一幣種進行歷史回測
    用過去 K 線模擬訊號，計算勝率

    勝利條件：先到達 TP1（止盈1）
    失敗條件：先到達 止損
    """
    timeframe = select_timeframe(symbol)
    data      = fetch_klines(symbol, timeframe, periods + 60)

    if data is None or len(data) < 80:
        return {"symbol": symbol, "winrate": 50.0, "trades": 0, "error": "資料不足"}

    wins   = 0
    losses = 0
    trades = []

    # 滑動窗口回測（每次用前 60 根計算訊號，後續根驗證結果）
    for start in range(0, min(periods, len(data) - 30), 5):
        end = start + 60
        if end + 10 > len(data):
            break

        hist_data = data[start:end]
        future    = data[end:end+20]   # 未來20根驗證

        try:
            h = hist_data[:, 2]; l = hist_data[:, 3]
            c = hist_data[:, 4]; v = hist_data[:, 5]
            o = hist_data[:, 1]

            price = float(c[-1])
            rsi   = calc_rsi(c)
            macd, ms, hist_val = calc_macd(c)
            ema20 = float(calc_ema(c, 20)[-1])
            ema50 = float(calc_ema(c, 50)[-1])
            ema200= float(calc_ema(c, min(200, max(2, len(c)-1)))[-1])
            atr   = calc_atr(h, l, c)
            adx_r = calc_adx(h, l, c)

            if adx_r["adx"] < _adaptive_params["min_adx"]:
                continue

            long_s = short_s = 0
            if rsi < 30:          long_s  += 2
            elif rsi < 45:        long_s  += 1
            elif rsi > 70:        short_s += 2
            elif rsi > 55:        short_s += 1
            if hist_val > 0 and macd > ms: long_s  += 2
            elif hist_val < 0 and macd < ms: short_s += 2
            if price > ema20 > ema50 > ema200: long_s += 3
            elif price < ema20 < ema50 < ema200: short_s += 3

            diff = abs(long_s - short_s)
            if diff < _adaptive_params["min_score_diff"]:
                continue
            if max(long_s, short_s) < _adaptive_params["min_score_total"]:
                continue

            direction = "long" if long_s > short_s else "short"

            swing_high = float(h[-50:].max()) if len(h) >= 50 else float(h.max())
            swing_low  = float(l[-50:].min()) if len(l) >= 50 else float(l.min())
            exits = get_fib_exits(swing_high, swing_low, price, direction)

            if exits["risk_reward"] < _adaptive_params["min_rr"]:
                continue

            tp1 = exits["tp1"]
            sl  = exits["stop_loss"]

            # 驗證：看未來20根哪個先被觸及
            won = None
            for k in range(len(future)):
                fh = float(future[k, 2])
                fl = float(future[k, 3])
                if direction == "long":
                    if fh >= tp1: won = True;  break
                    if fl <= sl:  won = False; break
                else:
                    if fl <= tp1: won = True;  break
                    if fh >= sl:  won = False; break

            if won is True:
                wins += 1
                trades.append({"result": "win", "rr": exits["risk_reward"]})
            elif won is False:
                losses += 1
                trades.append({"result": "loss", "rr": exits["risk_reward"]})

        except Exception:
            continue

    total = wins + losses
    wr    = round(wins / max(total, 1) * 100, 1)

    return {
        "symbol":   symbol,
        "winrate":  wr,
        "trades":   total,
        "wins":     wins,
        "losses":   losses,
    }


def quick_backtest(symbols: list = None, periods: int = 80) -> dict:
    """
    快速回測多個幣種，回傳整體勝率
    用於自適應參數更新
    """
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "XRP", "BNB"]

    all_wins = all_total = 0
    results  = []

    for sym in symbols:
        r = backtest_symbol(sym, periods)
        if r["trades"] >= 3:
            all_wins  += r["wins"]
            all_total += r["trades"]
            results.append(r)

    overall_wr = round(all_wins / max(all_total, 1) * 100, 1)

    return {
        "overall_winrate": overall_wr,
        "total_trades":    all_total,
        "details":         results,
    }


# ──────────────────────────────────────────────
# K 線抓取
# ──────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "4h",
                 limit: int = 200, retries: int = 2) -> Optional[np.ndarray]:
    """
    自動嘗試所有備用域名，解決 Binance 451 地理限制問題
    每個域名最多重試 retries 次
    """
    pair   = f"{symbol}USDT"
    params = {"symbol": pair, "interval": interval, "limit": limit}

    for url in BINANCE_KLINE_URLS:
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=10)

                # 451 = 地理限制，直接換下一個域名
                if resp.status_code == 451:
                    print(f"  🌐 {url} 地理限制(451)，嘗試備用域名...")
                    break

                resp.raise_for_status()
                raw = resp.json()

                if not raw or isinstance(raw, dict):
                    return None

                data = np.array(
                    [[float(k[0]), float(k[1]), float(k[2]),
                      float(k[3]), float(k[4]), float(k[5])]
                     for k in raw],
                    dtype=np.float64
                )

                if data.ndim != 2 or data.shape[1] < 6:
                    return None
                if np.isnan(data).any() or np.isinf(data).any():
                    return None
                if data[:, 5].sum() < EPS:
                    return None

                return data   # ✅ 成功

            except requests.exceptions.HTTPError as e:
                if "451" in str(e):
                    print(f"  🌐 {url} 地理限制，換域名...")
                    break
                print(f"  ⚠️ {symbol} {interval} [{url}] 第{attempt+1}次失敗：{e}")
                if attempt < retries - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"  ⚠️ {symbol} {interval} [{url}] 第{attempt+1}次失敗：{e}")
                if attempt < retries - 1:
                    time.sleep(1)

    print(f"  ❌ {symbol} {interval} 所有域名均失敗")
    return None


def select_timeframe(symbol: str) -> str:
    return COIN_TIMEFRAME.get(symbol.upper(), "1h")


def is_duplicate_signal(symbol: str, direction: str,
                        price: float, atr: float) -> bool:
    key        = f"{symbol}_{direction}"
    last_price = _signal_cache.get(key)
    if last_price is None:
        _signal_cache[key] = price
        return False
    if abs(price - last_price) > atr * 1.5:
        _signal_cache[key] = price
        return False
    return True


# ──────────────────────────────────────────────
# 技術指標
# ──────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes.astype(np.float64))
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
    if avg_loss < EPS:
        return 100.0
    return round(100.0 - (100.0 / (1.0 + avg_gain / avg_loss)), 2)


def calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
    closes = closes.astype(np.float64)
    ema    = np.zeros_like(closes)
    k      = 2.0 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def calc_macd(closes: np.ndarray):
    ml   = calc_ema(closes, 12) - calc_ema(closes, 26)
    ms   = calc_ema(ml, 9)
    mh   = ml - ms
    return float(ml[-1]), float(ms[-1]), float(mh[-1])


def calc_bollinger(closes: np.ndarray, period: int = 20, sd: float = 2.0):
    if len(closes) < period:
        m = float(closes[-1])
        return m, m, m
    w = closes[-period:].astype(np.float64)
    m = float(w.mean())
    s = float(w.std())
    return m + sd * s, m, m - sd * s


def calc_atr(highs, lows, closes, period: int = 14) -> float:
    if len(closes) < 2:
        return max(float(abs(highs[-1] - lows[-1])) if len(highs) > 0 else 1.0, EPS)
    n   = min(len(closes), len(highs), len(lows))
    trs = []
    for i in range(1, n):
        tr = max(
            float(highs[i])  - float(lows[i]),
            abs(float(highs[i])  - float(closes[i-1])),
            abs(float(lows[i])   - float(closes[i-1]))
        )
        trs.append(max(tr, EPS))
    if not trs:
        return 1.0
    arr = np.array(trs[-period:] if len(trs) >= period else trs)
    return float(arr.mean())


def calc_vwap(highs, lows, closes, volumes) -> float:
    tp  = (highs.astype(np.float64) + lows.astype(np.float64) + closes.astype(np.float64)) / 3.0
    tv  = float(np.sum(volumes))
    if tv < EPS:
        return float(closes[-1])
    return round(float(np.sum(tp * volumes.astype(np.float64)) / tv), 8)


def calc_adx(highs, lows, closes, period: int = 14) -> dict:
    n = min(len(closes), len(highs), len(lows))
    if n < period * 2 + 1:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️", "adx_score": -3, "pdi": 0.0, "mdi": 0.0}

    plus_dm = []; minus_dm = []; trs = []
    for i in range(1, n):
        hd = float(highs[i]) - float(highs[i-1])
        ld = float(lows[i-1]) - float(lows[i])
        plus_dm.append(hd  if (hd > ld  and hd > 0)  else 0.0)
        minus_dm.append(ld if (ld > hd  and ld > 0)  else 0.0)
        tr = max(float(highs[i]-lows[i]),
                 abs(float(highs[i]-closes[i-1])),
                 abs(float(lows[i]-closes[i-1])))
        trs.append(max(tr, EPS))

    def ws(arr, p):
        if len(arr) < p:
            return [max(sum(arr), EPS)]
        result = [sum(arr[:p])]
        for v in arr[p:]:
            result.append(result[-1] - result[-1]/p + v)
        return result

    a14 = ws(trs, period); p14 = ws(plus_dm, period); m14 = ws(minus_dm, period)
    ml  = min(len(a14), len(p14), len(m14))
    dx_list = []; lp = lm = 0.0

    for i in range(ml):
        a   = max(a14[i], EPS)
        pdi = 100.0 * p14[i] / a
        mdi = 100.0 * m14[i] / a
        s   = max(pdi + mdi, EPS)
        dx_list.append(100.0 * abs(pdi - mdi) / s)
        lp = pdi; lm = mdi

    if not dx_list:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️", "adx_score": -3, "pdi": 0.0, "mdi": 0.0}

    adx = round(float(np.mean(dx_list[-period:])), 2)

    if adx >= 40:   s, sc = "強趨勢🔥",  3
    elif adx >= 25: s, sc = "趨勢中🟡",  1
    elif adx >= 15: s, sc = "弱趨勢🟠",  0
    else:           s, sc = "橫盤⚪️",   -3

    return {"adx": adx, "trend_strength": s, "adx_score": sc,
            "pdi": round(lp, 2), "mdi": round(lm, 2)}


# ──────────────────────────────────────────────
# 市場結構
# ──────────────────────────────────────────────

def detect_market_structure(highs, lows, closes, atr: float) -> dict:
    if len(closes) < 30:
        return {"structure": "unknown", "label": "結構不明⚪️", "strategy": "觀望"}

    h = highs[-30:].astype(np.float64)
    l = lows[-30:].astype(np.float64)

    sh, sl = [], []
    for i in range(2, len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]:
            sh.append(float(h[i]))
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]:
            sl.append(float(l[i]))

    if len(sh) >= 2 and len(sl) >= 2:
        hh = bool(sh[-1] > sh[-2]); hl = bool(sl[-1] > sl[-2])
        lh = bool(sh[-1] < sh[-2]); ll = bool(sl[-1] < sl[-2])
        if hh and hl: return {"structure": "trending_up",   "label": "多頭趨勢📈", "strategy": "順勢做多，回調進場"}
        if lh and ll: return {"structure": "trending_down",  "label": "空頭趨勢📉", "strategy": "順勢做空，反彈進場"}
        if hh and ll: return {"structure": "expanding",      "label": "擴張震盪⚡", "strategy": "觀望，等待方向確認"}

    pr = float(h.max() - h.min()) if len(h) > 0 else 0
    if pr < atr * 8:
        return {"structure": "ranging", "label": "震盪整理↔️", "strategy": "等待突破"}

    return {"structure": "unknown", "label": "結構不明⚪️", "strategy": "觀望"}


# ──────────────────────────────────────────────
# 突破偵測（收盤確認）
# ──────────────────────────────────────────────

def detect_breakout(highs, lows, closes, volumes, atr: float) -> dict:
    if len(closes) < 25:
        return {"breakout": None, "breakout_label": "無突破", "breakout_level": None}

    cc  = float(closes[-2])
    lv  = float(volumes[-2])
    av  = float(volumes[-22:-2].mean()) if len(volumes) >= 22 else float(volumes[:-2].mean() + EPS)
    vr  = lv / max(av, EPS)
    rh  = float(highs[-22:-2].max())  if len(highs) >= 22 else float(highs[:-2].max())
    rl  = float(lows[-22:-2].min())   if len(lows)  >= 22 else float(lows[:-2].min())

    vol_ok = _adaptive_params["vol_threshold"]
    bull = bool(cc > rh and vr >= vol_ok and (cc - rh) >= atr * 0.3)
    bear = bool(cc < rl  and vr >= vol_ok and (rl - cc) >= atr * 0.3)

    if bull: return {"breakout": "bullish", "breakout_label": f"🚀 看漲突破（量比{vr:.1f}x）", "breakout_level": round(rh, 8)}
    if bear: return {"breakout": "bearish", "breakout_label": f"💥 看跌突破（量比{vr:.1f}x）", "breakout_level": round(rl, 8)}
    return {"breakout": None, "breakout_label": "無突破", "breakout_level": None}


# ──────────────────────────────────────────────
# 最終否決機制
# ──────────────────────────────────────────────

def check_veto(direction: str, rsi: float, macd: float, macd_sig: float,
               hist: float, adx: float, funding_rate: float,
               market_structure: str, upper_trend: Optional[str]) -> dict:
    vetoes = []

    if adx < _adaptive_params["min_adx"]:
        vetoes.append(f"ADX={adx:.1f}<{_adaptive_params['min_adx']}趨勢太弱")
    if direction == "long"  and rsi > 80: vetoes.append(f"RSI={rsi}嚴重超買")
    if direction == "short" and rsi < 20: vetoes.append(f"RSI={rsi}嚴重超賣")
    if direction == "long"  and macd < macd_sig and hist < 0 and macd < 0: vetoes.append("MACD空頭排列")
    if direction == "short" and macd > macd_sig and hist > 0 and macd > 0: vetoes.append("MACD多頭排列")
    if direction == "long"  and market_structure == "trending_down": vetoes.append("市場結構空頭")
    if direction == "short" and market_structure == "trending_up":   vetoes.append("市場結構多頭")
    if upper_trend == "down" and direction == "long":  vetoes.append("上層週期空頭")
    if upper_trend == "up"   and direction == "short": vetoes.append("上層週期多頭")
    if direction == "long"  and funding_rate > 0.3: vetoes.append(f"資金費率{funding_rate}%多方擁擠")
    if direction == "short" and funding_rate < -0.3: vetoes.append(f"資金費率{funding_rate}%空方擁擠")

    return {"vetoed": bool(vetoes), "reasons": vetoes}


# ──────────────────────────────────────────────
# 移動止損
# ──────────────────────────────────────────────

def calc_trailing_stop(entry: float, current_price: float,
                       atr: float, direction: str,
                       tp1_hit: bool = False) -> dict:
    if direction == "long":
        isl = round(entry - atr * 2.0, 8)
        be  = round(entry + atr * 0.1, 8)
        tsl = round(current_price - atr * 1.0, 8)
        if not tp1_hit: return {"trailing_sl": isl, "sl_type": "初始止損"}
        elif current_price < entry * 1.02: return {"trailing_sl": be,  "sl_type": "保本止損"}
        else:                              return {"trailing_sl": tsl, "sl_type": "追蹤止損"}
    else:
        isl = round(entry + atr * 2.0, 8)
        be  = round(entry - atr * 0.1, 8)
        tsl = round(current_price + atr * 1.0, 8)
        if not tp1_hit: return {"trailing_sl": isl, "sl_type": "初始止損"}
        elif current_price > entry * 0.98: return {"trailing_sl": be,  "sl_type": "保本止損"}
        else:                              return {"trailing_sl": tsl, "sl_type": "追蹤止損"}


# ──────────────────────────────────────────────
# 倉位管理
# ──────────────────────────────────────────────

def calc_position_size(winrate_pct: float, rr: float,
                       structure: str, adx: float) -> dict:
    """
    Kelly Criterion 簡化版
    最佳倉位 = (勝率 × 風報比 - 敗率) / 風報比
    限制在 3%~25% 之間
    """
    wr      = winrate_pct / 100.0
    lr      = 1.0 - wr
    kelly   = (wr * rr - lr) / max(rr, EPS)
    half_k  = kelly * 0.5   # Half-Kelly（更保守）

    base = max(3.0, min(25.0, half_k * 100))

    # 市場結構加成
    if structure in ("trending_up", "trending_down"): base = min(25, base * 1.2)
    elif structure == "ranging":                       base = min(25, base * 0.7)
    elif structure == "expanding":                     base = min(25, base * 0.5)

    # ADX 加成
    if adx >= 40:   base = min(25, base * 1.1)
    elif adx < 20:  base = min(25, base * 0.8)

    size = max(3.0, min(25.0, base))
    size = round(size, 1)

    if size >= 20:   label = "🟢 積極"
    elif size >= 15: label = "🟡 標準"
    elif size >= 10: label = "🟠 保守"
    else:            label = "🔴 觀望"

    return {"position_pct": size, "risk_label": label}


# ──────────────────────────────────────────────
# 支撐壓力
# ──────────────────────────────────────────────

def detect_support_resistance(highs, lows, closes, lookback=100, n=3) -> dict:
    sl  = min(lookback, len(highs), len(lows), len(closes))
    if sl < 10:
        return {"supports": [], "resistances": [], "nearest_sup": None,
                "nearest_res": None, "dist_to_sup": None, "dist_to_res": None}
    h     = highs[-sl:].astype(np.float64)
    l     = lows[-sl:].astype(np.float64)
    price = float(closes[-1])
    rs, ss = [], []
    for i in range(2, len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]: rs.append(float(h[i]))
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]: ss.append(float(l[i]))
    rs = sorted(set(r for r in rs if r > price))[:n]
    ss = sorted(set(s for s in ss if s < price), reverse=True)[:n]
    nr = rs[0] if rs else None; ns = ss[0] if ss else None
    dr = round((nr-price)/max(price,EPS)*100,2) if nr else None
    ds = round((price-ns)/max(price,EPS)*100,2) if ns else None
    return {"supports": ss, "resistances": rs, "nearest_sup": ns,
            "nearest_res": nr, "dist_to_sup": ds, "dist_to_res": dr}


# ──────────────────────────────────────────────
# 成交量
# ──────────────────────────────────────────────

def calc_volume_confirmation(volumes: np.ndarray, closes: np.ndarray) -> dict:
    if len(volumes) < 21:
        return {"bullish_vol": False, "bearish_vol": False, "vol_ratio": 1.0}
    av  = float(volumes[-21:-1].mean())
    lv  = float(volumes[-1])
    vr  = round(lv / max(av, EPS), 2)
    ok  = bool(vr >= _adaptive_params["vol_threshold"])
    up  = bool(float(closes[-1]) > float(closes[-2]))
    return {"bullish_vol": bool(ok and up), "bearish_vol": bool(ok and not up), "vol_ratio": vr}


# ──────────────────────────────────────────────
# K 線型態
# ──────────────────────────────────────────────

def detect_candle_pattern(opens, highs, lows, closes) -> dict:
    r = {"hammer": False, "engulfing_bull": False, "doji_bull": False,
         "shooting_star": False, "engulfing_bear": False, "name": "—"}
    if len(closes) < 3: return r
    o1=float(opens[-2]); c1=float(closes[-2])
    o2=float(opens[-1]); h2=float(highs[-1]); l2=float(lows[-1]); c2=float(closes[-1])
    body2=abs(c2-o2); rng2=h2-l2
    uw2=h2-max(o2,c2); lw2=min(o2,c2)-l2
    if rng2 < EPS: return r
    if body2>EPS and lw2>body2*2 and uw2<body2*0.5 and c2>o2 and c1<o1:
        r["hammer"]=True; r["name"]="🔨 錘子線"
    elif body2>EPS and uw2>body2*2 and lw2<body2*0.5 and c2<o2 and c1>o1:
        r["shooting_star"]=True; r["name"]="💫 流星線"
    elif c1<o1 and c2>o2 and o2<=c1 and c2>=o1:
        r["engulfing_bull"]=True; r["name"]="📈 看漲吞噬"
    elif c1>o1 and c2<o2 and o2>=c1 and c2<=o1:
        r["engulfing_bear"]=True; r["name"]="📉 看跌吞噬"
    elif body2<rng2*0.1 and lw2>rng2*0.6:
        r["doji_bull"]=True; r["name"]="✙ 十字星"
    return r


# ──────────────────────────────────────────────
# 多週期
# ──────────────────────────────────────────────

def get_upper_trend(symbol: str, main_tf: str) -> Optional[str]:
    utf = UPPER_TIMEFRAME.get(main_tf)
    if not utf: return None
    data = fetch_klines(symbol, utf, 60)
    if data is None or len(data) < 55: return None
    ema50 = calc_ema(data[:, 4], 50)
    return "up" if float(ema50[-1]) > float(ema50[-5]) else "down"


# ──────────────────────────────────────────────
# 背離
# ──────────────────────────────────────────────

def detect_divergence(highs, lows, closes, lookback=30) -> dict:
    r = {"rsi_bull_div": False, "rsi_bear_div": False, "macd_bull_div": False, "divergence": "無"}
    n = min(lookback, len(closes), len(highs), len(lows))
    if n < 20: return r
    c=closes[-n:].astype(np.float64); h=highs[-n:].astype(np.float64); l=lows[-n:].astype(np.float64)
    rs=[calc_rsi(c[:i+1]) for i in range(14, len(c))]
    if len(rs) < 8: return r
    pl=[(i,float(l[i])) for i in range(2,len(l)-2) if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]]
    ph=[(i,float(h[i])) for i in range(2,len(h)-2) if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]]
    if len(pl)>=2:
        p1i,p1p=pl[-2]; p2i,p2p=pl[-1]
        r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p<p1p and rs[r2i]>rs[r1i]:
            r["rsi_bull_div"]=True; r["divergence"]="📈 RSI看漲背離"
    if len(ph)>=2:
        p1i,p1p=ph[-2]; p2i,p2p=ph[-1]
        r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p>p1p and rs[r2i]<rs[r1i]:
            r["rsi_bear_div"]=True; r["divergence"]="📉 RSI看跌背離"
    return r


# ──────────────────────────────────────────────
# 進場點
# ──────────────────────────────────────────────

def get_best_entry(price: float, obs: list, fvgs: list,
                   direction: str, atr: float) -> float:
    cands=[]; md=atr*2.0
    if direction=="long":
        for o in obs:
            if o.get("type")=="bullish_ob":
                m=(o["high"]+o["low"])/2.0
                if m<price: cands.append(m)
        for f in fvgs:
            if f.get("type")=="bullish_fvg":
                m=(f["top"]+f["bottom"])/2.0
                if m<price: cands.append(m)
        if cands:
            b=max(cands)
            if price-b<=md: return round(b,8)
    else:
        for o in obs:
            if o.get("type")=="bearish_ob":
                m=(o["high"]+o["low"])/2.0
                if m>price: cands.append(m)
        for f in fvgs:
            if f.get("type")=="bearish_fvg":
                m=(f["top"]+f["bottom"])/2.0
                if m>price: cands.append(m)
        if cands:
            b=min(cands)
            if b-price<=md: return round(b,8)
    return round(price,8)


# ──────────────────────────────────────────────
# 斐波納契
# ──────────────────────────────────────────────

def find_swing_points(highs, lows, lookback=50):
    s=min(lookback,len(highs),len(lows))
    if s<2: return float(highs[-1]),float(lows[-1])
    return float(highs[-s:].max()), float(lows[-s:].min())


def get_fib_exits(swing_high, swing_low, entry, direction) -> dict:
    if swing_high<swing_low: swing_high,swing_low=swing_low,swing_high
    diff=max(swing_high-swing_low, entry*0.001)
    if direction=="long":
        sl=swing_high-diff*0.618
        if sl>=entry: sl=swing_high-diff*0.786
        if sl>=entry: sl=entry-diff*0.618
        tps=sorted([swing_high+diff*(e-1.0) for e in [1.272,1.618,2.618]])
        tps=[max(t,entry*1.001) for t in tps]
    else:
        sl=swing_low+diff*0.618
        if sl<=entry: sl=swing_low+diff*0.786
        if sl<=entry: sl=entry+diff*0.618
        tps=sorted([swing_low-diff*(e-1.0) for e in [1.272,1.618,2.618]],reverse=True)
        tps=[min(t,entry*0.999) for t in tps]
    tp1,tp2,tp3=tps
    risk=abs(entry-sl); rew=abs(tp1-entry)
    rr=round(rew/max(risk,EPS),2)
    return {"stop_loss":round(sl,8),"tp1":round(tp1,8),"tp2":round(tp2,8),"tp3":round(tp3,8),"risk_reward":rr}


# ──────────────────────────────────────────────
# SMC
# ──────────────────────────────────────────────

def detect_order_blocks(opens, highs, lows, closes, n=5) -> list:
    obs=[]; sn=min(len(closes),len(opens),len(highs),len(lows))
    for i in range(2,sn-1):
        body=abs(float(closes[i])-float(opens[i])); pr=float(highs[i-1])-float(lows[i-1])
        if body<EPS or pr<EPS: continue
        if closes[i-1]<opens[i-1] and closes[i]>highs[i-1] and body>pr*0.5:
            obs.append({"type":"bullish_ob","high":float(highs[i-1]),"low":float(lows[i-1]),"index":i})
        elif closes[i-1]>opens[i-1] and closes[i]<lows[i-1] and body>pr*0.5:
            obs.append({"type":"bearish_ob","high":float(highs[i-1]),"low":float(lows[i-1]),"index":i})
    return obs[-n:] if obs else []


def detect_fvg(highs,lows,closes) -> list:
    fvgs=[]; sn=min(len(closes),len(highs),len(lows))
    for i in range(2,sn):
        if float(lows[i])>float(highs[i-2]):
            fvgs.append({"type":"bullish_fvg","top":float(lows[i]),"bottom":float(highs[i-2])})
        elif float(highs[i])<float(lows[i-2]):
            fvgs.append({"type":"bearish_fvg","top":float(lows[i-2]),"bottom":float(highs[i])})
    return fvgs[-3:] if fvgs else []


def detect_bos_choch(highs,lows,closes,lookback=20):
    s=min(lookback+10,len(closes),len(highs),len(lows))
    if s<lookback+5: return None,None
    rh=float(highs[-lookback:-1].max()); rl=float(lows[-lookback:-1].min())
    lc=float(closes[-1])
    ema50=calc_ema(closes,50); slope=float(ema50[-1])-float(ema50[-10])
    pt="up" if slope>0 else "down"
    bos=choch=None
    if lc>rh:
        if pt=="up": bos="bullish_bos"
        else:        choch="bullish_choch"
    elif lc<rl:
        if pt=="down": bos="bearish_bos"
        else:          choch="bearish_choch"
    return bos,choch


def check_harmonic(X,A,B,C,D):
    def r(a,b): return abs(a)/max(abs(b),EPS)
    XA=A-X; AB=B-A; BC=C-B; CD=D-C
    if abs(XA)<EPS or abs(AB)<EPS or abs(BC)<EPS: return None
    abxa=r(AB,XA); bcab=r(BC,AB); cdbc=r(CD,BC); xdxa=r(D-X,XA)
    def near(v,t,tol=0.08): return bool(abs(v-t)<=tol)
    d="long" if XA>0 else "short"
    if near(abxa,.618) and (near(bcab,.382) or near(bcab,.886)) and near(cdbc,1.272) and near(xdxa,.786): return ("Gartley",d)
    if near(abxa,.382) and (near(bcab,.382) or near(bcab,.886)) and near(cdbc,2.0)   and near(xdxa,.886): return ("Bat",d)
    if near(abxa,.786) and (near(bcab,.382) or near(bcab,.886)) and near(cdbc,1.618) and near(xdxa,1.272): return ("Butterfly",d)
    if near(abxa,.382) and (near(bcab,.382) or near(bcab,.886)) and near(cdbc,3.618) and near(xdxa,1.618): return ("Crab",d)
    return None


def scan_harmonics(highs,lows,n=80):
    s=min(n,len(highs),len(lows))
    if s<10: return None
    h=list(highs[-s:].astype(float)); l=list(lows[-s:].astype(float)); pivots=[]
    for i in range(2,len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i+1]: pivots.append(h[i])
        elif l[i]<l[i-1] and l[i]<l[i+1]: pivots.append(l[i])
    if len(pivots)<5: return None
    for i in range(len(pivots)-4):
        res=check_harmonic(*pivots[i:i+5])
        if res: return res
    return None


# ──────────────────────────────────────────────
# 外部資料
# ──────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    try:
        resp=requests.get("https://api.alternative.me/fng/?limit=1",timeout=8); resp.raise_for_status()
        d=resp.json()["data"][0]; v=int(d["value"])
        if v<=24: lb="😱 極度恐懼"
        elif v<=49: lb="😨 恐懼"
        elif v<=74: lb="😊 貪婪"
        else: lb="🤑 極度貪婪"
        return {"value":v,"label":lb}
    except Exception: return {"value":-1,"label":"無法取得"}


def fetch_funding_rate(symbol: str) -> dict:
    for base in FUTURES_BASE_URLS:
        try:
            r = requests.get(f"{base}/premiumIndex",
                             params={"symbol": f"{symbol}USDT"}, timeout=8)
            if r.status_code == 451:
                continue
            r.raise_for_status()
            rate = round(float(r.json()["lastFundingRate"]) * 100, 4)
            sig  = "多方擁擠⚠️" if rate > 0.1 else ("空方擁擠⚠️" if rate < -0.1 else "市場平衡✅")
            return {"rate": rate, "signal": sig}
        except Exception:
            continue
    return {"rate": 0.0, "signal": "無法取得"}


def fetch_open_interest(symbol: str) -> dict:
    for base in FUTURES_BASE_URLS:
        try:
            r1 = requests.get(f"{base}/openInterest",
                              params={"symbol": f"{symbol}USDT"}, timeout=8)
            if r1.status_code == 451:
                continue
            r1.raise_for_status()
            ci = float(r1.json()["openInterest"])

            r2 = requests.get(f"{base}/openInterestHist",
                              params={"symbol": f"{symbol}USDT", "period": "1h", "limit": 2},
                              timeout=8)
            if r2.status_code == 451:
                continue
            r2.raise_for_status()
            hist = r2.json()
            oc   = 0.0
            if isinstance(hist, list) and len(hist) >= 2:
                prev = float(hist[0]["sumOpenInterest"])
                oc   = round((ci - prev) / max(prev, EPS) * 100, 2)
            sig = "OI急增📈" if oc > 2 else ("OI急減📉" if oc < -2 else "OI穩定➡️")
            return {"oi": round(ci, 2), "oi_change": oc, "oi_signal": sig}
        except Exception:
            continue
    return {"oi": 0.0, "oi_change": 0.0, "oi_signal": "無法取得"}


# ──────────────────────────────────────────────
# 主分析函數
# ──────────────────────────────────────────────

def full_analysis(symbol: str) -> Optional[dict]:
    try:
        # 系統4：風控暫停檢查
        if _risk_control["paused"]:
            print(f"  ⚠️ 風控暫停中，跳過 {symbol}")
            return None

        tf   = select_timeframe(symbol)
        data = fetch_klines(symbol, tf, 200)
        if data is None or len(data) < 60: return None

        o=data[:,1]; h=data[:,2]; l=data[:,3]; c=data[:,4]; v=data[:,5]
        price=float(c[-1])

        rsi=calc_rsi(c); macd,ms,hist=calc_macd(c)
        bbu,_,bbl=calc_bollinger(c)
        ema20=float(calc_ema(c,20)[-1]); ema50=float(calc_ema(c,50)[-1]); ema200=float(calc_ema(c,200)[-1])
        atr=calc_atr(h,l,c); vwap=calc_vwap(h,l,c,v)
        adx=calc_adx(h,l,c); ms_info=detect_market_structure(h,l,c,atr)
        bo=detect_breakout(h,l,c,v,atr); sr=detect_support_resistance(h,l,c)
        div=detect_divergence(h,l,c); funding=fetch_funding_rate(symbol); oi=fetch_open_interest(symbol)
        sh,sl=find_swing_points(h,l,50); obs=detect_order_blocks(o,h,l,c)
        fvgs=detect_fvg(h,l,c); bos,choch=detect_bos_choch(h,l,c)
        harmonic=scan_harmonics(h,l,80); vi=calc_volume_confirmation(v,c)
        candle=detect_candle_pattern(o,h,l,c); ut=get_upper_trend(symbol,tf)

        long_s=short_s=0

        # RSI
        if rsi<30: long_s+=2
        elif rsi<45: long_s+=1
        elif rsi>70: short_s+=2
        elif rsi>55: short_s+=1
        # MACD
        if hist>0 and macd>ms: long_s+=2
        elif hist<0 and macd<ms: short_s+=2
        # EMA
        if price>ema20>ema50>ema200: long_s+=3
        elif price<ema20<ema50<ema200: short_s+=3
        elif price>ema50: long_s+=1
        elif price<ema50: short_s+=1
        # BB
        if price<bbl: long_s+=1
        elif price>bbu: short_s+=1
        # VWAP
        if price>vwap: long_s+=1
        elif price<vwap: short_s+=1
        # ADX
        sc=adx["adx_score"]
        if sc>0: long_s+=sc; short_s+=sc
        else: long_s=max(0,long_s+sc); short_s=max(0,short_s+sc)
        if adx["pdi"]>adx["mdi"]: long_s+=1
        else: short_s+=1
        # 市場結構
        if ms_info["structure"]=="trending_up": long_s+=2
        elif ms_info["structure"]=="trending_down": short_s+=2
        # 突破
        if bo["breakout"]=="bullish": long_s+=4
        elif bo["breakout"]=="bearish": short_s+=4
        # OB/FVG
        if any(x.get("type")=="bullish_ob"  for x in obs): long_s+=1
        if any(x.get("type")=="bearish_ob"  for x in obs): short_s+=1
        if any(x.get("type")=="bullish_fvg" for x in fvgs): long_s+=1
        if any(x.get("type")=="bearish_fvg" for x in fvgs): short_s+=1
        # BOS/CHoCH
        if bos=="bullish_bos": long_s+=2
        elif bos=="bearish_bos": short_s+=2
        if choch=="bullish_choch": long_s+=3
        elif choch=="bearish_choch": short_s+=3
        # 和諧
        if harmonic:
            if harmonic[1]=="long": long_s+=2
            else: short_s+=2
        # 成交量
        if vi["bullish_vol"]: long_s+=2
        if vi["bearish_vol"]: short_s+=2
        # K線
        if candle["hammer"] or candle["engulfing_bull"] or candle["doji_bull"]: long_s+=2
        if candle["shooting_star"] or candle["engulfing_bear"]: short_s+=2
        # 多週期
        ua=False
        if ut=="up": long_s+=2; ua=True
        elif ut=="down": short_s+=2; ua=True
        # 背離
        hd=False
        if div["rsi_bull_div"] or div["macd_bull_div"]: long_s+=3; hd=True
        if div["rsi_bear_div"]: short_s+=3; hd=True
        # 籌碼
        if funding["rate"]>0.15: short_s+=1
        if funding["rate"]<-0.15: long_s+=1
        if oi["oi_change"]>2 and price>float(c[-2]): long_s+=1
        elif oi["oi_change"]>2 and price<float(c[-2]): short_s+=1

        # 過濾
        diff=abs(long_s-short_s); mx=max(long_s,short_s)
        if long_s==short_s: return None
        if diff<_adaptive_params["min_score_diff"]: return None
        if mx<_adaptive_params["min_score_total"]: return None

        direction="long" if long_s>short_s else "short"

        # 否決
        veto=check_veto(direction,rsi,macd,ms,hist,adx["adx"],funding["rate"],ms_info["structure"],ut)
        if veto["vetoed"]:
            print(f"  ⛔ {symbol} 被否決：{veto['reasons']}")
            return None

        best_entry=get_best_entry(price,obs,fvgs,direction,atr)
        exits=get_fib_exits(sh,sl,best_entry,direction)
        if exits["risk_reward"]<_adaptive_params["min_rr"]: return None
        if is_duplicate_signal(symbol,direction,price,atr): return None

        atr_sl=(round(best_entry-atr*2.0,8) if direction=="long" else round(best_entry+atr*2.0,8))
        trailing=calc_trailing_stop(best_entry,price,atr,direction,False)

        # 系統3：ML 預測勝率（0~100%）
        hv  = bool(vi["bullish_vol"] or vi["bearish_vol"])
        hc  = bool(any([candle["hammer"],candle["engulfing_bull"],candle["doji_bull"],
                        candle["shooting_star"],candle["engulfing_bear"]]))
        hbo = bool(bo["breakout"] is not None)
        itr = bool(ms_info["structure"] in ("trending_up","trending_down"))

        features   = build_ml_features(diff,adx["adx"],exits["risk_reward"],
                                       hv,hc,ua,hbo,itr,rsi,direction,hist,macd,ms,hd)
        winrate_pct = ml_predict_winrate(features)

        # 系統2：倉位（Kelly Criterion）
        position = calc_position_size(winrate_pct, exits["risk_reward"],
                                      ms_info["structure"], adx["adx"])

        return {
            "symbol":           str(symbol),
            "timeframe":        str(tf),
            "price":            round(float(price),8),
            "direction":        str(direction),
            "long_score":       int(long_s),
            "short_score":      int(short_s),
            "winrate_pct":      float(winrate_pct),        # 系統3：ML勝率
            "entry":            round(float(best_entry),8),
            "entry_source":     "OB/FVG中點" if best_entry!=round(price,8) else "現價",
            "stop_loss":        float(exits["stop_loss"]),
            "atr_stop_loss":    float(atr_sl),
            "trailing_sl":      float(trailing["trailing_sl"]),
            "sl_type":          str(trailing["sl_type"]),
            "tp1":              float(exits["tp1"]),
            "tp2":              float(exits["tp2"]),
            "tp3":              float(exits["tp3"]),
            "risk_reward":      float(exits["risk_reward"]),
            "position_pct":     float(position["position_pct"]),
            "risk_label":       str(position["risk_label"]),
            "market_structure": str(ms_info["label"]),
            "market_strategy":  str(ms_info["strategy"]),
            "breakout":         str(bo["breakout_label"]),
            "rsi":              float(rsi),
            "macd":             round(float(macd),8),
            "macd_hist":        round(float(hist),8),
            "ema20":            round(float(ema20),8),
            "ema50":            round(float(ema50),8),
            "ema200":           round(float(ema200),8),
            "bb_upper":         round(float(bbu),8),
            "bb_lower":         round(float(bbl),8),
            "vwap":             round(float(vwap),8),
            "atr":              round(float(atr),8),
            "adx":              float(adx["adx"]),
            "trend_strength":   str(adx["trend_strength"]),
            "pdi":              float(adx["pdi"]),
            "mdi":              float(adx["mdi"]),
            "divergence":       str(div["divergence"]),
            "nearest_sup":      sr["nearest_sup"],
            "nearest_res":      sr["nearest_res"],
            "dist_to_sup":      sr["dist_to_sup"],
            "dist_to_res":      sr["dist_to_res"],
            "funding_rate":     float(funding["rate"]),
            "funding_signal":   str(funding["signal"]),
            "oi_change":        float(oi["oi_change"]),
            "oi_signal":        str(oi["oi_signal"]),
            "swing_high":       round(float(sh),8),
            "swing_low":        round(float(sl),8),
            "order_blocks":     obs,
            "fvg":              fvgs,
            "bos":              bos,
            "choch":            choch,
            "candle_pattern":   str(candle["name"]),
            "vol_ratio":        float(vi["vol_ratio"]),
            "upper_trend":      str(ut) if ut else "—",
            "harmonic":         harmonic,
            "ml_features":      features,   # 儲存供後續更新ML使用
            "veto_reasons":     veto["reasons"],
        }

    except Exception as e:
        print(f"  ❌ full_analysis({symbol}) 錯誤：{e}")
        return None


# ──────────────────────────────────────────────
# 格式化
# ──────────────────────────────────────────────

def format_signal(r: dict) -> str:
    try:
        de  = "🟢 做多 LONG" if r["direction"]=="long" else "🔴 做空 SHORT"
        hs  = f"{r['harmonic'][0]} ({r['harmonic'][1]})" if r.get("harmonic") else "無"
        bs  = r.get("bos")   or "—"
        cs  = r.get("choch") or "—"
        ue  = "📈" if r.get("upper_trend")=="up" else ("📉" if r.get("upper_trend")=="down" else "—")
        ss  = f"`{r['nearest_sup']}` ({r['dist_to_sup']}%)" if r.get("nearest_sup") else "—"
        rs  = f"`{r['nearest_res']}` ({r['dist_to_res']}%)" if r.get("nearest_res") else "—"

        rsi=r["rsi"]
        if rsi<30: rl="超賣🔥"
        elif rsi>70: rl="超買❄️"
        elif rsi<45: rl="偏弱⬇️"
        elif rsi>55: rl="偏強⬆️"
        else: rl="中性⚪️"

        # 系統3：勝率顏色
        wr=r["winrate_pct"]
        if wr>=75: wr_emoji="🟢"
        elif wr>=60: wr_emoji="🟡"
        else: wr_emoji="🔴"

        rsk=get_risk_status()

        return (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} ｜ {de}\n"
            f"🎯 *預測勝率*：{wr_emoji} *{wr}%*\n"
            f"💼 倉位建議：{r['position_pct']}% {r['risk_label']}\n"
            f"🏛 市場結構：{r['market_structure']}\n"
            f"💡 策略：{r['market_strategy']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *進場*：`{r['entry']}` ({r['entry_source']})\n"
            f"\n"
            f"📐 *出場價位*\n"
            f"  🛑 Fib止損     ：`{r['stop_loss']}`\n"
            f"  🛑 ATR止損     ：`{r['atr_stop_loss']}`\n"
            f"  🛑 {r['sl_type']}：`{r['trailing_sl']}`\n"
            f"  🎯 TP1 (1.272)：`{r['tp1']}`\n"
            f"  🎯 TP2 (1.618)：`{r['tp2']}`\n"
            f"  🎯 TP3 (2.618)：`{r['tp3']}`\n"
            f"  ⚖️  風報比 R:R ：`1 : {r['risk_reward']}`\n"
            f"\n"
            f"🚀 突破：{r['breakout']}\n"
            f"🔀 背離：{r['divergence']}\n"
            f"\n"
            f"📊 *指標*\n"
            f"  RSI `{r['rsi']}` {rl} ｜ MACD Hist `{r['macd_hist']}`\n"
            f"  EMA 20:`{r['ema20']}` 50:`{r['ema50']}` 200:`{r['ema200']}`\n"
            f"  BB上:`{r['bb_upper']}` 下:`{r['bb_lower']}`\n"
            f"  VWAP:`{r['vwap']}` ATR:`{r['atr']}`\n"
            f"  ADX:`{r['adx']}` {r['trend_strength']} PDI:`{r['pdi']}` MDI:`{r['mdi']}`\n"
            f"\n"
            f"📍 支撐：{ss} ｜ 壓力：{rs}\n"
            f"🕯 K線：{r['candle_pattern']} ｜ 量比：`{r['vol_ratio']}x`\n"
            f"🌐 上層週期：{ue} {r['upper_trend']}\n"
            f"\n"
            f"💹 *籌碼*\n"
            f"  費率：`{r['funding_rate']}%` {r['funding_signal']}\n"
            f"  OI：`{r['oi_change']}%` {r['oi_signal']}\n"
            f"\n"
            f"🏗️ BOS：{bs} CHoCH：{cs}\n"
            f"  OB：{len(r.get('order_blocks',[]))}個 FVG：{len(r.get('fvg',[]))}個\n"
            f"🔷 和諧：{hs}\n"
            f"\n"
            f"📈 評分 多{r['long_score']} / 空{r['short_score']}\n"
            f"📊 系統實際勝率：`{rsk['actual_winrate']}%` "
            f"({rsk['winning_signals']}/{rsk['total_signals']}筆)\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as e:
        return f"❌ 格式化錯誤：{e}"


# 初始化
load_risk_control()
load_ml_weights()
