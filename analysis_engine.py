"""
analysis_engine.py v5.0
========================
新增功能：
  A. 支撐壓力位（自動抓關鍵歷史高低點）
  B. VWAP（機構常用指標，判斷主力方向）
  C. ADX 趨勢強度（過濾橫盤假訊號）
  D. ATR 動態止損（比固定 Fib 更合理）
  E. 恐懼貪婪指數（市場情緒）
  F. 資金費率（永續合約籌碼）
  G. 未平倉量 OI 變化（主力方向）
  H. RSI/MACD 背離偵測（抓反轉機會）
  I. 止盈/止損價格監控（推播用）
"""

import time
import requests
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# 幣種清單 & 週期設定
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

MIN_SCORE_DIFF  = 3
MIN_SCORE_TOTAL = 5

BINANCE_URL  = "https://api.binance.com/api/v3/klines"
FUTURES_URL  = "https://fapi.binance.com/fapi/v1"

_signal_cache: dict = {}
SIGNAL_COOLDOWN = 3600


def is_duplicate_signal(symbol: str, direction: str) -> bool:
    key = f"{symbol}_{direction}"
    now = time.time()
    if key in _signal_cache and now - _signal_cache[key] < SIGNAL_COOLDOWN:
        return True
    _signal_cache[key] = now
    return False


# ──────────────────────────────────────────────
# K 線抓取
# ──────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "4h",
                 limit: int = 200, retries: int = 3) -> Optional[np.ndarray]:
    pair = f"{symbol}USDT"
    for attempt in range(retries):
        try:
            resp = requests.get(
                BINANCE_URL,
                params={"symbol": pair, "interval": interval, "limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            raw = resp.json()
            if not raw or isinstance(raw, dict):
                return None
            data = np.array([[float(k[0]), float(k[1]), float(k[2]),
                               float(k[3]), float(k[4]), float(k[5])]
                              for k in raw])
            if data[:, 5].sum() == 0:
                return None
            return data
        except Exception as e:
            print(f"  ⚠️ {symbol} {interval} 第{attempt+1}次失敗：{e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def select_timeframe(symbol: str) -> str:
    return COIN_TIMEFRAME.get(symbol.upper(), "1h")


# ──────────────────────────────────────────────
# 基礎指標
# ──────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes)
    gains    = np.where(deltas > 0, deltas, 0.0)
    losses   = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)


def calc_ema(closes: np.ndarray, period: int) -> np.ndarray:
    ema    = np.zeros_like(closes)
    k      = 2 / (period + 1)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = closes[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_macd(closes: np.ndarray):
    macd_line   = calc_ema(closes, 12) - calc_ema(closes, 26)
    macd_signal = calc_ema(macd_line, 9)
    macd_hist   = macd_line - macd_signal
    return float(macd_line[-1]), float(macd_signal[-1]), float(macd_hist[-1])


def calc_bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
    if len(closes) < period:
        m = float(closes[-1])
        return m, m, m
    w = closes[-period:]
    m = float(w.mean())
    s = float(w.std())
    return m + std_dev * s, m, m - std_dev * s


def calc_atr(highs, lows, closes, period: int = 14) -> float:
    if len(closes) < period + 1:
        return float(highs[-1] - lows[-1])
    trs = [max(float(highs[i] - lows[i]),
               abs(float(highs[i] - closes[i - 1])),
               abs(float(lows[i]  - closes[i - 1])))
           for i in range(1, len(closes))]
    return float(np.array(trs[-period:]).mean())


# ──────────────────────────────────────────────
# A. 支撐壓力位
# ──────────────────────────────────────────────

def detect_support_resistance(highs, lows, closes, lookback=100, n=3) -> dict:
    """
    自動抓取關鍵支撐壓力位
    用 pivot high/low 偵測，取最近 n 個重要位置
    判斷現價距離最近支撐/壓力的距離
    """
    h = highs[-lookback:]
    l = lows[-lookback:]
    price = float(closes[-1])

    resistances = []
    supports    = []

    for i in range(2, len(h) - 2):
        # Pivot High = 壓力
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            resistances.append(float(h[i]))
        # Pivot Low = 支撐
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            supports.append(float(l[i]))

    # 只保留比現價高的壓力 / 比現價低的支撐
    resistances = sorted([r for r in resistances if r > price])[:n]
    supports    = sorted([s for s in supports    if s < price], reverse=True)[:n]

    nearest_res = resistances[0] if resistances else None
    nearest_sup = supports[0]    if supports    else None

    # 現價距支撐/壓力的百分比
    dist_to_res = round((nearest_res - price) / price * 100, 2) if nearest_res else None
    dist_to_sup = round((price - nearest_sup) / price * 100, 2) if nearest_sup else None

    return {
        "supports":    supports,
        "resistances": resistances,
        "nearest_sup": nearest_sup,
        "nearest_res": nearest_res,
        "dist_to_sup": dist_to_sup,   # % 距離支撐
        "dist_to_res": dist_to_res,   # % 距離壓力
    }


# ──────────────────────────────────────────────
# B. VWAP
# ──────────────────────────────────────────────

def calc_vwap(highs, lows, closes, volumes) -> float:
    """
    VWAP = 成交量加權平均價
    價格 > VWAP → 多方佔優
    價格 < VWAP → 空方佔優
    """
    typical_price = (highs + lows + closes) / 3
    vwap = float(np.sum(typical_price * volumes) / np.sum(volumes)) if np.sum(volumes) > 0 else float(closes[-1])
    return round(vwap, 6)


# ──────────────────────────────────────────────
# C. ADX 趨勢強度
# ──────────────────────────────────────────────

def calc_adx(highs, lows, closes, period: int = 14) -> dict:
    """
    ADX < 20  → 橫盤，訊號不可靠
    ADX 20-40 → 趨勢形成
    ADX > 40  → 強趨勢，訊號可靠
    """
    if len(closes) < period * 2:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️"}

    plus_dm  = []
    minus_dm = []
    trs      = []

    for i in range(1, len(closes)):
        high_diff = float(highs[i]) - float(highs[i-1])
        low_diff  = float(lows[i-1]) - float(lows[i])
        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
        tr = max(float(highs[i]) - float(lows[i]),
                 abs(float(highs[i]) - float(closes[i-1])),
                 abs(float(lows[i])  - float(closes[i-1])))
        trs.append(tr)

    def wilder_smooth(arr, p):
        result = [sum(arr[:p])]
        for v in arr[p:]:
            result.append(result[-1] - result[-1]/p + v)
        return result

    atr14    = wilder_smooth(trs, period)
    plus14   = wilder_smooth(plus_dm, period)
    minus14  = wilder_smooth(minus_dm, period)

    dx_list  = []
    for i in range(len(atr14)):
        pdi = 100 * plus14[i]  / atr14[i] if atr14[i] > 0 else 0
        mdi = 100 * minus14[i] / atr14[i] if atr14[i] > 0 else 0
        dx  = 100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) > 0 else 0
        dx_list.append(dx)

    adx = round(sum(dx_list[-period:]) / period, 2)

    if adx < 20:
        strength = "橫盤⚪️"
    elif adx < 40:
        strength = "趨勢中🟡"
    else:
        strength = "強趨勢🔥"

    return {"adx": adx, "trend_strength": strength}


# ──────────────────────────────────────────────
# D. ATR 動態止損
# ──────────────────────────────────────────────

def calc_atr_stoploss(entry: float, atr: float, direction: str,
                      multiplier: float = 2.0) -> float:
    """
    動態止損 = 進場價 ± ATR × multiplier
    比固定 Fib 更貼近市場波動
    multiplier=2.0 → 正常波動不觸發
    multiplier=1.5 → 較緊（適合強趨勢）
    """
    if direction == "long":
        return round(entry - atr * multiplier, 6)
    else:
        return round(entry + atr * multiplier, 6)


# ──────────────────────────────────────────────
# E. 恐懼貪婪指數
# ──────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    """
    使用 alternative.me API 抓取恐懼貪婪指數
    0-24  → 極度恐懼（做多機會）
    25-49 → 恐懼
    50-74 → 貪婪
    75-100 → 極度貪婪（做空機會）
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=8
        )
        data  = resp.json()["data"][0]
        value = int(data["value"])
        label = data["value_classification"]

        if value <= 24:
            emoji = "😱 極度恐懼"
        elif value <= 49:
            emoji = "😨 恐懼"
        elif value <= 74:
            emoji = "😊 貪婪"
        else:
            emoji = "🤑 極度貪婪"

        return {"value": value, "label": emoji}
    except Exception:
        return {"value": -1, "label": "無法取得"}


# ──────────────────────────────────────────────
# F. 資金費率
# ──────────────────────────────────────────────

def fetch_funding_rate(symbol: str) -> dict:
    """
    資金費率 > 0.1%  → 多方擁擠，注意回調
    資金費率 < -0.1% → 空方擁擠，注意反彈
    資金費率接近 0   → 市場平衡
    """
    try:
        resp = requests.get(
            f"{FUTURES_URL}/premiumIndex",
            params={"symbol": f"{symbol}USDT"},
            timeout=8
        )
        data = resp.json()
        rate = round(float(data["lastFundingRate"]) * 100, 4)

        if rate > 0.1:
            signal = "多方擁擠⚠️"
        elif rate < -0.1:
            signal = "空方擁擠⚠️"
        else:
            signal = "市場平衡✅"

        return {"rate": rate, "signal": signal}
    except Exception:
        return {"rate": 0.0, "signal": "無法取得"}


# ──────────────────────────────────────────────
# G. 未平倉量 OI
# ──────────────────────────────────────────────

def fetch_open_interest(symbol: str) -> dict:
    """
    OI 增加 + 價格上漲 → 多頭進場，趨勢延續
    OI 增加 + 價格下跌 → 空頭進場，趨勢延續
    OI 減少           → 主力出場，趨勢可能結束
    """
    try:
        # 目前 OI
        resp1 = requests.get(
            f"{FUTURES_URL}/openInterest",
            params={"symbol": f"{symbol}USDT"},
            timeout=8
        )
        current_oi = float(resp1.json()["openInterest"])

        # 歷史 OI（抓最近2筆做比較）
        resp2 = requests.get(
            f"{FUTURES_URL}/openInterestHist",
            params={"symbol": f"{symbol}USDT", "period": "1h", "limit": 2},
            timeout=8
        )
        hist = resp2.json()
        if len(hist) >= 2:
            prev_oi  = float(hist[0]["sumOpenInterest"])
            oi_change = round((current_oi - prev_oi) / prev_oi * 100, 2) if prev_oi > 0 else 0
        else:
            oi_change = 0.0

        if oi_change > 2:
            oi_signal = "OI急增📈"
        elif oi_change < -2:
            oi_signal = "OI急減📉"
        else:
            oi_signal = "OI穩定➡️"

        return {
            "oi":        round(current_oi, 2),
            "oi_change": oi_change,
            "oi_signal": oi_signal,
        }
    except Exception:
        return {"oi": 0.0, "oi_change": 0.0, "oi_signal": "無法取得"}


# ──────────────────────────────────────────────
# H. RSI / MACD 背離偵測
# ──────────────────────────────────────────────

def detect_divergence(highs, lows, closes, lookback=30) -> dict:
    """
    看漲背離：價格創新低，RSI 未創新低 → 做多訊號
    看跌背離：價格創新高，RSI 未創新高 → 做空訊號
    MACD 背離同理
    """
    result = {
        "rsi_bull_div":  False,
        "rsi_bear_div":  False,
        "macd_bull_div": False,
        "macd_bear_div": False,
        "divergence":    "無",
    }

    if len(closes) < lookback + 5:
        return result

    c  = closes[-lookback:]
    h  = highs[-lookback:]
    l  = lows[-lookback:]
    n  = len(c)

    # 計算整段的 RSI 序列（簡化版）
    rsi_series = []
    for i in range(14, n):
        rsi_series.append(calc_rsi(c[:i+1]))

    if len(rsi_series) < 10:
        return result

    # MACD hist 序列
    macd_hist_series = []
    for i in range(26, n):
        _, _, hist = calc_macd(c[:i+1])
        macd_hist_series.append(hist)

    # 價格最近兩個低點
    price_lows  = [(i, float(l[i])) for i in range(2, n-2)
                   if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]]
    price_highs = [(i, float(h[i])) for i in range(2, n-2)
                   if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]]

    # 看漲背離：價格低點下降，RSI低點上升
    if len(price_lows) >= 2:
        p1_idx, p1_price = price_lows[-2]
        p2_idx, p2_price = price_lows[-1]
        r1_idx = p1_idx - 14
        r2_idx = p2_idx - 14
        if (r1_idx >= 0 and r2_idx >= 0 and r2_idx < len(rsi_series) and
                p2_price < p1_price and rsi_series[r2_idx] > rsi_series[r1_idx]):
            result["rsi_bull_div"] = True
            result["divergence"]   = "📈 RSI看漲背離"

    # 看跌背離：價格高點上升，RSI高點下降
    if len(price_highs) >= 2:
        p1_idx, p1_price = price_highs[-2]
        p2_idx, p2_price = price_highs[-1]
        r1_idx = p1_idx - 14
        r2_idx = p2_idx - 14
        if (r1_idx >= 0 and r2_idx >= 0 and r2_idx < len(rsi_series) and
                p2_price > p1_price and rsi_series[r2_idx] < rsi_series[r1_idx]):
            result["rsi_bear_div"] = True
            result["divergence"]   = "📉 RSI看跌背離"

    # MACD 看漲背離
    if len(price_lows) >= 2 and len(macd_hist_series) >= 10:
        p1_idx, p1_price = price_lows[-2]
        p2_idx, p2_price = price_lows[-1]
        m1_idx = max(0, p1_idx - 26)
        m2_idx = max(0, p2_idx - 26)
        if (m2_idx < len(macd_hist_series) and
                p2_price < p1_price and
                macd_hist_series[m2_idx] > macd_hist_series[m1_idx]):
            result["macd_bull_div"] = True
            if result["divergence"] == "無":
                result["divergence"] = "📈 MACD看漲背離"

    return result


# ──────────────────────────────────────────────
# 成交量確認
# ──────────────────────────────────────────────

def calc_volume_confirmation(volumes: np.ndarray, closes: np.ndarray) -> dict:
    if len(volumes) < 21:
        return {"bullish_vol": False, "bearish_vol": False, "vol_ratio": 1.0}
    avg_vol   = float(volumes[-21:-1].mean())
    last_vol  = float(volumes[-1])
    vol_ratio = round(last_vol / avg_vol if avg_vol > 0 else 1.0, 2)
    is_strong = vol_ratio >= 1.5
    price_up  = float(closes[-1]) > float(closes[-2])
    return {
        "bullish_vol": is_strong and price_up,
        "bearish_vol": is_strong and not price_up,
        "vol_ratio":   vol_ratio,
    }


# ──────────────────────────────────────────────
# K 線型態
# ──────────────────────────────────────────────

def detect_candle_pattern(opens, highs, lows, closes) -> dict:
    result = {
        "hammer": False, "engulfing_bull": False, "doji_bull": False,
        "shooting_star": False, "engulfing_bear": False, "name": "—",
    }
    if len(closes) < 3:
        return result

    o1 = float(opens[-2]);  c1 = float(closes[-2])
    o2 = float(opens[-1]);  h2 = float(highs[-1])
    l2 = float(lows[-1]);   c2 = float(closes[-1])

    body2       = abs(c2 - o2)
    rng2        = h2 - l2
    upper_wick2 = h2 - max(o2, c2)
    lower_wick2 = min(o2, c2) - l2

    if rng2 == 0:
        return result

    if (body2 > 0 and lower_wick2 > body2 * 2
            and upper_wick2 < body2 * 0.5 and c2 > o2 and c1 < o1):
        result["hammer"] = True; result["name"] = "🔨 錘子線"
    elif (body2 > 0 and upper_wick2 > body2 * 2
            and lower_wick2 < body2 * 0.5 and c2 < o2 and c1 > o1):
        result["shooting_star"] = True; result["name"] = "💫 流星線"
    elif (c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1):
        result["engulfing_bull"] = True; result["name"] = "📈 看漲吞噬"
    elif (c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1):
        result["engulfing_bear"] = True; result["name"] = "📉 看跌吞噬"
    elif (body2 < rng2 * 0.1 and lower_wick2 > rng2 * 0.6):
        result["doji_bull"] = True; result["name"] = "✙ 十字星"

    return result


# ──────────────────────────────────────────────
# 多週期趨勢確認
# ──────────────────────────────────────────────

def get_upper_trend(symbol: str, main_tf: str) -> Optional[str]:
    upper_tf = UPPER_TIMEFRAME.get(main_tf)
    if not upper_tf:
        return None
    data = fetch_klines(symbol, upper_tf, 60)
    if data is None or len(data) < 55:
        return None
    closes = data[:, 4]
    ema50  = calc_ema(closes, 50)
    slope  = float(ema50[-1]) - float(ema50[-5])
    return "up" if slope > 0 else "down"


# ──────────────────────────────────────────────
# 進場點優化
# ──────────────────────────────────────────────

def get_best_entry(price: float, obs: list, fvgs: list,
                   direction: str, atr: float) -> float:
    max_dist   = atr * 2
    candidates = []

    if direction == "long":
        for ob in obs:
            if ob["type"] == "bullish_ob":
                mid = (ob["high"] + ob["low"]) / 2
                if mid < price: candidates.append(mid)
        for fvg in fvgs:
            if fvg["type"] == "bullish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2
                if mid < price: candidates.append(mid)
        if candidates:
            best = max(candidates)
            if price - best <= max_dist:
                return round(best, 6)
    else:
        for ob in obs:
            if ob["type"] == "bearish_ob":
                mid = (ob["high"] + ob["low"]) / 2
                if mid > price: candidates.append(mid)
        for fvg in fvgs:
            if fvg["type"] == "bearish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2
                if mid > price: candidates.append(mid)
        if candidates:
            best = min(candidates)
            if best - price <= max_dist:
                return round(best, 6)

    return round(price, 6)


# ──────────────────────────────────────────────
# 信心度
# ──────────────────────────────────────────────

def get_confidence(score_diff: int, has_volume: bool,
                   has_candle: bool, upper_agree: bool,
                   adx: float, has_divergence: bool) -> str:
    bonus = sum([has_volume, has_candle, upper_agree,
                 adx >= 25, has_divergence])
    if score_diff >= 8 and bonus >= 3:
        return "🔥 極高"
    elif score_diff >= 5 and bonus >= 2:
        return "✅ 高"
    elif score_diff >= 3 and bonus >= 1:
        return "🟡 中"
    else:
        return "⚪️ 低"


# ──────────────────────────────────────────────
# 斐波納契計算
# ──────────────────────────────────────────────

def find_swing_points(highs, lows, lookback: int = 50):
    return float(highs[-lookback:].max()), float(lows[-lookback:].min())


def fibonacci_levels(swing_high: float, swing_low: float, direction: str):
    if swing_high < swing_low:
        swing_high, swing_low = swing_low, swing_high
    diff = swing_high - swing_low or swing_high * 0.01

    if direction == "long":
        retracements = {r: swing_high - diff * r         for r in FIB_RETRACEMENT}
        extensions   = {e: swing_high + diff * (e - 1.0) for e in FIB_EXTENSION}
    else:
        retracements = {r: swing_low  + diff * r         for r in FIB_RETRACEMENT}
        extensions   = {e: swing_low  - diff * (e - 1.0) for e in FIB_EXTENSION}

    return {"retracements": retracements, "extensions": extensions}


def get_fib_exits(swing_high: float, swing_low: float,
                  entry: float, direction: str) -> dict:
    fib  = fibonacci_levels(swing_high, swing_low, direction)
    ret  = fib["retracements"]
    ext  = fib["extensions"]
    diff = abs(swing_high - swing_low)

    if direction == "long":
        sl = ret[0.618]
        if sl >= entry: sl = ret[0.786]
        if sl >= entry: sl = entry - diff * 0.618
        tps = sorted([ext[1.272], ext[1.618], ext[2.618]])
        tps = [max(t, entry * 1.001) for t in tps]
    else:
        sl = ret[0.618]
        if sl <= entry: sl = ret[0.786]
        if sl <= entry: sl = entry + diff * 0.618
        tps = sorted([ext[1.272], ext[1.618], ext[2.618]], reverse=True)
        tps = [min(t, entry * 0.999) for t in tps]

    tp1, tp2, tp3 = tps
    risk    = abs(entry - sl)
    reward1 = abs(tp1 - entry)
    rr      = round(reward1 / risk, 2) if risk > 0 else 0

    return {
        "stop_loss":   round(sl, 6),
        "tp1":         round(tp1, 6),
        "tp2":         round(tp2, 6),
        "tp3":         round(tp3, 6),
        "risk_reward": rr,
    }


# ──────────────────────────────────────────────
# SMC 分析
# ──────────────────────────────────────────────

def detect_order_blocks(opens, highs, lows, closes, n=5):
    obs = []
    for i in range(2, len(closes) - 1):
        body       = abs(float(closes[i]) - float(opens[i]))
        prev_range = float(highs[i-1]) - float(lows[i-1])
        if body == 0 or prev_range == 0:
            continue
        if (closes[i-1] < opens[i-1] and closes[i] > highs[i-1]
                and body > prev_range * 0.5):
            obs.append({"type": "bullish_ob",
                        "high": float(highs[i-1]),
                        "low":  float(lows[i-1]), "index": i})
        elif (closes[i-1] > opens[i-1] and closes[i] < lows[i-1]
              and body > prev_range * 0.5):
            obs.append({"type": "bearish_ob",
                        "high": float(highs[i-1]),
                        "low":  float(lows[i-1]), "index": i})
    return obs[-n:] if obs else []


def detect_fvg(highs, lows, closes):
    fvgs = []
    for i in range(2, len(closes)):
        if float(lows[i]) > float(highs[i-2]):
            fvgs.append({"type": "bullish_fvg",
                         "top": float(lows[i]), "bottom": float(highs[i-2])})
        elif float(highs[i]) < float(lows[i-2]):
            fvgs.append({"type": "bearish_fvg",
                         "top": float(lows[i-2]), "bottom": float(highs[i])})
    return fvgs[-3:] if fvgs else []


def detect_bos_choch(highs, lows, closes, lookback=20):
    if len(closes) < lookback + 10:
        return None, None
    recent_high = float(highs[-lookback:-1].max())
    recent_low  = float(lows[-lookback:-1].min())
    last_close  = float(closes[-1])
    ema50       = calc_ema(closes, 50)
    slope       = float(ema50[-1]) - float(ema50[-10])
    prior_trend = "up" if slope > 0 else "down"

    bos = choch = None
    if last_close > recent_high:
        if prior_trend == "up":   bos   = "bullish_bos"
        else:                     choch = "bullish_choch"
    elif last_close < recent_low:
        if prior_trend == "down": bos   = "bearish_bos"
        else:                     choch = "bearish_choch"
    return bos, choch


# ──────────────────────────────────────────────
# 和諧型態
# ──────────────────────────────────────────────

def check_harmonic(X, A, B, C, D):
    def ratio(a, b): return abs(a)/abs(b) if abs(b) > 1e-12 else 0
    XA = A-X; AB = B-A; BC = C-B; CD = D-C
    if XA == 0 or AB == 0 or BC == 0: return None
    AB_XA = ratio(AB,XA); BC_AB = ratio(BC,AB)
    CD_BC = ratio(CD,BC); XD_XA = ratio(D-X,XA)
    def near(v,t,tol=0.08): return abs(v-t)<=tol
    d = "long" if XA > 0 else "short"
    if near(AB_XA,.618) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.272) and near(XD_XA,.786): return ("Gartley",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,2.0)   and near(XD_XA,.886): return ("Bat",d)
    if near(AB_XA,.786) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.618) and near(XD_XA,1.272): return ("Butterfly",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,3.618) and near(XD_XA,1.618): return ("Crab",d)
    return None


def scan_harmonics(highs, lows, n=80):
    h = list(highs[-n:]); l = list(lows[-n:])
    pivots = []
    for i in range(2, len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i+1]: pivots.append(h[i])
        elif l[i]<l[i-1] and l[i]<l[i+1]: pivots.append(l[i])
    if len(pivots) < 5: return None
    for i in range(len(pivots)-4):
        r = check_harmonic(*pivots[i:i+5])
        if r: return r
    return None


# ──────────────────────────────────────────────
# 主分析函數
# ──────────────────────────────────────────────

def full_analysis(symbol: str) -> Optional[dict]:
    timeframe = select_timeframe(symbol)
    data      = fetch_klines(symbol, timeframe, 200)
    if data is None or len(data) < 50:
        return None

    opens  = data[:, 1]; highs  = data[:, 2]
    lows   = data[:, 3]; closes = data[:, 4]
    vols   = data[:, 5]
    price  = float(closes[-1])

    # 指標計算
    rsi                        = calc_rsi(closes)
    macd, macd_sig, hist       = calc_macd(closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    ema20  = float(calc_ema(closes, 20)[-1])
    ema50  = float(calc_ema(closes, 50)[-1])
    ema200 = float(calc_ema(closes, 200)[-1])
    atr    = calc_atr(highs, lows, closes)

    # 新增指標
    vwap      = calc_vwap(highs, lows, closes, vols)
    adx_info  = calc_adx(highs, lows, closes)
    sr        = detect_support_resistance(highs, lows, closes)
    div       = detect_divergence(highs, lows, closes)
    funding   = fetch_funding_rate(symbol)
    oi        = fetch_open_interest(symbol)

    # 其他
    swing_high, swing_low = find_swing_points(highs, lows, 50)
    obs        = detect_order_blocks(opens, highs, lows, closes)
    fvgs       = detect_fvg(highs, lows, closes)
    bos, choch = detect_bos_choch(highs, lows, closes)
    harmonic   = scan_harmonics(highs, lows, 80)
    vol_info   = calc_volume_confirmation(vols, closes)
    candle     = detect_candle_pattern(opens, highs, lows, closes)
    upper_trend = get_upper_trend(symbol, timeframe)

    # ── 多空評分 ──
    long_score  = 0
    short_score = 0

    # RSI
    if rsi < 30:        long_score  += 2
    elif rsi < 45:      long_score  += 1
    elif rsi > 70:      short_score += 2
    elif rsi > 55:      short_score += 1

    # MACD
    if hist > 0 and macd > macd_sig:    long_score  += 2
    elif hist < 0 and macd < macd_sig:  short_score += 2

    # EMA
    if price > ema20 > ema50 > ema200:    long_score  += 3
    elif price < ema20 < ema50 < ema200:  short_score += 3
    elif price > ema50:                   long_score  += 1
    elif price < ema50:                   short_score += 1

    # Bollinger
    if price < bb_lower:   long_score  += 1
    elif price > bb_upper: short_score += 1

    # B. VWAP
    if price > vwap:   long_score  += 1
    elif price < vwap: short_score += 1

    # OB / FVG
    if any(o["type"] == "bullish_ob"  for o in obs): long_score  += 1
    if any(o["type"] == "bearish_ob"  for o in obs): short_score += 1
    if any(f["type"] == "bullish_fvg" for f in fvgs): long_score  += 1
    if any(f["type"] == "bearish_fvg" for f in fvgs): short_score += 1

    # BOS / CHoCH
    if bos   == "bullish_bos":     long_score  += 2
    elif bos == "bearish_bos":     short_score += 2
    if choch == "bullish_choch":   long_score  += 3
    elif choch == "bearish_choch": short_score += 3

    # 和諧型態
    if harmonic:
        if harmonic[1] == "long":  long_score  += 2
        else:                      short_score += 2

    # 成交量
    if vol_info["bullish_vol"]: long_score  += 2
    if vol_info["bearish_vol"]: short_score += 2

    # K 線型態
    if candle["hammer"] or candle["engulfing_bull"] or candle["doji_bull"]:
        long_score  += 2
    if candle["shooting_star"] or candle["engulfing_bear"]:
        short_score += 2

    # 多週期確認
    upper_agree = False
    if upper_trend == "up":
        long_score  += 2; upper_agree = True
    elif upper_trend == "down":
        short_score += 2; upper_agree = True

    # H. 背離
    has_divergence = False
    if div["rsi_bull_div"] or div["macd_bull_div"]:
        long_score += 3; has_divergence = True
    if div["rsi_bear_div"]:
        short_score += 3; has_divergence = True

    # C. ADX 過濾橫盤（ADX < 20 時大幅降低分數）
    if adx_info["adx"] < 20:
        long_score  = int(long_score  * 0.6)
        short_score = int(short_score * 0.6)

    # 資金費率警示（不加分，但加入訊號）
    if funding["rate"] > 0.15:  short_score += 1  # 多方過熱
    if funding["rate"] < -0.15: long_score  += 1  # 空方過熱

    # OI 確認
    if oi["oi_change"] > 2 and price > float(closes[-2]):
        long_score  += 1
    elif oi["oi_change"] > 2 and price < float(closes[-2]):
        short_score += 1

    # 訊號過濾
    score_diff = abs(long_score - short_score)
    max_score  = max(long_score, short_score)

    if long_score == short_score:   return None
    if score_diff < MIN_SCORE_DIFF: return None
    if max_score < MIN_SCORE_TOTAL: return None

    direction  = "long" if long_score > short_score else "short"

    if is_duplicate_signal(symbol, direction):
        return None

    best_entry  = get_best_entry(price, obs, fvgs, direction, atr)
    exits       = get_fib_exits(swing_high, swing_low, best_entry, direction)
    atr_sl      = calc_atr_stoploss(best_entry, atr, direction)  # D. ATR動態止損

    has_candle = any([candle["hammer"], candle["engulfing_bull"],
                      candle["doji_bull"], candle["shooting_star"],
                      candle["engulfing_bear"]])
    has_vol    = vol_info["bullish_vol"] or vol_info["bearish_vol"]
    confidence = get_confidence(score_diff, has_vol, has_candle,
                                upper_agree, adx_info["adx"], has_divergence)

    return {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "price":          round(price, 6),
        "direction":      direction,
        "long_score":     long_score,
        "short_score":    short_score,
        "confidence":     confidence,
        "entry":          best_entry,
        "entry_source":   "OB/FVG中點" if best_entry != round(price, 6) else "現價",
        "stop_loss":      exits["stop_loss"],
        "atr_stop_loss":  atr_sl,
        "tp1":            exits["tp1"],
        "tp2":            exits["tp2"],
        "tp3":            exits["tp3"],
        "risk_reward":    exits["risk_reward"],
        "rsi":            rsi,
        "macd":           round(macd, 6),
        "macd_hist":      round(hist, 6),
        "ema20":          round(ema20, 6),
        "ema50":          round(ema50, 6),
        "ema200":         round(ema200, 6),
        "bb_upper":       round(bb_upper, 6),
        "bb_lower":       round(bb_lower, 6),
        "atr":            round(atr, 6),
        "vwap":           vwap,
        "adx":            adx_info["adx"],
        "trend_strength": adx_info["trend_strength"],
        "divergence":     div["divergence"],
        "nearest_sup":    sr["nearest_sup"],
        "nearest_res":    sr["nearest_res"],
        "dist_to_sup":    sr["dist_to_sup"],
        "dist_to_res":    sr["dist_to_res"],
        "funding_rate":   funding["rate"],
        "funding_signal": funding["signal"],
        "oi_change":      oi["oi_change"],
        "oi_signal":      oi["oi_signal"],
        "swing_high":     round(swing_high, 6),
        "swing_low":      round(swing_low, 6),
        "order_blocks":   obs,
        "fvg":            fvgs,
        "bos":            bos,
        "choch":          choch,
        "candle_pattern": candle["name"],
        "vol_ratio":      vol_info["vol_ratio"],
        "upper_trend":    upper_trend or "—",
        "harmonic":       harmonic,
    }


# ──────────────────────────────────────────────
# 訊號格式化
# ──────────────────────────────────────────────

def format_signal(r: dict) -> str:
    direction_emoji = "🟢 做多 LONG" if r["direction"] == "long" else "🔴 做空 SHORT"
    harmonic_str    = f"{r['harmonic'][0]} ({r['harmonic'][1]})" if r["harmonic"] else "無"
    bos_str         = r["bos"]   or "—"
    choch_str       = r["choch"] or "—"
    upper_emoji     = "📈" if r["upper_trend"] == "up" else ("📉" if r["upper_trend"] == "down" else "—")
    sup_str = f"`{r['nearest_sup']}` ({r['dist_to_sup']}%)" if r["nearest_sup"] else "—"
    res_str = f"`{r['nearest_res']}` ({r['dist_to_res']}%)" if r["nearest_res"] else "—"

    rsi = r["rsi"]
    if rsi < 30:   rsi_label = "超賣🔥"
    elif rsi > 70: rsi_label = "超買❄️"
    elif rsi < 45: rsi_label = "偏弱⬇️"
    elif rsi > 55: rsi_label = "偏強⬆️"
    else:          rsi_label = "中性⚪️"

    return (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} ｜ {direction_emoji}\n"
        f"🎯 信心度：{r['confidence']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *進場*：`{r['entry']}` ({r['entry_source']})\n"
        f"\n"
        f"📐 *出場價位*\n"
        f"  🛑 Fib止損  (0.618)：`{r['stop_loss']}`\n"
        f"  🛑 ATR動態止損     ：`{r['atr_stop_loss']}`\n"
        f"  🎯 止盈1   (1.272)：`{r['tp1']}`\n"
        f"  🎯 止盈2   (1.618)：`{r['tp2']}`\n"
        f"  🎯 止盈3   (2.618)：`{r['tp3']}`\n"
        f"  ⚖️  風報比 R:R     ：`1 : {r['risk_reward']}`\n"
        f"\n"
        f"📊 *技術指標*\n"
        f"  RSI    ：`{r['rsi']}` {rsi_label}\n"
        f"  MACD   ：`{r['macd']}` ｜ Hist `{r['macd_hist']}`\n"
        f"  EMA20  ：`{r['ema20']}`\n"
        f"  EMA50  ：`{r['ema50']}`\n"
        f"  EMA200 ：`{r['ema200']}`\n"
        f"  BB上軌 ：`{r['bb_upper']}` / 下軌：`{r['bb_lower']}`\n"
        f"  VWAP   ：`{r['vwap']}`\n"
        f"  ATR    ：`{r['atr']}`\n"
        f"  ADX    ：`{r['adx']}` {r['trend_strength']}\n"
        f"\n"
        f"📍 *支撐壓力*\n"
        f"  最近支撐：{sup_str}\n"
        f"  最近壓力：{res_str}\n"
        f"\n"
        f"🔀 *背離*：{r['divergence']}\n"
        f"🕯 *K線型態*：{r['candle_pattern']}\n"
        f"📦 *成交量比*：`{r['vol_ratio']}x`\n"
        f"🌐 *上層週期*：{upper_emoji} {r['upper_trend']}\n"
        f"\n"
        f"💹 *籌碼分析*\n"
        f"  資金費率：`{r['funding_rate']}%` {r['funding_signal']}\n"
        f"  OI變化  ：`{r['oi_change']}%` {r['oi_signal']}\n"
        f"\n"
        f"🏗️ *SMC 結構*\n"
        f"  BOS  ：{bos_str}\n"
        f"  CHoCH：{choch_str}\n"
        f"  OB   ：{len(r['order_blocks'])} 個\n"
        f"  FVG  ：{len(r['fvg'])} 個\n"
        f"\n"
        f"🔷 *和諧型態*：{harmonic_str}\n"
        f"\n"
        f"📈 *評分*：多 {r['long_score']} ／ 空 {r['short_score']}\n"
        f"🕯 Swing High：`{r['swing_high']}` / Low：`{r['swing_low']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )


if __name__ == "__main__":
    result = full_analysis("BTC")
    if result:
        print(format_signal(result))
    else:
        print("無強力訊號")
