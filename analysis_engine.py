"""
analysis_engine.py v3.0
========================
修正與改善項目：

【潛在問題修正】
  1. RSI 使用 Wilder's Smoothing（標準算法）
  2. 平局不輸出訊號
  3. 週期查表，不額外發 API
  4. 訊號門檻：差距≥3、總分≥5
  5. fetch_klines 加入重試機制（3次）
  6. 成交量為 0 時跳過（異常資料防護）
  7. BOS/CHoCH 改用 EMA50 斜率判斷趨勢

【進場精確度提升】
  8.  成交量確認（量增價漲/跌才加分）
  9.  K線型態識別（錘子/吞噬/流星/十字星）
  10. 多週期確認（主週期 + 上層週期方向一致加分）
  11. 進場點優先取 OB/FVG 中點，不直接追現價
  12. 新增進場信心度（極高/高/中/低）
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

# 上層週期（多週期確認用）
UPPER_TIMEFRAME = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}

FIB_RETRACEMENT = [0.236, 0.382, 0.5, 0.618, 0.786]
FIB_EXTENSION   = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]

MIN_SCORE_DIFF  = 3
MIN_SCORE_TOTAL = 5

BINANCE_URL = "https://api.binance.com/api/v3/klines"


# ──────────────────────────────────────────────
# K 線抓取（含重試）
# ──────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "4h",
                 limit: int = 200, retries: int = 3) -> Optional[np.ndarray]:
    """修正5：重試機制 / 修正6：成交量異常防護"""
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
    """修正1：Wilder's Smoothing 標準 RSI"""
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
    macd   = calc_ema(closes, 12) - calc_ema(closes, 26)
    signal = calc_ema(macd, 9)
    hist   = macd - signal
    return float(macd[-1]), float(signal[-1]), float(hist[-1])


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
# 成交量確認（進場精確度提升）
# ──────────────────────────────────────────────

def calc_volume_confirmation(volumes: np.ndarray, closes: np.ndarray) -> dict:
    """
    修正8：量增價漲 → 多頭確認 / 量增價跌 → 空頭確認
    vol_ratio ≥ 1.5 才算有效放量，避免假突破
    """
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
# K 線型態識別（進場精確度提升）
# ──────────────────────────────────────────────

def detect_candle_pattern(opens, highs, lows, closes) -> dict:
    """
    修正9：識別看多/看空 K 線型態
    錘子線、看漲吞噬、十字星（看多）
    流星線、看跌吞噬（看空）
    """
    result = {
        "hammer":        False,
        "engulfing_bull": False,
        "doji_bull":     False,
        "shooting_star": False,
        "engulfing_bear": False,
        "name":          "—",
    }
    if len(closes) < 3:
        return result

    o1 = float(opens[-2]);  h1 = float(highs[-2])
    l1 = float(lows[-2]);   c1 = float(closes[-2])
    o2 = float(opens[-1]);  h2 = float(highs[-1])
    l2 = float(lows[-1]);   c2 = float(closes[-1])

    body2        = abs(c2 - o2)
    rng2         = h2 - l2
    upper_wick2  = h2 - max(o2, c2)
    lower_wick2  = min(o2, c2) - l2

    if rng2 == 0:
        return result

    # 錘子線：下影 > 實體2倍，上影小，收陽，前根收陰
    if (lower_wick2 > body2 * 2 and upper_wick2 < body2 * 0.5
            and c2 > o2 and c1 < o1):
        result["hammer"] = True
        result["name"]   = "🔨 錘子線"

    # 流星線：上影 > 實體2倍，下影小，收陰，前根收陽
    elif (upper_wick2 > body2 * 2 and lower_wick2 < body2 * 0.5
            and c2 < o2 and c1 > o1):
        result["shooting_star"] = True
        result["name"]          = "💫 流星線"

    # 看漲吞噬：大陽線完全覆蓋前根陰線
    elif (c1 < o1 and c2 > o2 and o2 <= c1 and c2 >= o1):
        result["engulfing_bull"] = True
        result["name"]           = "📈 看漲吞噬"

    # 看跌吞噬：大陰線完全覆蓋前根陽線
    elif (c1 > o1 and c2 < o2 and o2 >= c1 and c2 <= o1):
        result["engulfing_bear"] = True
        result["name"]           = "📉 看跌吞噬"

    # 十字星看多：實體極小，下影線長（低位出現）
    elif (body2 < rng2 * 0.1 and lower_wick2 > rng2 * 0.6):
        result["doji_bull"] = True
        result["name"]      = "✙ 十字星"

    return result


# ──────────────────────────────────────────────
# 多週期趨勢確認（進場精確度提升）
# ──────────────────────────────────────────────

def get_upper_trend(symbol: str, main_tf: str) -> Optional[str]:
    """
    修正10：抓上層週期 EMA50 斜率判斷趨勢方向
    主週期與上層週期一致 → 訊號更可靠
    """
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
# 進場點優化（取 OB/FVG 中點）
# ──────────────────────────────────────────────

def get_best_entry(price: float, obs: list, fvgs: list,
                   direction: str, atr: float) -> float:
    """
    修正11：進場點優先取最近 OB 或 FVG 中點
    若中點距現價 > 1×ATR 則不用（太遠），直接用現價
    """
    candidates = []

    if direction == "long":
        # 看多：取價格下方最近的 bullish OB 或 bullish FVG 中點
        for ob in obs:
            if ob["type"] == "bullish_ob":
                mid = (ob["high"] + ob["low"]) / 2
                if mid < price:
                    candidates.append(mid)
        for fvg in fvgs:
            if fvg["type"] == "bullish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2
                if mid < price:
                    candidates.append(mid)
        if candidates:
            best = max(candidates)   # 最近的（最高的）
            if price - best <= atr:
                return round(best, 6)
    else:
        # 看空：取價格上方最近的 bearish OB 或 bearish FVG 中點
        for ob in obs:
            if ob["type"] == "bearish_ob":
                mid = (ob["high"] + ob["low"]) / 2
                if mid > price:
                    candidates.append(mid)
        for fvg in fvgs:
            if fvg["type"] == "bearish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2
                if mid > price:
                    candidates.append(mid)
        if candidates:
            best = min(candidates)   # 最近的（最低的）
            if best - price <= atr:
                return round(best, 6)

    return round(price, 6)


# ──────────────────────────────────────────────
# 進場信心度
# ──────────────────────────────────────────────

def get_confidence(score_diff: int, has_volume: bool,
                   has_candle: bool, upper_agree: bool) -> str:
    """
    修正12：綜合多個確認因子給出信心度
    """
    bonus = sum([has_volume, has_candle, upper_agree])
    if score_diff >= 8 and bonus >= 2:
        return "🔥 極高"
    elif score_diff >= 5 and bonus >= 1:
        return "✅ 高"
    elif score_diff >= 3:
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
        "fib_levels": {
            "swing_high": swing_high,
            "swing_low":  swing_low,
            **{f"ret_{int(k*1000)}": round(v, 6) for k, v in ret.items()},
            **{f"ext_{int(k*1000)}": round(v, 6) for k, v in ext.items()},
        },
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
                         "top":    float(lows[i]),
                         "bottom": float(highs[i-2])})
        elif float(highs[i]) < float(lows[i-2]):
            fvgs.append({"type": "bearish_fvg",
                         "top":    float(lows[i-2]),
                         "bottom": float(highs[i])})
    return fvgs[-3:] if fvgs else []


def detect_bos_choch(highs, lows, closes, lookback=20):
    """修正7：EMA50 斜率判斷趨勢"""
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
    def ratio(a, b):
        return abs(a) / abs(b) if abs(b) > 1e-12 else 0
    XA = A-X; AB = B-A; BC = C-B; CD = D-C
    if XA == 0 or AB == 0 or BC == 0:
        return None
    AB_XA = ratio(AB, XA); BC_AB = ratio(BC, AB)
    CD_BC = ratio(CD, BC); XD_XA = ratio(D-X, XA)

    def near(v, t, tol=0.08): return abs(v-t) <= tol
    d = "long" if XA > 0 else "short"

    if near(AB_XA,.618) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.272) and near(XD_XA,.786):
        return ("Gartley", d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,2.0)   and near(XD_XA,.886):
        return ("Bat", d)
    if near(AB_XA,.786) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.618) and near(XD_XA,1.272):
        return ("Butterfly", d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,3.618) and near(XD_XA,1.618):
        return ("Crab", d)
    return None


def scan_harmonics(highs, lows, n=80):
    h = list(highs[-n:]); l = list(lows[-n:])
    pivots = []
    for i in range(2, len(h)-2):
        if h[i] > h[i-1] and h[i] > h[i+1]: pivots.append(h[i])
        elif l[i] < l[i-1] and l[i] < l[i+1]: pivots.append(l[i])
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

    # ── 指標 ──
    rsi                        = calc_rsi(closes)
    macd, signal, hist         = calc_macd(closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    ema20  = float(calc_ema(closes, 20)[-1])
    ema50  = float(calc_ema(closes, 50)[-1])
    ema200 = float(calc_ema(closes, 200)[-1])
    atr    = calc_atr(highs, lows, closes)

    # ── 斐波納契 ──
    swing_high, swing_low = find_swing_points(highs, lows, 50)

    # ── SMC ──
    obs        = detect_order_blocks(opens, highs, lows, closes)
    fvgs       = detect_fvg(highs, lows, closes)
    bos, choch = detect_bos_choch(highs, lows, closes)

    # ── 和諧型態 ──
    harmonic = scan_harmonics(highs, lows, 80)

    # ── 進場精確度加強 ──
    vol_info = calc_volume_confirmation(vols, closes)     # 修正8
    candle   = detect_candle_pattern(opens, highs, lows, closes)  # 修正9
    upper_trend = get_upper_trend(symbol, timeframe)      # 修正10

    # ──────────────────────────────────────────
    # 多空評分
    # ──────────────────────────────────────────
    long_score  = 0
    short_score = 0

    # RSI
    if rsi < 30:        long_score  += 2
    elif rsi < 45:      long_score  += 1
    elif rsi > 70:      short_score += 2
    elif rsi > 55:      short_score += 1

    # MACD
    if hist > 0 and macd > signal:    long_score  += 2
    elif hist < 0 and macd < signal:  short_score += 2

    # EMA 排列
    if price > ema20 > ema50 > ema200:    long_score  += 3
    elif price < ema20 < ema50 < ema200:  short_score += 3
    elif price > ema50:                   long_score  += 1
    elif price < ema50:                   short_score += 1

    # Bollinger
    if price < bb_lower:   long_score  += 1
    elif price > bb_upper: short_score += 1

    # SMC OB
    if any(o["type"] == "bullish_ob" for o in obs): long_score  += 1
    if any(o["type"] == "bearish_ob" for o in obs): short_score += 1

    # SMC FVG
    if any(f["type"] == "bullish_fvg" for f in fvgs): long_score  += 1
    if any(f["type"] == "bearish_fvg" for f in fvgs): short_score += 1

    # BOS / CHoCH
    if bos   == "bullish_bos":    long_score  += 2
    elif bos == "bearish_bos":    short_score += 2
    if choch == "bullish_choch":  long_score  += 3
    elif choch == "bearish_choch":short_score += 3

    # 和諧型態
    if harmonic:
        if harmonic[1] == "long":  long_score  += 2
        else:                      short_score += 2

    # 修正8：成交量確認
    if vol_info["bullish_vol"]: long_score  += 2
    if vol_info["bearish_vol"]: short_score += 2

    # 修正9：K 線型態
    if candle["hammer"] or candle["engulfing_bull"] or candle["doji_bull"]:
        long_score  += 2
    if candle["shooting_star"] or candle["engulfing_bear"]:
        short_score += 2

    # 修正10：多週期確認
    upper_agree = False
    if upper_trend == "up":
        long_score  += 2
        upper_agree = True
    elif upper_trend == "down":
        short_score += 2
        upper_agree = True

    # ── 訊號過濾 ──
    score_diff = abs(long_score - short_score)
    max_score  = max(long_score, short_score)

    if long_score == short_score:       return None  # 修正2
    if score_diff < MIN_SCORE_DIFF:     return None  # 修正5
    if max_score  < MIN_SCORE_TOTAL:    return None  # 修正6

    direction = "long" if long_score > short_score else "short"

    # 修正11：最佳進場點
    best_entry = get_best_entry(price, obs, fvgs, direction, atr)
    exits      = get_fib_exits(swing_high, swing_low, best_entry, direction)

    # 修正12：信心度
    has_candle = (candle["hammer"] or candle["engulfing_bull"] or
                  candle["doji_bull"] or candle["shooting_star"] or
                  candle["engulfing_bear"])
    has_vol    = vol_info["bullish_vol"] or vol_info["bearish_vol"]
    confidence = get_confidence(score_diff, has_vol, has_candle, upper_agree)

    return {
        "symbol":       symbol,
        "timeframe":    timeframe,
        "price":        round(price, 6),
        "direction":    direction,
        "long_score":   long_score,
        "short_score":  short_score,
        "confidence":   confidence,
        # 進場
        "entry":        best_entry,
        "entry_source": "OB/FVG中點" if best_entry != round(price, 6) else "現價",
        # 技術指標
        "rsi":          rsi,
        "macd":         round(macd, 6),
        "macd_hist":    round(hist, 6),
        "ema20":        round(ema20, 6),
        "ema50":        round(ema50, 6),
        "ema200":       round(ema200, 6),
        "bb_upper":     round(bb_upper, 6),
        "bb_lower":     round(bb_lower, 6),
        "atr":          round(atr, 6),
        # 斐波納契出場
        "swing_high":   round(swing_high, 6),
        "swing_low":    round(swing_low, 6),
        "stop_loss":    exits["stop_loss"],
        "tp1":          exits["tp1"],
        "tp2":          exits["tp2"],
        "tp3":          exits["tp3"],
        "risk_reward":  exits["risk_reward"],
        "fib_levels":   exits["fib_levels"],
        # SMC
        "order_blocks": obs,
        "fvg":          fvgs,
        "bos":          bos,
        "choch":        choch,
        # 進場精確度
        "candle_pattern": candle["name"],
        "vol_ratio":      vol_info["vol_ratio"],
        "upper_trend":    upper_trend or "—",
        # 和諧型態
        "harmonic":     harmonic,
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

    rsi = r["rsi"]
    if rsi < 30:      rsi_label = "超賣🔥"
    elif rsi > 70:    rsi_label = "超買❄️"
    elif rsi < 45:    rsi_label = "偏弱⬇️"
    elif rsi > 55:    rsi_label = "偏強⬆️"
    else:             rsi_label = "中性⚪️"

    msg = (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} ｜ {direction_emoji}\n"
        f"🎯 信心度：{r['confidence']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *進場價格*：`{r['entry']}` ({r['entry_source']})\n"
        f"\n"
        f"📐 *斐波納契出場*\n"
        f"  🛑 止損  (Fib 0.618)：`{r['stop_loss']}`\n"
        f"  🎯 止盈1 (Fib 1.272)：`{r['tp1']}`\n"
        f"  🎯 止盈2 (Fib 1.618)：`{r['tp2']}`\n"
        f"  🎯 止盈3 (Fib 2.618)：`{r['tp3']}`\n"
        f"  ⚖️  風報比 R:R        ：`1 : {r['risk_reward']}`\n"
        f"\n"
        f"📊 *技術指標*\n"
        f"  RSI      ：`{r['rsi']}` {rsi_label}\n"
        f"  MACD     ：`{r['macd']}` ｜ Hist `{r['macd_hist']}`\n"
        f"  EMA20    ：`{r['ema20']}`\n"
        f"  EMA50    ：`{r['ema50']}`\n"
        f"  EMA200   ：`{r['ema200']}`\n"
        f"  BB上軌   ：`{r['bb_upper']}` / 下軌：`{r['bb_lower']}`\n"
        f"  ATR      ：`{r['atr']}`\n"
        f"\n"
        f"🕯 *K線型態*：{r['candle_pattern']}\n"
        f"📦 *成交量比*：`{r['vol_ratio']}x`\n"
        f"🌐 *上層週期*：{upper_emoji} {r['upper_trend']}\n"
        f"\n"
        f"🏗️ *SMC 結構*\n"
        f"  BOS      ：{bos_str}\n"
        f"  CHoCH    ：{choch_str}\n"
        f"  OB 數量  ：{len(r['order_blocks'])} 個\n"
        f"  FVG 數量 ：{len(r['fvg'])} 個\n"
        f"\n"
        f"🔷 *和諧型態*：{harmonic_str}\n"
        f"\n"
        f"📈 *評分*：多 {r['long_score']} ／ 空 {r['short_score']}\n"
        f"🕯 *Swing High*：`{r['swing_high']}`\n"
        f"🕯 *Swing Low* ：`{r['swing_low']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━"
    )
    return msg


if __name__ == "__main__":
    print("=== 分析 BTC ===")
    result = full_analysis("BTC")
    if result:
        print(format_signal(result))
    else:
        print("無強力訊號")
