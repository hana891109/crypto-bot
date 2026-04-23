"""
analysis_engine.py v6.0
========================
專業交易員級別重構

核心改進：
  1. 防重複改為「價格突破冷卻」而非時間冷卻
  2. 新增突破偵測（量價配合的關鍵位突破）
  3. 移動止損機制（追蹤止損保護利潤）
  4. 市場結構判斷（趨勢市 vs 震盪市）
  5. 倉位管理建議（根據信心度給出建議%）
  6. ADX 打折改為分級而非一刀切
  7. 防重複改為突破價位確認，不用時間限制

目標勝率：70-80%
"""

import time
import requests
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# 幣種清單 & 週期
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

BINANCE_URL = "https://api.binance.com/api/v3/klines"
FUTURES_URL = "https://fapi.binance.com/fapi/v1"

# ──────────────────────────────────────────────
# 修正1：防重複改為價格突破冷卻
# 記錄上次訊號的進場價，只有價格移動超過 ATR*1.5 才視為新訊號
# ──────────────────────────────────────────────
_signal_cache: dict = {}   # {symbol_direction: entry_price}


def is_duplicate_signal(symbol: str, direction: str,
                        current_price: float, atr: float) -> bool:
    key      = f"{symbol}_{direction}"
    last_entry = _signal_cache.get(key)
    if last_entry is None:
        _signal_cache[key] = current_price
        return False
    # 價格移動超過 1.5 ATR → 視為新訊號
    if abs(current_price - last_entry) > atr * 1.5:
        _signal_cache[key] = current_price
        return False
    return True   # 價格沒有明顯移動 → 重複訊號，跳過


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


def calc_vwap(highs, lows, closes, volumes) -> float:
    typical = (highs + lows + closes) / 3
    total_vol = float(np.sum(volumes))
    if total_vol == 0:
        return float(closes[-1])
    return round(float(np.sum(typical * volumes) / total_vol), 6)


# ──────────────────────────────────────────────
# 修正4：市場結構判斷（趨勢市 vs 震盪市）
# ──────────────────────────────────────────────

def detect_market_structure(highs, lows, closes, atr: float) -> dict:
    """
    判斷目前市場狀態：
    - trending_up   → 多頭趨勢市（順勢做多）
    - trending_down → 空頭趨勢市（順勢做空）
    - ranging       → 震盪市（逆勢策略，在支撐買/壓力賣）

    判斷方法：
    1. 用 HH/HL（更高高點/更高低點）判斷上升趨勢
    2. 用 LH/LL（更低高點/更低低點）判斷下降趨勢
    3. 結合 ATR 與價格區間比較
    """
    if len(closes) < 30:
        return {"structure": "unknown", "label": "未知⚪️", "strategy": "觀望"}

    h = highs[-30:]
    l = lows[-30:]
    c = closes[-30:]

    # 抓最近的 swing high/low
    swing_highs = [float(h[i]) for i in range(2, len(h)-2)
                   if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]]
    swing_lows  = [float(l[i]) for i in range(2, len(l)-2)
                   if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]]

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1] > swing_highs[-2]   # 更高高點
        hl = swing_lows[-1]  > swing_lows[-2]    # 更高低點
        lh = swing_highs[-1] < swing_highs[-2]   # 更低高點
        ll = swing_lows[-1]  < swing_lows[-2]    # 更低低點

        if hh and hl:
            return {
                "structure": "trending_up",
                "label":     "多頭趨勢📈",
                "strategy":  "順勢做多，回調進場",
            }
        elif lh and ll:
            return {
                "structure": "trending_down",
                "label":     "空頭趨勢📉",
                "strategy":  "順勢做空，反彈進場",
            }

    # 用價格區間 vs ATR 判斷震盪
    price_range = float(h.max() - l.min())
    if price_range < atr * 8:
        return {
            "structure": "ranging",
            "label":     "震盪整理↔️",
            "strategy":  "等待突破，區間高賣低買",
        }

    return {
        "structure": "unknown",
        "label":     "結構不明⚪️",
        "strategy":  "觀望為主",
    }


# ──────────────────────────────────────────────
# 修正2：突破偵測
# ──────────────────────────────────────────────

def detect_breakout(highs, lows, closes, volumes, atr: float) -> dict:
    """
    偵測關鍵位突破：
    - 價格突破近期高點 + 成交量放大 → 看漲突破
    - 價格跌破近期低點 + 成交量放大 → 看跌突破
    - 突破必須配合成交量，否則視為假突破
    """
    if len(closes) < 25:
        return {"breakout": None, "breakout_label": "無突破"}

    recent_high = float(highs[-21:-1].max())   # 過去20根高點
    recent_low  = float(lows[-21:-1].min())    # 過去20根低點
    last_close  = float(closes[-1])
    last_vol    = float(volumes[-1])
    avg_vol     = float(volumes[-21:-1].mean())
    vol_ratio   = last_vol / avg_vol if avg_vol > 0 else 1.0

    # 突破條件：價格突破 + 成交量 ≥ 1.5倍均量
    bull_breakout = (last_close > recent_high and vol_ratio >= 1.5)
    bear_breakout = (last_close < recent_low  and vol_ratio >= 1.5)

    # 假突破過濾：突破幅度需超過 0.3 ATR
    if bull_breakout and (last_close - recent_high) < atr * 0.3:
        bull_breakout = False
    if bear_breakout and (recent_low - last_close) < atr * 0.3:
        bear_breakout = False

    if bull_breakout:
        return {
            "breakout":       "bullish",
            "breakout_label": f"🚀 看漲突破（量比{vol_ratio:.1f}x）",
            "breakout_level": round(recent_high, 6),
        }
    elif bear_breakout:
        return {
            "breakout":       "bearish",
            "breakout_label": f"💥 看跌突破（量比{vol_ratio:.1f}x）",
            "breakout_level": round(recent_low, 6),
        }
    else:
        return {"breakout": None, "breakout_label": "無突破"}


# ──────────────────────────────────────────────
# 修正3：移動止損計算
# ──────────────────────────────────────────────

def calc_trailing_stop(entry: float, current_price: float,
                       atr: float, direction: str,
                       tp1_hit: bool = False) -> dict:
    """
    移動止損邏輯：
    - 未到TP1：止損 = 進場 ± ATR*2.0（初始止損）
    - 到達TP1：止損移至進場價（保本止損）
    - 超過TP1：止損 = 最高/低點 ∓ ATR*1.0（追蹤利潤）

    回傳建議的移動止損價
    """
    if direction == "long":
        initial_sl  = round(entry - atr * 2.0, 6)
        breakeven   = round(entry + atr * 0.1, 6)   # 微利保本
        trailing_sl = round(current_price - atr * 1.0, 6)

        if not tp1_hit:
            return {"trailing_sl": initial_sl,  "sl_type": "初始止損"}
        elif current_price < entry * 1.02:
            return {"trailing_sl": breakeven,   "sl_type": "保本止損"}
        else:
            return {"trailing_sl": trailing_sl, "sl_type": "追蹤止損"}
    else:
        initial_sl  = round(entry + atr * 2.0, 6)
        breakeven   = round(entry - atr * 0.1, 6)
        trailing_sl = round(current_price + atr * 1.0, 6)

        if not tp1_hit:
            return {"trailing_sl": initial_sl,  "sl_type": "初始止損"}
        elif current_price > entry * 0.98:
            return {"trailing_sl": breakeven,   "sl_type": "保本止損"}
        else:
            return {"trailing_sl": trailing_sl, "sl_type": "追蹤止損"}


# ──────────────────────────────────────────────
# 修正5：倉位管理建議
# ──────────────────────────────────────────────

def calc_position_size(confidence: str, structure: str,
                       risk_reward: float, adx: float) -> dict:
    """
    根據信心度、市場結構、風報比給出建議倉位

    基礎倉位：10%（單筆最大風險資金的10%）
    - 極高信心 + 趨勢市 + RR≥3 → 最高25%
    - 高信心              → 15%
    - 中信心              → 10%
    - 低信心              → 5%（或觀望）

    注意：這是建議倉位佔總資金的百分比
    """
    base = 10

    # 信心度加成
    conf_bonus = {"🔥 極高": 10, "✅ 高": 5, "🟡 中": 0, "⚪️ 低": -5}
    bonus = conf_bonus.get(confidence, 0)

    # 趨勢市加成
    if structure in ("trending_up", "trending_down"):
        bonus += 5
    elif structure == "ranging":
        bonus -= 3

    # 風報比加成
    if risk_reward >= 3.0:
        bonus += 5
    elif risk_reward >= 2.0:
        bonus += 2
    elif risk_reward < 1.5:
        bonus -= 5

    # ADX 加成（修正6：分級而非一刀切）
    if adx >= 40:
        bonus += 3
    elif adx >= 25:
        bonus += 1
    elif adx < 20:
        bonus -= 5

    size = max(5, min(25, base + bonus))   # 最少5%，最多25%

    if size >= 20:
        risk_label = "🟢 積極"
    elif size >= 15:
        risk_label = "🟡 標準"
    elif size >= 10:
        risk_label = "🟠 保守"
    else:
        risk_label = "🔴 觀望"

    return {
        "position_pct": size,
        "risk_label":   risk_label,
        "note":         f"建議倉位：總資金 {size}%（{risk_label}）",
    }


# ──────────────────────────────────────────────
# ADX（修正6：分級計算，不再一刀切打折）
# ──────────────────────────────────────────────

def calc_adx(highs, lows, closes, period: int = 14) -> dict:
    if len(closes) < period * 2:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️", "adx_score": -1}

    plus_dm  = []
    minus_dm = []
    trs      = []

    for i in range(1, len(closes)):
        hd = float(highs[i]) - float(highs[i-1])
        ld = float(lows[i-1]) - float(lows[i])
        plus_dm.append(hd if hd > ld and hd > 0 else 0)
        minus_dm.append(ld if ld > hd and ld > 0 else 0)
        tr = max(float(highs[i]-lows[i]),
                 abs(float(highs[i]-closes[i-1])),
                 abs(float(lows[i]-closes[i-1])))
        trs.append(tr)

    def ws(arr, p):
        r = [sum(arr[:p])]
        for v in arr[p:]:
            r.append(r[-1] - r[-1]/p + v)
        return r

    atr14   = ws(trs,      period)
    plus14  = ws(plus_dm,  period)
    minus14 = ws(minus_dm, period)

    dx_list = []
    for i in range(len(atr14)):
        pdi = 100 * plus14[i]  / atr14[i] if atr14[i] > 0 else 0
        mdi = 100 * minus14[i] / atr14[i] if atr14[i] > 0 else 0
        dx  = 100 * abs(pdi-mdi) / (pdi+mdi) if (pdi+mdi) > 0 else 0
        dx_list.append(dx)

    adx = round(sum(dx_list[-period:]) / period, 2)

    # 修正6：分級，不再打折
    if adx >= 40:
        strength = "強趨勢🔥"
        score    = 2
    elif adx >= 25:
        strength = "趨勢中🟡"
        score    = 1
    elif adx >= 20:
        strength = "弱趨勢🟠"
        score    = 0
    else:
        strength = "橫盤⚪️"
        score    = -2   # 橫盤時扣分而非打折

    return {"adx": adx, "trend_strength": strength, "adx_score": score}


# ──────────────────────────────────────────────
# 支撐壓力位
# ──────────────────────────────────────────────

def detect_support_resistance(highs, lows, closes, lookback=100, n=3) -> dict:
    h     = highs[-lookback:]
    l     = lows[-lookback:]
    price = float(closes[-1])

    resistances = []
    supports    = []

    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            resistances.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            supports.append(float(l[i]))

    resistances = sorted([r for r in resistances if r > price])[:n]
    supports    = sorted([s for s in supports    if s < price], reverse=True)[:n]

    nearest_res = resistances[0] if resistances else None
    nearest_sup = supports[0]    if supports    else None
    dist_to_res = round((nearest_res - price) / price * 100, 2) if nearest_res else None
    dist_to_sup = round((price - nearest_sup) / price * 100, 2) if nearest_sup else None

    return {
        "supports":    supports,
        "resistances": resistances,
        "nearest_sup": nearest_sup,
        "nearest_res": nearest_res,
        "dist_to_sup": dist_to_sup,
        "dist_to_res": dist_to_res,
    }


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

    if body2 > 0 and lower_wick2 > body2*2 and upper_wick2 < body2*0.5 and c2>o2 and c1<o1:
        result["hammer"] = True;        result["name"] = "🔨 錘子線"
    elif body2>0 and upper_wick2>body2*2 and lower_wick2<body2*0.5 and c2<o2 and c1>o1:
        result["shooting_star"] = True; result["name"] = "💫 流星線"
    elif c1<o1 and c2>o2 and o2<=c1 and c2>=o1:
        result["engulfing_bull"] = True; result["name"] = "📈 看漲吞噬"
    elif c1>o1 and c2<o2 and o2>=c1 and c2<=o1:
        result["engulfing_bear"] = True; result["name"] = "📉 看跌吞噬"
    elif body2 < rng2*0.1 and lower_wick2 > rng2*0.6:
        result["doji_bull"] = True;     result["name"] = "✙ 十字星"

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
# 背離偵測
# ──────────────────────────────────────────────

def detect_divergence(highs, lows, closes, lookback=30) -> dict:
    result = {
        "rsi_bull_div": False, "rsi_bear_div": False,
        "macd_bull_div": False, "divergence": "無",
    }
    if len(closes) < lookback + 5:
        return result

    c = closes[-lookback:]
    h = highs[-lookback:]
    l = lows[-lookback:]
    n = len(c)

    rsi_series = [calc_rsi(c[:i+1]) for i in range(14, n)]
    if len(rsi_series) < 10:
        return result

    price_lows  = [(i, float(l[i])) for i in range(2, n-2)
                   if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]]
    price_highs = [(i, float(h[i])) for i in range(2, n-2)
                   if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]]

    if len(price_lows) >= 2:
        p1i, p1p = price_lows[-2]; p2i, p2p = price_lows[-1]
        r1i = p1i - 14; r2i = p2i - 14
        if (r1i >= 0 and r2i >= 0 and r2i < len(rsi_series) and
                p2p < p1p and rsi_series[r2i] > rsi_series[r1i]):
            result["rsi_bull_div"] = True
            result["divergence"]   = "📈 RSI看漲背離"

    if len(price_highs) >= 2:
        p1i, p1p = price_highs[-2]; p2i, p2p = price_highs[-1]
        r1i = p1i - 14; r2i = p2i - 14
        if (r1i >= 0 and r2i >= 0 and r2i < len(rsi_series) and
                p2p > p1p and rsi_series[r2i] < rsi_series[r1i]):
            result["rsi_bear_div"] = True
            result["divergence"]   = "📉 RSI看跌背離"

    return result


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
# 信心度（整合市場結構）
# ──────────────────────────────────────────────

def get_confidence(score_diff: int, has_volume: bool, has_candle: bool,
                   upper_agree: bool, adx: float, has_divergence: bool,
                   has_breakout: bool, structure: str) -> str:
    bonus = sum([has_volume, has_candle, upper_agree,
                 adx >= 25, has_divergence, has_breakout,
                 structure in ("trending_up", "trending_down")])
    if score_diff >= 8 and bonus >= 4:
        return "🔥 極高"
    elif score_diff >= 6 and bonus >= 3:
        return "✅ 高"
    elif score_diff >= 4 and bonus >= 2:
        return "🟡 中"
    elif score_diff >= 3:
        return "🟠 偏低"
    else:
        return "⚪️ 低"


# ──────────────────────────────────────────────
# 斐波納契
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


def get_fib_exits(swing_high, swing_low, entry, direction) -> dict:
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
        "stop_loss": round(sl, 6), "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),     "tp3": round(tp3, 6),
        "risk_reward": rr,
    }


# ──────────────────────────────────────────────
# SMC
# ──────────────────────────────────────────────

def detect_order_blocks(opens, highs, lows, closes, n=5):
    obs = []
    for i in range(2, len(closes) - 1):
        body       = abs(float(closes[i]) - float(opens[i]))
        prev_range = float(highs[i-1]) - float(lows[i-1])
        if body == 0 or prev_range == 0:
            continue
        if closes[i-1]<opens[i-1] and closes[i]>highs[i-1] and body>prev_range*0.5:
            obs.append({"type":"bullish_ob","high":float(highs[i-1]),"low":float(lows[i-1]),"index":i})
        elif closes[i-1]>opens[i-1] and closes[i]<lows[i-1] and body>prev_range*0.5:
            obs.append({"type":"bearish_ob","high":float(highs[i-1]),"low":float(lows[i-1]),"index":i})
    return obs[-n:] if obs else []


def detect_fvg(highs, lows, closes):
    fvgs = []
    for i in range(2, len(closes)):
        if float(lows[i]) > float(highs[i-2]):
            fvgs.append({"type":"bullish_fvg","top":float(lows[i]),"bottom":float(highs[i-2])})
        elif float(highs[i]) < float(lows[i-2]):
            fvgs.append({"type":"bearish_fvg","top":float(lows[i-2]),"bottom":float(highs[i])})
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
    def ratio(a, b): return abs(a)/abs(b) if abs(b)>1e-12 else 0
    XA=A-X; AB=B-A; BC=C-B; CD=D-C
    if XA==0 or AB==0 or BC==0: return None
    AB_XA=ratio(AB,XA); BC_AB=ratio(BC,AB); CD_BC=ratio(CD,BC); XD_XA=ratio(D-X,XA)
    def near(v,t,tol=0.08): return abs(v-t)<=tol
    d="long" if XA>0 else "short"
    if near(AB_XA,.618) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.272) and near(XD_XA,.786): return ("Gartley",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,2.0)   and near(XD_XA,.886): return ("Bat",d)
    if near(AB_XA,.786) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.618) and near(XD_XA,1.272): return ("Butterfly",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,3.618) and near(XD_XA,1.618): return ("Crab",d)
    return None


def scan_harmonics(highs, lows, n=80):
    h=list(highs[-n:]); l=list(lows[-n:]); pivots=[]
    for i in range(2,len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i+1]: pivots.append(h[i])
        elif l[i]<l[i-1] and l[i]<l[i+1]: pivots.append(l[i])
    if len(pivots)<5: return None
    for i in range(len(pivots)-4):
        r=check_harmonic(*pivots[i:i+5])
        if r: return r
    return None


# ──────────────────────────────────────────────
# 外部資料
# ──────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    try:
        resp  = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        data  = resp.json()["data"][0]
        value = int(data["value"])
        if value <= 24:   emoji = "😱 極度恐懼"
        elif value <= 49: emoji = "😨 恐懼"
        elif value <= 74: emoji = "😊 貪婪"
        else:             emoji = "🤑 極度貪婪"
        return {"value": value, "label": emoji}
    except Exception:
        return {"value": -1, "label": "無法取得"}


def fetch_funding_rate(symbol: str) -> dict:
    try:
        resp = requests.get(f"{FUTURES_URL}/premiumIndex",
                            params={"symbol": f"{symbol}USDT"}, timeout=8)
        rate = round(float(resp.json()["lastFundingRate"]) * 100, 4)
        if rate > 0.1:   signal = "多方擁擠⚠️"
        elif rate < -0.1: signal = "空方擁擠⚠️"
        else:             signal = "市場平衡✅"
        return {"rate": rate, "signal": signal}
    except Exception:
        return {"rate": 0.0, "signal": "無法取得"}


def fetch_open_interest(symbol: str) -> dict:
    try:
        r1 = requests.get(f"{FUTURES_URL}/openInterest",
                          params={"symbol": f"{symbol}USDT"}, timeout=8)
        current_oi = float(r1.json()["openInterest"])
        r2 = requests.get(f"{FUTURES_URL}/openInterestHist",
                          params={"symbol": f"{symbol}USDT", "period": "1h", "limit": 2}, timeout=8)
        hist = r2.json()
        if len(hist) >= 2:
            prev_oi   = float(hist[0]["sumOpenInterest"])
            oi_change = round((current_oi - prev_oi) / prev_oi * 100, 2) if prev_oi > 0 else 0
        else:
            oi_change = 0.0
        oi_signal = "OI急增📈" if oi_change>2 else ("OI急減📉" if oi_change<-2 else "OI穩定➡️")
        return {"oi": round(current_oi, 2), "oi_change": oi_change, "oi_signal": oi_signal}
    except Exception:
        return {"oi": 0.0, "oi_change": 0.0, "oi_signal": "無法取得"}


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

    # 指標
    rsi                        = calc_rsi(closes)
    macd, macd_sig, hist       = calc_macd(closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    ema20  = float(calc_ema(closes, 20)[-1])
    ema50  = float(calc_ema(closes, 50)[-1])
    ema200 = float(calc_ema(closes, 200)[-1])
    atr    = calc_atr(highs, lows, closes)
    vwap   = calc_vwap(highs, lows, closes, vols)

    # 新功能
    adx_info   = calc_adx(highs, lows, closes)
    market_st  = detect_market_structure(highs, lows, closes, atr)  # 修正4
    breakout   = detect_breakout(highs, lows, closes, vols, atr)    # 修正2
    sr         = detect_support_resistance(highs, lows, closes)
    div        = detect_divergence(highs, lows, closes)
    funding    = fetch_funding_rate(symbol)
    oi         = fetch_open_interest(symbol)

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

    # VWAP
    if price > vwap:   long_score  += 1
    elif price < vwap: short_score += 1

    # ADX（修正6：分級加減分）
    long_score  += adx_info["adx_score"] if adx_info["adx_score"] > 0 else 0
    short_score += adx_info["adx_score"] if adx_info["adx_score"] > 0 else 0
    if adx_info["adx_score"] < 0:
        long_score  = max(0, long_score  + adx_info["adx_score"])
        short_score = max(0, short_score + adx_info["adx_score"])

    # 市場結構（修正4：趨勢市加分，震盪市不加分）
    if market_st["structure"] == "trending_up":
        long_score  += 2
    elif market_st["structure"] == "trending_down":
        short_score += 2

    # 突破（修正2：突破加重分）
    if breakout["breakout"] == "bullish":
        long_score  += 4
    elif breakout["breakout"] == "bearish":
        short_score += 4

    # OB / FVG
    if any(o["type"]=="bullish_ob"  for o in obs): long_score  += 1
    if any(o["type"]=="bearish_ob"  for o in obs): short_score += 1
    if any(f["type"]=="bullish_fvg" for f in fvgs): long_score  += 1
    if any(f["type"]=="bearish_fvg" for f in fvgs): short_score += 1

    # BOS / CHoCH
    if bos   == "bullish_bos":     long_score  += 2
    elif bos == "bearish_bos":     short_score += 2
    if choch == "bullish_choch":   long_score  += 3
    elif choch == "bearish_choch": short_score += 3

    # 和諧型態
    if harmonic:
        if harmonic[1]=="long": long_score  += 2
        else:                   short_score += 2

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

    # 背離
    has_divergence = False
    if div["rsi_bull_div"] or div["macd_bull_div"]:
        long_score += 3; has_divergence = True
    if div["rsi_bear_div"]:
        short_score += 3; has_divergence = True

    # 資金費率
    if funding["rate"] > 0.15:  short_score += 1
    if funding["rate"] < -0.15: long_score  += 1

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

    direction = "long" if long_score > short_score else "short"

    # 修正1：價格突破冷卻（取代時間冷卻）
    if is_duplicate_signal(symbol, direction, price, atr):
        return None

    best_entry = get_best_entry(price, obs, fvgs, direction, atr)
    exits      = get_fib_exits(swing_high, swing_low, best_entry, direction)
    atr_sl     = (round(best_entry - atr * 2.0, 6) if direction == "long"
                  else round(best_entry + atr * 2.0, 6))

    has_candle = any([candle["hammer"], candle["engulfing_bull"], candle["doji_bull"],
                      candle["shooting_star"], candle["engulfing_bear"]])
    has_vol    = vol_info["bullish_vol"] or vol_info["bearish_vol"]
    has_bo     = breakout["breakout"] is not None
    confidence = get_confidence(score_diff, has_vol, has_candle, upper_agree,
                                adx_info["adx"], has_divergence, has_bo,
                                market_st["structure"])

    # 修正5：倉位管理
    position   = calc_position_size(confidence, market_st["structure"],
                                    exits["risk_reward"], adx_info["adx"])

    # 修正3：初始移動止損
    trailing   = calc_trailing_stop(best_entry, price, atr, direction, False)

    return {
        "symbol":           symbol,
        "timeframe":        timeframe,
        "price":            round(price, 6),
        "direction":        direction,
        "long_score":       long_score,
        "short_score":      short_score,
        "confidence":       confidence,
        "entry":            best_entry,
        "entry_source":     "OB/FVG中點" if best_entry != round(price,6) else "現價",
        "stop_loss":        exits["stop_loss"],
        "atr_stop_loss":    atr_sl,
        "trailing_sl":      trailing["trailing_sl"],
        "sl_type":          trailing["sl_type"],
        "tp1":              exits["tp1"],
        "tp2":              exits["tp2"],
        "tp3":              exits["tp3"],
        "risk_reward":      exits["risk_reward"],
        "position_pct":     position["position_pct"],
        "risk_label":       position["risk_label"],
        "market_structure": market_st["label"],
        "market_strategy":  market_st["strategy"],
        "breakout":         breakout["breakout_label"],
        "rsi":              rsi,
        "macd":             round(macd, 6),
        "macd_hist":        round(hist, 6),
        "ema20":            round(ema20, 6),
        "ema50":            round(ema50, 6),
        "ema200":           round(ema200, 6),
        "bb_upper":         round(bb_upper, 6),
        "bb_lower":         round(bb_lower, 6),
        "vwap":             vwap,
        "atr":              round(atr, 6),
        "adx":              adx_info["adx"],
        "trend_strength":   adx_info["trend_strength"],
        "divergence":       div["divergence"],
        "nearest_sup":      sr["nearest_sup"],
        "nearest_res":      sr["nearest_res"],
        "dist_to_sup":      sr["dist_to_sup"],
        "dist_to_res":      sr["dist_to_res"],
        "funding_rate":     funding["rate"],
        "funding_signal":   funding["signal"],
        "oi_change":        oi["oi_change"],
        "oi_signal":        oi["oi_signal"],
        "swing_high":       round(swing_high, 6),
        "swing_low":        round(swing_low, 6),
        "order_blocks":     obs,
        "fvg":              fvgs,
        "bos":              bos,
        "choch":            choch,
        "candle_pattern":   candle["name"],
        "vol_ratio":        vol_info["vol_ratio"],
        "upper_trend":      upper_trend or "—",
        "harmonic":         harmonic,
    }


# ──────────────────────────────────────────────
# 訊號格式化
# ──────────────────────────────────────────────

def format_signal(r: dict) -> str:
    d_emoji  = "🟢 做多 LONG" if r["direction"] == "long" else "🔴 做空 SHORT"
    h_str    = f"{r['harmonic'][0]} ({r['harmonic'][1]})" if r["harmonic"] else "無"
    bos_str  = r["bos"]   or "—"
    choch_str= r["choch"] or "—"
    up_emoji = "📈" if r["upper_trend"]=="up" else ("📉" if r["upper_trend"]=="down" else "—")
    sup_str  = f"`{r['nearest_sup']}` ({r['dist_to_sup']}%)" if r["nearest_sup"] else "—"
    res_str  = f"`{r['nearest_res']}` ({r['dist_to_res']}%)" if r["nearest_res"] else "—"

    rsi = r["rsi"]
    if rsi < 30:   rl = "超賣🔥"
    elif rsi > 70: rl = "超買❄️"
    elif rsi < 45: rl = "偏弱⬇️"
    elif rsi > 55: rl = "偏強⬆️"
    else:          rl = "中性⚪️"

    return (
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} ｜ {d_emoji}\n"
        f"🎯 信心度：{r['confidence']} ｜ 倉位：{r['position_pct']}% {r['risk_label']}\n"
        f"🏛 市場結構：{r['market_structure']}\n"
        f"💡 策略建議：{r['market_strategy']}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *進場*：`{r['entry']}` ({r['entry_source']})\n"
        f"\n"
        f"📐 *出場價位*\n"
        f"  🛑 Fib止損        ：`{r['stop_loss']}`\n"
        f"  🛑 ATR止損        ：`{r['atr_stop_loss']}`\n"
        f"  🛑 {r['sl_type']}：`{r['trailing_sl']}`\n"
        f"  🎯 止盈1 (1.272) ：`{r['tp1']}`\n"
        f"  🎯 止盈2 (1.618) ：`{r['tp2']}`\n"
        f"  🎯 止盈3 (2.618) ：`{r['tp3']}`\n"
        f"  ⚖️  風報比 R:R    ：`1 : {r['risk_reward']}`\n"
        f"\n"
        f"🚀 *突破訊號*：{r['breakout']}\n"
        f"🔀 *背離*    ：{r['divergence']}\n"
        f"\n"
        f"📊 *技術指標*\n"
        f"  RSI    ：`{r['rsi']}` {rl}\n"
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
        f"🕯 *K線型態*：{r['candle_pattern']}\n"
        f"📦 *成交量比*：`{r['vol_ratio']}x`\n"
        f"🌐 *上層週期*：{up_emoji} {r['upper_trend']}\n"
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
        f"🔷 *和諧型態*：{h_str}\n"
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
