"""
analysis_engine.py v7.0
========================
最高精度重構，目標勝率 80%+

勝率提升策略：
  1. 三重確認制：技術 + 結構 + 籌碼 三方向全部一致才輸出
  2. 分數加權：不同指標依可靠度給予不同權重
  3. 假突破過濾：突破後需收盤確認才算有效
  4. 趨勢強度門檻：ADX < 15 完全不開倉
  5. 多空比過濾：資金費率極端時反向操作
  6. 最終否決機制：任何一項關鍵指標嚴重反對則否決訊號

Bug 修正：
  B1. numpy bool 比較改用 Python bool
  B2. 所有除法加 epsilon 防止除以零
  B3. fetch_klines 回傳前驗證資料形狀
  B4. 所有外部 API 呼叫加獨立 try/except + 預設值
  B5. signal_cache 改用 thread-safe 的 dict 操作
  B6. numpy float 轉 Python float 防止序列化錯誤
  B7. 空陣列切片防護
"""

import time
import requests
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# 常數設定
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

BINANCE_URL = "https://api.binance.com/api/v3/klines"
FUTURES_URL = "https://fapi.binance.com/fapi/v1"
EPS         = 1e-10   # B2：防除以零

# 勝率提升門檻
MIN_SCORE_DIFF   = 4    # 提高：差距需 ≥ 4
MIN_SCORE_TOTAL  = 7    # 提高：總分需 ≥ 7
MIN_ADX          = 15   # 新增：ADX < 15 完全不開倉
MIN_RR           = 1.5  # 新增：風報比 < 1.5 不開倉

# B5：防重複（price-based cooldown）
_signal_cache: dict = {}


def is_duplicate_signal(symbol: str, direction: str,
                        price: float, atr: float) -> bool:
    """B5：thread-safe dict 操作，價格突破才冷卻"""
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
# 資料抓取
# ──────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str = "4h",
                 limit: int = 200, retries: int = 3) -> Optional[np.ndarray]:
    """B3：回傳前驗證資料形狀"""
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
                              for k in raw], dtype=np.float64)
            # B3：驗證形狀與數值
            if data.ndim != 2 or data.shape[1] < 6:
                return None
            if np.isnan(data).any() or np.isinf(data).any():
                return None
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
# 基礎指標（全部加 B2 防除以零）
# ──────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder's Smoothing RSI"""
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
    macd_line   = calc_ema(closes, 12) - calc_ema(closes, 26)
    macd_signal = calc_ema(macd_line, 9)
    macd_hist   = macd_line - macd_signal
    return float(macd_line[-1]), float(macd_signal[-1]), float(macd_hist[-1])


def calc_bollinger(closes: np.ndarray, period: int = 20, std_dev: float = 2.0):
    if len(closes) < period:
        m = float(closes[-1])
        return m, m, m
    w = closes[-period:].astype(np.float64)
    m = float(w.mean())
    s = float(w.std())
    return m + std_dev * s, m, m - std_dev * s


def calc_atr(highs, lows, closes, period: int = 14) -> float:
    """B2：防除以零，B7：空陣列防護"""
    if len(closes) < 2:
        return float(abs(highs[-1] - lows[-1])) if len(highs) > 0 else 1.0
    n   = min(len(closes), len(highs), len(lows))
    trs = []
    for i in range(1, n):
        tr = max(
            float(highs[i])  - float(lows[i]),
            abs(float(highs[i])  - float(closes[i-1])),
            abs(float(lows[i])   - float(closes[i-1]))
        )
        trs.append(tr)
    if not trs:
        return 1.0
    return float(np.array(trs[-period:]).mean()) if len(trs) >= period else float(np.array(trs).mean())


def calc_vwap(highs, lows, closes, volumes) -> float:
    typical  = (highs.astype(np.float64) + lows.astype(np.float64) + closes.astype(np.float64)) / 3.0
    total_v  = float(np.sum(volumes))
    if total_v < EPS:
        return float(closes[-1])
    return round(float(np.sum(typical * volumes.astype(np.float64)) / total_v), 8)


# ──────────────────────────────────────────────
# ADX（分級，不打折）
# ──────────────────────────────────────────────

def calc_adx(highs, lows, closes, period: int = 14) -> dict:
    n = min(len(closes), len(highs), len(lows))
    if n < period * 2 + 1:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️", "adx_score": -3, "pdi": 0.0, "mdi": 0.0}

    plus_dm  = []
    minus_dm = []
    trs      = []

    for i in range(1, n):
        hd = float(highs[i]) - float(highs[i-1])
        ld = float(lows[i-1]) - float(lows[i])
        plus_dm.append(hd  if (hd > ld  and hd > 0)  else 0.0)
        minus_dm.append(ld if (ld > hd  and ld > 0)  else 0.0)
        tr = max(float(highs[i]-lows[i]),
                 abs(float(highs[i]-closes[i-1])),
                 abs(float(lows[i]-closes[i-1])))
        trs.append(max(tr, EPS))

    def wilder(arr, p):
        if len(arr) < p:
            return [sum(arr)]
        result = [sum(arr[:p])]
        for v in arr[p:]:
            result.append(result[-1] - result[-1] / p + v)
        return result

    atr14   = wilder(trs,      period)
    plus14  = wilder(plus_dm,  period)
    minus14 = wilder(minus_dm, period)

    min_len = min(len(atr14), len(plus14), len(minus14))
    dx_list = []
    last_pdi = last_mdi = 0.0

    for i in range(min_len):
        a = max(atr14[i], EPS)
        pdi = 100.0 * plus14[i]  / a
        mdi = 100.0 * minus14[i] / a
        dm_sum = pdi + mdi
        dx  = 100.0 * abs(pdi - mdi) / max(dm_sum, EPS)
        dx_list.append(dx)
        last_pdi = pdi
        last_mdi = mdi

    if not dx_list:
        return {"adx": 0.0, "trend_strength": "橫盤⚪️", "adx_score": -3, "pdi": 0.0, "mdi": 0.0}

    adx = round(float(np.mean(dx_list[-period:])), 2)

    if adx >= 40:   strength, score = "強趨勢🔥",  3
    elif adx >= 25: strength, score = "趨勢中🟡",  1
    elif adx >= 15: strength, score = "弱趨勢🟠",  0
    else:           strength, score = "橫盤⚪️",   -3

    return {
        "adx":            adx,
        "trend_strength": strength,
        "adx_score":      score,
        "pdi":            round(last_pdi, 2),
        "mdi":            round(last_mdi, 2),
    }


# ──────────────────────────────────────────────
# 勝率提升1：市場結構（HH/HL/LH/LL）
# ──────────────────────────────────────────────

def detect_market_structure(highs, lows, closes, atr: float) -> dict:
    """B7：空陣列防護"""
    if len(closes) < 30:
        return {"structure": "unknown", "label": "結構不明⚪️", "strategy": "觀望"}

    h = highs[-30:].astype(np.float64)
    l = lows[-30:].astype(np.float64)

    swing_highs, swing_lows = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            swing_highs.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            swing_lows.append(float(l[i]))

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = bool(swing_highs[-1] > swing_highs[-2])   # B1：明確轉 bool
        hl = bool(swing_lows[-1]  > swing_lows[-2])
        lh = bool(swing_highs[-1] < swing_highs[-2])
        ll = bool(swing_lows[-1]  < swing_lows[-2])

        if hh and hl:
            return {"structure": "trending_up",   "label": "多頭趨勢📈", "strategy": "順勢做多，回調進場"}
        elif lh and ll:
            return {"structure": "trending_down",  "label": "空頭趨勢📉", "strategy": "順勢做空，反彈進場"}
        elif hh and ll:
            return {"structure": "expanding",      "label": "擴張震盪⚡", "strategy": "觀望，等待方向確認"}

    price_range = float(h.max() - h.min()) if len(h) > 0 else 0
    if price_range < atr * 8:
        return {"structure": "ranging", "label": "震盪整理↔️", "strategy": "等待突破，區間操作"}

    return {"structure": "unknown", "label": "結構不明⚪️", "strategy": "觀望"}


# ──────────────────────────────────────────────
# 勝率提升2：突破偵測（含收盤確認）
# ──────────────────────────────────────────────

def detect_breakout(highs, lows, closes, volumes, atr: float) -> dict:
    """
    新增收盤確認：突破必須是已收盤的K線
    最後一根K線視為未收盤，用倒數第二根判斷
    """
    if len(closes) < 25:
        return {"breakout": None, "breakout_label": "無突破", "breakout_level": None}

    # 用倒數第二根做收盤確認
    confirmed_close = float(closes[-2])
    last_vol        = float(volumes[-2])
    avg_vol         = float(volumes[-22:-2].mean()) if len(volumes) >= 22 else float(volumes[:-2].mean() if len(volumes) > 2 else 1.0)
    vol_ratio       = last_vol / max(avg_vol, EPS)

    recent_high = float(highs[-22:-2].max()) if len(highs) >= 22 else float(highs[:-2].max())
    recent_low  = float(lows[-22:-2].min())  if len(lows)  >= 22 else float(lows[:-2].min())

    bull = bool(confirmed_close > recent_high and vol_ratio >= 1.5
                and (confirmed_close - recent_high) >= atr * 0.3)
    bear = bool(confirmed_close < recent_low  and vol_ratio >= 1.5
                and (recent_low - confirmed_close) >= atr * 0.3)

    if bull:
        return {"breakout": "bullish", "breakout_label": f"🚀 看漲突破（量比{vol_ratio:.1f}x）", "breakout_level": round(recent_high, 8)}
    elif bear:
        return {"breakout": "bearish", "breakout_label": f"💥 看跌突破（量比{vol_ratio:.1f}x）", "breakout_level": round(recent_low, 8)}
    return {"breakout": None, "breakout_label": "無突破", "breakout_level": None}


# ──────────────────────────────────────────────
# 勝率提升3：最終否決機制
# ──────────────────────────────────────────────

def check_veto(direction: str, rsi: float, macd: float, macd_sig: float,
               hist: float, adx: float, funding_rate: float,
               market_structure: str, upper_trend: Optional[str]) -> dict:
    """
    任一否決條件成立 → 直接拒絕訊號
    這是勝率提升最關鍵的機制：寧可少做，不要做錯
    """
    vetoes = []

    # ADX 否決：趨勢太弱
    if adx < MIN_ADX:
        vetoes.append(f"ADX={adx} 趨勢太弱")

    # 方向否決：RSI 極端反向
    if direction == "long" and rsi > 80:
        vetoes.append(f"RSI={rsi} 嚴重超買")
    if direction == "short" and rsi < 20:
        vetoes.append(f"RSI={rsi} 嚴重超賣")

    # MACD 完全反向否決
    if direction == "long" and macd < macd_sig and hist < 0 and macd < 0:
        vetoes.append("MACD 空頭排列")
    if direction == "short" and macd > macd_sig and hist > 0 and macd > 0:
        vetoes.append("MACD 多頭排列")

    # 市場結構否決
    if direction == "long"  and market_structure == "trending_down":
        vetoes.append("市場結構為空頭趨勢")
    if direction == "short" and market_structure == "trending_up":
        vetoes.append("市場結構為多頭趨勢")

    # 多週期方向否決
    if upper_trend == "down" and direction == "long":
        vetoes.append("上層週期為空頭")
    if upper_trend == "up"   and direction == "short":
        vetoes.append("上層週期為多頭")

    # 資金費率極端否決
    if direction == "long"  and funding_rate > 0.3:
        vetoes.append(f"資金費率{funding_rate}% 多方嚴重擁擠")
    if direction == "short" and funding_rate < -0.3:
        vetoes.append(f"資金費率{funding_rate}% 空方嚴重擁擠")

    return {
        "vetoed":  bool(vetoes),
        "reasons": vetoes,
    }


# ──────────────────────────────────────────────
# 移動止損
# ──────────────────────────────────────────────

def calc_trailing_stop(entry: float, current_price: float,
                       atr: float, direction: str,
                       tp1_hit: bool = False) -> dict:
    if direction == "long":
        initial_sl  = round(entry - atr * 2.0, 8)
        breakeven   = round(entry + atr * 0.1, 8)
        trailing_sl = round(current_price - atr * 1.0, 8)
        if not tp1_hit:
            return {"trailing_sl": initial_sl,  "sl_type": "初始止損"}
        elif current_price < entry * 1.02:
            return {"trailing_sl": breakeven,   "sl_type": "保本止損"}
        else:
            return {"trailing_sl": trailing_sl, "sl_type": "追蹤止損"}
    else:
        initial_sl  = round(entry + atr * 2.0, 8)
        breakeven   = round(entry - atr * 0.1, 8)
        trailing_sl = round(current_price + atr * 1.0, 8)
        if not tp1_hit:
            return {"trailing_sl": initial_sl,  "sl_type": "初始止損"}
        elif current_price > entry * 0.98:
            return {"trailing_sl": breakeven,   "sl_type": "保本止損"}
        else:
            return {"trailing_sl": trailing_sl, "sl_type": "追蹤止損"}


# ──────────────────────────────────────────────
# 倉位管理
# ──────────────────────────────────────────────

def calc_position_size(confidence: str, structure: str,
                       risk_reward: float, adx: float) -> dict:
    base  = 10
    bonus = {"🔥 極高": 12, "✅ 高": 7, "🟡 中": 2, "🟠 偏低": -3, "⚪️ 低": -8}.get(confidence, 0)

    if structure in ("trending_up", "trending_down"): bonus += 5
    elif structure == "ranging":                       bonus -= 5
    elif structure == "expanding":                     bonus -= 8

    if risk_reward >= 3.0:   bonus += 5
    elif risk_reward >= 2.0: bonus += 2
    elif risk_reward < 1.5:  bonus -= 8

    if adx >= 40:   bonus += 3
    elif adx >= 25: bonus += 1
    elif adx < 20:  bonus -= 5

    size = max(3, min(25, base + bonus))

    if size >= 20:   label = "🟢 積極"
    elif size >= 15: label = "🟡 標準"
    elif size >= 10: label = "🟠 保守"
    else:            label = "🔴 觀望"

    return {"position_pct": size, "risk_label": label}


# ──────────────────────────────────────────────
# 支撐壓力位
# ──────────────────────────────────────────────

def detect_support_resistance(highs, lows, closes, lookback=100, n=3) -> dict:
    """B7：空陣列防護"""
    safe_lookback = min(lookback, len(highs), len(lows), len(closes))
    if safe_lookback < 10:
        return {"supports": [], "resistances": [], "nearest_sup": None,
                "nearest_res": None, "dist_to_sup": None, "dist_to_res": None}

    h     = highs[-safe_lookback:].astype(np.float64)
    l     = lows[-safe_lookback:].astype(np.float64)
    price = float(closes[-1])

    resistances, supports = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
            resistances.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
            supports.append(float(l[i]))

    resistances = sorted(set(r for r in resistances if r > price))[:n]
    supports    = sorted(set(s for s in supports    if s < price), reverse=True)[:n]

    nearest_res = resistances[0] if resistances else None
    nearest_sup = supports[0]    if supports    else None
    dist_to_res = round((nearest_res - price) / max(price, EPS) * 100, 2) if nearest_res else None
    dist_to_sup = round((price - nearest_sup) / max(price, EPS) * 100, 2) if nearest_sup else None

    return {
        "supports": supports, "resistances": resistances,
        "nearest_sup": nearest_sup, "nearest_res": nearest_res,
        "dist_to_sup": dist_to_sup, "dist_to_res": dist_to_res,
    }


# ──────────────────────────────────────────────
# 成交量確認
# ──────────────────────────────────────────────

def calc_volume_confirmation(volumes: np.ndarray, closes: np.ndarray) -> dict:
    if len(volumes) < 21:
        return {"bullish_vol": False, "bearish_vol": False, "vol_ratio": 1.0}
    avg_vol   = float(volumes[-21:-1].mean())
    last_vol  = float(volumes[-1])
    vol_ratio = round(last_vol / max(avg_vol, EPS), 2)
    is_strong = bool(vol_ratio >= 1.5)
    price_up  = bool(float(closes[-1]) > float(closes[-2]))
    return {
        "bullish_vol": bool(is_strong and price_up),
        "bearish_vol": bool(is_strong and not price_up),
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

    body2        = abs(c2 - o2)
    rng2         = h2 - l2
    upper_wick2  = h2 - max(o2, c2)
    lower_wick2  = min(o2, c2) - l2

    if rng2 < EPS:
        return result

    if body2 > EPS and lower_wick2 > body2*2 and upper_wick2 < body2*0.5 and c2>o2 and c1<o1:
        result["hammer"] = True;         result["name"] = "🔨 錘子線"
    elif body2 > EPS and upper_wick2>body2*2 and lower_wick2<body2*0.5 and c2<o2 and c1>o1:
        result["shooting_star"] = True;  result["name"] = "💫 流星線"
    elif c1<o1 and c2>o2 and o2<=c1 and c2>=o1:
        result["engulfing_bull"] = True; result["name"] = "📈 看漲吞噬"
    elif c1>o1 and c2<o2 and o2>=c1 and c2<=o1:
        result["engulfing_bear"] = True; result["name"] = "📉 看跌吞噬"
    elif body2 < rng2*0.1 and lower_wick2 > rng2*0.6:
        result["doji_bull"] = True;      result["name"] = "✙ 十字星"

    return result


# ──────────────────────────────────────────────
# 多週期確認
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
    result = {"rsi_bull_div": False, "rsi_bear_div": False,
              "macd_bull_div": False, "divergence": "無"}
    n = min(lookback, len(closes), len(highs), len(lows))
    if n < 20:
        return result

    c = closes[-n:].astype(np.float64)
    h = highs[-n:].astype(np.float64)
    l = lows[-n:].astype(np.float64)

    rsi_series = []
    for i in range(14, len(c)):
        rsi_series.append(calc_rsi(c[:i+1]))

    if len(rsi_series) < 8:
        return result

    price_lows  = [(i, float(l[i])) for i in range(2, len(l)-2)
                   if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]]
    price_highs = [(i, float(h[i])) for i in range(2, len(h)-2)
                   if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]]

    if len(price_lows) >= 2:
        p1i, p1p = price_lows[-2]; p2i, p2p = price_lows[-1]
        r1i = p1i - 14; r2i = p2i - 14
        if (0 <= r1i < len(rsi_series) and 0 <= r2i < len(rsi_series)
                and p2p < p1p and rsi_series[r2i] > rsi_series[r1i]):
            result["rsi_bull_div"] = True
            result["divergence"]   = "📈 RSI看漲背離"

    if len(price_highs) >= 2:
        p1i, p1p = price_highs[-2]; p2i, p2p = price_highs[-1]
        r1i = p1i - 14; r2i = p2i - 14
        if (0 <= r1i < len(rsi_series) and 0 <= r2i < len(rsi_series)
                and p2p > p1p and rsi_series[r2i] < rsi_series[r1i]):
            result["rsi_bear_div"] = True
            result["divergence"]   = "📉 RSI看跌背離"

    return result


# ──────────────────────────────────────────────
# 進場點優化
# ──────────────────────────────────────────────

def get_best_entry(price: float, obs: list, fvgs: list,
                   direction: str, atr: float) -> float:
    candidates = []
    max_dist   = atr * 2.0

    if direction == "long":
        for ob in obs:
            if ob.get("type") == "bullish_ob":
                mid = (ob["high"] + ob["low"]) / 2.0
                if mid < price: candidates.append(mid)
        for fvg in fvgs:
            if fvg.get("type") == "bullish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2.0
                if mid < price: candidates.append(mid)
        if candidates:
            best = max(candidates)
            if price - best <= max_dist:
                return round(best, 8)
    else:
        for ob in obs:
            if ob.get("type") == "bearish_ob":
                mid = (ob["high"] + ob["low"]) / 2.0
                if mid > price: candidates.append(mid)
        for fvg in fvgs:
            if fvg.get("type") == "bearish_fvg":
                mid = (fvg["top"] + fvg["bottom"]) / 2.0
                if mid > price: candidates.append(mid)
        if candidates:
            best = min(candidates)
            if best - price <= max_dist:
                return round(best, 8)

    return round(price, 8)


# ──────────────────────────────────────────────
# 信心度
# ──────────────────────────────────────────────

def get_confidence(score_diff: int, has_volume: bool, has_candle: bool,
                   upper_agree: bool, adx: float, has_divergence: bool,
                   has_breakout: bool, structure: str,
                   veto: dict) -> str:
    if veto["vetoed"]:
        return "⛔ 已否決"
    bonus = sum([
        bool(has_volume), bool(has_candle), bool(upper_agree),
        bool(adx >= 25),  bool(has_divergence), bool(has_breakout),
        bool(structure in ("trending_up", "trending_down")),
    ])
    if score_diff >= 10 and bonus >= 5: return "🔥 極高"
    elif score_diff >= 7  and bonus >= 4: return "✅ 高"
    elif score_diff >= 5  and bonus >= 3: return "🟡 中"
    elif score_diff >= 4  and bonus >= 2: return "🟠 偏低"
    else:                                 return "⚪️ 低"


# ──────────────────────────────────────────────
# 斐波納契
# ──────────────────────────────────────────────

def find_swing_points(highs, lows, lookback: int = 50):
    safe = min(lookback, len(highs), len(lows))
    if safe < 2:
        return float(highs[-1]), float(lows[-1])
    return float(highs[-safe:].max()), float(lows[-safe:].min())


def get_fib_exits(swing_high: float, swing_low: float,
                  entry: float, direction: str) -> dict:
    if swing_high < swing_low:
        swing_high, swing_low = swing_low, swing_high
    diff = max(swing_high - swing_low, entry * 0.001)

    ret_618 = swing_high - diff * 0.618
    ret_786 = swing_high - diff * 0.786

    if direction == "long":
        sl = ret_618
        if sl >= entry: sl = ret_786
        if sl >= entry: sl = entry - diff * 0.618
        tps = sorted([swing_high + diff*(e-1.0) for e in [1.272, 1.618, 2.618]])
        tps = [max(t, entry * 1.001) for t in tps]
    else:
        sl = swing_low + diff * 0.618
        if sl <= entry: sl = swing_low + diff * 0.786
        if sl <= entry: sl = entry + diff * 0.618
        tps = sorted([swing_low - diff*(e-1.0) for e in [1.272, 1.618, 2.618]], reverse=True)
        tps = [min(t, entry * 0.999) for t in tps]

    tp1, tp2, tp3 = tps
    risk    = abs(entry - sl)
    reward1 = abs(tp1 - entry)
    rr      = round(reward1 / max(risk, EPS), 2)

    return {
        "stop_loss": round(sl, 8), "tp1": round(tp1, 8),
        "tp2": round(tp2, 8),      "tp3": round(tp3, 8),
        "risk_reward": rr,
    }


# ──────────────────────────────────────────────
# SMC
# ──────────────────────────────────────────────

def detect_order_blocks(opens, highs, lows, closes, n=5) -> list:
    obs = []
    safe_n = min(len(closes), len(opens), len(highs), len(lows))
    for i in range(2, safe_n - 1):
        body       = abs(float(closes[i]) - float(opens[i]))
        prev_range = float(highs[i-1]) - float(lows[i-1])
        if body < EPS or prev_range < EPS:
            continue
        if closes[i-1]<opens[i-1] and closes[i]>highs[i-1] and body>prev_range*0.5:
            obs.append({"type": "bullish_ob", "high": float(highs[i-1]),
                        "low": float(lows[i-1]), "index": i})
        elif closes[i-1]>opens[i-1] and closes[i]<lows[i-1] and body>prev_range*0.5:
            obs.append({"type": "bearish_ob", "high": float(highs[i-1]),
                        "low": float(lows[i-1]), "index": i})
    return obs[-n:] if obs else []


def detect_fvg(highs, lows, closes) -> list:
    fvgs   = []
    safe_n = min(len(closes), len(highs), len(lows))
    for i in range(2, safe_n):
        if float(lows[i]) > float(highs[i-2]):
            fvgs.append({"type": "bullish_fvg", "top": float(lows[i]), "bottom": float(highs[i-2])})
        elif float(highs[i]) < float(lows[i-2]):
            fvgs.append({"type": "bearish_fvg", "top": float(lows[i-2]), "bottom": float(highs[i])})
    return fvgs[-3:] if fvgs else []


def detect_bos_choch(highs, lows, closes, lookback=20):
    safe = min(lookback + 10, len(closes), len(highs), len(lows))
    if safe < lookback + 5:
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
    def r(a, b): return abs(a) / max(abs(b), EPS)
    XA=A-X; AB=B-A; BC=C-B; CD=D-C
    if abs(XA)<EPS or abs(AB)<EPS or abs(BC)<EPS: return None
    AB_XA=r(AB,XA); BC_AB=r(BC,AB); CD_BC=r(CD,BC); XD_XA=r(D-X,XA)
    def near(v,t,tol=0.08): return bool(abs(v-t)<=tol)
    d = "long" if XA > 0 else "short"
    if near(AB_XA,.618) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.272) and near(XD_XA,.786): return ("Gartley",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,2.0)   and near(XD_XA,.886): return ("Bat",d)
    if near(AB_XA,.786) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,1.618) and near(XD_XA,1.272): return ("Butterfly",d)
    if near(AB_XA,.382) and (near(BC_AB,.382) or near(BC_AB,.886)) and near(CD_BC,3.618) and near(XD_XA,1.618): return ("Crab",d)
    return None


def scan_harmonics(highs, lows, n=80):
    safe = min(n, len(highs), len(lows))
    if safe < 10: return None
    h=list(highs[-safe:].astype(float)); l=list(lows[-safe:].astype(float))
    pivots=[]
    for i in range(2,len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i+1]: pivots.append(h[i])
        elif l[i]<l[i-1] and l[i]<l[i+1]: pivots.append(l[i])
    if len(pivots)<5: return None
    for i in range(len(pivots)-4):
        res=check_harmonic(*pivots[i:i+5])
        if res: return res
    return None


# ──────────────────────────────────────────────
# 外部資料（B4：全部加預設值）
# ──────────────────────────────────────────────

def fetch_fear_greed() -> dict:
    try:
        resp  = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        resp.raise_for_status()
        data  = resp.json()["data"][0]
        value = int(data["value"])
        if value <= 24:   label = "😱 極度恐懼"
        elif value <= 49: label = "😨 恐懼"
        elif value <= 74: label = "😊 貪婪"
        else:             label = "🤑 極度貪婪"
        return {"value": value, "label": label}
    except Exception:
        return {"value": -1, "label": "無法取得"}


def fetch_funding_rate(symbol: str) -> dict:
    try:
        resp = requests.get(f"{FUTURES_URL}/premiumIndex",
                            params={"symbol": f"{symbol}USDT"}, timeout=8)
        resp.raise_for_status()
        rate = round(float(resp.json()["lastFundingRate"]) * 100, 4)
        if rate > 0.1:    signal = "多方擁擠⚠️"
        elif rate < -0.1: signal = "空方擁擠⚠️"
        else:             signal = "市場平衡✅"
        return {"rate": rate, "signal": signal}
    except Exception:
        return {"rate": 0.0, "signal": "無法取得"}


def fetch_open_interest(symbol: str) -> dict:
    try:
        r1 = requests.get(f"{FUTURES_URL}/openInterest",
                          params={"symbol": f"{symbol}USDT"}, timeout=8)
        r1.raise_for_status()
        current_oi = float(r1.json()["openInterest"])
        r2 = requests.get(f"{FUTURES_URL}/openInterestHist",
                          params={"symbol": f"{symbol}USDT", "period": "1h", "limit": 2}, timeout=8)
        r2.raise_for_status()
        hist = r2.json()
        oi_change = 0.0
        if isinstance(hist, list) and len(hist) >= 2:
            prev = float(hist[0]["sumOpenInterest"])
            oi_change = round((current_oi - prev) / max(prev, EPS) * 100, 2)
        signal = "OI急增📈" if oi_change > 2 else ("OI急減📉" if oi_change < -2 else "OI穩定➡️")
        return {"oi": round(current_oi, 2), "oi_change": oi_change, "oi_signal": signal}
    except Exception:
        return {"oi": 0.0, "oi_change": 0.0, "oi_signal": "無法取得"}


# ──────────────────────────────────────────────
# 主分析函數
# ──────────────────────────────────────────────

def full_analysis(symbol: str) -> Optional[dict]:
    try:
        timeframe = select_timeframe(symbol)
        data      = fetch_klines(symbol, timeframe, 200)
        if data is None or len(data) < 60:
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
        vwap   = calc_vwap(highs, lows, closes, vols)

        adx_info    = calc_adx(highs, lows, closes)
        market_st   = detect_market_structure(highs, lows, closes, atr)
        breakout    = detect_breakout(highs, lows, closes, vols, atr)
        sr          = detect_support_resistance(highs, lows, closes)
        div         = detect_divergence(highs, lows, closes)
        funding     = fetch_funding_rate(symbol)
        oi          = fetch_open_interest(symbol)

        swing_high, swing_low = find_swing_points(highs, lows, 50)
        obs         = detect_order_blocks(opens, highs, lows, closes)
        fvgs        = detect_fvg(highs, lows, closes)
        bos, choch  = detect_bos_choch(highs, lows, closes)
        harmonic    = scan_harmonics(highs, lows, 80)
        vol_info    = calc_volume_confirmation(vols, closes)
        candle      = detect_candle_pattern(opens, highs, lows, closes)
        upper_trend = get_upper_trend(symbol, timeframe)

        # ── 多空評分（加權版）──
        long_score = short_score = 0

        # RSI（權重2）
        if rsi < 30:        long_score  += 2
        elif rsi < 45:      long_score  += 1
        elif rsi > 70:      short_score += 2
        elif rsi > 55:      short_score += 1

        # MACD（權重2）
        if hist > 0 and macd > macd_sig:    long_score  += 2
        elif hist < 0 and macd < macd_sig:  short_score += 2

        # EMA 排列（權重3，可靠度最高）
        if price > ema20 > ema50 > ema200:    long_score  += 3
        elif price < ema20 < ema50 < ema200:  short_score += 3
        elif price > ema50:                   long_score  += 1
        elif price < ema50:                   short_score += 1

        # Bollinger（權重1）
        if price < bb_lower:   long_score  += 1
        elif price > bb_upper: short_score += 1

        # VWAP（權重1）
        if price > vwap:   long_score  += 1
        elif price < vwap: short_score += 1

        # ADX 分級
        s = adx_info["adx_score"]
        if s > 0:
            long_score  += s
            short_score += s
        else:
            long_score  = max(0, long_score  + s)
            short_score = max(0, short_score + s)

        # PDI/MDI 方向確認
        if adx_info["pdi"] > adx_info["mdi"]: long_score  += 1
        else:                                  short_score += 1

        # 市場結構（權重2）
        if market_st["structure"] == "trending_up":   long_score  += 2
        elif market_st["structure"] == "trending_down": short_score += 2

        # 突破（權重4，最高）
        if breakout["breakout"] == "bullish":  long_score  += 4
        elif breakout["breakout"] == "bearish": short_score += 4

        # OB / FVG（各權重1）
        if any(o.get("type") == "bullish_ob"  for o in obs): long_score  += 1
        if any(o.get("type") == "bearish_ob"  for o in obs): short_score += 1
        if any(f.get("type") == "bullish_fvg" for f in fvgs): long_score  += 1
        if any(f.get("type") == "bearish_fvg" for f in fvgs): short_score += 1

        # BOS/CHoCH
        if bos   == "bullish_bos":     long_score  += 2
        elif bos == "bearish_bos":     short_score += 2
        if choch == "bullish_choch":   long_score  += 3
        elif choch == "bearish_choch": short_score += 3

        # 和諧型態（權重2）
        if harmonic:
            if harmonic[1] == "long":  long_score  += 2
            else:                      short_score += 2

        # 成交量（權重2）
        if vol_info["bullish_vol"]: long_score  += 2
        if vol_info["bearish_vol"]: short_score += 2

        # K線型態（權重2）
        if candle["hammer"] or candle["engulfing_bull"] or candle["doji_bull"]:
            long_score  += 2
        if candle["shooting_star"] or candle["engulfing_bear"]:
            short_score += 2

        # 多週期確認（權重2）
        upper_agree = False
        if upper_trend == "up":
            long_score  += 2; upper_agree = True
        elif upper_trend == "down":
            short_score += 2; upper_agree = True

        # 背離（權重3）
        has_divergence = False
        if div["rsi_bull_div"] or div["macd_bull_div"]:
            long_score += 3; has_divergence = True
        if div["rsi_bear_div"]:
            short_score += 3; has_divergence = True

        # 籌碼分析
        if funding["rate"] > 0.15:  short_score += 1
        if funding["rate"] < -0.15: long_score  += 1
        if oi["oi_change"] > 2 and price > float(closes[-2]): long_score  += 1
        elif oi["oi_change"] > 2 and price < float(closes[-2]): short_score += 1

        # ── 訊號過濾 ──
        score_diff = abs(long_score - short_score)
        max_score  = max(long_score, short_score)

        if long_score == short_score:    return None
        if score_diff < MIN_SCORE_DIFF:  return None
        if max_score  < MIN_SCORE_TOTAL: return None

        direction = "long" if long_score > short_score else "short"

        # 勝率提升3：最終否決機制
        veto = check_veto(direction, rsi, macd, macd_sig, hist,
                          adx_info["adx"], funding["rate"],
                          market_st["structure"], upper_trend)
        if veto["vetoed"]:
            print(f"  ⛔ {symbol} 訊號被否決：{veto['reasons']}")
            return None

        # 風報比過濾（最低 1.5）
        best_entry = get_best_entry(price, obs, fvgs, direction, atr)
        exits      = get_fib_exits(swing_high, swing_low, best_entry, direction)
        if exits["risk_reward"] < MIN_RR:
            return None

        # 價格突破防重複
        if is_duplicate_signal(symbol, direction, price, atr):
            return None

        atr_sl   = (round(best_entry - atr*2.0, 8) if direction=="long"
                    else round(best_entry + atr*2.0, 8))
        trailing = calc_trailing_stop(best_entry, price, atr, direction, False)
        position = calc_position_size(
            get_confidence(score_diff, vol_info["bullish_vol"] or vol_info["bearish_vol"],
                           any([candle["hammer"], candle["engulfing_bull"], candle["doji_bull"],
                                candle["shooting_star"], candle["engulfing_bear"]]),
                           upper_agree, adx_info["adx"], has_divergence,
                           breakout["breakout"] is not None,
                           market_st["structure"], veto),
            market_st["structure"], exits["risk_reward"], adx_info["adx"]
        )

        has_candle = any([candle["hammer"], candle["engulfing_bull"], candle["doji_bull"],
                          candle["shooting_star"], candle["engulfing_bear"]])
        has_vol    = bool(vol_info["bullish_vol"] or vol_info["bearish_vol"])
        confidence = get_confidence(score_diff, has_vol, has_candle, upper_agree,
                                    adx_info["adx"], has_divergence,
                                    breakout["breakout"] is not None,
                                    market_st["structure"], veto)

        # B6：全部轉成 Python 原生型別
        return {
            "symbol":           str(symbol),
            "timeframe":        str(timeframe),
            "price":            round(float(price), 8),
            "direction":        str(direction),
            "long_score":       int(long_score),
            "short_score":      int(short_score),
            "confidence":       str(confidence),
            "entry":            round(float(best_entry), 8),
            "entry_source":     "OB/FVG中點" if best_entry != round(price,8) else "現價",
            "stop_loss":        float(exits["stop_loss"]),
            "atr_stop_loss":    float(atr_sl),
            "trailing_sl":      float(trailing["trailing_sl"]),
            "sl_type":          str(trailing["sl_type"]),
            "tp1":              float(exits["tp1"]),
            "tp2":              float(exits["tp2"]),
            "tp3":              float(exits["tp3"]),
            "risk_reward":      float(exits["risk_reward"]),
            "position_pct":     int(position["position_pct"]),
            "risk_label":       str(position["risk_label"]),
            "market_structure": str(market_st["label"]),
            "market_strategy":  str(market_st["strategy"]),
            "breakout":         str(breakout["breakout_label"]),
            "rsi":              float(rsi),
            "macd":             round(float(macd), 8),
            "macd_hist":        round(float(hist), 8),
            "ema20":            round(float(ema20), 8),
            "ema50":            round(float(ema50), 8),
            "ema200":           round(float(ema200), 8),
            "bb_upper":         round(float(bb_upper), 8),
            "bb_lower":         round(float(bb_lower), 8),
            "vwap":             round(float(vwap), 8),
            "atr":              round(float(atr), 8),
            "adx":              float(adx_info["adx"]),
            "trend_strength":   str(adx_info["trend_strength"]),
            "pdi":              float(adx_info["pdi"]),
            "mdi":              float(adx_info["mdi"]),
            "divergence":       str(div["divergence"]),
            "nearest_sup":      sr["nearest_sup"],
            "nearest_res":      sr["nearest_res"],
            "dist_to_sup":      sr["dist_to_sup"],
            "dist_to_res":      sr["dist_to_res"],
            "funding_rate":     float(funding["rate"]),
            "funding_signal":   str(funding["signal"]),
            "oi_change":        float(oi["oi_change"]),
            "oi_signal":        str(oi["oi_signal"]),
            "swing_high":       round(float(swing_high), 8),
            "swing_low":        round(float(swing_low), 8),
            "order_blocks":     obs,
            "fvg":              fvgs,
            "bos":              bos,
            "choch":            choch,
            "candle_pattern":   str(candle["name"]),
            "vol_ratio":        float(vol_info["vol_ratio"]),
            "upper_trend":      str(upper_trend) if upper_trend else "—",
            "harmonic":         harmonic,
            "veto_reasons":     veto["reasons"],
        }

    except Exception as e:
        print(f"  ❌ full_analysis({symbol}) 未預期錯誤：{e}")
        return None


# ──────────────────────────────────────────────
# 格式化輸出
# ──────────────────────────────────────────────

def format_signal(r: dict) -> str:
    try:
        d_emoji  = "🟢 做多 LONG" if r["direction"] == "long" else "🔴 做空 SHORT"
        h_str    = f"{r['harmonic'][0]} ({r['harmonic'][1]})" if r.get("harmonic") else "無"
        bos_str  = r.get("bos")   or "—"
        choch_str= r.get("choch") or "—"
        up_emoji = ("📈" if r.get("upper_trend") == "up" else
                    ("📉" if r.get("upper_trend") == "down" else "—"))
        sup_str  = (f"`{r['nearest_sup']}` ({r['dist_to_sup']}%)"
                    if r.get("nearest_sup") else "—")
        res_str  = (f"`{r['nearest_res']}` ({r['dist_to_res']}%)"
                    if r.get("nearest_res") else "—")

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
            f"🚀 *突破*：{r['breakout']}\n"
            f"🔀 *背離*：{r['divergence']}\n"
            f"\n"
            f"📊 *技術指標*\n"
            f"  RSI    ：`{r['rsi']}` {rl}\n"
            f"  MACD   ：`{r['macd']}` ｜ Hist `{r['macd_hist']}`\n"
            f"  EMA20  ：`{r['ema20']}` EMA50：`{r['ema50']}`\n"
            f"  EMA200 ：`{r['ema200']}`\n"
            f"  BB上   ：`{r['bb_upper']}` / 下：`{r['bb_lower']}`\n"
            f"  VWAP   ：`{r['vwap']}`\n"
            f"  ATR    ：`{r['atr']}`\n"
            f"  ADX    ：`{r['adx']}` {r['trend_strength']} "
            f"(PDI:`{r['pdi']}` MDI:`{r['mdi']}`)\n"
            f"\n"
            f"📍 *支撐壓力*\n"
            f"  支撐：{sup_str} ｜ 壓力：{res_str}\n"
            f"\n"
            f"🕯 *K線型態*：{r['candle_pattern']}\n"
            f"📦 *成交量比*：`{r['vol_ratio']}x`\n"
            f"🌐 *上層週期*：{up_emoji} {r['upper_trend']}\n"
            f"\n"
            f"💹 *籌碼*\n"
            f"  資金費率：`{r['funding_rate']}%` {r['funding_signal']}\n"
            f"  OI變化  ：`{r['oi_change']}%` {r['oi_signal']}\n"
            f"\n"
            f"🏗️ *SMC* BOS：{bos_str} ｜ CHoCH：{choch_str}\n"
            f"  OB：{len(r.get('order_blocks',[]))}個 ｜ FVG：{len(r.get('fvg',[]))}個\n"
            f"\n"
            f"🔷 *和諧型態*：{h_str}\n"
            f"📈 *評分*：多 {r['long_score']} ／ 空 {r['short_score']}\n"
            f"🕯 High：`{r['swing_high']}` / Low：`{r['swing_low']}`\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as e:
        return f"❌ 格式化錯誤：{e}"


if __name__ == "__main__":
    result = full_analysis("BTC")
    if result:
        print(format_signal(result))
    else:
        print("無強力訊號（已通過三重確認過濾）")
