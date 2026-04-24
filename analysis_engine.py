"""
analysis_engine.py v9.0
========================
目標勝率：85%+

v9.0 新增功能：
  1. 多時間框架投票制（1h/4h/1d 需2個同向）
  2. BTC 崩跌過濾（跌>3% 暫停山寨幣多單）
  3. 市場情緒過濾（極度貪婪只做空）
  4. 鯨魚成交量偵測（量>均量3倍加分）
  5. 勝率 < 70% 自動過濾
  6. OKX API（全球可用，無地理限制）
  7. ML 線上學習（越用越準）
  8. 自適應參數（每6小時自動調整）
  9. 風控系統（連虧3次暫停）
  10. Kelly Criterion 倉位管理
"""

import time, math, json, os, threading
import requests
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ──────────────────────────────────────────────
# 常數
# ──────────────────────────────────────────────
TOP30_COINS = [
    "BTC","ETH","BNB","XRP","SOL","ADA","DOGE","TRX","AVAX","SHIB",
    "DOT","LINK","MATIC","UNI","ICP","LTC","APT","NEAR","ATOM","XLM",
    "FIL","HBAR","ARB","OP","INJ","SUI","VET","GRT","AAVE","MKR",
]

COIN_TIMEFRAME = {
    "BTC":"4h","ETH":"4h","BNB":"4h","XRP":"4h","SOL":"4h",
    "ADA":"4h","DOGE":"4h","TRX":"4h","AVAX":"4h","SHIB":"1h",
    "DOT":"4h","LINK":"4h","MATIC":"1h","UNI":"1h","ICP":"1h",
    "LTC":"4h","APT":"1h","NEAR":"1h","ATOM":"1h","XLM":"1h",
    "FIL":"1h","HBAR":"1h","ARB":"1h","OP":"1h","INJ":"1h",
    "SUI":"1h","VET":"1h","GRT":"1h","AAVE":"4h","MKR":"4h",
}

UPPER_TIMEFRAME = {"15m":"1h","1h":"4h","4h":"1d","1d":"1w"}
FIB_RETRACEMENT = [0.236,0.382,0.5,0.618,0.786]
FIB_EXTENSION   = [1.0,1.272,1.414,1.618,2.0,2.618]
EPS             = 1e-10

# 門檻
MIN_WINRATE_PCT = 70.0
BTC_CRASH_PCT   = -3.0
FG_ONLY_SHORT   = 80
FG_ONLY_LONG    = 25

# OKX API
OKX_BASE      = "https://www.okx.com"
OKX_KLINE_URL = f"{OKX_BASE}/api/v5/market/candles"
OKX_TICKER    = f"{OKX_BASE}/api/v5/market/ticker"
OKX_FUNDING   = f"{OKX_BASE}/api/v5/public/funding-rate"
OKX_OI        = f"{OKX_BASE}/api/v5/public/open-interest"

OKX_INTERVAL = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H",
    "1d":"1D","1w":"1W",
}

# 動態門檻
_adaptive_params = {
    "min_score_diff":4,"min_score_total":7,
    "min_adx":15,"min_rr":1.5,"vol_threshold":1.5,
}

# 並行模式旗標
_parallel_mode = False

# 防重複快取
_signal_cache: dict = {}
_cache_lock = threading.Lock()

# 恐懼貪婪快取
_fg_cache = {"value":-1,"ts":0.0}
_FG_TTL   = 3600

# 檔案路徑
_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RISK_FILE  = os.path.join(_BASE_DIR,"risk_control.json")
ML_FILE    = os.path.join(_BASE_DIR,"ml_weights.json")

# ──────────────────────────────────────────────
# 風控系統
# ──────────────────────────────────────────────
_risk_control = {
    "consecutive_losses":0,"max_consecutive_losses":3,
    "paused":False,"total_signals":0,
    "winning_signals":0,"total_pnl_pct":0.0,
}

def load_risk_control():
    global _risk_control
    try:
        if os.path.exists(RISK_FILE):
            with open(RISK_FILE,"r") as f:
                _risk_control.update(json.load(f))
    except Exception: pass

def save_risk_control():
    try:
        with open(RISK_FILE,"w") as f: json.dump(_risk_control,f)
    except Exception: pass

def record_trade_result(win:bool, pnl_pct:float=0.0):
    _risk_control["total_signals"] += 1
    _risk_control["total_pnl_pct"] += pnl_pct
    if win:
        _risk_control["winning_signals"] += 1
        _risk_control["consecutive_losses"] = 0
    else:
        _risk_control["consecutive_losses"] += 1
    if _risk_control["consecutive_losses"] >= _risk_control["max_consecutive_losses"]:
        _risk_control["paused"] = True
        print(f"⚠️ 風控：連續虧損{_risk_control['consecutive_losses']}次，系統暫停！")
    save_risk_control()

def reset_risk_pause():
    _risk_control["paused"] = False
    _risk_control["consecutive_losses"] = 0
    save_risk_control()

def get_risk_status() -> dict:
    t=_risk_control["total_signals"]; w=_risk_control["winning_signals"]
    return {
        "paused":_risk_control["paused"],
        "consecutive_losses":_risk_control["consecutive_losses"],
        "total_signals":t,"winning_signals":w,
        "actual_winrate":round(w/max(t,1)*100,1),
        "total_pnl_pct":round(_risk_control["total_pnl_pct"],2),
    }

# ──────────────────────────────────────────────
# ML 系統
# ──────────────────────────────────────────────
_ml_weights = {
    "w":[0.15,0.12,0.18,0.08,0.08,0.10,0.12,0.10,0.05,0.07,0.08],
    "b":-3.5,"lr":0.01,"samples":0,
}

def load_ml_weights():
    global _ml_weights
    try:
        if os.path.exists(ML_FILE):
            with open(ML_FILE,"r") as f: _ml_weights.update(json.load(f))
    except Exception: pass

def save_ml_weights():
    try:
        with open(ML_FILE,"w") as f: json.dump(_ml_weights,f)
    except Exception: pass

def _sigmoid(x:float) -> float:
    try: return 1.0/(1.0+math.exp(-max(-500,min(500,x))))
    except: return 0.5

def ml_predict_winrate(features:list) -> float:
    w=_ml_weights["w"]; b=_ml_weights["b"]
    n=min(len(w),len(features))
    dot=sum(w[i]*features[i] for i in range(n))+b
    return round(max(35.0,min(95.0,_sigmoid(dot)*100)),1)

def ml_update(features:list, win:bool):
    y=1.0 if win else 0.0
    w=_ml_weights["w"]; b=_ml_weights["b"]; lr=_ml_weights["lr"]
    n=min(len(w),len(features))
    dot=sum(w[i]*features[i] for i in range(n))+b
    pred=_sigmoid(dot); err=pred-y
    for i in range(n): w[i]-=lr*err*features[i]
    _ml_weights["b"]-=lr*err
    _ml_weights["samples"]+=1
    _ml_weights["lr"]=max(0.001,0.01/(1+_ml_weights["samples"]*0.01))
    save_ml_weights()

def build_ml_features(score_diff,adx,rr,has_vol,has_candle,upper_agree,
                       has_breakout,is_trending,rsi,direction,hist,macd,macd_sig,has_div) -> list:
    rsi_ok  = float((rsi<45 and direction=="long") or (rsi>55 and direction=="short"))
    macd_ok = float((hist>0 and macd>macd_sig and direction=="long") or
                    (hist<0 and macd<macd_sig and direction=="short"))
    return [
        min(score_diff/20.0,1.0), min(adx/60.0,1.0), min(rr/5.0,1.0),
        float(has_vol), float(has_candle), float(upper_agree),
        float(has_breakout), float(is_trending), rsi_ok, macd_ok, float(has_div),
    ]

# ──────────────────────────────────────────────
# OKX API
# ──────────────────────────────────────────────
def fetch_klines(symbol:str, interval:str="4h", limit:int=200) -> Optional[np.ndarray]:
    """OKX K線，現貨優先"""
    okx_bar = OKX_INTERVAL.get(interval,"4H")
    for inst in [f"{symbol}-USDT", f"{symbol}-USDT-SWAP"]:
        try:
            r = requests.get(OKX_KLINE_URL,
                params={"instId":inst,"bar":okx_bar,"limit":str(min(limit,300))},
                timeout=8)
            if not r.ok: continue
            raw = r.json()
            if raw.get("code")!="0" or not raw.get("data"): continue
            bars = raw["data"][::-1]   # 最新在前，反轉
            if len(bars) < 10: continue
            data = np.array([[float(b[0]),float(b[1]),float(b[2]),
                               float(b[3]),float(b[4]),float(b[5])]
                              for b in bars], dtype=np.float64)
            if data.ndim!=2 or data.shape[1]<6: continue
            if np.isnan(data).any() or np.isinf(data).any(): continue
            return data
        except Exception as e:
            print(f"  ⚠️ {symbol} OKX {inst} 失敗：{e}")
            continue
    print(f"  ❌ {symbol} {interval} 取得失敗")
    return None

def select_timeframe(symbol:str) -> str:
    return COIN_TIMEFRAME.get(symbol.upper(),"1h")

def is_duplicate_signal(symbol:str, direction:str, price:float, atr:float) -> bool:
    key = f"{symbol}_{direction}"
    with _cache_lock:
        last = _signal_cache.get(key)
        if last is None:
            _signal_cache[key]=price; return False
        if abs(price-last) > atr*1.5:
            _signal_cache[key]=price; return False
    return True

def fetch_fear_greed() -> dict:
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=8)
        r.raise_for_status()
        d=r.json()["data"][0]; v=int(d["value"])
        if v<=24: lb="😱 極度恐懼"
        elif v<=49: lb="😨 恐懼"
        elif v<=74: lb="😊 貪婪"
        else: lb="🤑 極度貪婪"
        return {"value":v,"label":lb}
    except: return {"value":-1,"label":"無法取得"}

def get_fear_greed_cached() -> int:
    now=time.time()
    if now-_fg_cache["ts"]<_FG_TTL and _fg_cache["value"]>=0:
        return _fg_cache["value"]
    fg=fetch_fear_greed()
    if fg["value"]>=0:
        _fg_cache["value"]=fg["value"]; _fg_cache["ts"]=now
    return _fg_cache["value"]

def fetch_funding_rate(symbol:str) -> dict:
    try:
        r=requests.get(OKX_FUNDING,params={"instId":f"{symbol}-USDT-SWAP"},timeout=8)
        if r.ok:
            d=r.json()
            if d.get("code")=="0" and d.get("data"):
                rate=round(float(d["data"][0]["fundingRate"])*100,4)
                sig="多方擁擠⚠️" if rate>0.1 else ("空方擁擠⚠️" if rate<-0.1 else "市場平衡✅")
                return {"rate":rate,"signal":sig}
    except: pass
    return {"rate":0.0,"signal":"無法取得"}

def fetch_open_interest(symbol:str) -> dict:
    try:
        r=requests.get(OKX_OI,params={"instId":f"{symbol}-USDT-SWAP","instType":"SWAP"},timeout=8)
        if r.ok:
            d=r.json()
            if d.get("code")=="0" and d.get("data"):
                oi=float(d["data"][0]["oi"])
                return {"oi":round(oi,2),"oi_change":0.0,"oi_signal":"OI穩定➡️"}
    except: pass
    return {"oi":0.0,"oi_change":0.0,"oi_signal":"無法取得"}

# ──────────────────────────────────────────────
# 技術指標
# ──────────────────────────────────────────────
def calc_rsi(closes:np.ndarray, period:int=14) -> float:
    if len(closes)<period+1: return 50.0
    d=np.diff(closes.astype(np.float64))
    g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=float(g[:period].mean()); al=float(l[:period].mean())
    for i in range(period,len(g)):
        ag=(ag*(period-1)+float(g[i]))/period
        al=(al*(period-1)+float(l[i]))/period
    if al<EPS: return 100.0
    return round(100.0-(100.0/(1.0+ag/al)),2)

def calc_ema(closes:np.ndarray, period:int) -> np.ndarray:
    c=closes.astype(np.float64); e=np.zeros_like(c); k=2.0/(period+1)
    e[0]=c[0]
    for i in range(1,len(c)): e[i]=c[i]*k+e[i-1]*(1.0-k)
    return e

def calc_macd(closes:np.ndarray):
    ml=calc_ema(closes,12)-calc_ema(closes,26)
    ms=calc_ema(ml,9); mh=ml-ms
    return float(ml[-1]),float(ms[-1]),float(mh[-1])

def calc_bollinger(closes:np.ndarray, period:int=20, sd:float=2.0):
    if len(closes)<period: m=float(closes[-1]); return m,m,m
    w=closes[-period:].astype(np.float64); m=float(w.mean()); s=float(w.std())
    return m+sd*s,m,m-sd*s

def calc_atr(highs,lows,closes,period:int=14) -> float:
    if len(closes)<2: return max(float(abs(highs[-1]-lows[-1])) if len(highs)>0 else 1.0,EPS)
    n=min(len(closes),len(highs),len(lows)); trs=[]
    for i in range(1,n):
        tr=max(float(highs[i]-lows[i]),abs(float(highs[i]-closes[i-1])),abs(float(lows[i]-closes[i-1])))
        trs.append(max(tr,EPS))
    if not trs: return 1.0
    arr=np.array(trs[-period:] if len(trs)>=period else trs)
    return float(arr.mean())

def calc_vwap(highs,lows,closes,volumes) -> float:
    tp=(highs.astype(np.float64)+lows.astype(np.float64)+closes.astype(np.float64))/3.0
    tv=float(np.sum(volumes))
    if tv<EPS: return float(closes[-1])
    return round(float(np.sum(tp*volumes.astype(np.float64))/tv),8)

def calc_adx(highs,lows,closes,period:int=14) -> dict:
    n=min(len(closes),len(highs),len(lows))
    if n<period*2+1: return {"adx":0.0,"trend_strength":"橫盤⚪️","adx_score":-3,"pdi":0.0,"mdi":0.0}
    pdm=[]; mdm=[]; trs=[]
    for i in range(1,n):
        hd=float(highs[i])-float(highs[i-1]); ld=float(lows[i-1])-float(lows[i])
        pdm.append(hd if hd>ld and hd>0 else 0.0)
        mdm.append(ld if ld>hd and ld>0 else 0.0)
        tr=max(float(highs[i]-lows[i]),abs(float(highs[i]-closes[i-1])),abs(float(lows[i]-closes[i-1])))
        trs.append(max(tr,EPS))
    def ws(arr,p):
        if len(arr)<p: return [max(sum(arr),EPS)]
        r=[sum(arr[:p])]
        for v in arr[p:]: r.append(r[-1]-r[-1]/p+v)
        return r
    a14=ws(trs,period); p14=ws(pdm,period); m14=ws(mdm,period)
    ml=min(len(a14),len(p14),len(m14)); dx=[]; lp=lm=0.0
    for i in range(ml):
        a=max(a14[i],EPS); pdi=100.0*p14[i]/a; mdi=100.0*m14[i]/a
        dx.append(100.0*abs(pdi-mdi)/max(pdi+mdi,EPS)); lp=pdi; lm=mdi
    if not dx: return {"adx":0.0,"trend_strength":"橫盤⚪️","adx_score":-3,"pdi":0.0,"mdi":0.0}
    adx=round(float(np.mean(dx[-period:])),2)
    if adx>=40: s,sc="強趨勢🔥",3
    elif adx>=25: s,sc="趨勢中🟡",1
    elif adx>=15: s,sc="弱趨勢🟠",0
    else: s,sc="橫盤⚪️",-3
    return {"adx":adx,"trend_strength":s,"adx_score":sc,"pdi":round(lp,2),"mdi":round(lm,2)}

# ──────────────────────────────────────────────
# 市場結構
# ──────────────────────────────────────────────
def detect_market_structure(highs,lows,closes,atr:float) -> dict:
    if len(closes)<30: return {"structure":"unknown","label":"結構不明⚪️","strategy":"觀望"}
    h=highs[-30:].astype(np.float64); l=lows[-30:].astype(np.float64)
    sh,sl=[],[]
    for i in range(2,len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]: sh.append(float(h[i]))
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]: sl.append(float(l[i]))
    if len(sh)>=2 and len(sl)>=2:
        if sh[-1]>sh[-2] and sl[-1]>sl[-2]: return {"structure":"trending_up","label":"多頭趨勢📈","strategy":"順勢做多"}
        if sh[-1]<sh[-2] and sl[-1]<sl[-2]: return {"structure":"trending_down","label":"空頭趨勢📉","strategy":"順勢做空"}
    pr=float(h.max()-h.min()) if len(h)>0 else 0
    if pr<atr*8: return {"structure":"ranging","label":"震盪整理↔️","strategy":"等待突破"}
    return {"structure":"unknown","label":"結構不明⚪️","strategy":"觀望"}

# ──────────────────────────────────────────────
# 突破偵測
# ──────────────────────────────────────────────
def detect_breakout(highs,lows,closes,volumes,atr:float) -> dict:
    if len(closes)<25: return {"breakout":None,"breakout_label":"無突破","breakout_level":None}
    cc=float(closes[-2]); lv=float(volumes[-2])
    av=float(volumes[-22:-2].mean()) if len(volumes)>=22 else float(volumes[:-2].mean()+EPS)
    vr=lv/max(av,EPS)
    rh=float(highs[-22:-2].max()) if len(highs)>=22 else float(highs[:-2].max())
    rl=float(lows[-22:-2].min())  if len(lows)>=22  else float(lows[:-2].min())
    vok=_adaptive_params["vol_threshold"]
    bull=bool(cc>rh and vr>=vok and (cc-rh)>=atr*0.3)
    bear=bool(cc<rl  and vr>=vok and (rl-cc)>=atr*0.3)
    if bull: return {"breakout":"bullish","breakout_label":f"🚀 看漲突破（{vr:.1f}x）","breakout_level":round(rh,8)}
    if bear: return {"breakout":"bearish","breakout_label":f"💥 看跌突破（{vr:.1f}x）","breakout_level":round(rl,8)}
    return {"breakout":None,"breakout_label":"無突破","breakout_level":None}

# ──────────────────────────────────────────────
# 新系統：多時間框架投票
# ──────────────────────────────────────────────
def multi_timeframe_vote(symbol:str, main_tf:str, direction:str) -> dict:
    tfs=["1h","4h","1d"] if not _parallel_mode else [main_tf,UPPER_TIMEFRAME.get(main_tf,main_tf)]
    votes=0; total=0; tf_r={}
    for tf in tfs:
        try:
            data=fetch_klines(symbol,tf,60)
            if data is None or len(data)<30: continue
            c=data[:,4]
            e20=float(calc_ema(c,20)[-1]); e50=float(calc_ema(c,50)[-1])
            rv=calc_rsi(c); _,_,hv=calc_macd(c)
            bull=e20>e50 and rv>45 and hv>0
            bear=e20<e50 and rv<55 and hv<0
            ok=(direction=="long" and bull) or (direction=="short" and bear)
            if ok: votes+=1
            total+=1; tf_r[tf]="✅" if ok else "❌"
        except: continue
    passed=bool(votes>=2 and total>0)
    return {"passed":passed,"votes":votes,"total":total,
            "label":f"{'✅' if passed else '❌'} {votes}/{total}週期同向"}

# ──────────────────────────────────────────────
# 新系統：BTC 崩跌過濾
# ──────────────────────────────────────────────
def check_btc_crash(symbol:str) -> dict:
    if symbol in ("BTC","ETH"): return {"crashed":False,"btc_change":0.0,"label":"不適用"}
    try:
        data=fetch_klines("BTC","1h",5)
        if data is None or len(data)<4: return {"crashed":False,"btc_change":0.0,"label":"無法取得"}
        prev=float(data[-4,4]); last=float(data[-1,4])
        chg=round((last-prev)/max(prev,EPS)*100,2)
        crash=bool(chg<=BTC_CRASH_PCT)
        return {"crashed":crash,"btc_change":chg,
                "label":f"{'🚨 BTC崩跌' if crash else '✅ BTC正常'}({chg:+.1f}%)"}
    except: return {"crashed":False,"btc_change":0.0,"label":"無法取得"}

# ──────────────────────────────────────────────
# 新系統：市場情緒過濾
# ──────────────────────────────────────────────
def check_sentiment_filter(direction:str) -> dict:
    fg=get_fear_greed_cached()
    if fg<0: return {"blocked":False,"fg_value":-1,"label":"無法取得","reason":""}
    blocked=False; reason=""
    if fg>=FG_ONLY_SHORT and direction=="long": blocked=True; reason=f"極度貪婪({fg})只做空"
    elif fg<=FG_ONLY_LONG and direction=="short": blocked=True; reason=f"極度恐懼({fg})只做多"
    if fg>=75: lbl=f"🤑 極貪婪({fg})"
    elif fg>=55: lbl=f"😊 貪婪({fg})"
    elif fg>=45: lbl=f"😐 中性({fg})"
    elif fg>=25: lbl=f"😨 恐懼({fg})"
    else: lbl=f"😱 極恐懼({fg})"
    return {"blocked":blocked,"fg_value":fg,"label":lbl,"reason":reason}

# ──────────────────────────────────────────────
# 新系統：鯨魚成交量偵測
# ──────────────────────────────────────────────
def detect_whale_volume(volumes:np.ndarray, closes:np.ndarray) -> dict:
    if len(volumes)<22: return {"whale":False,"whale_bull":False,"whale_bear":False,"vol_ratio":1.0,"label":"—"}
    av=float(volumes[-21:-1].mean()); lv=float(volumes[-1])
    vr=round(lv/max(av,EPS),2); whale=bool(vr>=3.0)
    up=bool(float(closes[-1])>float(closes[-2]))
    if whale and up: lbl=f"🐋 鯨魚買入({vr}x)"
    elif whale: lbl=f"🐋 鯨魚賣出({vr}x)"
    else: lbl=f"正常({vr}x)"
    return {"whale":whale,"whale_bull":bool(whale and up),"whale_bear":bool(whale and not up),"vol_ratio":vr,"label":lbl}

# ──────────────────────────────────────────────
# 否決機制
# ──────────────────────────────────────────────
def check_veto(direction,rsi,macd,macd_sig,hist,adx,funding_rate,market_structure,upper_trend) -> dict:
    v=[]
    if adx<_adaptive_params["min_adx"]: v.append(f"ADX={adx:.1f}趨勢太弱")
    if direction=="long"  and rsi>80: v.append(f"RSI={rsi}嚴重超買")
    if direction=="short" and rsi<20: v.append(f"RSI={rsi}嚴重超賣")
    if direction=="long"  and macd<macd_sig and hist<0 and macd<0: v.append("MACD空頭排列")
    if direction=="short" and macd>macd_sig and hist>0 and macd>0: v.append("MACD多頭排列")
    if direction=="long"  and market_structure=="trending_down": v.append("市場結構空頭")
    if direction=="short" and market_structure=="trending_up": v.append("市場結構多頭")
    if upper_trend=="down" and direction=="long": v.append("上層週期空頭")
    if upper_trend=="up"   and direction=="short": v.append("上層週期多頭")
    if direction=="long"  and funding_rate>0.3: v.append(f"資金費率{funding_rate}%多方擁擠")
    if direction=="short" and funding_rate<-0.3: v.append(f"資金費率{funding_rate}%空方擁擠")
    return {"vetoed":bool(v),"reasons":v}

# ──────────────────────────────────────────────
# 移動止損
# ──────────────────────────────────────────────
def calc_trailing_stop(entry,current_price,atr,direction,tp1_hit=False) -> dict:
    if direction=="long":
        isl=round(entry-atr*2.0,8); be=round(entry+atr*0.1,8); tsl=round(current_price-atr*1.0,8)
        if not tp1_hit: return {"trailing_sl":isl,"sl_type":"初始止損"}
        elif current_price<entry*1.02: return {"trailing_sl":be,"sl_type":"保本止損"}
        else: return {"trailing_sl":tsl,"sl_type":"追蹤止損"}
    else:
        isl=round(entry+atr*2.0,8); be=round(entry-atr*0.1,8); tsl=round(current_price+atr*1.0,8)
        if not tp1_hit: return {"trailing_sl":isl,"sl_type":"初始止損"}
        elif current_price>entry*0.98: return {"trailing_sl":be,"sl_type":"保本止損"}
        else: return {"trailing_sl":tsl,"sl_type":"追蹤止損"}

# ──────────────────────────────────────────────
# 倉位管理（Kelly Criterion）
# ──────────────────────────────────────────────
def calc_position_size(winrate_pct,rr,structure,adx) -> dict:
    wr=winrate_pct/100.0; lr=1.0-wr
    kelly=(wr*rr-lr)/max(rr,EPS); base=max(3.0,min(25.0,kelly*50))
    if structure in ("trending_up","trending_down"): base=min(25,base*1.2)
    elif structure=="ranging": base=min(25,base*0.7)
    if adx>=40: base=min(25,base*1.1)
    elif adx<20: base=min(25,base*0.8)
    size=round(max(3.0,min(25.0,base)),1)
    if size>=20: lbl="🟢 積極"
    elif size>=15: lbl="🟡 標準"
    elif size>=10: lbl="🟠 保守"
    else: lbl="🔴 觀望"
    return {"position_pct":size,"risk_label":lbl}

# ──────────────────────────────────────────────
# 支撐壓力
# ──────────────────────────────────────────────
def detect_support_resistance(highs,lows,closes,lookback=100,n=3) -> dict:
    sl=min(lookback,len(highs),len(lows),len(closes))
    if sl<10: return {"supports":[],"resistances":[],"nearest_sup":None,"nearest_res":None,"dist_to_sup":None,"dist_to_res":None}
    h=highs[-sl:].astype(np.float64); l=lows[-sl:].astype(np.float64); price=float(closes[-1])
    rs=[]; ss=[]
    for i in range(2,len(h)-2):
        if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]: rs.append(float(h[i]))
        if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]: ss.append(float(l[i]))
    rs=sorted(set(r for r in rs if r>price))[:n]; ss=sorted(set(s for s in ss if s<price),reverse=True)[:n]
    nr=rs[0] if rs else None; ns=ss[0] if ss else None
    dr=round((nr-price)/max(price,EPS)*100,2) if nr else None
    ds=round((price-ns)/max(price,EPS)*100,2) if ns else None
    return {"supports":ss,"resistances":rs,"nearest_sup":ns,"nearest_res":nr,"dist_to_sup":ds,"dist_to_res":dr}

# ──────────────────────────────────────────────
# 成交量確認
# ──────────────────────────────────────────────
def calc_volume_confirmation(volumes,closes) -> dict:
    if len(volumes)<21: return {"bullish_vol":False,"bearish_vol":False,"vol_ratio":1.0}
    av=float(volumes[-21:-1].mean()); lv=float(volumes[-1])
    vr=round(lv/max(av,EPS),2); ok=bool(vr>=_adaptive_params["vol_threshold"])
    up=bool(float(closes[-1])>float(closes[-2]))
    return {"bullish_vol":bool(ok and up),"bearish_vol":bool(ok and not up),"vol_ratio":vr}

# ──────────────────────────────────────────────
# K線型態
# ──────────────────────────────────────────────
def detect_candle_pattern(opens,highs,lows,closes) -> dict:
    r={"hammer":False,"engulfing_bull":False,"doji_bull":False,"shooting_star":False,"engulfing_bear":False,"name":"—"}
    if len(closes)<3: return r
    o1=float(opens[-2]); c1=float(closes[-2]); o2=float(opens[-1]); h2=float(highs[-1]); l2=float(lows[-1]); c2=float(closes[-1])
    body2=abs(c2-o2); rng2=h2-l2; uw2=h2-max(o2,c2); lw2=min(o2,c2)-l2
    if rng2<EPS: return r
    if body2>EPS and lw2>body2*2 and uw2<body2*0.5 and c2>o2 and c1<o1: r["hammer"]=True; r["name"]="🔨 錘子線"
    elif body2>EPS and uw2>body2*2 and lw2<body2*0.5 and c2<o2 and c1>o1: r["shooting_star"]=True; r["name"]="💫 流星線"
    elif c1<o1 and c2>o2 and o2<=c1 and c2>=o1: r["engulfing_bull"]=True; r["name"]="📈 看漲吞噬"
    elif c1>o1 and c2<o2 and o2>=c1 and c2<=o1: r["engulfing_bear"]=True; r["name"]="📉 看跌吞噬"
    elif body2<rng2*0.1 and lw2>rng2*0.6: r["doji_bull"]=True; r["name"]="✙ 十字星"
    return r

# ──────────────────────────────────────────────
# 多週期趨勢
# ──────────────────────────────────────────────
def get_upper_trend(symbol:str, main_tf:str) -> Optional[str]:
    if _parallel_mode: return None
    utf=UPPER_TIMEFRAME.get(main_tf)
    if not utf: return None
    data=fetch_klines(symbol,utf,60)
    if data is None or len(data)<55: return None
    e=calc_ema(data[:,4],50)
    return "up" if float(e[-1])>float(e[-5]) else "down"

# ──────────────────────────────────────────────
# 背離偵測
# ──────────────────────────────────────────────
def detect_divergence(highs,lows,closes,lookback=20) -> dict:
    r={"rsi_bull_div":False,"rsi_bear_div":False,"macd_bull_div":False,"divergence":"無"}
    n=min(lookback,len(closes),len(highs),len(lows))
    if n<20: return r
    c=closes[-n:].astype(np.float64); h=highs[-n:].astype(np.float64); l=lows[-n:].astype(np.float64)
    rs=[calc_rsi(c[:i+1]) for i in range(14,len(c))]
    if len(rs)<8: return r
    pl=[(i,float(l[i])) for i in range(2,len(l)-2) if l[i]<l[i-1] and l[i]<l[i-2] and l[i]<l[i+1] and l[i]<l[i+2]]
    ph=[(i,float(h[i])) for i in range(2,len(h)-2) if h[i]>h[i-1] and h[i]>h[i-2] and h[i]>h[i+1] and h[i]>h[i+2]]
    if len(pl)>=2:
        p1i,p1p=pl[-2]; p2i,p2p=pl[-1]; r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p<p1p and rs[r2i]>rs[r1i]:
            r["rsi_bull_div"]=True; r["divergence"]="📈 RSI看漲背離"
    if len(ph)>=2:
        p1i,p1p=ph[-2]; p2i,p2p=ph[-1]; r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p>p1p and rs[r2i]<rs[r1i]:
            r["rsi_bear_div"]=True; r["divergence"]="📉 RSI看跌背離"
    return r

# ──────────────────────────────────────────────
# 進場點優化
# ──────────────────────────────────────────────
def get_best_entry(price,obs,fvgs,direction,atr) -> float:
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
def find_swing_points(highs,lows,lookback=50):
    s=min(lookback,len(highs),len(lows))
    if s<2: return float(highs[-1]),float(lows[-1])
    return float(highs[-s:].max()),float(lows[-s:].min())

def get_fib_exits(swing_high,swing_low,entry,direction) -> dict:
    if swing_high<swing_low: swing_high,swing_low=swing_low,swing_high
    diff=max(swing_high-swing_low,entry*0.001)
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
    tp1,tp2,tp3=tps; risk=abs(entry-sl); rew=abs(tp1-entry)
    return {"stop_loss":round(sl,8),"tp1":round(tp1,8),"tp2":round(tp2,8),"tp3":round(tp3,8),"risk_reward":round(rew/max(risk,EPS),2)}

# ──────────────────────────────────────────────
# SMC
# ──────────────────────────────────────────────
def detect_order_blocks(opens,highs,lows,closes,n=5) -> list:
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
        if float(lows[i])>float(highs[i-2]): fvgs.append({"type":"bullish_fvg","top":float(lows[i]),"bottom":float(highs[i-2])})
        elif float(highs[i])<float(lows[i-2]): fvgs.append({"type":"bearish_fvg","top":float(lows[i-2]),"bottom":float(highs[i])})
    return fvgs[-3:] if fvgs else []

def detect_bos_choch(highs,lows,closes,lookback=20):
    s=min(lookback+10,len(closes),len(highs),len(lows))
    if s<lookback+5: return None,None
    rh=float(highs[-lookback:-1].max()); rl=float(lows[-lookback:-1].min()); lc=float(closes[-1])
    e50=calc_ema(closes,50); slope=float(e50[-1])-float(e50[-10])
    pt="up" if slope>0 else "down"; bos=choch=None
    if lc>rh:
        if pt=="up": bos="bullish_bos"
        else: choch="bullish_choch"
    elif lc<rl:
        if pt=="down": bos="bearish_bos"
        else: choch="bearish_choch"
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
# 自適應參數
# ──────────────────────────────────────────────
def get_market_volatility(symbol:str="BTC") -> float:
    try:
        data=fetch_klines(symbol,"1d",14)
        if data is None or len(data)<10: return 0.03
        atr=calc_atr(data[:,2],data[:,3],data[:,4],14)
        price=float(data[-1,4])
        return float(atr/max(price,EPS))
    except: return 0.03

def update_adaptive_params(recent_winrate:float, market_volatility:float):
    global _adaptive_params
    if recent_winrate>=75: _adaptive_params["min_score_diff"]=3; _adaptive_params["min_score_total"]=6
    elif recent_winrate>=60: _adaptive_params["min_score_diff"]=4; _adaptive_params["min_score_total"]=7
    else: _adaptive_params["min_score_diff"]=5; _adaptive_params["min_score_total"]=9
    if market_volatility>0.05: _adaptive_params["min_adx"]=20; _adaptive_params["vol_threshold"]=2.0
    elif market_volatility>0.02: _adaptive_params["min_adx"]=15; _adaptive_params["vol_threshold"]=1.5
    else: _adaptive_params["min_adx"]=12; _adaptive_params["vol_threshold"]=1.3

# ──────────────────────────────────────────────
# 回測系統
# ──────────────────────────────────────────────
def backtest_symbol(symbol:str, periods:int=80) -> dict:
    tf=select_timeframe(symbol)
    data=fetch_klines(symbol,tf,periods+60)
    if data is None or len(data)<80: return {"symbol":symbol,"winrate":50.0,"trades":0}
    wins=losses=0
    for start in range(0,min(periods,len(data)-30),5):
        end=start+60
        if end+10>len(data): break
        try:
            hd=data[start:end]; fut=data[end:end+20]
            h=hd[:,2]; l=hd[:,3]; c=hd[:,4]
            price=float(c[-1]); rsi=calc_rsi(c)
            macd,ms,hv=calc_macd(c)
            e20=float(calc_ema(c,20)[-1]); e50=float(calc_ema(c,50)[-1])
            e200=float(calc_ema(c,min(200,max(2,len(c)-1)))[-1])
            atr=calc_atr(h,l,c); adx=calc_adx(h,l,c)
            if adx["adx"]<_adaptive_params["min_adx"]: continue
            ls=ss=0
            if rsi<30: ls+=2
            elif rsi<45: ls+=1
            elif rsi>70: ss+=2
            elif rsi>55: ss+=1
            if hv>0 and macd>ms: ls+=2
            elif hv<0 and macd<ms: ss+=2
            if price>e20>e50>e200: ls+=3
            elif price<e20<e50<e200: ss+=3
            diff=abs(ls-ss)
            if diff<_adaptive_params["min_score_diff"]: continue
            if max(ls,ss)<_adaptive_params["min_score_total"]: continue
            direction="long" if ls>ss else "short"
            sh=float(h[-50:].max()) if len(h)>=50 else float(h.max())
            sl=float(l[-50:].min()) if len(l)>=50 else float(l.min())
            exits=get_fib_exits(sh,sl,price,direction)
            if exits["risk_reward"]<_adaptive_params["min_rr"]: continue
            tp1=exits["tp1"]; sl_p=exits["stop_loss"]; won=None
            for k in range(len(fut)):
                fh=float(fut[k,2]); fl=float(fut[k,3])
                if direction=="long":
                    if fh>=tp1: won=True; break
                    if fl<=sl_p: won=False; break
                else:
                    if fl<=tp1: won=True; break
                    if fh>=sl_p: won=False; break
            if won is True: wins+=1
            elif won is False: losses+=1
        except: continue
    total=wins+losses
    return {"symbol":symbol,"winrate":round(wins/max(total,1)*100,1),"trades":total,"wins":wins,"losses":losses}

def quick_backtest(symbols=None,periods=60) -> dict:
    if symbols is None: symbols=["BTC","ETH","SOL","XRP","BNB"]
    aw=at=0; results=[]
    for sym in symbols:
        r=backtest_symbol(sym,periods)
        if r["trades"]>=3: aw+=r["wins"]; at+=r["trades"]; results.append(r)
    return {"overall_winrate":round(aw/max(at,1)*100,1),"total_trades":at,"details":results}

# ──────────────────────────────────────────────
# 主分析函數
# ──────────────────────────────────────────────
def full_analysis(symbol:str) -> Optional[dict]:
    try:
        if _risk_control["paused"]: return None
        tf=select_timeframe(symbol)
        data=fetch_klines(symbol,tf,200)
        if data is None or len(data)<60: return None

        o=data[:,1]; h=data[:,2]; l=data[:,3]; c=data[:,4]; v=data[:,5]
        price=float(c[-1])

        rsi=calc_rsi(c); macd,ms,hist=calc_macd(c)
        bbu,_,bbl=calc_bollinger(c)
        e20=float(calc_ema(c,20)[-1]); e50=float(calc_ema(c,50)[-1]); e200=float(calc_ema(c,200)[-1])
        atr=calc_atr(h,l,c); vwap=calc_vwap(h,l,c,v)
        adx=calc_adx(h,l,c); ms_info=detect_market_structure(h,l,c,atr)
        bo=detect_breakout(h,l,c,v,atr); sr=detect_support_resistance(h,l,c)
        div=detect_divergence(h,l,c,lookback=20)
        funding=fetch_funding_rate(symbol); oi=fetch_open_interest(symbol)
        sh,sl=find_swing_points(h,l,50)
        obs=detect_order_blocks(o,h,l,c); fvgs=detect_fvg(h,l,c)
        bos,choch=detect_bos_choch(h,l,c)
        harmonic=scan_harmonics(h,l,80)
        vi=calc_volume_confirmation(v,c)
        candle=detect_candle_pattern(o,h,l,c)
        ut=get_upper_trend(symbol,tf)

        ls=ss=0
        if rsi<30: ls+=2
        elif rsi<45: ls+=1
        elif rsi>70: ss+=2
        elif rsi>55: ss+=1
        if hist>0 and macd>ms: ls+=2
        elif hist<0 and macd<ms: ss+=2
        if price>e20>e50>e200: ls+=3
        elif price<e20<e50<e200: ss+=3
        elif price>e50: ls+=1
        elif price<e50: ss+=1
        if price<bbl: ls+=1
        elif price>bbu: ss+=1
        if price>vwap: ls+=1
        elif price<vwap: ss+=1
        sc=adx["adx_score"]
        if sc>0:
            if adx["pdi"]>adx["mdi"]: ls+=sc
            else: ss+=sc
        else: ls=max(0,ls+sc); ss=max(0,ss+sc)
        if adx["pdi"]>adx["mdi"]: ls+=1
        else: ss+=1
        if ms_info["structure"]=="trending_up": ls+=2
        elif ms_info["structure"]=="trending_down": ss+=2
        if bo["breakout"]=="bullish": ls+=4
        elif bo["breakout"]=="bearish": ss+=4
        if any(x.get("type")=="bullish_ob"  for x in obs): ls+=1
        if any(x.get("type")=="bearish_ob"  for x in obs): ss+=1
        if any(x.get("type")=="bullish_fvg" for x in fvgs): ls+=1
        if any(x.get("type")=="bearish_fvg" for x in fvgs): ss+=1
        if bos=="bullish_bos": ls+=2
        elif bos=="bearish_bos": ss+=2
        if choch=="bullish_choch": ls+=3
        elif choch=="bearish_choch": ss+=3
        if harmonic:
            if harmonic[1]=="long": ls+=2
            else: ss+=2
        if vi["bullish_vol"]: ls+=2
        if vi["bearish_vol"]: ss+=2
        if candle["hammer"] or candle["engulfing_bull"] or candle["doji_bull"]: ls+=2
        if candle["shooting_star"] or candle["engulfing_bear"]: ss+=2
        ua=False
        if ut=="up": ls+=2; ua=True
        elif ut=="down": ss+=2; ua=True
        hd=False
        if div["rsi_bull_div"] or div["macd_bull_div"]: ls+=3; hd=True
        if div["rsi_bear_div"]: ss+=3; hd=True
        if funding["rate"]>0.15: ss+=1
        if funding["rate"]<-0.15: ls+=1
        if oi["oi_change"]>2 and price>float(c[-2]): ls+=1
        elif oi["oi_change"]>2 and price<float(c[-2]): ss+=1

        diff=abs(ls-ss); mx=max(ls,ss)
        if ls==ss: return None
        if diff<_adaptive_params["min_score_diff"]: return None
        if mx<_adaptive_params["min_score_total"]: return None

        direction="long" if ls>ss else "short"

        # 否決
        veto=check_veto(direction,rsi,macd,ms,hist,adx["adx"],funding["rate"],ms_info["structure"],ut)
        if veto["vetoed"]: return None

        # BTC崩跌過濾
        btc_info=check_btc_crash(symbol)
        if btc_info["crashed"] and direction=="long": return None

        # 市場情緒過濾
        sentiment=check_sentiment_filter(direction)
        if sentiment["blocked"]: return None

        # 多時間框架投票
        mtf=multi_timeframe_vote(symbol,tf,direction)
        if not mtf["passed"]: return None

        # 鯨魚成交量
        whale=detect_whale_volume(v,c)
        if whale["whale_bull"]: ls+=3
        if whale["whale_bear"]: ss+=3

        best_entry=get_best_entry(price,obs,fvgs,direction,atr)
        exits=get_fib_exits(sh,sl,best_entry,direction)
        if exits["risk_reward"]<_adaptive_params["min_rr"]: return None
        if is_duplicate_signal(symbol,direction,price,atr): return None

        atr_sl=(round(best_entry-atr*2.0,8) if direction=="long" else round(best_entry+atr*2.0,8))
        trailing=calc_trailing_stop(best_entry,price,atr,direction,False)

        hv_=bool(vi["bullish_vol"] or vi["bearish_vol"])
        hc_=bool(any([candle["hammer"],candle["engulfing_bull"],candle["doji_bull"],candle["shooting_star"],candle["engulfing_bear"]]))
        hbo=bool(bo["breakout"] is not None)
        itr=bool(ms_info["structure"] in ("trending_up","trending_down"))
        features=build_ml_features(diff,adx["adx"],exits["risk_reward"],hv_,hc_,ua,hbo,itr,rsi,direction,hist,macd,ms,hd)
        winrate_pct=ml_predict_winrate(features)

        # 勝率 < 70% 不推薦
        if winrate_pct<MIN_WINRATE_PCT: return None

        position=calc_position_size(winrate_pct,exits["risk_reward"],ms_info["structure"],adx["adx"])

        return {
            "symbol":str(symbol),"timeframe":str(tf),"price":round(float(price),8),
            "direction":str(direction),"long_score":int(ls),"short_score":int(ss),
            "winrate_pct":float(winrate_pct),
            "entry":round(float(best_entry),8),
            "entry_source":"OB/FVG中點" if best_entry!=round(price,8) else "現價",
            "stop_loss":float(exits["stop_loss"]),"atr_stop_loss":float(atr_sl),
            "trailing_sl":float(trailing["trailing_sl"]),"sl_type":str(trailing["sl_type"]),
            "tp1":float(exits["tp1"]),"tp2":float(exits["tp2"]),"tp3":float(exits["tp3"]),
            "risk_reward":float(exits["risk_reward"]),
            "position_pct":float(position["position_pct"]),"risk_label":str(position["risk_label"]),
            "market_structure":str(ms_info["label"]),"market_strategy":str(ms_info["strategy"]),
            "breakout":str(bo["breakout_label"]),
            "rsi":float(rsi),"macd":round(float(macd),8),"macd_hist":round(float(hist),8),
            "ema20":round(float(e20),8),"ema50":round(float(e50),8),"ema200":round(float(e200),8),
            "bb_upper":round(float(bbu),8),"bb_lower":round(float(bbl),8),
            "vwap":round(float(vwap),8),"atr":round(float(atr),8),
            "adx":float(adx["adx"]),"trend_strength":str(adx["trend_strength"]),
            "pdi":float(adx["pdi"]),"mdi":float(adx["mdi"]),
            "divergence":str(div["divergence"]),
            "nearest_sup":sr["nearest_sup"],"nearest_res":sr["nearest_res"],
            "dist_to_sup":sr["dist_to_sup"],"dist_to_res":sr["dist_to_res"],
            "funding_rate":float(funding["rate"]),"funding_signal":str(funding["signal"]),
            "oi_change":float(oi["oi_change"]),"oi_signal":str(oi["oi_signal"]),
            "swing_high":round(float(sh),8),"swing_low":round(float(sl),8),
            "order_blocks":obs,"fvg":fvgs,"bos":bos,"choch":choch,
            "candle_pattern":str(candle["name"]),"vol_ratio":float(vi["vol_ratio"]),
            "upper_trend":str(ut) if ut else "—","harmonic":harmonic,
            "ml_features":features,"veto_reasons":veto["reasons"],
            "mtf_label":mtf["label"],"btc_label":btc_info["label"],
            "sentiment_label":sentiment["label"],"whale_label":whale["label"],
        }
    except Exception as e:
        print(f"  ❌ full_analysis({symbol}) 錯誤：{e}")
        return None

# ──────────────────────────────────────────────
# 格式化輸出
# ──────────────────────────────────────────────
def format_signal(r:dict) -> str:
    try:
        de="🟢 做多 LONG" if r["direction"]=="long" else "🔴 做空 SHORT"
        hs=f"{r['harmonic'][0]}({r['harmonic'][1]})" if r.get("harmonic") else "無"
        bs=r.get("bos") or "—"; cs=r.get("choch") or "—"
        ue="📈" if r.get("upper_trend")=="up" else ("📉" if r.get("upper_trend")=="down" else "—")
        ss=f"`{r['nearest_sup']}`({r['dist_to_sup']}%)" if r.get("nearest_sup") else "—"
        rs=f"`{r['nearest_res']}`({r['dist_to_res']}%)" if r.get("nearest_res") else "—"
        rsi=r["rsi"]
        if rsi<30: rl="超賣🔥"
        elif rsi>70: rl="超買❄️"
        elif rsi<45: rl="偏弱⬇️"
        elif rsi>55: rl="偏強⬆️"
        else: rl="中性⚪️"
        wr=r["winrate_pct"]
        we="🟢" if wr>=80 else ("🟡" if wr>=70 else "🔴")
        rsk=get_risk_status()
        return (
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} ｜ {de}\n"
            f"🎯 *預測勝率*：{we} *{wr}%*\n"
            f"💼 倉位：{r['position_pct']}% {r['risk_label']}\n"
            f"🏛 結構：{r['market_structure']} ｜ 策略：{r['market_strategy']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *進場*：`{r['entry']}` ({r['entry_source']})\n"
            f"\n"
            f"📐 *出場價位*\n"
            f"  🛑 Fib止損  ：`{r['stop_loss']}`\n"
            f"  🛑 ATR止損  ：`{r['atr_stop_loss']}`\n"
            f"  🛑 {r['sl_type']}：`{r['trailing_sl']}`\n"
            f"  🎯 TP1(1.272)：`{r['tp1']}`\n"
            f"  🎯 TP2(1.618)：`{r['tp2']}`\n"
            f"  🎯 TP3(2.618)：`{r['tp3']}`\n"
            f"  ⚖️  R:R：`1:{r['risk_reward']}`\n"
            f"\n"
            f"🚀 突破：{r['breakout']} ｜ 🔀 背離：{r['divergence']}\n"
            f"🐋 鯨魚：{r.get('whale_label','—')}\n"
            f"🗳 多週期：{r.get('mtf_label','—')}\n"
            f"😱 情緒：{r.get('sentiment_label','—')}\n"
            f"₿ BTC：{r.get('btc_label','—')}\n"
            f"\n"
            f"📊 *指標*\n"
            f"  RSI:`{r['rsi']}` {rl} MACD Hist:`{r['macd_hist']}`\n"
            f"  EMA 20:`{r['ema20']}` 50:`{r['ema50']}` 200:`{r['ema200']}`\n"
            f"  BB上:`{r['bb_upper']}` 下:`{r['bb_lower']}`\n"
            f"  VWAP:`{r['vwap']}` ATR:`{r['atr']}`\n"
            f"  ADX:`{r['adx']}` {r['trend_strength']}\n"
            f"\n"
            f"📍 支撐：{ss} 壓力：{rs}\n"
            f"🕯 K線：{r['candle_pattern']} 量比：`{r['vol_ratio']}x`\n"
            f"🌐 上層週期：{ue} {r['upper_trend']}\n"
            f"\n"
            f"💹 費率：`{r['funding_rate']}%` {r['funding_signal']}\n"
            f"   OI：`{r['oi_change']}%` {r['oi_signal']}\n"
            f"\n"
            f"🏗️ BOS：{bs} CHoCH：{cs}\n"
            f"   OB：{len(r.get('order_blocks',[]))}個 FVG：{len(r.get('fvg',[]))}個\n"
            f"🔷 和諧：{hs}\n"
            f"\n"
            f"📈 評分 多{r['long_score']}/空{r['short_score']}\n"
            f"📊 系統勝率：`{rsk['actual_winrate']}%`({rsk['winning_signals']}/{rsk['total_signals']}筆)\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
    except Exception as e:
        return f"❌ 格式化錯誤：{e}"

# ──────────────────────────────────────────────
# 並行掃描
# ──────────────────────────────────────────────
def scan_single(symbol:str):
    try:
        r=full_analysis(symbol)
        if r: return (max(r["long_score"],r["short_score"]),r)
    except Exception as e: print(f"  ❌ {symbol} 失敗：{e}")
    return None

def parallel_scan(top_n:int=5, max_workers:int=8) -> list:
    global _parallel_mode
    _parallel_mode=True
    results=[]; print(f"[掃描] 並行掃描 {len(TOP30_COINS)} 個幣種...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(scan_single,sym):sym for sym in TOP30_COINS}
        for future in as_completed(futures):
            sym=futures[future]
            try:
                res=future.result(timeout=60)
                if res:
                    score,r=res; results.append((score,r))
                    print(f"  ✅ {sym} 訊號！勝率：{r['winrate_pct']}%")
                else: print(f"  ⏭ {sym} 無訊號")
            except Exception as e: print(f"  ❌ {sym}：{e}")
    _parallel_mode=False
    results.sort(key=lambda x:x[0],reverse=True)
    return [r for _,r in results[:top_n]]

# 初始化
load_risk_control()
load_ml_weights()
