"""
analysis_engine.py v10.0
=========================
完整優化版本

優化項目：
  1. 消除 full_analysis / full_analysis_tf 重複程式碼
     → 統一用 _analyze_core() 核心函數
  2. OKX API 加入連線測試與快取
  3. ML 特徵向量加入週期權重
  4. 自適應參數依週期分開管理
  5. 評分系統重構（加權評分，不同指標有不同可靠度）
  6. 新增趨勢確認：Supertrend 指標
  7. 新增動量指標：Stochastic RSI
  8. 訊號品質等級（A/B/C/D）
  9. 所有函數加入完整防錯處理
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

ALL_TIMEFRAMES = ["5m","15m","1h","4h","1d"]

TF_TYPE = {
    "5m":"超短線⚡","15m":"短線🔵","1h":"短中線🟡","4h":"中線🟠","1d":"長線🔴"
}

TF_SL_MULTIPLIER  = {"5m":0.8,"15m":1.0,"1h":1.5,"4h":2.0,"1d":3.0}
TF_MAX_POSITION   = {"5m":5.0,"15m":8.0,"1h":12.0,"4h":18.0,"1d":25.0}
TF_MIN_BARS       = {"5m":50,"15m":50,"1h":60,"4h":80,"1d":50}

COIN_TIMEFRAME = {
    "BTC":"4h","ETH":"4h","BNB":"4h","XRP":"4h","SOL":"4h",
    "ADA":"4h","DOGE":"4h","TRX":"4h","AVAX":"4h","SHIB":"1h",
    "DOT":"4h","LINK":"4h","MATIC":"1h","UNI":"1h","ICP":"1h",
    "LTC":"4h","APT":"1h","NEAR":"1h","ATOM":"1h","XLM":"1h",
    "FIL":"1h","HBAR":"1h","ARB":"1h","OP":"1h","INJ":"1h",
    "SUI":"1h","VET":"1h","GRT":"1h","AAVE":"4h","MKR":"4h",
}

UPPER_TIMEFRAME = {"5m":"15m","15m":"1h","1h":"4h","4h":"1d","1d":"1w"}
EPS = 1e-10

# OKX API
OKX_BASE      = "https://www.okx.com"
OKX_KLINE_URL = f"{OKX_BASE}/api/v5/market/candles"
OKX_FUNDING   = f"{OKX_BASE}/api/v5/public/funding-rate"
OKX_OI        = f"{OKX_BASE}/api/v5/public/open-interest"
OKX_INTERVAL  = {
    "1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m",
    "1h":"1H","2h":"2H","4h":"4H","6h":"6H","12h":"12H",
    "1d":"1D","1w":"1W",
}

# 動態門檻
_adaptive_params = {
    "min_score_diff":2,
    "min_score_total":4,   # 4分即可
    "min_adx":5,           # ADX>5就分析
    "min_rr":1.0,          # 風報比1.0即可
    "vol_threshold":1.0,   # 成交量不限
}

# 訊號品質門檻
GRADE_A = 80.0  # ML 勝率 ≥ 80%
GRADE_B = 70.0
GRADE_C = 65.0  # 最低門檻

MIN_WINRATE_PCT = GRADE_C
BTC_CRASH_PCT   = -8.0   # 只有BTC單日跌8%才暫停
FG_ONLY_SHORT   = 90   # 放寬：只有極度貪婪才限制做多
FG_ONLY_LONG    = 15   # 放寬：只有極度恐懼才限制做空

_parallel_mode = False
_signal_cache: dict = {}
_cache_lock = threading.Lock()
_fg_cache   = {"value":-1,"ts":0.0}
_FG_TTL     = 3600

# 檔案路徑
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_FILE = os.path.join(_BASE_DIR,"risk_control.json")
ML_FILE   = os.path.join(_BASE_DIR,"ml_weights.json")

# ──────────────────────────────────────────────
# 風控系統
# ──────────────────────────────────────────────
_risk = {
    "consecutive_losses":0,"max_consecutive_losses":3,
    "paused":False,"total_signals":0,
    "winning_signals":0,"total_pnl_pct":0.0,
}

def load_risk_control():
    global _risk
    try:
        if os.path.exists(RISK_FILE):
            with open(RISK_FILE,"r") as f: _risk.update(json.load(f))
    except Exception: pass

def save_risk_control():
    try:
        with open(RISK_FILE,"w") as f: json.dump(_risk,f)
    except Exception: pass

def record_trade_result(win:bool, pnl_pct:float=0.0):
    _risk["total_signals"] += 1
    _risk["total_pnl_pct"] += pnl_pct
    if win:
        _risk["winning_signals"] += 1
        _risk["consecutive_losses"] = 0
    else:
        _risk["consecutive_losses"] += 1
    if _risk["consecutive_losses"] >= _risk["max_consecutive_losses"]:
        _risk["paused"] = True
    save_risk_control()

def reset_risk_pause():
    _risk["paused"] = False
    _risk["consecutive_losses"] = 0
    save_risk_control()

def get_risk_status() -> dict:
    t=_risk["total_signals"]; w=_risk["winning_signals"]
    return {
        "paused":_risk["paused"],
        "consecutive_losses":_risk["consecutive_losses"],
        "total_signals":t,"winning_signals":w,
        "actual_winrate":round(w/max(t,1)*100,1),
        "total_pnl_pct":round(_risk["total_pnl_pct"],2),
    }

# ──────────────────────────────────────────────
# ML 系統
# ──────────────────────────────────────────────
_ml = {
    "w":[0.15,0.12,0.18,0.08,0.08,0.10,0.12,0.10,0.05,0.07,0.08,0.05],
    "b":-3.5,"lr":0.01,"samples":0,
}

def load_ml_weights():
    global _ml
    try:
        if os.path.exists(ML_FILE):
            with open(ML_FILE,"r") as f: _ml.update(json.load(f))
    except Exception: pass

def save_ml_weights():
    try:
        with open(ML_FILE,"w") as f: json.dump(_ml,f)
    except Exception: pass

def _sigmoid(x:float) -> float:
    try: return 1.0/(1.0+math.exp(-max(-500,min(500,x))))
    except: return 0.5

def ml_predict_winrate(features:list) -> float:
    w=_ml["w"]; b=_ml["b"]
    n=min(len(w),len(features))
    dot=sum(w[i]*features[i] for i in range(n))+b
    return round(max(35.0,min(95.0,_sigmoid(dot)*100)),1)

def ml_update(features:list, win:bool):
    y=1.0 if win else 0.0
    w=_ml["w"]; b=_ml["b"]; lr=_ml["lr"]
    n=min(len(w),len(features))
    dot=sum(w[i]*features[i] for i in range(n))+b
    pred=_sigmoid(dot); err=pred-y
    for i in range(n): w[i]-=lr*err*features[i]
    _ml["b"]-=lr*err
    _ml["samples"]+=1
    _ml["lr"]=max(0.001,0.01/(1+_ml["samples"]*0.01))
    save_ml_weights()

def build_ml_features(score_diff,adx,rr,has_vol,has_candle,upper_agree,
                       has_breakout,is_trending,rsi,direction,hist,macd,
                       macd_sig,has_div,tf_weight) -> list:
    """加入 tf_weight：不同週期給予不同特徵權重"""
    rsi_ok  = float((rsi<45 and direction=="long") or (rsi>55 and direction=="short"))
    macd_ok = float((hist>0 and macd>macd_sig and direction=="long") or
                    (hist<0 and macd<macd_sig and direction=="short"))
    return [
        min(score_diff/20.0,1.0), min(adx/60.0,1.0),
        min(rr/5.0,1.0),           float(has_vol),
        float(has_candle),          float(upper_agree),
        float(has_breakout),        float(is_trending),
        rsi_ok, macd_ok,            float(has_div),
        float(tf_weight),
    ]

# ──────────────────────────────────────────────
# OKX API
# ──────────────────────────────────────────────
def fetch_klines(symbol:str, interval:str="4h", limit:int=200) -> Optional[np.ndarray]:
    okx_bar = OKX_INTERVAL.get(interval,"4H")
    for inst in [f"{symbol}-USDT", f"{symbol}-USDT-SWAP"]:
        try:
            r = requests.get(OKX_KLINE_URL,
                params={"instId":inst,"bar":okx_bar,"limit":str(min(limit,300))},
                timeout=8)
            if not r.ok: continue
            raw = r.json()
            if raw.get("code")!="0" or not raw.get("data"): continue
            bars = raw["data"][::-1]
            if len(bars)<10: continue
            data = np.array([[float(b[0]),float(b[1]),float(b[2]),
                               float(b[3]),float(b[4]),float(b[5])]
                              for b in bars], dtype=np.float64)
            if data.ndim!=2 or data.shape[1]<6: continue
            if np.isnan(data).any() or np.isinf(data).any(): continue
            return data
        except Exception as e:
            print(f"  ⚠️ {symbol} OKX {interval} 失敗：{e}")
    return None

def select_timeframe(symbol:str) -> str:
    return COIN_TIMEFRAME.get(symbol.upper(),"1h")

def is_duplicate_signal(key:str, direction:str, price:float, atr:float) -> bool:
    k = f"{key}_{direction}"
    with _cache_lock:
        last = _signal_cache.get(k)
        if last is None:
            _signal_cache[k]=price; return False
        if abs(price-last)>atr*0.3:   # 極度放寬，幾乎不重複過濾
            _signal_cache[k]=price; return False
    return True

def fetch_fear_greed() -> dict:
    try:
        r=requests.get("https://api.alternative.me/fng/?limit=1",timeout=8)
        r.raise_for_status()
        d=r.json()["data"][0]; v=int(d["value"])
        lb=("😱 極度恐懼" if v<=24 else "😨 恐懼" if v<=49 else "😊 貪婪" if v<=74 else "🤑 極度貪婪")
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
def calc_rsi(c:np.ndarray, p:int=14) -> float:
    if len(c)<p+1: return 50.0
    d=np.diff(c.astype(np.float64))
    g=np.where(d>0,d,0.0); l=np.where(d<0,-d,0.0)
    ag=float(g[:p].mean()); al=float(l[:p].mean())
    for i in range(p,len(g)):
        ag=(ag*(p-1)+float(g[i]))/p; al=(al*(p-1)+float(l[i]))/p
    if al<EPS: return 100.0
    return round(100.0-(100.0/(1.0+ag/al)),2)

def calc_stoch_rsi(c:np.ndarray, rsi_p:int=14, stoch_p:int=14) -> dict:
    """Stochastic RSI：RSI 的隨機指標，更靈敏的超買超賣"""
    if len(c)<rsi_p+stoch_p+1: return {"k":50.0,"d":50.0}
    rsi_vals=[calc_rsi(c[:i+1]) for i in range(rsi_p,len(c))]
    if len(rsi_vals)<stoch_p: return {"k":50.0,"d":50.0}
    window=rsi_vals[-stoch_p:]
    low_r=min(window); high_r=max(window)
    if high_r-low_r<EPS: return {"k":50.0,"d":50.0}
    k=round((rsi_vals[-1]-low_r)/(high_r-low_r)*100,2)
    d=round(sum([
        (rsi_vals[-i]-min(rsi_vals[-stoch_p:-i+1] if i>1 else [rsi_vals[-1]]))/
        max(max(rsi_vals[-stoch_p:-i+1] if i>1 else [rsi_vals[-1]])-
            min(rsi_vals[-stoch_p:-i+1] if i>1 else [rsi_vals[-1]]),EPS)*100
        for i in range(1,4)])/3,2)
    return {"k":k,"d":d}

def calc_ema(c:np.ndarray, p:int) -> np.ndarray:
    c=c.astype(np.float64); e=np.zeros_like(c); k=2.0/(p+1)
    e[0]=c[0]
    for i in range(1,len(c)): e[i]=c[i]*k+e[i-1]*(1.0-k)
    return e

def calc_macd(c:np.ndarray):
    ml=calc_ema(c,12)-calc_ema(c,26); ms=calc_ema(ml,9); mh=ml-ms
    return float(ml[-1]),float(ms[-1]),float(mh[-1])

def calc_bollinger(c:np.ndarray, p:int=20, sd:float=2.0):
    if len(c)<p: m=float(c[-1]); return m,m,m
    w=c[-p:].astype(np.float64); m=float(w.mean()); s=float(w.std())
    return m+sd*s,m,m-sd*s

def calc_atr(h,l,c,p:int=14) -> float:
    if len(c)<2: return max(float(abs(h[-1]-l[-1])) if len(h)>0 else 1.0,EPS)
    n=min(len(c),len(h),len(l)); trs=[]
    for i in range(1,n):
        tr=max(float(h[i]-l[i]),abs(float(h[i]-c[i-1])),abs(float(l[i]-c[i-1])))
        trs.append(max(tr,EPS))
    if not trs: return 1.0
    return float(np.array(trs[-p:] if len(trs)>=p else trs).mean())

def calc_vwap(h,l,c,v) -> float:
    tp=(h.astype(np.float64)+l.astype(np.float64)+c.astype(np.float64))/3.0
    tv=float(np.sum(v))
    if tv<EPS: return float(c[-1])
    return round(float(np.sum(tp*v.astype(np.float64))/tv),8)

def calc_adx(h,l,c,p:int=14) -> dict:
    n=min(len(c),len(h),len(l))
    if n<p*2+1: return {"adx":0.0,"trend_strength":"橫盤⚪️","adx_score":-3,"pdi":0.0,"mdi":0.0}
    pdm=[]; mdm=[]; trs=[]
    for i in range(1,n):
        hd=float(h[i])-float(h[i-1]); ld=float(l[i-1])-float(l[i])
        pdm.append(hd if hd>ld and hd>0 else 0.0)
        mdm.append(ld if ld>hd and ld>0 else 0.0)
        tr=max(float(h[i]-l[i]),abs(float(h[i]-c[i-1])),abs(float(l[i]-c[i-1])))
        trs.append(max(tr,EPS))
    def ws(arr,pp):
        if len(arr)<pp: return [max(sum(arr),EPS)]
        r=[sum(arr[:pp])]
        for v in arr[pp:]: r.append(r[-1]-r[-1]/pp+v)
        return r
    a14=ws(trs,p); p14=ws(pdm,p); m14=ws(mdm,p)
    ml=min(len(a14),len(p14),len(m14)); dx=[]; lp=lm=0.0
    for i in range(ml):
        a=max(a14[i],EPS); pdi=100.0*p14[i]/a; mdi=100.0*m14[i]/a
        dx.append(100.0*abs(pdi-mdi)/max(pdi+mdi,EPS)); lp=pdi; lm=mdi
    if not dx: return {"adx":0.0,"trend_strength":"橫盤⚪️","adx_score":-3,"pdi":0.0,"mdi":0.0}
    adx=round(float(np.mean(dx[-p:])),2)
    if adx>=40: s,sc="強趨勢🔥",3
    elif adx>=25: s,sc="趨勢中🟡",1
    elif adx>=15: s,sc="弱趨勢🟠",0
    else: s,sc="橫盤⚪️",-3
    return {"adx":adx,"trend_strength":s,"adx_score":sc,"pdi":round(lp,2),"mdi":round(lm,2)}

def calc_supertrend(h,l,c,p:int=10,mult:float=3.0) -> dict:
    """
    Supertrend：趨勢追蹤指標
    direction=1  → 多頭（加3分）
    direction=-1 → 空頭（加3分）
    """
    if len(c)<p+2:
        return {"direction":0,"value":float(c[-1]),"label":"無法計算"}
    n=min(len(c),len(h),len(l))
    atr_vals=[calc_atr(h[max(0,i-p):i+1],l[max(0,i-p):i+1],c[max(0,i-p):i+1],p)
              for i in range(p,n)]
    if not atr_vals: return {"direction":0,"value":float(c[-1]),"label":"無法計算"}
    hl2=[(float(h[p+i])+float(l[p+i]))/2 for i in range(len(atr_vals))]
    up  =[hl2[i]+mult*atr_vals[i] for i in range(len(atr_vals))]
    dn  =[hl2[i]-mult*atr_vals[i] for i in range(len(atr_vals))]
    closes=[float(c[p+i]) for i in range(len(atr_vals))]
    st_dir=1; final_up=up[0]; final_dn=dn[0]
    for i in range(1,len(closes)):
        final_up=min(up[i],final_up) if closes[i-1]>final_up else up[i]
        final_dn=max(dn[i],final_dn) if closes[i-1]<final_dn else dn[i]
        if closes[i]>final_up: st_dir=1
        elif closes[i]<final_dn: st_dir=-1
    val=round(final_dn if st_dir==1 else final_up,8)
    lbl="📈 多頭" if st_dir==1 else "📉 空頭"
    return {"direction":st_dir,"value":val,"label":lbl}

# ──────────────────────────────────────────────
# 市場分析模組
# ──────────────────────────────────────────────
def detect_market_structure(h,l,c,atr:float) -> dict:
    if len(c)<30: return {"structure":"unknown","label":"結構不明⚪️","strategy":"觀望"}
    hh=h[-30:].astype(np.float64); ll=l[-30:].astype(np.float64)
    sh,sl=[],[]
    for i in range(2,len(hh)-2):
        if hh[i]>hh[i-1] and hh[i]>hh[i-2] and hh[i]>hh[i+1] and hh[i]>hh[i+2]: sh.append(float(hh[i]))
        if ll[i]<ll[i-1] and ll[i]<ll[i-2] and ll[i]<ll[i+1] and ll[i]<ll[i+2]: sl.append(float(ll[i]))
    if len(sh)>=2 and len(sl)>=2:
        if sh[-1]>sh[-2] and sl[-1]>sl[-2]: return {"structure":"trending_up","label":"多頭趨勢📈","strategy":"順勢做多"}
        if sh[-1]<sh[-2] and sl[-1]<sl[-2]: return {"structure":"trending_down","label":"空頭趨勢📉","strategy":"順勢做空"}
    pr=float(hh.max()-hh.min()) if len(hh)>0 else 0
    if pr<atr*8: return {"structure":"ranging","label":"震盪整理↔️","strategy":"等待突破"}
    return {"structure":"unknown","label":"結構不明⚪️","strategy":"觀望"}

def detect_breakout(h,l,c,v,atr:float) -> dict:
    if len(c)<25: return {"breakout":None,"breakout_label":"無突破","breakout_level":None}
    cc=float(c[-2]); lv=float(v[-2])
    av=float(v[-22:-2].mean()) if len(v)>=22 else float(v[:-2].mean()+EPS)
    vr=lv/max(av,EPS); vok=_adaptive_params["vol_threshold"]
    rh=float(h[-22:-2].max()) if len(h)>=22 else float(h[:-2].max())
    rl=float(l[-22:-2].min()) if len(l)>=22 else float(l[:-2].min())
    bull=bool(cc>rh and vr>=vok and (cc-rh)>=atr*0.3)
    bear=bool(cc<rl  and vr>=vok and (rl-cc)>=atr*0.3)
    if bull: return {"breakout":"bullish","breakout_label":f"🚀 看漲突破({vr:.1f}x)","breakout_level":round(rh,8)}
    if bear: return {"breakout":"bearish","breakout_label":f"💥 看跌突破({vr:.1f}x)","breakout_level":round(rl,8)}
    return {"breakout":None,"breakout_label":"無突破","breakout_level":None}

def calc_volume_confirmation(v,c) -> dict:
    if len(v)<21: return {"bullish_vol":False,"bearish_vol":False,"vol_ratio":1.0}
    av=float(v[-21:-1].mean()); lv=float(v[-1])
    vr=round(lv/max(av,EPS),2); ok=bool(vr>=_adaptive_params["vol_threshold"])
    up=bool(float(c[-1])>float(c[-2]))
    return {"bullish_vol":bool(ok and up),"bearish_vol":bool(ok and not up),"vol_ratio":vr}

def detect_whale_volume(v,c) -> dict:
    if len(v)<22: return {"whale":False,"whale_bull":False,"whale_bear":False,"vol_ratio":1.0,"label":"—"}
    av=float(v[-21:-1].mean()); lv=float(v[-1])
    vr=round(lv/max(av,EPS),2); whale=bool(vr>=3.0)
    up=bool(float(c[-1])>float(c[-2]))
    if whale and up: lbl=f"🐋 鯨魚買入({vr}x)"
    elif whale: lbl=f"🐋 鯨魚賣出({vr}x)"
    else: lbl=f"正常({vr}x)"
    return {"whale":whale,"whale_bull":bool(whale and up),"whale_bear":bool(whale and not up),"vol_ratio":vr,"label":lbl}

def detect_candle_pattern(o,h,l,c) -> dict:
    r={"hammer":False,"engulfing_bull":False,"doji_bull":False,"shooting_star":False,"engulfing_bear":False,"name":"—"}
    if len(c)<3: return r
    o1=float(o[-2]); c1=float(c[-2]); o2=float(o[-1]); h2=float(h[-1]); l2=float(l[-1]); c2=float(c[-1])
    b2=abs(c2-o2); rng2=h2-l2; uw2=h2-max(o2,c2); lw2=min(o2,c2)-l2
    if rng2<EPS: return r
    if b2>EPS and lw2>b2*2 and uw2<b2*0.5 and c2>o2 and c1<o1: r["hammer"]=True; r["name"]="🔨 錘子線"
    elif b2>EPS and uw2>b2*2 and lw2<b2*0.5 and c2<o2 and c1>o1: r["shooting_star"]=True; r["name"]="💫 流星線"
    elif c1<o1 and c2>o2 and o2<=c1 and c2>=o1: r["engulfing_bull"]=True; r["name"]="📈 看漲吞噬"
    elif c1>o1 and c2<o2 and o2>=c1 and c2<=o1: r["engulfing_bear"]=True; r["name"]="📉 看跌吞噬"
    elif b2<rng2*0.1 and lw2>rng2*0.6: r["doji_bull"]=True; r["name"]="✙ 十字星"
    return r

# ──────────────────────────────────────────────
# PA 核心：完整 K 線型態偵測
# ──────────────────────────────────────────────
def detect_candle_patterns_advanced(o, h, l, c) -> dict:
    """
    進階 PA 型態偵測：
    - 晨星 / 暮星（Morning/Evening Star）三根K線反轉
    - 三白兵 / 三黑鴉（Three White Soldiers / Crows）趨勢延續
    - 孕線（Inside Bar）整理後突破
    - 外包線（Outside Bar）強勢反轉
    - 釘頭棒（Pin Bar）高可靠度反轉
    """
    r = {
        "morning_star": False, "evening_star": False,
        "three_soldiers": False, "three_crows": False,
        "inside_bar": False, "outside_bar_bull": False,
        "outside_bar_bear": False, "pin_bar_bull": False,
        "pin_bar_bear": False, "name": "—", "signal": "neutral",
    }
    if len(c) < 4: return r

    # 最近4根K線
    o1,c1,h1,l1 = float(o[-4]),float(c[-4]),float(h[-4]),float(l[-4])
    o2,c2,h2,l2 = float(o[-3]),float(c[-3]),float(h[-3]),float(l[-3])
    o3,c3,h3,l3 = float(o[-2]),float(c[-2]),float(h[-2]),float(l[-2])
    o4,c4,h4,l4 = float(o[-1]),float(c[-1]),float(h[-1]),float(l[-1])

    b2=abs(c2-o2); b3=abs(c3-o3); b4=abs(c4-o4)
    rng2=h2-l2; rng3=h3-l3; rng4=h4-l4

    # 晨星（3根：大陰線 + 小實體 + 大陽線）
    if (c2<o2 and b2>rng2*0.6 and       # 大陰線
        b3<b2*0.3 and                    # 小實體
        c4>o4 and b4>rng4*0.6 and        # 大陽線
        c4>((o2+c2)/2)):                  # 陽線收盤超過第一根中點
        r["morning_star"]=True; r["name"]="🌅 晨星"; r["signal"]="bullish"

    # 暮星（3根：大陽線 + 小實體 + 大陰線）
    elif (c2>o2 and b2>rng2*0.6 and
          b3<b2*0.3 and
          c4<o4 and b4>rng4*0.6 and
          c4<((o2+c2)/2)):
        r["evening_star"]=True; r["name"]="🌆 暮星"; r["signal"]="bearish"

    # 三白兵（3根連續陽線，每根收盤更高）
    elif (c2>o2 and c3>o3 and c4>o4 and
          c2>c1 and c3>c2 and c4>c3 and
          o3>o2 and o4>o3):
        r["three_soldiers"]=True; r["name"]="⚔️ 三白兵"; r["signal"]="bullish"

    # 三黑鴉（3根連續陰線，每根收盤更低）
    elif (c2<o2 and c3<o3 and c4<o4 and
          c2<c1 and c3<c2 and c4<c3 and
          o3<o2 and o4<o3):
        r["three_crows"]=True; r["name"]="🐦‍⬛ 三黑鴉"; r["signal"]="bearish"

    else:
        # 孕線（Inside Bar：當前K線完全在前一根範圍內）
        if h4<=h3 and l4>=l3:
            r["inside_bar"]=True; r["name"]="📦 孕線（整理）"; r["signal"]="neutral"

        # 外包線（Outside Bar）
        elif h4>h3 and l4<l3:
            if c4>o4:
                r["outside_bar_bull"]=True; r["name"]="📈 外包吞噬（多）"; r["signal"]="bullish"
            else:
                r["outside_bar_bear"]=True; r["name"]="📉 外包吞噬（空）"; r["signal"]="bearish"

        # Pin Bar（釘頭棒：影線 > 實體3倍，反轉信號）
        elif rng4 > EPS:
            uw4=h4-max(o4,c4); lw4=min(o4,c4)-l4
            if lw4>b4*3 and uw4<b4*0.5:    # 下影線長，多頭PIN
                r["pin_bar_bull"]=True; r["name"]="📍 多頭Pin Bar"; r["signal"]="bullish"
            elif uw4>b4*3 and lw4<b4*0.5:  # 上影線長，空頭PIN
                r["pin_bar_bear"]=True; r["name"]="📍 空頭Pin Bar"; r["signal"]="bearish"

    return r


# ──────────────────────────────────────────────
# PA 核心：支撐壓力強度評估
# ──────────────────────────────────────────────
def calc_sr_strength(h, l, c, price: float, atr: float) -> dict:
    """
    評估支撐壓力強度：
    - 觸碰次數越多 = 越強的支撐/壓力
    - 距離越近 = 越重要（止損/止盈設置參考）
    - 回報多個支撐壓力位及其強度
    """
    lookback = min(100, len(h), len(l), len(c))
    if lookback < 20:
        return {"strong_sup": None, "strong_res": None,
                "sup_touches": 0, "res_touches": 0,
                "near_sup": None, "near_res": None}

    hh = h[-lookback:].astype(np.float64)
    ll = l[-lookback:].astype(np.float64)
    tolerance = atr * 0.5   # 在ATR範圍內算同一個位置

    # 收集swing points
    swing_highs, swing_lows = [], []
    for i in range(2, lookback-2):
        if hh[i]>hh[i-1] and hh[i]>hh[i-2] and hh[i]>hh[i+1] and hh[i]>hh[i+2]:
            swing_highs.append(float(hh[i]))
        if ll[i]<ll[i-1] and ll[i]<ll[i-2] and ll[i]<ll[i+1] and ll[i]<ll[i+2]:
            swing_lows.append(float(ll[i]))

    def cluster_levels(levels):
        """把相近的價位合併為同一個區域"""
        if not levels: return []
        levels = sorted(levels)
        clusters = []
        current = [levels[0]]
        for lv in levels[1:]:
            if lv - current[-1] <= tolerance:
                current.append(lv)
            else:
                clusters.append((sum(current)/len(current), len(current)))
                current = [lv]
        clusters.append((sum(current)/len(current), len(current)))
        return clusters   # (price_level, touch_count)

    res_clusters = [(lv, tc) for lv, tc in cluster_levels(swing_highs) if lv > price]
    sup_clusters = [(lv, tc) for lv, tc in cluster_levels(swing_lows)  if lv < price]

    # 找最強的支撐壓力（觸碰最多次）
    strong_res = max(res_clusters, key=lambda x: x[1], default=(None, 0))
    strong_sup = max(sup_clusters, key=lambda x: x[1], default=(None, 0))

    # 找最近的支撐壓力
    near_res = min(res_clusters, key=lambda x: abs(x[0]-price), default=(None, 0))
    near_sup = min(sup_clusters, key=lambda x: abs(x[0]-price), default=(None, 0))

    return {
        "strong_res":   round(strong_res[0], 8) if strong_res[0] else None,
        "res_touches":  strong_res[1],
        "strong_sup":   round(strong_sup[0], 8) if strong_sup[0] else None,
        "sup_touches":  strong_sup[1],
        "near_res":     round(near_res[0], 8) if near_res[0] else None,
        "near_sup":     round(near_sup[0], 8) if near_sup[0] else None,
        "dist_near_res": round((near_res[0]-price)/max(price,EPS)*100, 2) if near_res[0] else None,
        "dist_near_sup": round((price-near_sup[0])/max(price,EPS)*100, 2) if near_sup[0] else None,
    }


# ──────────────────────────────────────────────
# PA 核心：趨勢線偵測
# ──────────────────────────────────────────────
def detect_trendline(h, l, c, lookback: int = 50) -> dict:
    """
    簡化趨勢線偵測：
    - 上升趨勢線：連接近期低點的趨勢線（支撐）
    - 下降趨勢線：連接近期高點的趨勢線（壓力）
    - 偵測價格是否接近趨勢線（可能反彈或突破）
    """
    n = min(lookback, len(h), len(l), len(c))
    if n < 15:
        return {"up_trendline": None, "down_trendline": None, "near_up": False, "near_down": False}

    hh = h[-n:].astype(np.float64)
    ll = l[-n:].astype(np.float64)
    price = float(c[-1])
    atr_val = calc_atr(h[-n:], l[-n:], c[-n:], 14)

    # 上升趨勢線：用最近2個擺動低點
    swing_lows = [(i, float(ll[i])) for i in range(2, n-2)
                  if ll[i]<ll[i-1] and ll[i]<ll[i-2] and ll[i]<ll[i+1] and ll[i]<ll[i+2]]
    swing_highs = [(i, float(hh[i])) for i in range(2, n-2)
                   if hh[i]>hh[i-1] and hh[i]>hh[i-2] and hh[i]>hh[i+1] and hh[i]>hh[i+2]]

    up_tl = down_tl = None
    near_up = near_down = False

    # 上升趨勢線（連接兩個低點）
    if len(swing_lows) >= 2:
        i1, p1 = swing_lows[-2]; i2, p2 = swing_lows[-1]
        if i2 > i1 and p2 > p1:   # 更高低點
            slope = (p2 - p1) / (i2 - i1)
            tl_now = p2 + slope * (n-1 - i2)
            up_tl = round(tl_now, 8)
            near_up = bool(abs(price - tl_now) <= atr_val * 1.0)

    # 下降趨勢線（連接兩個高點）
    if len(swing_highs) >= 2:
        i1, p1 = swing_highs[-2]; i2, p2 = swing_highs[-1]
        if i2 > i1 and p2 < p1:   # 更低高點
            slope = (p2 - p1) / (i2 - i1)
            tl_now = p2 + slope * (n-1 - i2)
            down_tl = round(tl_now, 8)
            near_down = bool(abs(price - tl_now) <= atr_val * 1.0)

    return {
        "up_trendline":  up_tl,
        "down_trendline": down_tl,
        "near_up_tl":    near_up,    # 接近上升趨勢線（支撐）
        "near_down_tl":  near_down,  # 接近下降趨勢線（壓力）
    }


def detect_divergence(h,l,c,lookback=20) -> dict:
    r={"rsi_bull_div":False,"rsi_bear_div":False,"divergence":"無"}
    n=min(lookback,len(c),len(h),len(l))
    if n<20: return r
    cc=c[-n:].astype(np.float64); hh=h[-n:].astype(np.float64); ll=l[-n:].astype(np.float64)
    rs=[calc_rsi(cc[:i+1]) for i in range(14,len(cc))]
    if len(rs)<8: return r
    pl=[(i,float(ll[i])) for i in range(2,len(ll)-2) if ll[i]<ll[i-1] and ll[i]<ll[i-2] and ll[i]<ll[i+1] and ll[i]<ll[i+2]]
    ph=[(i,float(hh[i])) for i in range(2,len(hh)-2) if hh[i]>hh[i-1] and hh[i]>hh[i-2] and hh[i]>hh[i+1] and hh[i]>hh[i+2]]
    if len(pl)>=2:
        p1i,p1p=pl[-2]; p2i,p2p=pl[-1]; r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p<p1p and rs[r2i]>rs[r1i]:
            r["rsi_bull_div"]=True; r["divergence"]="📈 RSI看漲背離"
    if len(ph)>=2:
        p1i,p1p=ph[-2]; p2i,p2p=ph[-1]; r1i=p1i-14; r2i=p2i-14
        if 0<=r1i<len(rs) and 0<=r2i<len(rs) and p2p>p1p and rs[r2i]<rs[r1i]:
            r["rsi_bear_div"]=True; r["divergence"]="📉 RSI看跌背離"
    return r

def detect_order_blocks(o,h,l,c,n=5) -> list:
    obs=[]; sn=min(len(c),len(o),len(h),len(l))
    for i in range(2,sn-1):
        body=abs(float(c[i])-float(o[i])); pr=float(h[i-1])-float(l[i-1])
        if body<EPS or pr<EPS: continue
        if c[i-1]<o[i-1] and c[i]>h[i-1] and body>pr*0.5:
            obs.append({"type":"bullish_ob","high":float(h[i-1]),"low":float(l[i-1]),"index":i})
        elif c[i-1]>o[i-1] and c[i]<l[i-1] and body>pr*0.5:
            obs.append({"type":"bearish_ob","high":float(h[i-1]),"low":float(l[i-1]),"index":i})
    return obs[-n:] if obs else []

def detect_fvg(h,l,c) -> list:
    fvgs=[]; sn=min(len(c),len(h),len(l))
    for i in range(2,sn):
        if float(l[i])>float(h[i-2]): fvgs.append({"type":"bullish_fvg","top":float(l[i]),"bottom":float(h[i-2])})
        elif float(h[i])<float(l[i-2]): fvgs.append({"type":"bearish_fvg","top":float(l[i-2]),"bottom":float(h[i])})
    return fvgs[-3:] if fvgs else []

def detect_bos_choch(h,l,c,lookback=20):
    s=min(lookback+10,len(c),len(h),len(l))
    if s<lookback+5: return None,None
    rh=float(h[-lookback:-1].max()); rl=float(l[-lookback:-1].min()); lc=float(c[-1])
    e50=calc_ema(c,50); slope=float(e50[-1])-float(e50[-10])
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

def scan_harmonics(h,l,n=80):
    s=min(n,len(h),len(l))
    if s<10: return None
    hh=list(h[-s:].astype(float)); ll=list(l[-s:].astype(float)); pivots=[]
    for i in range(2,len(hh)-2):
        if hh[i]>hh[i-1] and hh[i]>hh[i+1]: pivots.append(hh[i])
        elif ll[i]<ll[i-1] and ll[i]<ll[i+1]: pivots.append(ll[i])
    if len(pivots)<5: return None
    for i in range(len(pivots)-4):
        res=check_harmonic(*pivots[i:i+5])
        if res: return res
    return None

def get_upper_trend(symbol:str, main_tf:str) -> Optional[str]:
    if _parallel_mode: return None
    utf=UPPER_TIMEFRAME.get(main_tf)
    if not utf: return None
    data=fetch_klines(symbol,utf,60)
    if data is None or len(data)<55: return None
    e=calc_ema(data[:,4],50)
    return "up" if float(e[-1])>float(e[-5]) else "down"

def multi_timeframe_vote(symbol:str, main_tf:str, direction:str) -> dict:
    tfs=["1h","4h","1d"] if not _parallel_mode else [main_tf,UPPER_TIMEFRAME.get(main_tf,main_tf)]
    votes=0; total=0; tf_r={}
    for tf in tfs:
        try:
            data=fetch_klines(symbol,tf,60)
            if data is None or len(data)<30: continue
            cc=data[:,4]
            e20=float(calc_ema(cc,20)[-1]); e50=float(calc_ema(cc,50)[-1])
            rv=calc_rsi(cc); _,_,hv=calc_macd(cc)
            bull=e20>e50 and rv>42 and hv>0
            bear=e20<e50 and rv<58 and hv<0
            ok=(direction=="long" and bull) or (direction=="short" and bear)
            if ok: votes+=1
            total+=1; tf_r[tf]="✅" if ok else "❌"
        except: continue
    needed=1 if _parallel_mode else 2
    passed=bool(votes>=needed and total>0)
    return {"passed":passed,"votes":votes,"total":total,"label":f"{'✅' if passed else '❌'} {votes}/{total}週期同向"}

def check_btc_crash(symbol:str) -> dict:
    if symbol in ("BTC","ETH"): return {"crashed":False,"btc_change":0.0,"label":"不適用"}
    try:
        data=fetch_klines("BTC","1h",5)
        if data is None or len(data)<4: return {"crashed":False,"btc_change":0.0,"label":"無法取得"}
        prev=float(data[-4,4]); last=float(data[-1,4])
        chg=round((last-prev)/max(prev,EPS)*100,2)
        crash=bool(chg<=BTC_CRASH_PCT)
        return {"crashed":crash,"btc_change":chg,"label":f"{'🚨 BTC崩跌' if crash else '✅ BTC正常'}({chg:+.1f}%)"}
    except: return {"crashed":False,"btc_change":0.0,"label":"無法取得"}

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

def check_veto(direction,rsi,macd,macd_sig,hist,adx,funding_rate,market_structure,upper_trend) -> dict:
    """只否決最極端情況，讓訊號能夠產生"""
    v=[]
    if direction=="long"  and rsi>90: v.append(f"RSI={rsi}極端超買")
    if direction=="short" and rsi<10: v.append(f"RSI={rsi}極端超賣")
    return {"vetoed":bool(v),"reasons":v}

def detect_support_resistance(h,l,c,lookback=100,n=3) -> dict:
    sl=min(lookback,len(h),len(l),len(c))
    if sl<10: return {"supports":[],"resistances":[],"nearest_sup":None,"nearest_res":None,"dist_to_sup":None,"dist_to_res":None}
    hh=h[-sl:].astype(np.float64); ll=l[-sl:].astype(np.float64); price=float(c[-1])
    rs=[]; ss=[]
    for i in range(2,len(hh)-2):
        if hh[i]>hh[i-1] and hh[i]>hh[i-2] and hh[i]>hh[i+1] and hh[i]>hh[i+2]: rs.append(float(hh[i]))
        if ll[i]<ll[i-1] and ll[i]<ll[i-2] and ll[i]<ll[i+1] and ll[i]<ll[i+2]: ss.append(float(ll[i]))
    rs=sorted(set(r for r in rs if r>price))[:n]; ss=sorted(set(s for s in ss if s<price),reverse=True)[:n]
    nr=rs[0] if rs else None; ns=ss[0] if ss else None
    dr=round((nr-price)/max(price,EPS)*100,2) if nr else None
    ds=round((price-ns)/max(price,EPS)*100,2) if ns else None
    return {"supports":ss,"resistances":rs,"nearest_sup":ns,"nearest_res":nr,"dist_to_sup":ds,"dist_to_res":dr}

def find_swing_points(h,l,lookback=50):
    s=min(lookback,len(h),len(l))
    if s<2: return float(h[-1]),float(l[-1])
    return float(h[-s:].max()),float(l[-s:].min())

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
    elif recent_winrate>=60: _adaptive_params["min_score_diff"]=3; _adaptive_params["min_score_total"]=6
    else: _adaptive_params["min_score_diff"]=4; _adaptive_params["min_score_total"]=8
    if market_volatility>0.05: _adaptive_params["min_adx"]=15; _adaptive_params["vol_threshold"]=1.8
    elif market_volatility>0.02: _adaptive_params["min_adx"]=12; _adaptive_params["vol_threshold"]=1.3
    else: _adaptive_params["min_adx"]=10; _adaptive_params["vol_threshold"]=1.1

# ──────────────────────────────────────────────
# 核心分析函數（消除重複程式碼）
# ──────────────────────────────────────────────

def _analyze_core(symbol:str, timeframe:str) -> Optional[dict]:
    """
    統一核心分析函數，供 full_analysis 和 full_analysis_tf 共用
    消除原本 300+ 行重複程式碼
    """
    try:
        if _risk["paused"]: return None

        min_bars = TF_MIN_BARS.get(timeframe, 60)
        data = fetch_klines(symbol, timeframe, 200)
        if data is None or len(data)<min_bars: return None

        o=data[:,1]; h=data[:,2]; l=data[:,3]; c=data[:,4]; v=data[:,5]
        price=float(c[-1])

        # 基礎指標
        rsi=calc_rsi(c); macd,ms,hist=calc_macd(c)
        stoch=calc_stoch_rsi(c)    # 新增
        bbu,_,bbl=calc_bollinger(c)
        e20=float(calc_ema(c,20)[-1]); e50=float(calc_ema(c,50)[-1])
        e200=float(calc_ema(c,min(200,max(2,len(c)-1)))[-1])
        atr=calc_atr(h,l,c); vwap=calc_vwap(h,l,c,v)
        adx=calc_adx(h,l,c); st=calc_supertrend(h,l,c)  # 新增 Supertrend

        ms_info=detect_market_structure(h,l,c,atr)
        bo=detect_breakout(h,l,c,v,atr)
        sr=detect_support_resistance(h,l,c)
        sr_strength=calc_sr_strength(h,l,c,price,atr)    # PA：支撐壓力強度
        candle_adv=detect_candle_patterns_advanced(o,h,l,c)  # PA：進階K線型態
        trendline=detect_trendline(h,l,c,50)              # PA：趨勢線

        # 短線跳過耗時 API
        if timeframe in ("5m","15m"):
            div={"rsi_bull_div":False,"rsi_bear_div":False,"divergence":"無"}
            funding={"rate":0.0,"signal":"—"}; oi={"oi":0.0,"oi_change":0.0,"oi_signal":"—"}
        else:
            div=detect_divergence(h,l,c,lookback=20)
            funding=fetch_funding_rate(symbol); oi=fetch_open_interest(symbol)

        sh,sl=find_swing_points(h,l,50)
        obs=detect_order_blocks(o,h,l,c); fvgs=detect_fvg(h,l,c)
        bos,choch=detect_bos_choch(h,l,c)
        harmonic=scan_harmonics(h,l,80)
        vi=calc_volume_confirmation(v,c); whale=detect_whale_volume(v,c)
        candle=detect_candle_pattern(o,h,l,c)
        ut=get_upper_trend(symbol,timeframe)

        # ══════════════════════════════════════════
        # 評分系統：PA 為主（70分）、SMC 為輔（30分）
        # PA  = Price Action：K線型態、趨勢結構、支撐壓力、成交量
        # SMC = Smart Money Concepts：OB、FVG、BOS、CHoCH
        # ══════════════════════════════════════════
        ls=ss=0

        # ── PA 主要分析（共70分上限）──

        # 1. 趨勢方向（PA核心，EMA排列，權重5）
        if price>e20>e50>e200: ls+=5
        elif price<e20<e50<e200: ss+=5
        elif price>e50>e200: ls+=3
        elif price<e50<e200: ss+=3
        elif price>e20: ls+=1
        elif price<e20: ss+=1

        # 2. Supertrend（趨勢追蹤，權重4）
        if st["direction"]==1: ls+=4
        elif st["direction"]==-1: ss+=4

        # 3. 市場結構（HH/HL vs LH/LL，PA核心，權重4）
        if ms_info["structure"]=="trending_up": ls+=4
        elif ms_info["structure"]=="trending_down": ss+=4
        elif ms_info["structure"]=="ranging": pass   # 震盪不加分

        # 4. K線型態（PA核心，權重3）
        if candle["hammer"] or candle["engulfing_bull"]: ls+=3
        elif candle["doji_bull"]: ls+=2
        if candle["shooting_star"] or candle["engulfing_bear"]: ss+=3

        # 5. 突破（量價配合，PA核心，權重4）
        if bo["breakout"]=="bullish": ls+=4
        elif bo["breakout"]=="bearish": ss+=4

        # 6. 成交量確認（PA核心，權重3）
        if vi["bullish_vol"]: ls+=3
        if vi["bearish_vol"]: ss+=3

        # 7. 鯨魚成交量（PA核心，異常量能，權重2）
        if whale["whale_bull"]: ls+=2
        if whale["whale_bear"]: ss+=2

        # 8. VWAP（價格相對機構成本，PA核心，權重2）
        if price>vwap: ls+=2
        elif price<vwap: ss+=2

        # 9. 動量（RSI + Stochastic RSI，權重3）
        if rsi<30: ls+=2
        elif rsi<42: ls+=1
        elif rsi>70: ss+=2
        elif rsi>58: ss+=1
        if stoch["k"]<20 and stoch["d"]<20: ls+=1
        elif stoch["k"]>80 and stoch["d"]>80: ss+=1

        # 10. MACD 動量方向（PA輔助，權重2）
        if hist>0 and macd>ms: ls+=2
        elif hist<0 and macd<ms: ss+=2

        # 11. Bollinger Band（PA均值回歸，權重1）
        if price<bbl: ls+=1
        elif price>bbu: ss+=1

        # 12. 背離（PA重要反轉信號，權重3）
        ua=False; hd=False
        if div["rsi_bull_div"]: ls+=3; hd=True
        if div["rsi_bear_div"]: ss+=3; hd=True

        # 13. 趨勢強度（ADX，PA輔助，依強度加分）
        sc=adx["adx_score"]
        if sc>0:
            if adx["pdi"]>adx["mdi"]: ls+=sc
            else: ss+=sc
        else:
            ls=max(0,ls+sc); ss=max(0,ss+sc)

        # 14. 多週期趨勢確認（PA核心，上層週期同向，權重3）
        if ut=="up":   ls+=3; ua=True
        elif ut=="down": ss+=3; ua=True

        # 15. 進階K線型態（PA核心，高可靠度型態，權重4）
        if candle_adv["signal"]=="bullish":
            if candle_adv["morning_star"] or candle_adv["three_soldiers"]: ls+=4
            elif candle_adv["pin_bar_bull"] or candle_adv["outside_bar_bull"]: ls+=3
        elif candle_adv["signal"]=="bearish":
            if candle_adv["evening_star"] or candle_adv["three_crows"]: ss+=4
            elif candle_adv["pin_bar_bear"] or candle_adv["outside_bar_bear"]: ss+=3

        # 16. 趨勢線（PA核心，接近趨勢線 = 關鍵位，權重2）
        #     先用初步方向判斷（ls vs ss）
        _tmp_dir = "long" if ls >= ss else "short"
        if trendline["near_up_tl"]   and _tmp_dir == "long":  ls += 2
        if trendline["near_down_tl"] and _tmp_dir == "short": ss += 2

        # 17. 強支撐壓力（PA核心，多次觸碰=越強，權重2）
        if sr_strength["sup_touches"] >= 3 and _tmp_dir == "long":  ls += 2
        if sr_strength["res_touches"] >= 3 and _tmp_dir == "short": ss += 2

        # ── SMC 次要分析（共30分上限）──
        # SMC 作為進場精確度輔助，不作為主要方向判斷

        # 15. 訂單塊 Order Block（SMC，權重2）
        #     多頭OB = 機構買入區域，空頭OB = 機構賣出區域
        if any(x.get("type")=="bullish_ob" for x in obs): ls+=2
        if any(x.get("type")=="bearish_ob" for x in obs): ss+=2

        # 16. 公平價值缺口 FVG（SMC，權重1）
        #     FVG = 機構快速移動留下的缺口，通常會回補
        if any(x.get("type")=="bullish_fvg" for x in fvgs): ls+=1
        if any(x.get("type")=="bearish_fvg" for x in fvgs): ss+=1

        # 17. 市場結構突破 BOS（SMC，確認趨勢，權重2）
        if bos=="bullish_bos": ls+=2
        elif bos=="bearish_bos": ss+=2

        # 18. 角色互換 CHoCH（SMC，最重要！趨勢反轉確認，權重4）
        #     CHoCH = 多頭轉空頭或空頭轉多頭的關鍵信號
        if choch=="bullish_choch": ls+=4
        elif choch=="bearish_choch": ss+=4

        # 19. 和諧型態（SMC/PA結合，權重2）
        if harmonic:
            if harmonic[1]=="long": ls+=2
            else: ss+=2

        # 20. 籌碼分析（輔助，資金費率+OI，權重1）
        if funding["rate"]>0.15: ss+=1
        if funding["rate"]<-0.15: ls+=1
        if oi["oi_change"]>2 and price>float(c[-2]): ls+=1
        elif oi["oi_change"]>2 and price<float(c[-2]): ss+=1

        # 過濾
        diff=abs(ls-ss); mx=max(ls,ss)
        if ls==ss: return None
        if diff<_adaptive_params["min_score_diff"]: return None
        if mx<_adaptive_params["min_score_total"]: return None

        direction="long" if ls>ss else "short"

        # 否決
        veto=check_veto(direction,rsi,macd,ms,hist,adx["adx"],funding["rate"],ms_info["structure"],ut)
        if veto["vetoed"]: return None

        # BTC崩跌（短線更敏感）
        btc_info=check_btc_crash(symbol)
        crash_pct=-3.0 if timeframe in ("5m","15m") else BTC_CRASH_PCT
        if btc_info["btc_change"]<=crash_pct and direction=="long": return None

        # 情緒過濾（只記錄，不否決）
        sentiment=check_sentiment_filter(direction)
        # 不再因情緒直接否決，只作為參考資訊顯示

        # 多時間框架投票（改為加分制，不直接否決）
        if timeframe not in ("5m","15m"):
            mtf=multi_timeframe_vote(symbol,timeframe,direction)
            mtf_label=mtf["label"]
            # 通過加2分，未通過不否決但不加分
            if mtf["passed"]:
                if direction=="long": ls+=2
                else: ss+=2
        else:
            mtf={"passed":True,"votes":1,"total":1}
            mtf_label="短線（跳過）"

        # 鯨魚加分
        if whale["whale_bull"]: ls+=3
        if whale["whale_bear"]: ss+=3

        best_entry=get_best_entry(price,obs,fvgs,direction,atr)
        exits=get_fib_exits(sh,sl,best_entry,direction)

        # 週期專屬止損
        sl_mult=TF_SL_MULTIPLIER.get(timeframe,2.0)
        atr_sl=(round(best_entry-atr*sl_mult,8) if direction=="long"
                else round(best_entry+atr*sl_mult,8))

        if exits["risk_reward"]<0.8: return None   # 只過濾極端差的風報比

        # 防重複
        cache_key=f"{symbol}_{timeframe}"
        if is_duplicate_signal(cache_key,direction,price,atr): return None

        trailing=calc_trailing_stop(best_entry,price,atr,direction,False)

        # ML 預測
        tf_weight={"5m":0.6,"15m":0.7,"1h":0.85,"4h":1.0,"1d":0.9}.get(timeframe,0.8)
        hv_=bool(vi["bullish_vol"] or vi["bearish_vol"])
        hc_=bool(any([candle["hammer"],candle["engulfing_bull"],candle["doji_bull"],
                      candle["shooting_star"],candle["engulfing_bear"]]))
        hbo=bool(bo["breakout"] is not None)
        itr=bool(ms_info["structure"] in ("trending_up","trending_down"))
        features=build_ml_features(diff,adx["adx"],exits["risk_reward"],
                                   hv_,hc_,ua,hbo,itr,rsi,direction,hist,macd,ms,hd,tf_weight)
        winrate_pct=ml_predict_winrate(features)

        # 勝率門檻（依訓練樣本數動態調整）
        samples = _ml.get("samples", 0)
        if samples < 50:
            min_wr = 45.0   # 樣本不足時幾乎不過濾，優先累積數據
        elif samples < 100:
            min_wr = 55.0
        else:
            min_wr = 60.0 if timeframe in ("5m","15m") else MIN_WINRATE_PCT
        if winrate_pct < min_wr: return None

        # 訊號品質等級
        if winrate_pct>=GRADE_A: grade="🏆 A級"
        elif winrate_pct>=GRADE_B: grade="⭐ B級"
        elif winrate_pct>=GRADE_C: grade="✅ C級"
        else: grade="⚪️ 觀察"

        # 週期專屬倉位上限
        max_pos=TF_MAX_POSITION.get(timeframe,15.0)
        position=calc_position_size(winrate_pct,exits["risk_reward"],ms_info["structure"],adx["adx"])
        position["position_pct"]=min(position["position_pct"],max_pos)

        tf_type=TF_TYPE.get(timeframe,"未知")

        return {
            "symbol":str(symbol),"timeframe":str(timeframe),
            "tf_type":str(tf_type),"grade":str(grade),
            "price":round(float(price),8),"direction":str(direction),
            "long_score":int(ls),"short_score":int(ss),
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
            "supertrend":str(st["label"]),"supertrend_val":float(st["value"]),
            "stoch_k":float(stoch["k"]),"stoch_d":float(stoch["d"]),
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
            "whale_label":str(whale["label"]),
            "upper_trend":str(ut) if ut else "—","harmonic":harmonic,
            "mtf_label":str(mtf_label),
            "btc_label":str(btc_info["label"]),
            "sentiment_label":str(sentiment["label"]),
            # PA 進階欄位
            "candle_adv_name":str(candle_adv["name"]),
            "candle_adv_signal":str(candle_adv["signal"]),
            "up_trendline":trendline["up_trendline"],
            "down_trendline":trendline["down_trendline"],
            "near_up_tl":bool(trendline["near_up_tl"]),
            "near_down_tl":bool(trendline["near_down_tl"]),
            "strong_sup":sr_strength["strong_sup"],
            "strong_res":sr_strength["strong_res"],
            "sup_touches":int(sr_strength["sup_touches"]),
            "res_touches":int(sr_strength["res_touches"]),
            "near_sup":sr_strength["near_sup"],
            "near_res":sr_strength["near_res"],
            "dist_near_sup":sr_strength["dist_near_sup"],
            "dist_near_res":sr_strength["dist_near_res"],
            "trendline_label": str(trendline.get("label","—")),
            "sr_label":f"支撐{sr_strength['sup_touches']}次/壓力{sr_strength['res_touches']}次",
            "pa_pattern":str(candle_adv.get("name","—")),
            "ml_features":features,"veto_reasons":veto["reasons"],
        }

    except Exception as e:
        print(f"  ❌ _analyze_core({symbol},{timeframe}) 錯誤：{e}")
        return None

# ──────────────────────────────────────────────
# 公開分析函數（使用核心函數）
# ──────────────────────────────────────────────

def full_analysis(symbol:str) -> Optional[dict]:
    """分析單一幣種的主力週期"""
    return _analyze_core(symbol, select_timeframe(symbol))

def full_analysis_tf(symbol:str, timeframe:str) -> Optional[dict]:
    """分析單一幣種的指定週期"""
    return _analyze_core(symbol, timeframe)

def full_tf_scan(symbol:str, top_n:int=5) -> list:
    """完整掃描單一幣種的所有週期（5m~1d）"""
    results=[]
    for tf in ALL_TIMEFRAMES:
        try:
            r=_analyze_core(symbol,tf)
            if r: results.append((max(r["long_score"],r["short_score"]),r))
        except Exception as e: print(f"  ❌ {symbol} {tf} 失敗：{e}")
    results.sort(key=lambda x:x[0],reverse=True)
    return [r for _,r in results[:top_n]]

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
            f"📌 *{r['symbol']}USDT* ｜ {r['timeframe']} {r.get('tf_type','')} ｜ {de}\n"
            f"🏆 訊號等級：{r.get('grade','—')}\n"
            f"🎯 *預測勝率*：{we} *{wr}%*\n"
            f"💼 倉位：{r['position_pct']}% {r['risk_label']}\n"
            f"🏛 結構：{r['market_structure']} ｜ 策略：{r['market_strategy']}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 *進場*：`{r['entry']}` ({r['entry_source']})\n"
            f"\n"
            f"📐 *出場價位*\n"
            f"  🛑 Fib止損   ：`{r['stop_loss']}`\n"
            f"  🛑 ATR止損   ：`{r['atr_stop_loss']}`\n"
            f"  🛑 {r['sl_type']}：`{r['trailing_sl']}`\n"
            f"  🎯 TP1(1.272)：`{r['tp1']}`\n"
            f"  🎯 TP2(1.618)：`{r['tp2']}`\n"
            f"  🎯 TP3(2.618)：`{r['tp3']}`\n"
            f"  ⚖️  R:R：`1:{r['risk_reward']}`\n"
            f"\n"
            f"📊 *技術指標*\n"
            f"  RSI：`{r['rsi']}` {rl} ｜ StochRSI K:`{r.get('stoch_k',0)}` D:`{r.get('stoch_d',0)}`\n"
            f"  MACD Hist：`{r['macd_hist']}`\n"
            f"  EMA 20:`{r['ema20']}` 50:`{r['ema50']}` 200:`{r['ema200']}`\n"
            f"  BB上:`{r['bb_upper']}` 下:`{r['bb_lower']}`\n"
            f"  VWAP:`{r['vwap']}` ATR:`{r['atr']}`\n"
            f"  ADX:`{r['adx']}` {r['trend_strength']}\n"
            f"  🌊 Supertrend：{r.get('supertrend','—')}\n"
            f"\n"
            f"🚀 突破：{r['breakout']} ｜ 🔀 背離：{r['divergence']}\n"
            f"🐋 鯨魚：{r.get('whale_label','—')}\n"
            f"🗳 多週期：{r.get('mtf_label','—')}\n"
            f"😱 情緒：{r.get('sentiment_label','—')}\n"
            f"₿ BTC：{r.get('btc_label','—')}\n"
            f"\n"
            f"📍 支撐：{ss} 壓力：{rs}\n"
            f"🕯 K線：{r['candle_pattern']} 量比：`{r['vol_ratio']}x`\n"
            f"📊 PA型態：{r.get('candle_adv_name','—')}\\n"
            f"📏 趨勢線：{'✅接近支撐' if r.get('near_up_tl') else ('⚠️接近壓力' if r.get('near_down_tl') else '—')}\\n"
            f"🔒 強支撐：`{r.get('strong_sup','—')}`({r.get('sup_touches',0)}次) 強壓力：`{r.get('strong_res','—')}`({r.get('res_touches',0)}次)\\n"
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

def _scan_symbol_multitf(symbol:str) -> list:
    tfs=["1h","4h","1d"] if _parallel_mode else ALL_TIMEFRAMES
    results=[]
    for tf in tfs:
        try:
            r=_analyze_core(symbol,tf)
            if r: results.append((max(r["long_score"],r["short_score"]),r))
        except Exception as e: print(f"  ❌ {symbol} {tf}：{e}")
    return results

def parallel_scan(top_n:int=5, max_workers:int=8) -> list:
    global _parallel_mode
    _parallel_mode=True
    all_results=[]; print(f"[掃描] 多週期並行掃描 {len(TOP30_COINS)} 個幣種...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures={ex.submit(_scan_symbol_multitf,sym):sym for sym in TOP30_COINS}
        for future in as_completed(futures):
            sym=futures[future]
            try:
                results=future.result(timeout=90)
                if results:
                    for score,r in results:
                        all_results.append((score,r))
                        print(f"  ✅ {sym} {r['timeframe']} {r['grade']} 勝率{r['winrate_pct']}%")
                else: print(f"  ⏭ {sym} 無訊號")
            except Exception as e: print(f"  ❌ {sym}：{e}")
    _parallel_mode=False
    all_results.sort(key=lambda x:x[0],reverse=True)
    return [r for _,r in all_results[:top_n]]

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
            macd_v,ms_v,hv=calc_macd(c)
            e20=float(calc_ema(c,20)[-1]); e50=float(calc_ema(c,50)[-1])
            e200=float(calc_ema(c,min(200,max(2,len(c)-1)))[-1])
            atr=calc_atr(h,l,c); adx=calc_adx(h,l,c)
            if adx["adx"]<_adaptive_params["min_adx"]: continue
            ls_b=ss_b=0
            if rsi<30: ls_b+=2
            elif rsi<45: ls_b+=1
            elif rsi>70: ss_b+=2
            elif rsi>55: ss_b+=1
            if hv>0 and macd_v>ms_v: ls_b+=2
            elif hv<0 and macd_v<ms_v: ss_b+=2
            if price>e20>e50>e200: ls_b+=3
            elif price<e20<e50<e200: ss_b+=3
            diff=abs(ls_b-ss_b)
            if diff<_adaptive_params["min_score_diff"]: continue
            if max(ls_b,ss_b)<_adaptive_params["min_score_total"]: continue
            direction="long" if ls_b>ss_b else "short"
            sh_b=float(h[-50:].max()) if len(h)>=50 else float(h.max())
            sl_b=float(l[-50:].min()) if len(l)>=50 else float(l.min())
            exits=get_fib_exits(sh_b,sl_b,price,direction)
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

# 初始化
load_risk_control()
load_ml_weights()
