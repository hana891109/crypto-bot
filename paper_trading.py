"""
paper_trading.py v2.0
=====================
自主模擬學習交易系統
"""

import os, json, time, threading
from datetime import datetime, timezone, timedelta
from typing import Optional

TW_TZ           = timezone(timedelta(hours=8))
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PAPER_FILE      = os.path.join(BASE_DIR, "paper_trades.json")
INITIAL_CAPITAL = 10000.0
MIN_LEVERAGE    = 5
MAX_LEVERAGE    = 50
TAKER_FEE       = 0.0005
MAX_OPEN_TRADES = 10
MAX_POSITION_PCT = 0.08

_lock = threading.Lock()

def _default_data() -> dict:
    return {
        "capital": INITIAL_CAPITAL, "peak_capital": INITIAL_CAPITAL,
        "open_trades": [], "closed_trades": [],
        "total_pnl": 0.0, "total_fee": 0.0,
        "wins": 0, "losses": 0, "consecutive_losses": 0,
        "equity_curve": [INITIAL_CAPITAL],
        "tf_stats": {}, "symbol_stats": {},
        "created_at": _tw_now(), "last_updated": _tw_now(),
    }

def _load() -> dict:
    try:
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in _default_data().items():
                    if k not in data: data[k] = v
                return data
    except Exception: pass
    return _default_data()

def _save(data: dict):
    data["last_updated"] = _tw_now()
    try:
        with open(PAPER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[模擬] 儲存失敗：{e}")

def _tw_now() -> str:
    return datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M:%S")

def _hours_ago(time_str: str) -> float:
    try:
        t = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S").replace(tzinfo=TW_TZ)
        return (datetime.now(TW_TZ) - t).total_seconds() / 3600
    except Exception: return 0.0

def calc_leverage(winrate: float, adx: float, structure: str,
                  rr: float, consecutive_losses: int) -> int:
    base = 5
    if winrate >= 80:   base += 25
    elif winrate >= 70: base += 15
    elif winrate >= 60: base += 8
    elif winrate >= 50: base += 3
    if adx >= 40:   base += 10
    elif adx >= 25: base += 5
    elif adx < 15:  base -= 3
    if structure in ("trending_up", "trending_down"): base += 5
    elif structure == "ranging": base -= 5
    if rr >= 3.0:   base += 5
    elif rr >= 2.0: base += 3
    base -= consecutive_losses * 5
    return max(MIN_LEVERAGE, min(MAX_LEVERAGE, base))

def calc_kelly_position(capital: float, wins: int, losses: int, rr: float) -> float:
    total = wins + losses
    if total < 10: return capital * 0.05
    wr = wins / total; lr = 1.0 - wr
    kelly = (wr * rr - lr) / max(rr, 1e-10)
    pct = max(0.02, min(MAX_POSITION_PCT, kelly * 0.5))
    return round(capital * pct, 2)

def _update_stats(data: dict, trade: dict):
    tf = trade["timeframe"]; sym = trade["symbol"]; win = trade["pnl_usdt"] > 0
    for key, cat in [(tf, "tf_stats"), (sym, "symbol_stats")]:
        if key not in data[cat]:
            data[cat][key] = {"wins": 0, "losses": 0, "pnl": 0.0}
        data[cat][key]["wins" if win else "losses"] += 1
        data[cat][key]["pnl"] = round(data[cat][key]["pnl"] + trade["pnl_usdt"], 2)

def open_paper_trade(signal: dict) -> Optional[dict]:
    with _lock:
        data = _load()
        if data["capital"] <= 100: return None
        if len(data["open_trades"]) >= MAX_OPEN_TRADES: return None
        sym = signal["symbol"]; dir_ = signal["direction"]
        for t in data["open_trades"]:
            if t["symbol"] == sym and t["direction"] == dir_: return None

        entry     = float(signal["entry"])
        winrate   = float(signal.get("winrate_pct", 50.0))
        adx       = float(signal.get("adx", 20.0))
        structure = str(signal.get("market_structure", "unknown"))
        rr        = float(signal.get("risk_reward", 1.5))
        timeframe = str(signal.get("timeframe", "4h"))

        leverage      = calc_leverage(winrate, adx, structure, rr, data["consecutive_losses"])
        position_usdt = calc_kelly_position(data["capital"], data["wins"], data["losses"], rr)
        margin        = round(position_usdt / leverage, 2)
        open_fee      = round(position_usdt * TAKER_FEE, 4)

        trade = {
            "id": f"{sym}_{timeframe}_{int(time.time())}",
            "symbol": sym, "timeframe": timeframe, "direction": dir_,
            "entry_price": round(entry, 8),
            "position_usdt": round(position_usdt, 2),
            "position_size": round(position_usdt / entry, 6),
            "leverage": leverage, "margin": margin, "open_fee": open_fee,
            "tp1": round(float(signal["tp1"]), 8),
            "tp2": round(float(signal["tp2"]), 8),
            "tp3": round(float(signal["tp3"]), 8),
            "stop_loss": round(float(signal["stop_loss"]), 8),
            "trailing_sl": round(float(signal["stop_loss"]), 8),
            "tp1_hit": False, "tp2_hit": False,
            "winrate_pct": winrate, "adx": adx, "rr": rr, "structure": structure,
            "ml_features": signal.get("ml_features", []),
            "status": "open", "open_time": _tw_now(),
            "close_time": None, "close_price": None,
            "close_reason": None, "pnl_usdt": 0.0, "pnl_pct": 0.0,
            "max_loss_usdt": round(margin * 0.8, 2),
        }
        data["open_trades"].append(trade)
        data["capital"] = round(data["capital"] - open_fee, 2)
        _save(data)
        print(f"[模擬開倉] {sym} {timeframe} {dir_} @{entry} x{leverage} 保證金${margin}")
        return trade

def close_paper_trade(trade_id: str, close_price: float,
                       reason: str, auto_ml: bool = True) -> Optional[dict]:
    with _lock:
        data = _load()
        trade = next((t for t in data["open_trades"] if t["id"] == trade_id), None)
        if not trade: return None

        entry = trade["entry_price"]; leverage = trade["leverage"]
        margin = trade["margin"]; dir_ = trade["direction"]

        raw_pct = ((close_price - entry) / entry * leverage if dir_ == "long"
                   else (entry - close_price) / entry * leverage)
        close_fee = round(trade["position_usdt"] * TAKER_FEE, 4)
        pnl_usdt  = round(margin * raw_pct - close_fee, 2)
        pnl_pct   = round(raw_pct * 100, 2)

        if pnl_usdt < -trade["max_loss_usdt"]:
            pnl_usdt = -trade["max_loss_usdt"]
            pnl_pct  = round(-trade["max_loss_usdt"] / margin * 100, 2)

        win = bool(pnl_usdt > 0)
        trade.update({
            "status": "closed", "close_time": _tw_now(),
            "close_price": round(close_price, 8), "close_reason": reason,
            "close_fee": close_fee, "pnl_usdt": pnl_usdt, "pnl_pct": pnl_pct,
        })
        data["open_trades"]    = [t for t in data["open_trades"] if t["id"] != trade_id]
        data["closed_trades"].append(trade)
        data["capital"]        = round(data["capital"] + margin + pnl_usdt, 2)
        data["total_pnl"]      = round(data["total_pnl"] + pnl_usdt, 2)
        data["total_fee"]      = round(data["total_fee"] + close_fee + trade.get("open_fee", 0), 4)
        data["peak_capital"]   = max(data["peak_capital"], data["capital"])
        if win:
            data["wins"] += 1; data["consecutive_losses"] = 0
        else:
            data["losses"] += 1; data["consecutive_losses"] += 1
        data["equity_curve"].append(round(data["capital"], 2))
        if len(data["equity_curve"]) > 1000:
            data["equity_curve"] = data["equity_curve"][-500:]
        _update_stats(data, trade)
        _save(data)

    if auto_ml and trade.get("ml_features"):
        try:
            from analysis_engine import ml_update, record_trade_result
            ml_update(trade["ml_features"], win)
            record_trade_result(win, pnl_pct / 100)
            print(f"[模擬學習] ML更新 {trade['symbol']} {'獲利' if win else '虧損'} {pnl_pct}%")
        except Exception as e:
            print(f"[模擬學習] ML失敗：{e}")

    print(f"[模擬平倉] {trade['symbol']} {reason} @{close_price} PNL:{'+' if pnl_usdt>=0 else ''}{pnl_usdt}({pnl_pct}%)")
    return trade

def _update_trailing_sl(trade: dict, current_price: float) -> float:
    entry = trade["entry_price"]; dir_ = trade["direction"]; old_sl = trade["trailing_sl"]
    if not trade["tp1_hit"]: return old_sl
    if dir_ == "long":
        return max(round(entry * 1.001, 8), round(current_price * 0.99, 8), old_sl)
    else:
        return min(round(entry * 0.999, 8), round(current_price * 1.01, 8), old_sl)

def monitor_paper_trades():
    from analysis_engine import fetch_klines
    print("[模擬監控] 啟動，每30秒檢查持倉")
    while True:
        time.sleep(30)
        try:
            with _lock:
                data = _load()
                open_trades = list(data["open_trades"])
            if not open_trades: continue

            for trade in open_trades:
                try:
                    klines = fetch_klines(trade["symbol"], "1m", 3)
                    if klines is None or len(klines) < 1: continue
                    price = float(klines[-1, 4]); tid = trade["id"]; dir_ = trade["direction"]

                    new_sl = _update_trailing_sl(trade, price)
                    if new_sl != trade["trailing_sl"]:
                        with _lock:
                            d2 = _load()
                            for t in d2["open_trades"]:
                                if t["id"] == tid: t["trailing_sl"] = new_sl; break
                            _save(d2)
                        trade["trailing_sl"] = new_sl

                    if ((dir_=="long" and price<=trade["trailing_sl"]) or
                            (dir_=="short" and price>=trade["trailing_sl"])):
                        close_paper_trade(tid, price, "SL止損"); continue

                    if (trade.get("tp2_hit") and
                            ((dir_=="long" and price>=trade["tp3"]) or
                             (dir_=="short" and price<=trade["tp3"]))):
                        close_paper_trade(tid, price, "TP3全倉"); continue

                    if (trade.get("tp1_hit") and not trade.get("tp2_hit") and
                            ((dir_=="long" and price>=trade["tp2"]) or
                             (dir_=="short" and price<=trade["tp2"]))):
                        with _lock:
                            d2=_load()
                            for t in d2["open_trades"]:
                                if t["id"]==tid: t["tp2_hit"]=True; break
                            _save(d2)
                        trade["tp2_hit"] = True

                    elif not trade.get("tp1_hit") and (
                            (dir_=="long" and price>=trade["tp1"]) or
                            (dir_=="short" and price<=trade["tp1"])):
                        with _lock:
                            d2=_load()
                            for t in d2["open_trades"]:
                                if t["id"]==tid: t["tp1_hit"]=True; break
                            _save(d2)
                        trade["tp1_hit"] = True

                    tf_timeout = {"5m":4,"15m":8,"1h":24,"4h":72,"1d":168}
                    if _hours_ago(trade["open_time"]) > tf_timeout.get(trade["timeframe"], 48):
                        close_paper_trade(tid, price, "超時")

                except Exception as e:
                    print(f"[模擬監控] {trade.get('symbol','?')} 錯誤：{e}")
        except Exception as e:
            print(f"[模擬監控] 主循環錯誤：{e}")

def get_paper_stats() -> dict:
    with _lock: data = _load()
    total = data["wins"] + data["losses"]
    winrate = round(data["wins"] / max(total, 1) * 100, 1)
    pnl = data["total_pnl"]; pnl_pct = round(pnl / INITIAL_CAPITAL * 100, 2)
    curve = data["equity_curve"]
    peak = max(curve) if curve else INITIAL_CAPITAL
    trough = min(curve[curve.index(peak):]) if len(curve) > 1 else data["capital"]
    drawdown = round((peak - trough) / max(peak, 1) * 100, 2)
    all_t = data["closed_trades"] + data["open_trades"]
    avg_lev = round(sum(t.get("leverage", 10) for t in all_t) / max(len(all_t), 1), 1)
    tf_stats = data.get("tf_stats", {}); sym_stats = data.get("symbol_stats", {})
    best_tf   = max(tf_stats.items(),  key=lambda x: x[1]["pnl"], default=(None, None))
    worst_tf  = min(tf_stats.items(),  key=lambda x: x[1]["pnl"], default=(None, None))
    best_sym  = max(sym_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))
    worst_sym = min(sym_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))
    wins_pnl  = [t["pnl_usdt"] for t in data["closed_trades"] if t["pnl_usdt"] > 0]
    loss_pnl  = [t["pnl_usdt"] for t in data["closed_trades"] if t["pnl_usdt"] < 0]
    avg_win   = round(sum(wins_pnl) / len(wins_pnl), 2) if wins_pnl else 0.0
    avg_loss  = round(sum(loss_pnl) / len(loss_pnl), 2) if loss_pnl else 0.0
    expected  = round(winrate / 100 * avg_win + (1 - winrate / 100) * avg_loss, 2)
    return {
        "capital": data["capital"], "initial": INITIAL_CAPITAL,
        "total_pnl": pnl, "pnl_pct": pnl_pct,
        "total_fee": round(data.get("total_fee", 0), 2),
        "total_trades": total, "wins": data["wins"], "losses": data["losses"],
        "winrate": winrate, "open_trades": len(data["open_trades"]),
        "avg_leverage": avg_lev, "max_drawdown": drawdown,
        "consecutive_losses": data.get("consecutive_losses", 0),
        "avg_win": avg_win, "avg_loss": avg_loss, "expected_value": expected,
        "best_tf": best_tf, "worst_tf": worst_tf,
        "best_sym": best_sym, "worst_sym": worst_sym,
        "tf_stats": tf_stats, "sym_stats": sym_stats,
        "recent_trades": data["closed_trades"][-10:] if data["closed_trades"] else [],
        "equity_curve": curve[-20:],
    }

def format_paper_stats() -> str:
    s = get_paper_stats()
    pe = "📈" if s["total_pnl"] >= 0 else "📉"
    btf  = f"{s['best_tf'][0]}(+${s['best_tf'][1]['pnl']})"    if s["best_tf"][0]  else "—"
    wtf  = f"{s['worst_tf'][0]}(${s['worst_tf'][1]['pnl']})"   if s["worst_tf"][0] else "—"
    bsym = f"{s['best_sym'][0]}(+${s['best_sym'][1]['pnl']})"  if s["best_sym"][0]  else "—"
    wsym = f"{s['worst_sym'][0]}(${s['worst_sym'][1]['pnl']})" if s["worst_sym"][0] else "—"
    msg = (
        f"🤖 *模擬交易統計 v2.0*\n━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 初始：`${s['initial']:,.0f}` → 目前：`${s['capital']:,.2f}`\n"
        f"{pe} 總損益：`{'+' if s['total_pnl']>=0 else ''}{s['total_pnl']:,.2f}` USDT (`{s['pnl_pct']}%`)\n"
        f"手續費：`${s['total_fee']:,.2f}` 最大回撤：`{s['max_drawdown']}%`\n\n"
        f"📊 *交易統計*\n"
        f"  總：`{s['total_trades']}` 持倉：`{s['open_trades']}` 連虧：`{s['consecutive_losses']}`\n"
        f"  獲利：`{s['wins']}` 虧損：`{s['losses']}` *勝率：`{s['winrate']}%`*\n"
        f"  平均槓桿：`{s['avg_leverage']}x`\n\n"
        f"📐 *損益分析*\n"
        f"  平均獲利：`+${s['avg_win']}` 平均虧損：`${s['avg_loss']}`\n"
        f"  預期值：`{'+' if s['expected_value']>=0 else ''}{s['expected_value']} USDT/筆`\n\n"
        f"🏆 最佳週期：{btf} 最差：{wtf}\n"
        f"🏆 最佳幣種：{bsym} 最差：{wsym}\n"
    )
    if s["recent_trades"]:
        msg += "\n📋 *最近10筆*\n"
        for t in reversed(s["recent_trades"]):
            icon = "✅" if t["pnl_usdt"] >= 0 else "❌"
            d = "多" if t["direction"] == "long" else "空"
            msg += f"  {icon} {t['symbol']} {t['timeframe']} {d}x{t['leverage']} `{'+' if t['pnl_usdt']>=0 else ''}{t['pnl_usdt']}`({t['close_reason']})\n"
    msg += "━━━━━━━━━━━━━━━━━━━━"
    return msg

def format_open_trades() -> str:
    with _lock: data = _load()
    trades = data["open_trades"]
    if not trades: return "📋 目前無模擬持倉"
    msg = f"📋 *模擬持倉（{len(trades)} 筆）*\n━━━━━━━━━━━━━━━━━━━━\n"
    for t in trades:
        d = "🟢多" if t["direction"] == "long" else "🔴空"
        tp = "TP2✅" if t.get("tp2_hit") else ("TP1✅" if t.get("tp1_hit") else "持倉中")
        hrs = round(_hours_ago(t["open_time"]), 1)
        msg += (f"{d} *{t['symbol']}* {t['timeframe']} x{t['leverage']}\n"
                f"  進場：`{t['entry_price']}` 保證金：`${t['margin']}`\n"
                f"  移動止損：`{t['trailing_sl']}` TP1：`{t['tp1']}`\n"
                f"  狀態：{tp} 持倉{hrs}h\n\n")
    return msg

def format_tf_report() -> str:
    s = get_paper_stats(); tf = s["tf_stats"]
    if not tf: return "📊 尚無週期統計數據"
    msg = "📊 *週期績效報告*\n━━━━━━━━━━━━━━━━━━━━\n"
    for name, stat in sorted(tf.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total = stat["wins"] + stat["losses"]
        wr = round(stat["wins"] / max(total, 1) * 100, 1)
        pe = "📈" if stat["pnl"] >= 0 else "📉"
        msg += f"  {pe} *{name}*：勝率`{wr}%`({stat['wins']}勝{stat['losses']}敗) 損益`{'+' if stat['pnl']>=0 else ''}{stat['pnl']}`\n"
    msg += "━━━━━━━━━━━━━━━━━━━━"
    return msg

def get_daily_summary() -> str:
    s = get_paper_stats()
    growth = round((s["capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
    return (
        f"🧠 *模擬交易每日學習報告*\n\n"
        f"📈 資金成長：`{'+' if growth>=0 else ''}{growth}%`\n"
        f"🎯 累計勝率：`{s['winrate']}%` ({s['total_trades']}筆)\n"
        f"💡 預期值：`{s['expected_value']} USDT/筆`\n"
        f"📉 最大回撤：`{s['max_drawdown']}%`\n\n"
        f"🏆 最佳週期：{s['best_tf'][0] if s['best_tf'][0] else '—'}\n"
        f"🏆 最佳幣種：{s['best_sym'][0] if s['best_sym'][0] else '—'}\n\n"
        f"🤖 ML已從 {s['total_trades']} 筆模擬交易中學習"
    )

def reset_paper_trading() -> bool:
    with _lock: _save(_default_data())
    return True
