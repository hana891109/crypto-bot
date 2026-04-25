"""
paper_trading.py v2.0
=====================
自主模擬學習交易系統

優化項目：
  1. 修正 Lock 競態條件（_load_data 統一在 Lock 內執行）
  2. 槓桿計算改為保守模式（避免模擬爆倉導致數據失真）
  3. 平倉後自動更新 ML 權重（真正實現自主學習）
  4. 分週期/分幣種統計（找出最佳標的和週期）
  5. 連續虧損保護（連虧5次降低槓桿）
  6. 每日自動學習報告
  7. 凱利公式動態倉位（依模擬勝率調整每筆倉位）
  8. 移動止損追蹤（TP1到達後移動止損至進場價）
  9. 資金曲線記錄（可視化資金成長）
  10. 手續費模擬（Taker 0.05%，更貼近真實）
"""

import os, json, time, threading, math
from datetime import datetime, timezone, timedelta
from typing import Optional

TW_TZ           = timezone(timedelta(hours=8))
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PAPER_FILE      = os.path.join(BASE_DIR, "paper_trades.json")
INITIAL_CAPITAL = 10000.0
MIN_LEVERAGE    = 5
MAX_LEVERAGE    = 50        # 優化：最高50倍（原100倍太激進）
TAKER_FEE       = 0.0005   # 0.05% 手續費
MAX_OPEN_TRADES = 10       # 最多同時持有10筆
MAX_POSITION_PCT = 0.08    # 每筆最多用8%資金

_lock = threading.Lock()

# ──────────────────────────────────────────────
# 資料管理
# ──────────────────────────────────────────────

def _default_data() -> dict:
    return {
        "capital":          INITIAL_CAPITAL,
        "peak_capital":     INITIAL_CAPITAL,
        "open_trades":      [],
        "closed_trades":    [],
        "total_pnl":        0.0,
        "total_fee":        0.0,
        "wins":             0,
        "losses":           0,
        "consecutive_losses": 0,
        "equity_curve":     [INITIAL_CAPITAL],
        "tf_stats":         {},   # 分週期統計
        "symbol_stats":     {},   # 分幣種統計
        "created_at":       _tw_now(),
        "last_updated":     _tw_now(),
    }

def _load() -> dict:
    """在 Lock 內呼叫"""
    try:
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 補齊舊版缺少的欄位
                for k, v in _default_data().items():
                    if k not in data:
                        data[k] = v
                return data
    except Exception:
        pass
    return _default_data()

def _save(data: dict):
    """在 Lock 內呼叫"""
    data["last_updated"] = _tw_now()
    try:
        with open(PAPER_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[模擬] 儲存失敗：{e}")

def _tw_now() -> str:
    return datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M:%S")

def _hours_ago(time_str: str) -> float:
    """計算距離現在幾小時"""
    try:
        t = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S").replace(tzinfo=TW_TZ)
        return (datetime.now(TW_TZ) - t).total_seconds() / 3600
    except Exception:
        return 0.0

# ──────────────────────────────────────────────
# 槓桿計算（保守版）
# ──────────────────────────────────────────────

def calc_leverage(winrate: float, adx: float,
                  structure: str, rr: float,
                  consecutive_losses: int) -> int:
    """
    保守槓桿計算（5~50倍）

    設計原則：
    - 模擬階段以數據品質優先，不追求極端槓桿
    - 連續虧損時自動降槓桿
    - 橫盤市場限制槓桿
    """
    base = 5

    # 勝率加成
    if winrate >= 80:   base += 25
    elif winrate >= 70: base += 15
    elif winrate >= 60: base += 8
    elif winrate >= 50: base += 3

    # ADX 加成
    if adx >= 40:   base += 10
    elif adx >= 25: base += 5
    elif adx < 15:  base -= 3

    # 市場結構
    if structure in ("trending_up", "trending_down"): base += 5
    elif structure == "ranging": base -= 5

    # 風報比加成
    if rr >= 3.0:   base += 5
    elif rr >= 2.0: base += 3

    # 連續虧損懲罰（每連虧1次降5倍）
    base -= consecutive_losses * 5

    return max(MIN_LEVERAGE, min(MAX_LEVERAGE, base))

# ──────────────────────────────────────────────
# 凱利公式動態倉位
# ──────────────────────────────────────────────

def calc_kelly_position(capital: float, wins: int, losses: int,
                         rr: float) -> float:
    """
    凱利公式計算最佳倉位
    f = (wr × rr - lr) / rr
    限制在 2%~8% 之間
    """
    total = wins + losses
    if total < 10:
        return capital * 0.05   # 樣本不足時用固定5%

    wr = wins / total
    lr = 1.0 - wr
    kelly = (wr * rr - lr) / max(rr, 0.001)
    half_kelly = kelly * 0.5   # 半凱利（更保守）
    pct = max(0.02, min(MAX_POSITION_PCT, half_kelly))
    return round(capital * pct, 2)

# ──────────────────────────────────────────────
# 更新分類統計
# ──────────────────────────────────────────────

def _update_stats(data: dict, trade: dict):
    """更新週期和幣種統計"""
    tf  = trade["timeframe"]
    sym = trade["symbol"]
    win = trade["pnl_usdt"] > 0

    # 週期統計
    if tf not in data["tf_stats"]:
        data["tf_stats"][tf] = {"wins": 0, "losses": 0, "pnl": 0.0}
    data["tf_stats"][tf]["wins" if win else "losses"] += 1
    data["tf_stats"][tf]["pnl"] = round(data["tf_stats"][tf]["pnl"] + trade["pnl_usdt"], 2)

    # 幣種統計
    if sym not in data["symbol_stats"]:
        data["symbol_stats"][sym] = {"wins": 0, "losses": 0, "pnl": 0.0}
    data["symbol_stats"][sym]["wins" if win else "losses"] += 1
    data["symbol_stats"][sym]["pnl"] = round(data["symbol_stats"][sym]["pnl"] + trade["pnl_usdt"], 2)

# ──────────────────────────────────────────────
# 開倉
# ──────────────────────────────────────────────

def open_paper_trade(signal: dict) -> Optional[dict]:
    """
    根據訊號開立模擬倉位（不限勝率）
    自動計算槓桿和倉位大小
    """
    with _lock:
        data     = _load()
        capital  = data["capital"]

        if capital <= 100:
            print("[模擬] 資金不足，停止開倉")
            return None

        # 限制同時持倉數量
        if len(data["open_trades"]) >= MAX_OPEN_TRADES:
            print(f"[模擬] 已達最大持倉數 {MAX_OPEN_TRADES}")
            return None

        # 避免同一幣種+方向重複開倉
        sym = signal["symbol"]
        dir_ = signal["direction"]
        for t in data["open_trades"]:
            if t["symbol"] == sym and t["direction"] == dir_:
                print(f"[模擬] {sym} {dir_} 已有持倉，跳過")
                return None

        entry     = float(signal["entry"])
        tp1       = float(signal["tp1"])
        tp2       = float(signal["tp2"])
        tp3       = float(signal["tp3"])
        sl        = float(signal["stop_loss"])
        winrate   = float(signal.get("winrate_pct", 50.0))
        adx       = float(signal.get("adx", 20.0))
        structure = str(signal.get("market_structure", "unknown"))
        rr        = float(signal.get("risk_reward", 1.5))
        timeframe = str(signal.get("timeframe", "4h"))
        ml_feats  = signal.get("ml_features", [])

        # 連續虧損保護
        cons_loss = data["consecutive_losses"]

        # 槓桿計算
        leverage = calc_leverage(winrate, adx, structure, rr, cons_loss)

        # 凱利公式倉位
        position_usdt = calc_kelly_position(capital, data["wins"], data["losses"], rr)
        margin        = round(position_usdt / leverage, 2)
        position_size = round(position_usdt / entry, 6)

        # 手續費（開倉）
        open_fee = round(position_usdt * TAKER_FEE, 4)

        # 最大虧損（保證金的80%，防止爆倉）
        max_loss = round(margin * 0.8, 2)

        trade = {
            "id":               f"{sym}_{timeframe}_{int(time.time())}",
            "symbol":           sym,
            "timeframe":        timeframe,
            "direction":        dir_,
            "entry_price":      round(entry, 8),
            "position_usdt":    round(position_usdt, 2),
            "position_size":    position_size,
            "leverage":         leverage,
            "margin":           margin,
            "open_fee":         open_fee,
            "tp1":              round(tp1, 8),
            "tp2":              round(tp2, 8),
            "tp3":              round(tp3, 8),
            "stop_loss":        round(sl, 8),
            "trailing_sl":      round(sl, 8),   # 移動止損
            "tp1_hit":          False,
            "tp2_hit":          False,
            "winrate_pct":      winrate,
            "adx":              adx,
            "rr":               rr,
            "structure":        structure,
            "ml_features":      ml_feats,
            "status":           "open",
            "open_time":        _tw_now(),
            "close_time":       None,
            "close_price":      None,
            "close_reason":     None,
            "pnl_usdt":         0.0,
            "pnl_pct":          0.0,
            "max_loss_usdt":    max_loss,
        }

        data["open_trades"].append(trade)
        data["capital"] = round(capital - open_fee, 2)  # 扣除手續費
        _save(data)

        print(f"[模擬開倉] {sym} {timeframe} {dir_} @ {entry} "
              f"x{leverage} 保證金${margin} 勝率{winrate}%")
        return trade

# ──────────────────────────────────────────────
# 平倉
# ──────────────────────────────────────────────

def close_paper_trade(trade_id: str, close_price: float,
                       reason: str, auto_ml: bool = True) -> Optional[dict]:
    """
    平倉模擬倉位
    auto_ml=True：平倉後自動更新 ML 權重（自主學習）
    """
    with _lock:
        data  = _load()
        trade = next((t for t in data["open_trades"] if t["id"] == trade_id), None)

        if not trade:
            return None

        entry    = trade["entry_price"]
        leverage = trade["leverage"]
        margin   = trade["margin"]
        dir_     = trade["direction"]

        # 計算損益
        if dir_ == "long":
            raw_pct = (close_price - entry) / entry * leverage
        else:
            raw_pct = (entry - close_price) / entry * leverage

        pnl_pct  = round(raw_pct * 100, 2)
        pnl_usdt = round(margin * raw_pct, 2)

        # 手續費（平倉）
        close_fee = round(trade["position_usdt"] * TAKER_FEE, 4)
        pnl_usdt  = round(pnl_usdt - close_fee, 2)

        # 防止超過最大虧損
        if pnl_usdt < -trade["max_loss_usdt"]:
            pnl_usdt = -trade["max_loss_usdt"]
            pnl_pct  = round(-trade["max_loss_usdt"] / margin * 100, 2)

        win = bool(pnl_usdt > 0)

        # 更新交易記錄
        trade.update({
            "status":       "closed",
            "close_time":   _tw_now(),
            "close_price":  round(close_price, 8),
            "close_reason": reason,
            "close_fee":    close_fee,
            "pnl_usdt":     pnl_usdt,
            "pnl_pct":      pnl_pct,
        })

        # 從開倉移到已平倉
        data["open_trades"]   = [t for t in data["open_trades"] if t["id"] != trade_id]
        data["closed_trades"].append(trade)

        # 更新資金
        data["capital"]   = round(data["capital"] + margin + pnl_usdt, 2)
        data["total_pnl"] = round(data["total_pnl"] + pnl_usdt, 2)
        data["total_fee"] = round(data["total_fee"] + close_fee + trade.get("open_fee", 0), 4)
        data["peak_capital"] = max(data["peak_capital"], data["capital"])

        # 勝負統計
        if win:
            data["wins"] += 1
            data["consecutive_losses"] = 0
        else:
            data["losses"] += 1
            data["consecutive_losses"] += 1

        # 資金曲線
        data["equity_curve"].append(round(data["capital"], 2))
        if len(data["equity_curve"]) > 1000:
            data["equity_curve"] = data["equity_curve"][-500:]

        # 分類統計
        _update_stats(data, trade)

        _save(data)

    # 自動 ML 學習（在 Lock 外執行）
    if auto_ml and trade.get("ml_features"):
        try:
            from analysis_engine import ml_update, record_trade_result
            ml_update(trade["ml_features"], win)
            record_trade_result(win, pnl_pct / 100)
            print(f"[模擬學習] ML已更新 {trade['symbol']} {'獲利' if win else '虧損'} {pnl_pct}%")
        except Exception as e:
            print(f"[模擬學習] ML更新失敗：{e}")

    print(f"[模擬平倉] {trade['symbol']} {reason} @ {close_price} "
          f"PNL: {'+' if pnl_usdt>=0 else ''}{pnl_usdt} USDT ({pnl_pct}%)")
    return trade

# ──────────────────────────────────────────────
# 移動止損更新
# ──────────────────────────────────────────────

def _update_trailing_sl(trade: dict, current_price: float) -> float:
    """
    移動止損邏輯：
    - 未到TP1：維持原止損
    - 到達TP1：止損移至進場價（保本）
    - 超過TP1：止損跟隨價格（ATR×0.5）
    """
    entry = trade["entry_price"]
    dir_  = trade["direction"]
    old_sl = trade["trailing_sl"]

    if not trade["tp1_hit"]:
        return old_sl

    if dir_ == "long":
        # 保本止損（至少在進場價+0.1%）
        breakeven = round(entry * 1.001, 8)
        # 追蹤止損（當前價-1%）
        trailing  = round(current_price * 0.99, 8)
        new_sl = max(breakeven, trailing, old_sl)
        return new_sl
    else:
        breakeven = round(entry * 0.999, 8)
        trailing  = round(current_price * 1.01, 8)
        new_sl = min(breakeven, trailing, old_sl)
        return new_sl

# ──────────────────────────────────────────────
# 自動監控（核心學習循環）
# ──────────────────────────────────────────────

def monitor_paper_trades():
    """
    背景監控執行緒
    每30秒檢查一次（更即時）
    自動觸發 TP/SL/移動止損
    平倉後自動更新 ML
    """
    from analysis_engine import fetch_klines

    print("[模擬監控] 啟動，每30秒檢查持倉")

    while True:
        time.sleep(30)
        try:
            # 取得持倉快照（Lock 外操作避免阻塞）
            with _lock:
                data = _load()
                open_trades = list(data["open_trades"])

            if not open_trades:
                continue

            for trade in open_trades:
                try:
                    # 抓取最新價格（用1m K線）
                    klines = fetch_klines(trade["symbol"], "1m", 3)
                    if klines is None or len(klines) < 1:
                        continue

                    price = float(klines[-1, 4])
                    tid   = trade["id"]
                    dir_  = trade["direction"]

                    # 更新移動止損
                    new_sl = _update_trailing_sl(trade, price)
                    if new_sl != trade["trailing_sl"]:
                        with _lock:
                            data2 = _load()
                            for t in data2["open_trades"]:
                                if t["id"] == tid:
                                    t["trailing_sl"] = new_sl
                                    break
                            _save(data2)
                        trade["trailing_sl"] = new_sl

                    # 止損觸發（用移動止損）
                    sl_hit = ((dir_ == "long"  and price <= trade["trailing_sl"]) or
                              (dir_ == "short" and price >= trade["trailing_sl"]))
                    if sl_hit:
                        close_paper_trade(tid, price, "SL止損")
                        continue

                    # TP3 全部平倉
                    if (trade.get("tp2_hit") and
                            ((dir_ == "long"  and price >= trade["tp3"]) or
                             (dir_ == "short" and price <= trade["tp3"]))):
                        close_paper_trade(tid, price, "TP3全倉")
                        continue

                    # TP2
                    if (trade.get("tp1_hit") and not trade.get("tp2_hit") and
                            ((dir_ == "long"  and price >= trade["tp2"]) or
                             (dir_ == "short" and price <= trade["tp2"]))):
                        with _lock:
                            data2 = _load()
                            for t in data2["open_trades"]:
                                if t["id"] == tid:
                                    t["tp2_hit"] = True
                                    break
                            _save(data2)
                        trade["tp2_hit"] = True
                        print(f"[模擬] {trade['symbol']} 到達TP2 @ {price}")

                    # TP1
                    elif not trade.get("tp1_hit") and (
                            (dir_ == "long"  and price >= trade["tp1"]) or
                            (dir_ == "short" and price <= trade["tp1"])):
                        with _lock:
                            data2 = _load()
                            for t in data2["open_trades"]:
                                if t["id"] == tid:
                                    t["tp1_hit"] = True
                                    break
                            _save(data2)
                        trade["tp1_hit"] = True
                        print(f"[模擬] {trade['symbol']} 到達TP1 @ {price}")

                    # 超時平倉（依週期設定不同超時）
                    tf_timeout = {
                        "5m": 4, "15m": 8, "1h": 24, "4h": 72, "1d": 168
                    }
                    timeout_h = tf_timeout.get(trade["timeframe"], 48)
                    if _hours_ago(trade["open_time"]) > timeout_h:
                        close_paper_trade(tid, price, f"超時({timeout_h}h)")

                except Exception as e:
                    print(f"[模擬監控] {trade.get('symbol','?')} 錯誤：{e}")

        except Exception as e:
            print(f"[模擬監控] 主循環錯誤：{e}")

# ──────────────────────────────────────────────
# 統計分析
# ──────────────────────────────────────────────

def get_paper_stats() -> dict:
    """取得完整統計數據"""
    with _lock:
        data = _load()

    total   = data["wins"] + data["losses"]
    winrate = round(data["wins"] / max(total, 1) * 100, 1)
    capital = data["capital"]
    pnl     = data["total_pnl"]
    pnl_pct = round(pnl / INITIAL_CAPITAL * 100, 2)

    # 最大回撤
    curve   = data["equity_curve"]
    peak    = max(curve) if curve else INITIAL_CAPITAL
    trough  = min(curve[curve.index(peak):]) if len(curve) > 1 else capital
    drawdown = round((peak - trough) / max(peak, 1) * 100, 2)

    # 平均槓桿
    all_t   = data["closed_trades"] + data["open_trades"]
    avg_lev = round(sum(t.get("leverage", 10) for t in all_t) / max(len(all_t), 1), 1)

    # 最佳/最差週期
    tf_stats = data.get("tf_stats", {})
    best_tf  = max(tf_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))
    worst_tf = min(tf_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))

    # 最佳/最差幣種
    sym_stats = data.get("symbol_stats", {})
    best_sym  = max(sym_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))
    worst_sym = min(sym_stats.items(), key=lambda x: x[1]["pnl"], default=(None, None))

    # 連續虧損
    cons_loss = data.get("consecutive_losses", 0)

    # 近10筆
    recent = data["closed_trades"][-10:] if data["closed_trades"] else []

    # 預期值（每筆平均損益）
    avg_win  = 0.0; avg_loss = 0.0
    wins_pnl = [t["pnl_usdt"] for t in data["closed_trades"] if t["pnl_usdt"] > 0]
    loss_pnl = [t["pnl_usdt"] for t in data["closed_trades"] if t["pnl_usdt"] < 0]
    if wins_pnl: avg_win  = round(sum(wins_pnl) / len(wins_pnl), 2)
    if loss_pnl: avg_loss = round(sum(loss_pnl) / len(loss_pnl), 2)
    expected = round(winrate/100 * avg_win + (1-winrate/100) * avg_loss, 2)

    return {
        "capital":          capital,
        "initial":          INITIAL_CAPITAL,
        "total_pnl":        pnl,
        "pnl_pct":          pnl_pct,
        "total_fee":        round(data.get("total_fee", 0), 2),
        "total_trades":     total,
        "wins":             data["wins"],
        "losses":           data["losses"],
        "winrate":          winrate,
        "open_trades":      len(data["open_trades"]),
        "avg_leverage":     avg_lev,
        "max_drawdown":     drawdown,
        "consecutive_losses": cons_loss,
        "avg_win":          avg_win,
        "avg_loss":         avg_loss,
        "expected_value":   expected,
        "best_tf":          best_tf,
        "worst_tf":         worst_tf,
        "best_sym":         best_sym,
        "worst_sym":        worst_sym,
        "tf_stats":         tf_stats,
        "sym_stats":        sym_stats,
        "recent_trades":    recent,
        "equity_curve":     curve[-20:],
    }

# ──────────────────────────────────────────────
# 格式化輸出
# ──────────────────────────────────────────────

def format_paper_stats() -> str:
    s   = get_paper_stats()
    pe  = "📈" if s["total_pnl"] >= 0 else "📉"
    evs = "+" if s["expected_value"] >= 0 else ""

    # 最佳週期
    btf  = f"{s['best_tf'][0]}(+${s['best_tf'][1]['pnl']})"  if s["best_tf"][0]  else "—"
    wtf  = f"{s['worst_tf'][0]}(${s['worst_tf'][1]['pnl']})" if s["worst_tf"][0] else "—"
    bsym = f"{s['best_sym'][0]}(+${s['best_sym'][1]['pnl']})"  if s["best_sym"][0]  else "—"
    wsym = f"{s['worst_sym'][0]}(${s['worst_sym'][1]['pnl']})" if s["worst_sym"][0] else "—"

    msg = (
        f"🤖 *模擬交易統計 v2.0*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 *資金狀況*\n"
        f"  初始：`${s['initial']:,.0f}` → 目前：`${s['capital']:,.2f}`\n"
        f"  {pe} 總損益：`{'+' if s['total_pnl']>=0 else ''}{s['total_pnl']:,.2f}` USDT "
        f"(`{s['pnl_pct']}%`)\n"
        f"  手續費：`${s['total_fee']:,.2f}`\n"
        f"  最大回撤：`{s['max_drawdown']}%`\n"
        f"\n"
        f"📊 *交易統計*\n"
        f"  總交易：`{s['total_trades']}` 筆 ｜ 持倉中：`{s['open_trades']}` 筆\n"
        f"  獲利：`{s['wins']}` ｜ 虧損：`{s['losses']}` ｜ "
        f"連虧：`{s['consecutive_losses']}` 次\n"
        f"  *模擬勝率：`{s['winrate']}%`*\n"
        f"  平均槓桿：`{s['avg_leverage']}x`\n"
        f"\n"
        f"📐 *損益分析*\n"
        f"  平均獲利：`+${s['avg_win']}`\n"
        f"  平均虧損：`${s['avg_loss']}`\n"
        f"  預期值/筆：`{evs}{s['expected_value']} USDT`\n"
        f"\n"
        f"🏆 *最佳/最差分析*\n"
        f"  最佳週期：{btf}\n"
        f"  最差週期：{wtf}\n"
        f"  最佳幣種：{bsym}\n"
        f"  最差幣種：{wsym}\n"
    )

    if s["recent_trades"]:
        msg += f"\n📋 *最近10筆*\n"
        for t in reversed(s["recent_trades"]):
            icon = "✅" if t["pnl_usdt"] >= 0 else "❌"
            d    = "多" if t["direction"] == "long" else "空"
            msg += (f"  {icon} {t['symbol']} {t['timeframe']} {d}"
                    f"x{t['leverage']} `{'+' if t['pnl_usdt']>=0 else ''}"
                    f"{t['pnl_usdt']}`({t['close_reason']})\n")

    msg += f"━━━━━━━━━━━━━━━━━━━━"
    return msg

def format_open_trades() -> str:
    with _lock:
        data = _load()
    trades = data["open_trades"]

    if not trades:
        return "📋 目前無模擬持倉"

    msg = f"📋 *模擬持倉（{len(trades)} 筆）*\n━━━━━━━━━━━━━━━━━━━━\n"
    for t in trades:
        d   = "🟢多" if t["direction"] == "long" else "🔴空"
        tp  = "TP2✅" if t.get("tp2_hit") else ("TP1✅" if t.get("tp1_hit") else "持倉中")
        hrs = round(_hours_ago(t["open_time"]), 1)
        msg += (
            f"{d} *{t['symbol']}* {t['timeframe']} x{t['leverage']}\n"
            f"  進場：`{t['entry_price']}` 保證金：`${t['margin']}`\n"
            f"  移動止損：`{t['trailing_sl']}` TP1：`{t['tp1']}`\n"
            f"  狀態：{tp} 持倉{hrs}h\n\n"
        )
    return msg

def format_tf_report() -> str:
    """週期績效報告"""
    s  = get_paper_stats()
    tf = s["tf_stats"]

    if not tf:
        return "📊 尚無週期統計數據"

    msg = "📊 *週期績效報告*\n━━━━━━━━━━━━━━━━━━━━\n"
    for name, stat in sorted(tf.items(), key=lambda x: x[1]["pnl"], reverse=True):
        total  = stat["wins"] + stat["losses"]
        wr     = round(stat["wins"] / max(total, 1) * 100, 1)
        pe     = "📈" if stat["pnl"] >= 0 else "📉"
        msg   += (f"  {pe} *{name}*：勝率`{wr}%` "
                  f"({stat['wins']}勝{stat['losses']}敗) "
                  f"損益`{'+' if stat['pnl']>=0 else ''}{stat['pnl']}`\n")
    msg += "━━━━━━━━━━━━━━━━━━━━"
    return msg

def reset_paper_trading() -> bool:
    with _lock:
        _save(_default_data())
    return True

def get_daily_summary() -> str:
    """每日學習總結"""
    s = get_paper_stats()
    curve = s["equity_curve"]
    growth = round((s["capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)

    return (
        f"🧠 *模擬交易每日學習報告*\n\n"
        f"📈 資金成長：`{'+' if growth>=0 else ''}{growth}%`\n"
        f"🎯 累計勝率：`{s['winrate']}%` ({s['total_trades']}筆)\n"
        f"💡 預期值：`{s['expected_value']} USDT/筆`\n"
        f"📉 最大回撤：`{s['max_drawdown']}%`\n\n"
        f"🏆 最佳週期：{s['best_tf'][0] if s['best_tf'][0] else '—'}\n"
        f"🏆 最佳幣種：{s['best_sym'][0] if s['best_sym'][0] else '—'}\n\n"
        f"🤖 ML模型已從 {s['total_trades']} 筆模擬交易中學習"
    )
