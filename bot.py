"""
bot.py v11.0
============
完整優化版本
"""

import asyncio, threading, time, os, http.server, socketserver
from datetime import datetime, timezone, timedelta
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from paper_trading import (
    open_paper_trade, close_paper_trade,
    get_paper_stats, format_paper_stats,
    format_open_trades, reset_paper_trading,
    monitor_paper_trades, format_tf_report,
    get_daily_summary,
)
from analysis_engine import (
    full_analysis, full_analysis_tf, full_tf_scan,
    format_signal, TOP30_COINS,
    fetch_klines, fetch_fear_greed, get_fear_greed_cached,
    calc_atr, calc_trailing_stop,
    backtest_symbol, quick_backtest,
    get_market_volatility, update_adaptive_params,
    get_risk_status, reset_risk_pause, record_trade_result,
    ml_update, parallel_scan,
    ALL_TIMEFRAMES, TF_TYPE,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN","8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw")
TW_TZ     = timezone(timedelta(hours=8))

CHAT_IDS:             set  = set()
auto_push_active:     bool = False
quiet_mode:           bool = False
paper_trading_active: bool = False
SCAN_INTERVAL              = 300

_signals_lock  = threading.Lock()
active_signals: list = []
MAX_SIGNALS    = 50
_last_signals: dict = {}

def add_signal(sig:dict):
    with _signals_lock:
        if len(active_signals)>=MAX_SIGNALS: active_signals.pop(0)
        active_signals.append(sig)

def remove_signals(to_remove:list):
    with _signals_lock:
        for s in to_remove:
            if s in active_signals: active_signals.remove(s)

def get_signals_copy() -> list:
    with _signals_lock: return list(active_signals)

def tw_now() -> str:
    return datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M")

def scan_all(top_n:int=5) -> list:
    return parallel_scan(top_n=top_n, max_workers=8)

# ══════════════════════════════════════════════
# 指令
# ══════════════════════════════════════════════

async def cmd_start(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text(
        "👋 *加密貨幣分析 Bot v11.0*\n\n"
        "📌 *基本指令*：\n"
        "  /a BTC       → 分析所有週期（5m~1d）\n"
        "  /a BTC 4h    → 分析指定週期\n"
        "  /scan        → 掃描前30大幣種\n"
        "  /autoon      → 開啟自動推播\n"
        "  /autooff     → 關閉自動推播\n"
        "  /quiet       → 靜音模式\n"
        "  /loud        → 關閉靜音\n"
        "  /market      → 市場情緒指數\n"
        "  /status      → 監控中的訊號\n\n"
        "🔬 *進階指令*：\n"
        "  /backtest BTC → 回測單一幣種\n"
        "  /winrate      → 系統實際勝率\n"
        "  /riskstatus   → 風控狀態\n"
        "  /riskresume   → 解除風控暫停\n"
        "  /win BTC      → 記錄獲利\n"
        "  /lose BTC     → 記錄虧損\n\n"
        "🤖 *模擬交易指令*：\n"
        "  /paperon     → 開啟自動模擬交易\n"
        "  /paperoff    → 關閉模擬交易\n"
        "  /paperstats  → 模擬交易統計\n"
        "  /paperpos    → 目前模擬持倉\n"
        "  /papertf     → 週期績效報告\n"
        "  /paperlearn  → 每日學習總結\n"
        "  /paperreset  → 重置模擬數據\n\n"
        "🆕 *v10.0 優化*：\n"
        "  • Supertrend 趨勢指標\n"
        "  • Stochastic RSI\n"
        "  • 訊號品質等級 A/B/C\n"
        "  • 消除重複程式碼（更穩定）\n"
        "  • 加權評分系統",
        parse_mode="Markdown"
    )

async def cmd_analyse(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/a BTC"); return
    symbol=context.args[0].upper()

    if len(context.args)>=2 and context.args[1] in ALL_TIMEFRAMES:
        target_tf=context.args[1]
        await update.message.reply_text(f"⏳ 正在分析 {symbol} {target_tf}...")
        loop=asyncio.get_running_loop()
        r=await loop.run_in_executor(None,full_analysis_tf,symbol,target_tf)
        signals=[r] if r else []
    else:
        await update.message.reply_text(f"⏳ 正在分析 {symbol} 所有週期（5m~1d）...")
        loop=asyncio.get_running_loop()
        signals=await loop.run_in_executor(None,full_tf_scan,symbol,5)

    if signals:
        await update.message.reply_text(f"📊 *{symbol}* 找到 {len(signals)} 個訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r),parse_mode="Markdown")
            _last_signals[symbol]={"features":r.get("ml_features",[]),"direction":r["direction"]}
            add_signal({
                "symbol":symbol,"direction":r["direction"],"entry":r["entry"],"atr":r["atr"],
                "sl":r["stop_loss"],"tp1":r["tp1"],"tp2":r["tp2"],"tp3":r["tp3"],
                "tp1_hit":False,"tp2_hit":False,"last_trailing_sl":r["trailing_sl"],
                "chat_id":update.effective_chat.id,
            })
            if paper_trading_active:
                try: open_paper_trade(r)
                except Exception as e: print(f"[模擬開倉] {e}")
            await asyncio.sleep(0.5)
        await update.message.reply_text(
            f"🔔 已開啟 *{symbol}* 止盈止損監控！\n\n"
            f"💡 交易結束後：\n"
            f"  /win {symbol} → 記錄獲利\n"
            f"  /lose {symbol} → 記錄虧損",
            parse_mode="Markdown"
        )
    else:
        rsk=get_risk_status()
        if rsk["paused"]:
            await update.message.reply_text(
                f"⚠️ 風控暫停中！連續虧損{rsk['consecutive_losses']}次\n輸入 /riskresume 解除",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"⚠️ *{symbol}* 所有週期目前無強力訊號",
                parse_mode="Markdown"
            )

async def cmd_scan(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔍 多週期並行掃描中（約2分鐘）...")
    loop=asyncio.get_running_loop()
    signals=await loop.run_in_executor(None,scan_all)
    if signals:
        await update.message.reply_text(f"📊 找到 {len(signals)} 個高品質訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r),parse_mode="Markdown")
            _last_signals[r["symbol"]]={"features":r.get("ml_features",[]),"direction":r["direction"]}
            if paper_trading_active:
                try: open_paper_trade(r)
                except Exception as e: print(f"[模擬開倉] {e}")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無高品質訊號")

async def cmd_market(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    loop=asyncio.get_running_loop()
    fg=await loop.run_in_executor(None,fetch_fear_greed)
    await update.message.reply_text(
        f"🌡️ *市場情緒指數*\n\n"
        f"  指數：`{fg['value']}` / 100\n"
        f"  狀態：{fg['label']}\n\n"
        f"📊 解讀：\n"
        f"  0-24  😱 極度恐懼 → 做多機會\n"
        f"  25-49 😨 恐懼     → 偏多\n"
        f"  50-74 😊 貪婪     → 偏空\n"
        f"  75-100 🤑 極度貪婪 → 做空機會",
        parse_mode="Markdown"
    )

async def cmd_status(update:Update, context:ContextTypes.DEFAULT_TYPE):
    sigs=get_signals_copy()
    if not sigs:
        await update.message.reply_text("📋 目前沒有監控中的訊號"); return
    msg=f"📋 *監控中（{len(sigs)} 筆）*\n\n"
    for s in sigs:
        d="🟢多" if s["direction"]=="long" else "🔴空"
        tp="✅已達" if s["tp1_hit"] else "⏳等待"
        msg+=f"  {d} *{s['symbol']}* 進場:`{s['entry']}` 移動止損:`{s['last_trailing_sl']}` TP1:{tp}\n\n"
    await update.message.reply_text(msg,parse_mode="Markdown")

async def cmd_auto_on(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active=True
    await update.message.reply_text(
        "✅ *自動推播已開啟！*\n"
        "• 每5分鐘多週期掃描\n"
        "• 止盈/止損自動警告\n"
        "• 移動止損追蹤\n"
        "• 每日早8點市場總結",
        parse_mode="Markdown"
    )

async def cmd_auto_off(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global auto_push_active; auto_push_active=False
    await update.message.reply_text("🔕 自動推播已關閉")

async def cmd_quiet(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global quiet_mode; quiet_mode=True
    await update.message.reply_text("🔇 靜音模式開啟")

async def cmd_loud(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global quiet_mode; quiet_mode=False
    await update.message.reply_text("🔊 靜音模式關閉")

async def cmd_backtest(update:Update, context:ContextTypes.DEFAULT_TYPE):
    symbol=context.args[0].upper() if context.args else "BTC"
    await update.message.reply_text(f"⏳ 回測 {symbol} 中...")
    loop=asyncio.get_running_loop()
    r=await loop.run_in_executor(None,backtest_symbol,symbol,80)
    await update.message.reply_text(
        f"📊 *{symbol} 回測結果*\n\n"
        f"  總交易：{r['trades']} 筆\n"
        f"  勝利：{r.get('wins',0)} ｜ 虧損：{r.get('losses',0)}\n"
        f"  *歷史勝率：{r['winrate']}%*\n\n"
        f"{'⚠️ 樣本數不足' if r['trades']<5 else '✅ 樣本數足夠'}",
        parse_mode="Markdown"
    )

async def cmd_winrate(update:Update, context:ContextTypes.DEFAULT_TYPE):
    rsk=get_risk_status()
    await update.message.reply_text(
        f"📈 *系統勝率統計*\n\n"
        f"  總訊號：{rsk['total_signals']} 筆\n"
        f"  獲利：{rsk['winning_signals']} 筆\n"
        f"  *實際勝率：{rsk['actual_winrate']}%*\n"
        f"  累計損益：{rsk['total_pnl_pct']}%\n\n"
        f"🔒 風控：{'⚠️ 暫停中' if rsk['paused'] else '✅ 正常'}\n"
        f"  連續虧損：{rsk['consecutive_losses']} 次",
        parse_mode="Markdown"
    )

async def cmd_risk_status(update:Update, context:ContextTypes.DEFAULT_TYPE):
    rsk=get_risk_status()
    await update.message.reply_text(
        f"🔒 *風控系統狀態*\n\n"
        f"  狀態：{'⚠️ 暫停中' if rsk['paused'] else '✅ 正常運作'}\n"
        f"  連續虧損：{rsk['consecutive_losses']} / 3 次\n"
        f"  總交易：{rsk['total_signals']} 筆\n"
        f"  實際勝率：{rsk['actual_winrate']}%",
        parse_mode="Markdown"
    )

async def cmd_risk_resume(update:Update, context:ContextTypes.DEFAULT_TYPE):
    reset_risk_pause()
    await update.message.reply_text("✅ 風控暫停已解除，請謹慎交易",parse_mode="Markdown")

async def cmd_win(update:Update, context:ContextTypes.DEFAULT_TYPE):
    symbol=context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/win BTC"); return
    record_trade_result(win=True,pnl_pct=2.0)
    if symbol in _last_signals:
        ml_update(_last_signals[symbol]["features"],win=True)
        del _last_signals[symbol]
    await update.message.reply_text(
        f"✅ 已記錄 *{symbol}* 獲利！ML已更新\n當前勝率：{get_risk_status()['actual_winrate']}%",
        parse_mode="Markdown"
    )

async def cmd_lose(update:Update, context:ContextTypes.DEFAULT_TYPE):
    symbol=context.args[0].upper() if context.args else None
    if not symbol:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/lose BTC"); return
    record_trade_result(win=False,pnl_pct=-1.0)
    if symbol in _last_signals:
        ml_update(_last_signals[symbol]["features"],win=False)
        del _last_signals[symbol]
    rsk=get_risk_status()
    msg=f"❌ 已記錄 *{symbol}* 虧損，ML已更新\n連續虧損：{rsk['consecutive_losses']}/3次"
    if rsk["paused"]: msg+="\n\n⚠️ *風控暫停！* 輸入 /riskresume 可手動恢復"
    await update.message.reply_text(msg,parse_mode="Markdown")

# 模擬交易指令
async def cmd_paper_on(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global paper_trading_active; paper_trading_active=True
    await update.message.reply_text(
        "✅ *自動模擬交易已開啟！*\n\n"
        "系統會自動：\n"
        "• 掃描到訊號立即模擬開倉\n"
        "• 勝率10%~100%全部開倉\n"
        "• 系統自動判斷槓桿(5~50x)\n"
        "• 到達TP/SL自動平倉並更新ML\n"
        "• 凱利公式動態倉位管理\n\n"
        "使用 /paperstats 查看統計",
        parse_mode="Markdown"
    )

async def cmd_paper_off(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global paper_trading_active; paper_trading_active=False
    await update.message.reply_text("🔕 自動模擬交易已關閉")

async def cmd_paper_stats(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_paper_stats(),parse_mode="Markdown")

async def cmd_paper_positions(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_open_trades(),parse_mode="Markdown")

async def cmd_paper_tf(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(format_tf_report(),parse_mode="Markdown")

async def cmd_paper_summary(update:Update, context:ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(get_daily_summary(),parse_mode="Markdown")

async def cmd_paper_reset(update:Update, context:ContextTypes.DEFAULT_TYPE):
    reset_paper_trading()
    await update.message.reply_text(
        "🔄 *模擬交易已重置！*\n"
        "初始資金：$10,000 USDT\n"
        "所有持倉和歷史記錄已清空",
        parse_mode="Markdown"
    )

async def cmd_reset_ml(update:Update, context:ContextTypes.DEFAULT_TYPE):
    from analysis_engine import _ml, save_ml_weights
    _ml["w"]=[0.15,0.12,0.18,0.08,0.08,0.10,0.12,0.10,0.05,0.07,0.08,0.05]
    _ml["b"]=0.0; _ml["lr"]=0.01; _ml["samples"]=0; _ml["wins"]=0
    save_ml_weights(); reset_risk_pause()
    await update.message.reply_text(
        "🔄 *ML數據已重置！*\n\n"
        "• ML歸零重新學習\n"
        "• 風控已解除\n\n"
        "建議：開啟 /paperon 累積數據",
        parse_mode="Markdown"
    )
async def cmd_ml_status(update:Update, context:ContextTypes.DEFAULT_TYPE):
    from analysis_engine import _ml
    samples=_ml.get("samples",0); wins=_ml.get("wins",0)
    losses=samples-wins; wr=round(wins/max(samples,1)*100,1)
    if samples<30: stage="🔵 初期（不足30筆）"
    elif samples<100: stage="🟡 中期（累積中）"
    else: stage="🟢 成熟（充足）"
    warn=" ⚠️ 建議 /resetml 重置" if losses>wins*2 and samples>10 else ""
    await update.message.reply_text(
        f"🤖 *ML狀態*\n\n"
        f"  {stage}\n"
        f"  樣本：`{samples}`筆 勝：`{wins}` 敗：`{losses}`\n"
        f"  歷史勝率：`{wr}%`{warn}",
        parse_mode="Markdown"
    )

# ══════════════════════════════════════════════
# 背景：自動掃描
# ══════════════════════════════════════════════
def auto_push_loop(bot_token:str):
    async def push():
        bot=Bot(token=bot_token)
        if not quiet_mode:
            for cid in list(CHAT_IDS):
                try: await bot.send_message(cid,f"🔍 *自動掃描開始* ｜ {tw_now()}",parse_mode="Markdown")
                except: pass

        signals=scan_all()

        if signals:
            # 模擬交易自動開倉
            if paper_trading_active:
                for r in signals:
                    try: open_paper_trade(r)
                    except Exception as e: print(f"[模擬開倉] {e}")

            for cid in list(CHAT_IDS):
                try:
                    await bot.send_message(cid,f"🔔 *發現 {len(signals)} 個高品質訊號！*",parse_mode="Markdown")
                    for r in signals:
                        await bot.send_message(cid,format_signal(r),parse_mode="Markdown")
                        _last_signals[r["symbol"]]={"features":r.get("ml_features",[]),"direction":r["direction"]}
                        add_signal({
                            "symbol":r["symbol"],"direction":r["direction"],"entry":r["entry"],"atr":r["atr"],
                            "sl":r["stop_loss"],"tp1":r["tp1"],"tp2":r["tp2"],"tp3":r["tp3"],
                            "tp1_hit":False,"tp2_hit":False,"last_trailing_sl":r["trailing_sl"],"chat_id":cid
                        })
                        await asyncio.sleep(0.3)
                except Exception as e: print(f"  ❌ 推播失敗 {cid}：{e}")
        else:
            print("[掃描] 無高品質訊號")
            if not quiet_mode:
                for cid in list(CHAT_IDS):
                    try: await bot.send_message(cid,"📭 掃描完成，無高品質訊號",parse_mode="Markdown")
                    except: pass

    while True:
        time.sleep(SCAN_INTERVAL)
        if auto_push_active and CHAT_IDS:
            try: asyncio.run(push())
            except Exception as e: print(f"[推播錯誤] {e}")

# ══════════════════════════════════════════════
# 背景：止盈止損監控
# ══════════════════════════════════════════════
def price_monitor_loop(bot_token:str):
    async def check():
        bot=Bot(token=bot_token); to_remove=[]
        for sig in get_signals_copy():
            try:
                data=fetch_klines(sig["symbol"],"5m",20)
                if data is None or len(data)<2: continue
                price=float(data[-1,4]); atr=calc_atr(data[:,2],data[:,3],data[:,4])
                if atr<=0: atr=float(sig.get("atr",1.0))
                cid=sig["chat_id"]; d=sig["direction"]; sym=sig["symbol"]
                nt=calc_trailing_stop(sig["entry"],price,atr,d,sig["tp1_hit"])
                nsl=nt["trailing_sl"]; osl=sig["last_trailing_sl"]
                moved=((d=="long" and nsl>osl+atr*0.3) or (d=="short" and nsl<osl-atr*0.3))
                if moved:
                    try:
                        await bot.send_message(cid,f"📊 *{sym}USDT 移動止損更新*\n{nt['sl_type']}：`{osl}` → `{nsl}`",parse_mode="Markdown")
                        sig["last_trailing_sl"]=nsl
                    except: pass
                sl_hit=((d=="long" and price<=sig["last_trailing_sl"]) or (d=="short" and price>=sig["last_trailing_sl"]))
                if sl_hit:
                    try: await bot.send_message(cid,f"🚨 *{sym}USDT 止損觸發！*\n現價：`{price}` 止損：`{sig['last_trailing_sl']}`\n請記錄：/lose {sym}",parse_mode="Markdown")
                    except: pass
                    to_remove.append(sig); continue
                if not sig["tp1_hit"] and ((d=="long" and price>=sig["tp1"]) or (d=="short" and price<=sig["tp1"])):
                    try: await bot.send_message(cid,f"🎯 *{sym}USDT 到達止盈1！*\n現價：`{price}` TP1：`{sig['tp1']}`\n建議出場50%",parse_mode="Markdown")
                    except: pass
                    sig["tp1_hit"]=True
                elif sig["tp1_hit"] and not sig["tp2_hit"] and ((d=="long" and price>=sig["tp2"]) or (d=="short" and price<=sig["tp2"])):
                    try: await bot.send_message(cid,f"🎯🎯 *{sym}USDT 到達止盈2！*\n現價：`{price}` TP2：`{sig['tp2']}`\n建議再出場30%",parse_mode="Markdown")
                    except: pass
                    sig["tp2_hit"]=True
                elif sig["tp2_hit"] and ((d=="long" and price>=sig["tp3"]) or (d=="short" and price<=sig["tp3"])):
                    try: await bot.send_message(cid,f"🎯🎯🎯 *{sym}USDT 到達最終止盈！*\n現價：`{price}` TP3：`{sig['tp3']}`\n🏆 全部出場！請記錄：/win {sym}",parse_mode="Markdown")
                    except: pass
                    to_remove.append(sig)
            except Exception as e: print(f"  [監控錯誤] {sig.get('symbol','?')}：{e}")
        if to_remove: remove_signals(to_remove)

    while True:
        time.sleep(60)
        if get_signals_copy():
            try: asyncio.run(check())
            except Exception as e: print(f"[監控錯誤] {e}")

# ══════════════════════════════════════════════
# 背景：自適應參數（每6小時）
# ══════════════════════════════════════════════
def adaptive_update_loop():
    while True:
        time.sleep(21600)
        try:
            bt=quick_backtest(["BTC","ETH","SOL","XRP","BNB"],60)
            vol=get_market_volatility("BTC")
            update_adaptive_params(bt["overall_winrate"],vol)
            print(f"[自適應] 更新完成，回測勝率：{bt['overall_winrate']}%")
        except Exception as e: print(f"[自適應錯誤] {e}")

# ══════════════════════════════════════════════
# 背景：每日早上8點（台灣時間）
# ══════════════════════════════════════════════
def daily_summary_loop(bot_token:str):
    async def summary():
        bot=Bot(token=bot_token); fg=fetch_fear_greed(); rsk=get_risk_status()
        bt=quick_backtest(["BTC","ETH","SOL"],50)

        # 模擬學習報告
        try:
            paper_msg=get_daily_summary()
            for cid in list(CHAT_IDS):
                try: await bot.send_message(cid,paper_msg,parse_mode="Markdown")
                except: pass
        except Exception as e: print(f"[每日模擬] {e}")

        msg=(
            f"☀️ *每日市場總結 - {datetime.now(TW_TZ).strftime('%Y/%m/%d')}*\n\n"
            f"🌡️ 恐懼貪婪：`{fg['value']}` {fg['label']}\n\n"
            f"📊 *系統統計*\n"
            f"  實際勝率：`{rsk['actual_winrate']}%` ({rsk['winning_signals']}/{rsk['total_signals']}筆)\n"
            f"  回測勝率：`{bt['overall_winrate']}%` ({bt['total_trades']}筆)\n"
            f"  風控：{'⚠️ 暫停' if rsk['paused'] else '✅ 正常'}\n\n"
            f"📋 即將進行全市場掃描..."
        )
        for cid in list(CHAT_IDS):
            try: await bot.send_message(cid,msg,parse_mode="Markdown")
            except: pass

    while True:
        try:
            now=datetime.now(TW_TZ)
            target=now.replace(hour=8,minute=0,second=0,microsecond=0)
            if now>=target: target+=timedelta(days=1)
            secs=max(1.0,(target-now).total_seconds())
            print(f"[每日總結] 距離下次：{int(secs/3600)}小時")
            time.sleep(secs)
            if auto_push_active and CHAT_IDS: asyncio.run(summary())
        except Exception as e:
            print(f"[每日總結錯誤] {e}"); time.sleep(3600)

# ══════════════════════════════════════════════
# Render 免費方案 Keep-Alive
# ══════════════════════════════════════════════
def keep_alive_server():
    PORT=int(os.environ.get("PORT",10000))
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers()
            rsk=get_risk_status()
            ps=get_paper_stats()
            msg=(f"Bot v11.0 | 勝率:{rsk['actual_winrate']}% | "
                 f"模擬勝率:{ps['winrate']}% | {tw_now()}")
            self.wfile.write(msg.encode())
        def log_message(self,format,*args): pass
    with socketserver.TCPServer(("",PORT),Handler) as httpd:
        print(f"[Keep-Alive] 監聽 Port {PORT}")
        httpd.serve_forever()

# ══════════════════════════════════════════════
# 啟動
# ══════════════════════════════════════════════

def force_single_instance():
    """
    強制單一實例運行
    策略：持續嘗試取得 getUpdates 直到成功（代表舊實例已停止）
    最多等待 60 秒，確保新實例一定能啟動
    """
    import requests as req

    print("[啟動] 強制清除舊實例...")

    # 步驟1：刪除 webhook
    for _ in range(3):
        try:
            req.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook",
                json={"drop_pending_updates": True},
                timeout=10
            )
            break
        except Exception:
            time.sleep(2)

    # 步驟2：持續嘗試直到舊實例停止（最多等60秒）
    for attempt in range(20):
        try:
            resp = req.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                json={"offset": -1, "timeout": 1},
                timeout=8
            )
            data = resp.json()
            # 成功取得 updates 代表沒有衝突
            if data.get("ok"):
                print(f"[啟動] 舊實例已清除（嘗試{attempt+1}次）")
                return True
            # 409 Conflict = 舊實例還在，繼續等
            if resp.status_code == 409 or "Conflict" in str(data):
                print(f"[啟動] 舊實例還在，等待3秒... ({attempt+1}/20)")
                time.sleep(3)
                continue
        except Exception as e:
            print(f"[啟動] 嘗試{attempt+1}失敗：{e}")
            time.sleep(3)

    print("[啟動] 警告：等待逾時，強制啟動")
    return False


if __name__ == "__main__":
    print(f"Bot v11.0 啟動中... {tw_now()}")

    # 步驟1：初始化參數
    try:
        vol=get_market_volatility("BTC")
        rsk=get_risk_status()
        update_adaptive_params(rsk["actual_winrate"] if rsk["total_signals"]>10 else 65.0,vol)
        print(f"[啟動] 自適應參數初始化完成，波動率：{vol:.3f}")
    except Exception as e: print(f"[啟動] 初始化失敗：{e}")

    # 步驟2：啟動背景執行緒
    threads=[
        threading.Thread(target=auto_push_loop,      args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=price_monitor_loop,   args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=daily_summary_loop,   args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=adaptive_update_loop, daemon=True),
        threading.Thread(target=monitor_paper_trades, daemon=True),
        threading.Thread(target=keep_alive_server,    daemon=True),
    ]
    for t in threads: t.start()

    # 步驟3：建立 App
    app = (ApplicationBuilder()
           .token(BOT_TOKEN)
           .build())

    for cmd,fn in [
        ("start",cmd_start),("a",cmd_analyse),("scan",cmd_scan),
        ("autoon",cmd_auto_on),("autooff",cmd_auto_off),
        ("market",cmd_market),("status",cmd_status),
        ("quiet",cmd_quiet),("loud",cmd_loud),
        ("backtest",cmd_backtest),("winrate",cmd_winrate),
        ("riskstatus",cmd_risk_status),("riskresume",cmd_risk_resume),
        ("win",cmd_win),("lose",cmd_lose),
        ("paperon",cmd_paper_on),("paperoff",cmd_paper_off),
        ("paperstats",cmd_paper_stats),("paperpos",cmd_paper_positions),
        ("papertf",cmd_paper_tf),("paperlearn",cmd_paper_summary),
        ("paperreset",cmd_paper_reset),
        ("resetml",   cmd_reset_ml),
        ("mlstatus",  cmd_ml_status),
    ]:
        app.add_handler(CommandHandler(cmd,fn))

    print("✅ Bot v11.0 已啟動！")
    # allowed_updates=[] 讓 Telegram 自動清除舊連線
    # drop_pending_updates=True 忽略舊訊息
    # read_timeout=30 避免網路不穩斷線
    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message","callback_query"],
        read_timeout=30,
        write_timeout=30,
        connect_timeout=30,
        pool_timeout=30,
        close_loop=False,
    )
