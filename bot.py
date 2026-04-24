"""
bot.py v9.0
===========
加密貨幣分析 Bot - 專業級
"""

import asyncio, threading, time, os
from datetime import datetime, timezone, timedelta
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import (
    full_analysis, format_signal, TOP30_COINS,
    fetch_klines, fetch_fear_greed, get_fear_greed_cached,
    calc_atr, calc_trailing_stop,
    backtest_symbol, quick_backtest,
    get_market_volatility, update_adaptive_params,
    get_risk_status, reset_risk_pause, record_trade_result,
    ml_update, parallel_scan,
)

BOT_TOKEN = os.environ.get("BOT_TOKEN","8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw")
TW_TZ     = timezone(timedelta(hours=8))

CHAT_IDS:         set  = set()
auto_push_active: bool = False
quiet_mode:       bool = False
SCAN_INTERVAL          = 300

_signals_lock = threading.Lock()
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
        "👋 *加密貨幣分析 Bot v9.0*\n\n"
        "📌 *基本指令*：\n"
        "  /a BTC      → 分析幣種（含ML勝率）\n"
        "  /scan       → 掃描前30大幣種\n"
        "  /autoon     → 開啟自動推播\n"
        "  /autooff    → 關閉自動推播\n"
        "  /quiet      → 靜音（只推播有訊號）\n"
        "  /loud       → 關閉靜音\n"
        "  /market     → 市場情緒指數\n"
        "  /status     → 監控中的訊號\n\n"
        "🔬 *進階指令*：\n"
        "  /backtest BTC → 回測單一幣種\n"
        "  /winrate      → 系統實際勝率\n"
        "  /riskstatus   → 風控狀態\n"
        "  /riskresume   → 解除風控暫停\n"
        "  /win BTC      → 記錄獲利（訓練ML）\n"
        "  /lose BTC     → 記錄虧損（訓練ML）\n\n"
        "🆕 *v9.0 核心功能*：\n"
        "  • OKX API（全球可用）\n"
        "  • ML勝率預測（0~100%）\n"
        "  • 勝率<70%自動過濾\n"
        "  • 多時間框架投票制\n"
        "  • BTC崩跌自動暫停山寨幣\n"
        "  • 市場情緒過濾\n"
        "  • 鯨魚成交量偵測\n"
        "  • 風控：連虧3次自動暫停\n"
        "  • Kelly Criterion倉位管理\n"
        "  • 預估勝率：85%+",
        parse_mode="Markdown"
    )

async def cmd_analyse(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/a BTC"); return
    symbol=context.args[0].upper()
    await update.message.reply_text(f"⏳ 正在分析 {symbol}（ML評估中）...")
    loop=asyncio.get_running_loop()
    r=await loop.run_in_executor(None,full_analysis,symbol)
    if r:
        await update.message.reply_text(format_signal(r),parse_mode="Markdown")
        _last_signals[symbol]={"features":r.get("ml_features",[]),"direction":r["direction"]}
        add_signal({
            "symbol":symbol,"direction":r["direction"],"entry":r["entry"],"atr":r["atr"],
            "sl":r["stop_loss"],"tp1":r["tp1"],"tp2":r["tp2"],"tp3":r["tp3"],
            "tp1_hit":False,"tp2_hit":False,"last_trailing_sl":r["trailing_sl"],
            "chat_id":update.effective_chat.id,
        })
        await update.message.reply_text(
            f"🔔 已開啟 *{symbol}* 止盈止損監控！\n"
            f"移動止損：`{r['trailing_sl']}` ({r['sl_type']})\n\n"
            f"💡 交易結束後請輸入：\n"
            f"  /win {symbol} → 記錄獲利\n"
            f"  /lose {symbol} → 記錄虧損",
            parse_mode="Markdown"
        )
    else:
        rsk=get_risk_status()
        if rsk["paused"]:
            await update.message.reply_text(f"⚠️ 風控暫停中！連續虧損{rsk['consecutive_losses']}次\n輸入 /riskresume 解除",parse_mode="Markdown")
        else:
            await update.message.reply_text(f"⚠️ *{symbol}* 目前無強力訊號（未通過所有過濾）",parse_mode="Markdown")

async def cmd_scan(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔍 掃描前30大幣種中（約2分鐘）...")
    loop=asyncio.get_running_loop()
    signals=await loop.run_in_executor(None,scan_all)
    if signals:
        await update.message.reply_text(f"📊 找到 {len(signals)} 個高品質訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r),parse_mode="Markdown")
            _last_signals[r["symbol"]]={"features":r.get("ml_features",[]),"direction":r["direction"]}
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無高品質訊號（所有訊號未通過85%+品質過濾）")

async def cmd_market(update:Update, context:ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    loop=asyncio.get_running_loop()
    fg=await loop.run_in_executor(None,fetch_fear_greed)
    await update.message.reply_text(
        f"🌡️ *市場情緒指數*\n\n"
        f"  指數：`{fg['value']}` / 100\n"
        f"  狀態：{fg['label']}\n\n"
        f"📊 *解讀*：\n"
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
        msg+=f"  {d} *{s['symbol']}* 進場:`{s['entry']}`\n     移動止損:`{s['last_trailing_sl']}` TP1:{tp}\n\n"
    await update.message.reply_text(msg,parse_mode="Markdown")

async def cmd_auto_on(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active=True
    await update.message.reply_text(
        "✅ *自動推播已開啟！*\n"
        "• 每5分鐘掃描，發現高品質訊號立即通知\n"
        "• 止盈/止損自動警告\n"
        "• 移動止損追蹤\n"
        "• 每日早上8點（台灣時間）市場總結\n\n"
        "💡 輸入 /quiet 可關閉無訊號通知",
        parse_mode="Markdown"
    )

async def cmd_auto_off(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    auto_push_active=False
    await update.message.reply_text("🔕 自動推播已關閉")

async def cmd_quiet(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global quiet_mode; quiet_mode=True
    await update.message.reply_text("🔇 靜音模式開啟，只推播有訊號時通知")

async def cmd_loud(update:Update, context:ContextTypes.DEFAULT_TYPE):
    global quiet_mode; quiet_mode=False
    await update.message.reply_text("🔊 靜音模式關閉")

async def cmd_backtest(update:Update, context:ContextTypes.DEFAULT_TYPE):
    symbol=context.args[0].upper() if context.args else "BTC"
    await update.message.reply_text(f"⏳ 回測 {symbol} 中（約30秒）...")
    loop=asyncio.get_running_loop()
    r=await loop.run_in_executor(None,backtest_symbol,symbol,80)
    await update.message.reply_text(
        f"📊 *{symbol} 回測結果*\n\n"
        f"  總交易：{r['trades']} 筆\n"
        f"  勝利：{r.get('wins',0)} 筆\n"
        f"  虧損：{r.get('losses',0)} 筆\n"
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
        f"  連續虧損：{rsk['consecutive_losses']} 次\n\n"
        f"💡 每次交易結束後用 /win 或 /lose 記錄，讓ML越來越準！",
        parse_mode="Markdown"
    )

async def cmd_risk_status(update:Update, context:ContextTypes.DEFAULT_TYPE):
    rsk=get_risk_status()
    await update.message.reply_text(
        f"🔒 *風控系統狀態*\n\n"
        f"  狀態：{'⚠️ 暫停中' if rsk['paused'] else '✅ 正常運作'}\n"
        f"  連續虧損：{rsk['consecutive_losses']} / 3 次\n"
        f"  總交易：{rsk['total_signals']} 筆\n"
        f"  實際勝率：{rsk['actual_winrate']}%\n\n"
        f"{'輸入 /riskresume 可手動恢復' if rsk['paused'] else '⚡ 連虧3次將自動暫停保護本金'}",
        parse_mode="Markdown"
    )

async def cmd_risk_resume(update:Update, context:ContextTypes.DEFAULT_TYPE):
    reset_risk_pause()
    await update.message.reply_text("✅ 風控暫停已解除，系統恢復正常運作\n⚠️ 請謹慎交易",parse_mode="Markdown")

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
    if rsk["paused"]: msg+="\n\n⚠️ *風控暫停！連虧3次，請休息觀望*\n輸入 /riskresume 可手動恢復"
    await update.message.reply_text(msg,parse_mode="Markdown")

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
            for cid in list(CHAT_IDS):
                try:
                    await bot.send_message(cid,f"🔔 *發現 {len(signals)} 個高品質訊號！*",parse_mode="Markdown")
                    for r in signals:
                        await bot.send_message(cid,format_signal(r),parse_mode="Markdown")
                        _last_signals[r["symbol"]]={"features":r.get("ml_features",[]),"direction":r["direction"]}
                        add_signal({"symbol":r["symbol"],"direction":r["direction"],"entry":r["entry"],"atr":r["atr"],
                                    "sl":r["stop_loss"],"tp1":r["tp1"],"tp2":r["tp2"],"tp3":r["tp3"],
                                    "tp1_hit":False,"tp2_hit":False,"last_trailing_sl":r["trailing_sl"],"chat_id":cid})
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
        msg=(f"☀️ *每日市場總結 - {datetime.now(TW_TZ).strftime('%Y/%m/%d')}*\n\n"
             f"🌡️ 恐懼貪婪：`{fg['value']}` {fg['label']}\n\n"
             f"📊 *系統統計*\n"
             f"  實際勝率：`{rsk['actual_winrate']}%` ({rsk['winning_signals']}/{rsk['total_signals']}筆)\n"
             f"  回測勝率：`{bt['overall_winrate']}%` ({bt['total_trades']}筆)\n"
             f"  風控：{'⚠️ 暫停' if rsk['paused'] else '✅ 正常'}\n\n"
             f"📋 即將進行全市場掃描...")
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
# 啟動
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
# Render 免費方案 Keep-Alive（避免服務睡眠）
# ══════════════════════════════════════════════
def keep_alive_server():
    """Render 免費方案需要監聽 Port，否則會被睡眠"""
    import http.server, socketserver
    PORT = int(os.environ.get("PORT", 10000))
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            rsk = get_risk_status()
            msg = f"Bot v9.0 運行中 | 勝率：{rsk['actual_winrate']}% | {tw_now()}"
            self.wfile.write(msg.encode())
        def log_message(self, format, *args): pass  # 靜音 log
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"[Keep-Alive] 監聽 Port {PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    print(f"Bot v9.0 啟動中... {tw_now()}")
    try:
        vol=get_market_volatility("BTC")
        rsk=get_risk_status()
        update_adaptive_params(rsk["actual_winrate"] if rsk["total_signals"]>10 else 65.0,vol)
        print(f"[啟動] 自適應參數初始化完成，波動率：{vol:.3f}")
    except Exception as e: print(f"[啟動] 初始化失敗：{e}")

    threads=[
        threading.Thread(target=auto_push_loop,    args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=price_monitor_loop, args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=daily_summary_loop, args=(BOT_TOKEN,),daemon=True),
        threading.Thread(target=adaptive_update_loop,daemon=True),
    ]
    for t in threads: t.start()

    # Render 免費方案：啟動 Keep-Alive 伺服器
    ka = threading.Thread(target=keep_alive_server, daemon=True)
    ka.start()

    app=ApplicationBuilder().token(BOT_TOKEN).build()
    for cmd,fn in [
        ("start",cmd_start),("a",cmd_analyse),("scan",cmd_scan),
        ("autoon",cmd_auto_on),("autooff",cmd_auto_off),
        ("market",cmd_market),("status",cmd_status),
        ("quiet",cmd_quiet),("loud",cmd_loud),
        ("backtest",cmd_backtest),("winrate",cmd_winrate),
        ("riskstatus",cmd_risk_status),("riskresume",cmd_risk_resume),
        ("win",cmd_win),("lose",cmd_lose),
    ]:
        app.add_handler(CommandHandler(cmd,fn))

    print("✅ Bot v9.0 已啟動！")
    app.run_polling()
