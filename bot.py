"""
bot.py v5.1
===========
隱患修正：
  1. asyncio.get_event_loop() 在新版 Python 已棄用 → 改用 asyncio.get_running_loop()
  2. active_signals 多執行緒同時讀寫 → 加入 threading.Lock() 保護
  3. daily_summary 時間計算錯誤（負數情況）→ 改用精確計算
  4. Railway 雲端時區為 UTC → 自動換算為台灣時間（UTC+8）
  5. 掃描通知訊息過多（每5分鐘一則）→ 加入靜音模式選項
  6. active_signals 無上限，長期運行會無限增長 → 限制最多50筆
  7. Bot Token 明碼寫在程式碼 → 改從環境變數讀取
"""

import asyncio
import threading
import time
import os
from datetime import datetime, timezone, timedelta
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import (full_analysis, format_signal, TOP30_COINS,
                              fetch_klines, fetch_fear_greed)

# 修正7：從環境變數讀取 Token（Railway 上設定）
# 若環境變數不存在則用預設值（本機測試用）
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw")

# 修正4：台灣時區 UTC+8
TW_TZ = timezone(timedelta(hours=8))

CHAT_IDS:         set  = set()
auto_push_active: bool = False
quiet_mode:       bool = False   # 修正5：靜音模式（只推播有訊號時）
SCAN_INTERVAL          = 300     # 5 分鐘

# 修正2：加入 Lock 保護 active_signals
_signals_lock  = threading.Lock()
active_signals: list = []
MAX_SIGNALS    = 50   # 修正6：最多監控 50 筆


def add_signal(sig: dict):
    """修正2 & 6：加鎖新增監控訊號，超過上限自動移除最舊的"""
    with _signals_lock:
        if len(active_signals) >= MAX_SIGNALS:
            active_signals.pop(0)
        active_signals.append(sig)


def remove_signals(to_remove: list):
    """修正2：加鎖移除訊號"""
    with _signals_lock:
        for sig in to_remove:
            if sig in active_signals:
                active_signals.remove(sig)


def get_signals_copy() -> list:
    """修正2：加鎖讀取訊號列表快照"""
    with _signals_lock:
        return list(active_signals)


def tw_now() -> str:
    """修正4：取得台灣當前時間字串"""
    return datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M")


# ══════════════════════════════
# 掃描幣種
# ══════════════════════════════
def scan_all(top_n=5) -> list:
    results = []
    print(f"[掃描] 開始掃描 {len(TOP30_COINS)} 個幣種...")
    for symbol in TOP30_COINS:
        try:
            r = full_analysis(symbol)
            if r:
                score = max(r["long_score"], r["short_score"])
                results.append((score, r))
                print(f"  ✅ {symbol} 強力訊號！分數：{score}")
            else:
                print(f"  ⏭ {symbol} 無訊號")
        except Exception as e:
            print(f"  ❌ {symbol} 失敗：{e}")
    results.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in results[:top_n]]


# ══════════════════════════════
# 指令：/start
# ══════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text(
        "👋 *加密貨幣全方位分析 Bot v5.1*\n\n"
        "📌 *指令說明*：\n"
        "  /a BTC     → 分析單一幣種\n"
        "  /scan      → 立即掃描前30大幣種\n"
        "  /autoon    → 開啟自動推播\n"
        "  /autooff   → 關閉自動推播\n"
        "  /quiet     → 靜音模式（只推播有訊號時）\n"
        "  /loud      → 關閉靜音模式\n"
        "  /market    → 查看市場情緒\n"
        "  /status    → 查看目前監控中的訊號\n\n"
        "🔔 *自動推播功能*：\n"
        "  • 每5分鐘掃描，有強力訊號立即通知\n"
        "  • 到達止盈/止損自動警告\n"
        "  • 每日早上8點（台灣時間）市場總結",
        parse_mode="Markdown"
    )


# ══════════════════════════════
# 指令：/a BTC
# ══════════════════════════════
async def cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/a BTC")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ 正在分析 {symbol}，請稍候...")

    # 修正1：改用 asyncio.get_running_loop()
    loop = asyncio.get_running_loop()
    r    = await loop.run_in_executor(None, full_analysis, symbol)

    if r:
        await update.message.reply_text(format_signal(r), parse_mode="Markdown")
        add_signal({
            "symbol":    symbol,
            "direction": r["direction"],
            "entry":     r["entry"],
            "sl":        r["stop_loss"],
            "tp1":       r["tp1"],
            "tp2":       r["tp2"],
            "tp3":       r["tp3"],
            "tp1_hit":   False,
            "tp2_hit":   False,
            "chat_id":   update.effective_chat.id,
        })
        await update.message.reply_text(
            f"🔔 已開啟 *{symbol}* 止盈止損監控！",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"⚠️ *{symbol}* 目前無強力訊號",
            parse_mode="Markdown"
        )


# ══════════════════════════════
# 指令：/scan
# ══════════════════════════════
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔍 掃描前30大幣種中，約需1-2分鐘...")

    loop    = asyncio.get_running_loop()   # 修正1
    signals = await loop.run_in_executor(None, scan_all)

    if signals:
        await update.message.reply_text(f"📊 找到 {len(signals)} 個強力訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r), parse_mode="Markdown")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無強力訊號")


# ══════════════════════════════
# 指令：/market
# ══════════════════════════════
async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    loop = asyncio.get_running_loop()   # 修正1
    fg   = await loop.run_in_executor(None, fetch_fear_greed)
    msg  = (
        f"🌡️ *市場情緒指數*\n\n"
        f"  指數：`{fg['value']}` / 100\n"
        f"  狀態：{fg['label']}\n\n"
        f"📊 *解讀*：\n"
        f"  0-24   😱 極度恐懼 → 做多機會\n"
        f"  25-49  😨 恐懼     → 偏多\n"
        f"  50-74  😊 貪婪     → 偏空\n"
        f"  75-100 🤑 極度貪婪 → 做空機會"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


# ══════════════════════════════
# 指令：/status
# ══════════════════════════════
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sigs = get_signals_copy()
    if not sigs:
        await update.message.reply_text("📋 目前沒有監控中的訊號")
        return
    msg = f"📋 *目前監控中的訊號（共 {len(sigs)} 筆）*\n\n"
    for s in sigs:
        d = "🟢多" if s["direction"] == "long" else "🔴空"
        msg += (f"  {d} *{s['symbol']}* ｜ 進場：`{s['entry']}`\n"
                f"     止損：`{s['sl']}` ｜ TP1：`{s['tp1']}`\n\n")
    await update.message.reply_text(msg, parse_mode="Markdown")


# ══════════════════════════════
# 指令：/quiet / /loud
# ══════════════════════════════
async def cmd_quiet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quiet_mode
    quiet_mode = True
    await update.message.reply_text(
        "🔇 *靜音模式已開啟*\n掃描開始/無訊號時不再通知，只有強力訊號才推播",
        parse_mode="Markdown"
    )


async def cmd_loud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quiet_mode
    quiet_mode = False
    await update.message.reply_text("🔊 靜音模式已關閉，恢復所有通知")


# ══════════════════════════════
# 指令：/autoon / /autooff
# ══════════════════════════════
async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active = True
    await update.message.reply_text(
        "✅ *自動推播已開啟！*\n"
        "• 每5分鐘掃描，發現強力訊號立即通知\n"
        "• 到達止盈/止損自動警告\n"
        "• 每日早上8點（台灣時間）市場總結\n\n"
        "💡 輸入 /quiet 可開啟靜音模式（只推播有訊號時）",
        parse_mode="Markdown"
    )


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    auto_push_active = False
    await update.message.reply_text("🔕 自動推播已關閉")


# ══════════════════════════════
# 背景：自動掃描推播
# ══════════════════════════════
def auto_push_loop(bot_token: str):
    async def push_signals():
        bot = Bot(token=bot_token)
        now = tw_now()   # 修正4：台灣時間

        # 掃描開始通知（修正5：靜音模式下跳過）
        if not quiet_mode:
            for chat_id in list(CHAT_IDS):
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(f"🔍 *自動掃描開始* ｜ {now}\n"
                              f"正在掃描前30大幣種，請稍候..."),
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    print(f"  [掃描通知錯誤] {chat_id}：{e}")

        signals = scan_all()

        if signals and CHAT_IDS:
            for chat_id in list(CHAT_IDS):
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=f"🔔 *發現 {len(signals)} 個強力訊號！*",
                        parse_mode="Markdown"
                    )
                    for r in signals:
                        await bot.send_message(
                            chat_id=chat_id,
                            text=format_signal(r),
                            parse_mode="Markdown"
                        )
                        add_signal({   # 修正2：加鎖
                            "symbol":    r["symbol"],
                            "direction": r["direction"],
                            "entry":     r["entry"],
                            "sl":        r["stop_loss"],
                            "tp1":       r["tp1"],
                            "tp2":       r["tp2"],
                            "tp3":       r["tp3"],
                            "tp1_hit":   False,
                            "tp2_hit":   False,
                            "chat_id":   chat_id,
                        })
                        await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"  ❌ 推播失敗 {chat_id}：{e}")
        else:
            print("[掃描] 無強力訊號")
            # 修正5：靜音模式下不發無訊號通知
            if not quiet_mode:
                for chat_id in list(CHAT_IDS):
                    try:
                        await bot.send_message(
                            chat_id=chat_id,
                            text="📭 掃描完成，本次無強力訊號，靜待下次掃描...",
                            parse_mode="Markdown"
                        )
                    except Exception as e:
                        print(f"  [無訊號通知錯誤] {chat_id}：{e}")

    while True:
        time.sleep(SCAN_INTERVAL)
        if auto_push_active and CHAT_IDS:
            try:
                asyncio.run(push_signals())
            except Exception as e:
                print(f"[推播錯誤] {e}")


# ══════════════════════════════
# 背景：止盈止損監控
# ══════════════════════════════
def price_monitor_loop(bot_token: str):
    async def check_prices():
        bot       = Bot(token=bot_token)
        to_remove = []
        sigs      = get_signals_copy()   # 修正2：讀取快照

        for sig in sigs:
            try:
                data  = fetch_klines(sig["symbol"], "5m", 2)
                if data is None:
                    continue
                price = float(data[-1, 4])
                sym   = sig["symbol"]
                cid   = sig["chat_id"]
                d     = sig["direction"]

                # 止損
                if ((d == "long"  and price <= sig["sl"]) or
                        (d == "short" and price >= sig["sl"])):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🚨 *{sym}USDT 止損觸發！*\n"
                              f"現價：`{price}` / 止損：`{sig['sl']}`\n"
                              f"方向：{'做多' if d=='long' else '做空'}"),
                        parse_mode="Markdown"
                    )
                    to_remove.append(sig)
                    continue

                # TP1
                if (not sig["tp1_hit"] and
                        ((d == "long"  and price >= sig["tp1"]) or
                         (d == "short" and price <= sig["tp1"]))):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🎯 *{sym}USDT 到達止盈1！*\n"
                              f"現價：`{price}` / TP1：`{sig['tp1']}`\n"
                              f"建議出場 50% 倉位"),
                        parse_mode="Markdown"
                    )
                    sig["tp1_hit"] = True

                # TP2
                elif (sig["tp1_hit"] and not sig["tp2_hit"] and
                      ((d == "long"  and price >= sig["tp2"]) or
                       (d == "short" and price <= sig["tp2"]))):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🎯🎯 *{sym}USDT 到達止盈2！*\n"
                              f"現價：`{price}` / TP2：`{sig['tp2']}`\n"
                              f"建議再出場 30% 倉位"),
                        parse_mode="Markdown"
                    )
                    sig["tp2_hit"] = True

                # TP3
                elif (sig["tp2_hit"] and
                      ((d == "long"  and price >= sig["tp3"]) or
                       (d == "short" and price <= sig["tp3"]))):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🎯🎯🎯 *{sym}USDT 到達最終止盈！*\n"
                              f"現價：`{price}` / TP3：`{sig['tp3']}`\n"
                              f"🏆 全部出場！恭喜獲利！"),
                        parse_mode="Markdown"
                    )
                    to_remove.append(sig)

            except Exception as e:
                print(f"  [監控錯誤] {sig['symbol']}：{e}")

        if to_remove:
            remove_signals(to_remove)   # 修正2：加鎖移除

    while True:
        time.sleep(60)
        if get_signals_copy():
            try:
                asyncio.run(check_prices())
            except Exception as e:
                print(f"[監控錯誤] {e}")


# ══════════════════════════════
# 背景：每日早上 8 點（台灣時間）市場總結
# ══════════════════════════════
def daily_summary_loop(bot_token: str):
    async def send_summary():
        bot = Bot(token=bot_token)
        fg  = fetch_fear_greed()
        msg = (
            f"☀️ *每日市場總結 - {datetime.now(TW_TZ).strftime('%Y/%m/%d')}*\n\n"
            f"🌡️ 恐懼貪婪指數：`{fg['value']}` {fg['label']}\n\n"
            f"📋 即將進行全市場掃描，有強力訊號將立即推播..."
        )
        for chat_id in list(CHAT_IDS):
            try:
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
            except Exception as e:
                print(f"  [每日總結錯誤] {chat_id}：{e}")

    while True:
        # 修正3 & 4：精確計算距離台灣時間早上8點的秒數
        now_tw        = datetime.now(TW_TZ)
        target        = now_tw.replace(hour=8, minute=0, second=0, microsecond=0)
        if now_tw >= target:
            target = target + timedelta(days=1)
        seconds_until = (target - now_tw).total_seconds()

        print(f"[每日總結] 距離下次推播：{int(seconds_until/3600)} 小時")
        time.sleep(seconds_until)

        if auto_push_active and CHAT_IDS:
            try:
                asyncio.run(send_summary())
            except Exception as e:
                print(f"[每日總結錯誤] {e}")


# ══════════════════════════════
# 啟動
# ══════════════════════════════
if __name__ == "__main__":
    print("Bot v5.1 啟動中...")
    print(f"目前台灣時間：{tw_now()}")

    threads = [
        threading.Thread(target=auto_push_loop,    args=(BOT_TOKEN,), daemon=True),
        threading.Thread(target=price_monitor_loop, args=(BOT_TOKEN,), daemon=True),
        threading.Thread(target=daily_summary_loop, args=(BOT_TOKEN,), daemon=True),
    ]
    for t in threads:
        t.start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("a",       cmd_analyse))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("autoon",  cmd_auto_on))
    app.add_handler(CommandHandler("autooff", cmd_auto_off))
    app.add_handler(CommandHandler("market",  cmd_market))
    app.add_handler(CommandHandler("status",  cmd_status))
    app.add_handler(CommandHandler("quiet",   cmd_quiet))
    app.add_handler(CommandHandler("loud",    cmd_loud))

    print("✅ Bot v5.1 已啟動！")
    app.run_polling()
