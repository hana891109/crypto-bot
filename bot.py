"""
bot.py v5.0
===========
新增：
  - 止盈/止損價格監控，到達時自動推播
  - 每日早上 8 點市場總結
  - 恐懼貪婪指數每日推播
"""

import asyncio
import threading
import time
from datetime import datetime
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import (full_analysis, format_signal, TOP30_COINS,
                              fetch_klines, fetch_fear_greed)

BOT_TOKEN = "8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw"

CHAT_IDS:         set  = set()
auto_push_active: bool = False
SCAN_INTERVAL          = 300   # 5 分鐘掃描

# 監控中的訊號（止盈/止損追蹤）
# 格式：[{"symbol", "direction", "entry", "sl", "tp1", "tp2", "tp3", "chat_id"}]
active_signals: list = []


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
        "👋 *加密貨幣全方位分析 Bot v5.0*\n\n"
        "📌 *指令說明*：\n"
        "  /a BTC     → 分析單一幣種\n"
        "  /scan      → 立即掃描前30大幣種\n"
        "  /autoon    → 開啟自動推播\n"
        "  /autooff   → 關閉自動推播\n"
        "  /market    → 查看市場情緒\n\n"
        "🔔 *自動推播功能*：\n"
        "  • 每5分鐘掃描，有強力訊號立即通知\n"
        "  • 到達止盈/止損自動警告\n"
        "  • 每日早上8點市場總結\n\n"
        "🔬 *分析項目（共18項）*：\n"
        "  RSI・MACD・BB・EMA・VWAP・ADX\n"
        "  支撐壓力・背離・K線型態・成交量\n"
        "  多週期確認・資金費率・OI未平倉量\n"
        "  SMC(OB/FVG/BOS/CHoCH)・和諧型態\n"
        "  Fib止損・ATR動態止損・信心度評級",
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

    loop = asyncio.get_event_loop()
    r    = await loop.run_in_executor(None, full_analysis, symbol)

    if r:
        await update.message.reply_text(format_signal(r), parse_mode="Markdown")
        # 加入止盈止損監控
        active_signals.append({
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
            f"🔔 已開啟 *{symbol}* 止盈止損監控，到達目標價位時自動通知！",
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

    loop    = asyncio.get_event_loop()
    signals = await loop.run_in_executor(None, scan_all)

    if signals:
        await update.message.reply_text(f"📊 找到 {len(signals)} 個強力訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r), parse_mode="Markdown")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無強力訊號")


# ══════════════════════════════
# 指令：/market（市場情緒）
# ══════════════════════════════
async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    loop = asyncio.get_event_loop()
    fg   = await loop.run_in_executor(None, fetch_fear_greed)

    msg = (
        f"🌡️ *市場情緒指數*\n\n"
        f"  指數：`{fg['value']}` / 100\n"
        f"  狀態：{fg['label']}\n\n"
        f"📊 *解讀*：\n"
        f"  0-24  😱 極度恐懼 → 做多機會\n"
        f"  25-49 😨 恐懼     → 偏多\n"
        f"  50-74 😊 貪婪     → 偏空\n"
        f"  75-100 🤑 極度貪婪 → 做空機會"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


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
        "• 每日早上8點市場總結",
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
        bot  = Bot(token=bot_token)
        now  = datetime.now().strftime("%H:%M")

        # ── 掃描開始通知 ──
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

        # ── 掃描結果通知 ──
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
                        # 自動加入監控
                        active_signals.append({
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
        bot      = Bot(token=bot_token)
        to_remove = []

        for sig in active_signals:
            try:
                data  = fetch_klines(sig["symbol"], "5m", 2)
                if data is None:
                    continue
                price = float(data[-1, 4])
                sym   = sig["symbol"]
                cid   = sig["chat_id"]
                d     = sig["direction"]

                # 止損觸發
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

        for sig in to_remove:
            if sig in active_signals:
                active_signals.remove(sig)

    while True:
        time.sleep(60)   # 每分鐘檢查一次
        if active_signals:
            try:
                asyncio.run(check_prices())
            except Exception as e:
                print(f"[監控錯誤] {e}")


# ══════════════════════════════
# 背景：每日早上 8 點市場總結
# ══════════════════════════════
def daily_summary_loop(bot_token: str):
    async def send_summary():
        bot = Bot(token=bot_token)
        fg  = fetch_fear_greed()
        msg = (
            f"☀️ *每日市場總結 - {datetime.now().strftime('%Y/%m/%d')}*\n\n"
            f"🌡️ 恐懼貪婪指數：`{fg['value']}` {fg['label']}\n\n"
            f"📋 即將進行全市場掃描，有強力訊號將立即推播..."
        )
        for chat_id in list(CHAT_IDS):
            try:
                await bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
            except Exception as e:
                print(f"  [每日總結錯誤] {chat_id}：{e}")

    while True:
        now = datetime.now()
        # 等到早上 8:00
        target_hour = 8
        seconds_until = ((target_hour - now.hour - 1) * 3600 +
                         (60 - now.minute - 1) * 60 +
                         (60 - now.second)) % 86400
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
    print("Bot v5.0 啟動中...")

    # 背景執行緒
    threads = [
        threading.Thread(target=auto_push_loop,     args=(BOT_TOKEN,), daemon=True),
        threading.Thread(target=price_monitor_loop,  args=(BOT_TOKEN,), daemon=True),
        threading.Thread(target=daily_summary_loop,  args=(BOT_TOKEN,), daemon=True),
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

    print("✅ Bot v5.0 已啟動！")
    app.run_polling()
