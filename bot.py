"""
bot.py v6.0
===========
配合 analysis_engine v6.0
新增：移動止損追蹤更新通知
"""

import asyncio
import threading
import time
import os
from datetime import datetime, timezone, timedelta
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import (full_analysis, format_signal, TOP30_COINS,
                              fetch_klines, fetch_fear_greed,
                              calc_atr, calc_trailing_stop)

BOT_TOKEN  = os.environ.get("BOT_TOKEN", "8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw")
TW_TZ      = timezone(timedelta(hours=8))

CHAT_IDS:         set  = set()
auto_push_active: bool = False
quiet_mode:       bool = False
SCAN_INTERVAL          = 300

_signals_lock  = threading.Lock()
active_signals: list = []
MAX_SIGNALS    = 50


def add_signal(sig: dict):
    with _signals_lock:
        if len(active_signals) >= MAX_SIGNALS:
            active_signals.pop(0)
        active_signals.append(sig)


def remove_signals(to_remove: list):
    with _signals_lock:
        for sig in to_remove:
            if sig in active_signals:
                active_signals.remove(sig)


def get_signals_copy() -> list:
    with _signals_lock:
        return list(active_signals)


def tw_now() -> str:
    return datetime.now(TW_TZ).strftime("%Y/%m/%d %H:%M")


# ══════════════════════════════
# 掃描
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
# 指令
# ══════════════════════════════
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text(
        "👋 *加密貨幣全方位分析 Bot v6.0*\n\n"
        "📌 *指令說明*：\n"
        "  /a BTC     → 分析單一幣種\n"
        "  /scan      → 立即掃描前30大幣種\n"
        "  /autoon    → 開啟自動推播\n"
        "  /autooff   → 關閉自動推播\n"
        "  /quiet     → 靜音模式\n"
        "  /loud      → 關閉靜音\n"
        "  /market    → 市場情緒\n"
        "  /status    → 監控中的訊號\n\n"
        "🆕 *v6.0 新功能*：\n"
        "  • 突破偵測（量價配合）\n"
        "  • 市場結構判斷（趨勢/震盪）\n"
        "  • 移動止損追蹤\n"
        "  • 倉位管理建議\n"
        "  • 防重複改為價格突破確認",
        parse_mode="Markdown"
    )


async def cmd_analyse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    if not context.args:
        await update.message.reply_text("⚠️ 請輸入幣種，例如：/a BTC")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ 正在分析 {symbol}，請稍候...")

    loop = asyncio.get_running_loop()
    r    = await loop.run_in_executor(None, full_analysis, symbol)

    if r:
        await update.message.reply_text(format_signal(r), parse_mode="Markdown")
        add_signal({
            "symbol":    symbol,
            "direction": r["direction"],
            "entry":     r["entry"],
            "atr":       r["atr"],
            "sl":        r["stop_loss"],
            "tp1":       r["tp1"],
            "tp2":       r["tp2"],
            "tp3":       r["tp3"],
            "tp1_hit":   False,
            "tp2_hit":   False,
            "last_trailing_sl": r["trailing_sl"],
            "chat_id":   update.effective_chat.id,
        })
        await update.message.reply_text(
            f"🔔 已開啟 *{symbol}* 止盈止損監控！\n"
            f"移動止損：`{r['trailing_sl']}` ({r['sl_type']})",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(f"⚠️ *{symbol}* 目前無強力訊號", parse_mode="Markdown")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔍 掃描前30大幣種中，約需1-2分鐘...")
    loop    = asyncio.get_running_loop()
    signals = await loop.run_in_executor(None, scan_all)
    if signals:
        await update.message.reply_text(f"📊 找到 {len(signals)} 個強力訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r), parse_mode="Markdown")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無強力訊號")


async def cmd_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    loop = asyncio.get_running_loop()
    fg   = await loop.run_in_executor(None, fetch_fear_greed)
    await update.message.reply_text(
        f"🌡️ *市場情緒指數*\n\n"
        f"  指數：`{fg['value']}` / 100\n"
        f"  狀態：{fg['label']}\n\n"
        f"📊 *解讀*：\n"
        f"  0-24   😱 極度恐懼 → 做多機會\n"
        f"  25-49  😨 恐懼     → 偏多\n"
        f"  50-74  😊 貪婪     → 偏空\n"
        f"  75-100 🤑 極度貪婪 → 做空機會",
        parse_mode="Markdown"
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sigs = get_signals_copy()
    if not sigs:
        await update.message.reply_text("📋 目前沒有監控中的訊號")
        return
    msg = f"📋 *監控中的訊號（共 {len(sigs)} 筆）*\n\n"
    for s in sigs:
        d = "🟢多" if s["direction"] == "long" else "🔴空"
        msg += (f"  {d} *{s['symbol']}* ｜ 進場：`{s['entry']}`\n"
                f"     止損：`{s['sl']}` ｜ 移動止損：`{s['last_trailing_sl']}`\n"
                f"     TP1：`{s['tp1']}` ｜ TP2：`{s['tp2']}`\n\n")
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active = True
    await update.message.reply_text(
        "✅ *自動推播已開啟！*\n"
        "• 每5分鐘掃描，發現強力訊號立即通知\n"
        "• 移動止損追蹤\n"
        "• 到達止盈/止損自動警告\n"
        "• 每日早上8點（台灣時間）市場總結\n\n"
        "💡 輸入 /quiet 可開啟靜音模式",
        parse_mode="Markdown"
    )


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    auto_push_active = False
    await update.message.reply_text("🔕 自動推播已關閉")


async def cmd_quiet(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quiet_mode
    quiet_mode = True
    await update.message.reply_text("🔇 靜音模式開啟，只推播有訊號時通知", parse_mode="Markdown")


async def cmd_loud(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global quiet_mode
    quiet_mode = False
    await update.message.reply_text("🔊 靜音模式關閉")


# ══════════════════════════════
# 背景：自動掃描
# ══════════════════════════════
def auto_push_loop(bot_token: str):
    async def push_signals():
        bot = Bot(token=bot_token)
        now = tw_now()

        if not quiet_mode:
            for chat_id in list(CHAT_IDS):
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=f"🔍 *自動掃描開始* ｜ {now}\n正在掃描前30大幣種...",
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    print(f"  [通知錯誤] {chat_id}：{e}")

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
                        add_signal({
                            "symbol":    r["symbol"],
                            "direction": r["direction"],
                            "entry":     r["entry"],
                            "atr":       r["atr"],
                            "sl":        r["stop_loss"],
                            "tp1":       r["tp1"],
                            "tp2":       r["tp2"],
                            "tp3":       r["tp3"],
                            "tp1_hit":   False,
                            "tp2_hit":   False,
                            "last_trailing_sl": r["trailing_sl"],
                            "chat_id":   chat_id,
                        })
                        await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"  ❌ 推播失敗 {chat_id}：{e}")
        else:
            print("[掃描] 無強力訊號")
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
# 背景：止盈止損 + 移動止損監控
# ══════════════════════════════
def price_monitor_loop(bot_token: str):
    async def check_prices():
        bot       = Bot(token=bot_token)
        to_remove = []
        sigs      = get_signals_copy()

        for sig in sigs:
            try:
                data  = fetch_klines(sig["symbol"], "5m", 20)
                if data is None:
                    continue
                price  = float(data[-1, 4])
                highs  = data[:, 2]
                lows   = data[:, 3]
                closes = data[:, 4]
                atr    = calc_atr(highs, lows, closes)
                sym    = sig["symbol"]
                cid    = sig["chat_id"]
                d      = sig["direction"]

                # 移動止損更新（修正3）
                new_trailing = calc_trailing_stop(
                    sig["entry"], price, atr, d, sig["tp1_hit"]
                )
                new_sl = new_trailing["trailing_sl"]
                old_sl = sig["last_trailing_sl"]

                # 移動止損有意義地移動時才通知
                if d == "long" and new_sl > old_sl + atr * 0.3:
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"📊 *{sym}USDT 移動止損更新*\n"
                              f"  {new_trailing['sl_type']}\n"
                              f"  舊止損：`{old_sl}` → 新止損：`{new_sl}`"),
                        parse_mode="Markdown"
                    )
                    sig["last_trailing_sl"] = new_sl
                elif d == "short" and new_sl < old_sl - atr * 0.3:
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"📊 *{sym}USDT 移動止損更新*\n"
                              f"  {new_trailing['sl_type']}\n"
                              f"  舊止損：`{old_sl}` → 新止損：`{new_sl}`"),
                        parse_mode="Markdown"
                    )
                    sig["last_trailing_sl"] = new_sl

                # 止損觸發
                if ((d=="long"  and price <= sig["last_trailing_sl"]) or
                        (d=="short" and price >= sig["last_trailing_sl"])):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🚨 *{sym}USDT 止損觸發！*\n"
                              f"現價：`{price}` / 止損：`{sig['last_trailing_sl']}`\n"
                              f"方向：{'做多' if d=='long' else '做空'}"),
                        parse_mode="Markdown"
                    )
                    to_remove.append(sig)
                    continue

                # TP1
                if (not sig["tp1_hit"] and
                        ((d=="long" and price>=sig["tp1"]) or
                         (d=="short" and price<=sig["tp1"]))):
                    await bot.send_message(
                        chat_id=cid,
                        text=(f"🎯 *{sym}USDT 到達止盈1！*\n"
                              f"現價：`{price}` / TP1：`{sig['tp1']}`\n"
                              f"建議出場 50% 倉位，止損移至進場價"),
                        parse_mode="Markdown"
                    )
                    sig["tp1_hit"] = True

                # TP2
                elif (sig["tp1_hit"] and not sig["tp2_hit"] and
                      ((d=="long" and price>=sig["tp2"]) or
                       (d=="short" and price<=sig["tp2"]))):
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
                      ((d=="long" and price>=sig["tp3"]) or
                       (d=="short" and price<=sig["tp3"]))):
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
            remove_signals(to_remove)

    while True:
        time.sleep(60)
        if get_signals_copy():
            try:
                asyncio.run(check_prices())
            except Exception as e:
                print(f"[監控錯誤] {e}")


# ══════════════════════════════
# 背景：每日早上8點（台灣時間）
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
    print(f"Bot v6.0 啟動中... 台灣時間：{tw_now()}")

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

    print("✅ Bot v6.0 已啟動！")
    app.run_polling()
