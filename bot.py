"""
bot.py v4.0
===========
新增：有強力訊號自動推播（不需要等 30 分鐘）
  - 每 5 分鐘掃描一次全部幣種
  - 發現強力訊號立即推播
  - 相同幣種+方向 1 小時內不重複（由 analysis_engine 控制）
  - 支援多人同時使用
"""

import asyncio
import threading
import time
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import full_analysis, format_signal, TOP30_COINS

BOT_TOKEN = "8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw"

# 所有使用者 ID（支援多人）
CHAT_IDS: set = set()

# 自動推播開關
auto_push_active = False

# 掃描間隔（秒）
SCAN_INTERVAL = 300   # 5 分鐘掃描一次，有訊號立即推播


# ==============================
# 掃描所有幣種
# ==============================
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


# ==============================
# 指令：/start
# ==============================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text(
        "👋 *加密貨幣全方位分析 Bot v4.0*\n\n"
        "📌 *指令說明*：\n"
        "  /a BTC     → 分析單一幣種\n"
        "  /scan      → 立即掃描前30大幣種\n"
        "  /autoon    → 開啟自動推播（有訊號立即通知）\n"
        "  /autooff   → 關閉自動推播\n\n"
        "🔔 *自動推播說明*：\n"
        "  每5分鐘掃描一次，發現強力訊號立即推播\n"
        "  相同幣種同方向 1 小時內不重複通知\n\n"
        "🔬 *分析內容*：\n"
        "  • RSI / MACD / BB / EMA\n"
        "  • SMC（OB/FVG/BOS/CHoCH）\n"
        "  • 和諧型態 / K線型態 / 成交量\n"
        "  • 多週期確認 / 斐波納契出場\n"
        "  • 進場信心度評級",
        parse_mode="Markdown"
    )


# ==============================
# 指令：/a BTC
# ==============================
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
    else:
        await update.message.reply_text(
            f"⚠️ *{symbol}* 目前無強力訊號\n"
            f"（多空差距不足 3 分，或總分低於 5 分）",
            parse_mode="Markdown"
        )


# ==============================
# 指令：/scan
# ==============================
async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    CHAT_IDS.add(update.effective_chat.id)
    await update.message.reply_text("🔍 開始掃描前30大幣種，約需1-2分鐘...")

    loop    = asyncio.get_event_loop()
    signals = await loop.run_in_executor(None, scan_all)

    if signals:
        await update.message.reply_text(f"📊 掃描完成！找到 {len(signals)} 個強力訊號：")
        for r in signals:
            await update.message.reply_text(format_signal(r), parse_mode="Markdown")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無強力訊號")


# ==============================
# 指令：/autoon / /autooff
# ==============================
async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active = True
    await update.message.reply_text(
        "✅ *自動推播已開啟！*\n"
        "每5分鐘掃描一次，發現強力訊號立即通知你 🔔",
        parse_mode="Markdown"
    )


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    auto_push_active = False
    await update.message.reply_text("🔕 自動推播已關閉")


# ==============================
# 背景自動掃描（有訊號立即推播）
# ==============================
def auto_push_loop(bot_token: str):
    """
    每 SCAN_INTERVAL 秒掃描一次
    發現強力訊號 → 立即廣播給所有使用者
    無訊號 → 靜默等待下次掃描
    """
    async def push_signals():
        bot     = Bot(token=bot_token)
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
                        await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"  ❌ 推播失敗 {chat_id}：{e}")
        else:
            print(f"[掃描] 無強力訊號，靜默等待...")

    while True:
        time.sleep(SCAN_INTERVAL)
        if auto_push_active and CHAT_IDS:
            try:
                asyncio.run(push_signals())
            except Exception as e:
                print(f"[推播錯誤] {e}")


# ==============================
# 啟動
# ==============================
if __name__ == "__main__":
    print("Bot 啟動中...")

    # 背景執行自動掃描
    t = threading.Thread(target=auto_push_loop, args=(BOT_TOKEN,), daemon=True)
    t.start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("a",       cmd_analyse))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("autoon",  cmd_auto_on))
    app.add_handler(CommandHandler("autooff", cmd_auto_off))

    print("✅ Bot 已啟動！發送 /autoon 開啟自動推播")
    app.run_polling()
