"""
bot.py
======
Telegram 加密貨幣分析 Bot v2.0

修正：
  - CHAT_ID 改為 Set 集合，支援多人同時使用
  - 自動推播廣播給所有已註冊用戶，不再互相覆蓋
"""

import asyncio
import threading
import time
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from analysis_engine import full_analysis, format_signal, TOP30_COINS

BOT_TOKEN = "8556894585:AAFSzzBsMC-1f1VinHfAdbjY-QGu0zsB_Tw"

# ── 修正：改用 Set 儲存所有使用者 ID，支援多人 ──
CHAT_IDS: set = set()

auto_push_active = False


# ==============================
# 掃描前30大幣種，篩選最強訊號
# ==============================
def scan_all(top_n=5) -> list:
    results = []
    print(f"開始掃描 {len(TOP30_COINS)} 個幣種...")
    for symbol in TOP30_COINS:
        try:
            r = full_analysis(symbol)
            if r:
                score = max(r["long_score"], r["short_score"])
                results.append((score, r))
                print(f"  ✅ {symbol} 分析完成，最高分：{score}")
            else:
                print(f"  ⏭ {symbol} 無強力訊號，跳過")
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
        "👋 *加密貨幣全方位分析 Bot v2.0*\n\n"
        "📌 指令說明：\n"
        "  /a BTC     → 分析單一幣種\n"
        "  /scan      → 掃描前30大幣種\n"
        "  /autoon    → 開啟每30分鐘自動推播\n"
        "  /autooff   → 關閉自動推播\n\n"
        "🔬 分析包含：\n"
        "  • 傳統技術分析（RSI/MACD/BB/EMA）\n"
        "  • SMC（OB/FVG/BOS/CHoCH）\n"
        "  • 和諧型態（Gartley/Bat/Butterfly/Crab）\n"
        "  • 斐波納契出場（止損/止盈1/2/3）\n"
        "  • 多空評分過濾（差距≥3分才推播）",
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
        msg = format_signal(r)
        await update.message.reply_text(msg, parse_mode="Markdown")
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
            msg = format_signal(r)
            await update.message.reply_text(msg, parse_mode="Markdown")
            await asyncio.sleep(0.5)
    else:
        await update.message.reply_text("📭 本次掃描無強力訊號（所有幣種評分差距不足）")


# ==============================
# 指令：/autoon / /autooff
# ==============================
async def cmd_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    CHAT_IDS.add(update.effective_chat.id)
    auto_push_active = True
    await update.message.reply_text("✅ 已開啟自動推播！每30分鐘掃描一次")


async def cmd_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global auto_push_active
    auto_push_active = False
    await update.message.reply_text("🔕 已關閉自動推播")


# ==============================
# 背景定時推播（廣播給所有用戶）
# ==============================
def auto_push_loop(bot_token: str):
    async def push():
        bot     = Bot(token=bot_token)
        signals = scan_all()

        if signals and CHAT_IDS:
            for chat_id in CHAT_IDS:
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text="⏰ *自動掃描報告*",
                        parse_mode="Markdown"
                    )
                    for r in signals:
                        msg = format_signal(r)
                        await bot.send_message(
                            chat_id=chat_id,
                            text=msg,
                            parse_mode="Markdown"
                        )
                        await asyncio.sleep(0.3)
                except Exception as e:
                    print(f"推播失敗 {chat_id}：{e}")

    while True:
        time.sleep(1800)  # 30 分鐘
        if auto_push_active and CHAT_IDS:
            asyncio.run(push())


# ==============================
# 啟動
# ==============================
if __name__ == "__main__":
    print("Bot 啟動中...")

    t = threading.Thread(target=auto_push_loop, args=(BOT_TOKEN,), daemon=True)
    t.start()

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("a",       cmd_analyse))
    app.add_handler(CommandHandler("scan",    cmd_scan))
    app.add_handler(CommandHandler("autoon",  cmd_auto_on))
    app.add_handler(CommandHandler("autooff", cmd_auto_off))

    print("✅ Bot 已啟動！")
    app.run_polling()
