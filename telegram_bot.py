import asyncio
import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from orchestrator import Engine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
ALLOWED = set(int(x) for x in os.getenv("ALLOWED_CHAT_IDS", "").split(",") if x.strip())


def _ok(update):
    return not ALLOWED or update.effective_chat.id in ALLOWED


def _esc(s):
    for c in r"_*[]()~`>#+-=|{}.!":
        s = s.replace(c, f"\\{c}")
    return s


def _icon(m):
    return {"NORMAL": "🟢", "CRISIS": "🔴", "HALTED": "⛔"}.get(m, "?")


def _status(e: Engine):
    b     = e.core.book
    m     = e.core.mode.value
    price = _esc(f"{e.snap.mid:.4f}") if e.snap else "n/a"
    return (f"*STATUS*\n"
            f"{_icon(m)} `{m}`\n"
            f"price `{price}`\n"
            f"open `{b.open_count}`  trades `{len(b.closed)}`\n"
            f"pnl `{_esc(f'{b.pnl:+.4f}')}`\n")


def _positions(e: Engine):
    if not e.core.book._open:
        return "no open positions"
    lines = ["*POSITIONS*"]
    for pid, pos in e.core.book._open.items():
        age = int(time.time() - pos.opened_at)
        lines.append(f"`{pid}` {pos.side.value}\n"
                     f"  entry `{_esc(f'{pos.entry:.4f}')}` qty `{pos.qty:.2f}`\n"
                     f"  sl `{_esc(f'{pos.sl:.4f}')}` tp `{_esc(f'{pos.tp:.4f}')}`\n"
                     f"  pnl `{_esc(f'{pos.pnl:+.4f}')}` age `{age}s`")
    return "\n".join(lines)


def _pnl(e: Engine):
    trades = e.core.book.closed
    won    = sum(1 for t in trades if t["pnl"] > 0)
    wr     = won / len(trades) * 100 if trades else 0
    return (f"*PNL*\n"
            f"24h `{_esc(f'{e.core.book.pnl:+.4f}')}`\n"
            f"trades `{len(trades)}`  W `{won}` L `{len(trades)-won}`\n"
            f"win rate `{_esc(f'{wr:.1f}')}%`\n")


def _params(e: Engine):
    g = e.opt.params
    return (f"*PARAMS*\n"
            f"threshold `{g.sent_thresh:.3f}`\n"
            f"half\\-life `{g.half_life:.0f}s`\n"
            f"stop `{_esc(f'{g.stop_pct*100:.2f}')}%`\n"
            f"target `{_esc(f'{g.target_pct*100:.2f}')}%`\n"
            f"sharpe `{_esc(f'{g.fitness:.4f}')}`\n")


class _St:
    engine: Optional[Engine] = None
    task:   Optional[asyncio.Task] = None
    on:     bool = False


st = _St()


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _ok(update):
        return
    kb = [
        [InlineKeyboardButton("status",    callback_data="status"),
         InlineKeyboardButton("positions", callback_data="positions")],
        [InlineKeyboardButton("pnl",       callback_data="pnl"),
         InlineKeyboardButton("params",    callback_data="params")],
        [InlineKeyboardButton("▶ start",   callback_data="eng_start"),
         InlineKeyboardButton("■ stop",    callback_data="eng_stop")],
    ]
    await update.message.reply_text(
        "*sense\\-act*",
        reply_markup=InlineKeyboardMarkup(kb),
        parse_mode=ParseMode.MARKDOWN_V2,
    )


async def cmd_status(u, c):
    if not _ok(u) or not st.engine: return
    await u.message.reply_text(_status(st.engine), parse_mode=ParseMode.MARKDOWN_V2)

async def cmd_positions(u, c):
    if not _ok(u) or not st.engine: return
    await u.message.reply_text(_positions(st.engine), parse_mode=ParseMode.MARKDOWN_V2)

async def cmd_pnl(u, c):
    if not _ok(u) or not st.engine: return
    await u.message.reply_text(_pnl(st.engine), parse_mode=ParseMode.MARKDOWN_V2)

async def cmd_params(u, c):
    if not _ok(u) or not st.engine: return
    await u.message.reply_text(_params(st.engine), parse_mode=ParseMode.MARKDOWN_V2)

async def cmd_stop(u, c):
    if not _ok(u): return
    await _kill()
    await u.message.reply_text("stopped")


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    d = q.data

    if d == "status":      msg = _status(st.engine)    if st.engine else "not running"
    elif d == "positions": msg = _positions(st.engine) if st.engine else "not running"
    elif d == "pnl":       msg = _pnl(st.engine)       if st.engine else "not running"
    elif d == "params":    msg = _params(st.engine)     if st.engine else "not running"
    elif d == "eng_start":
        if st.on:
            msg = "already running"
        else:
            await q.edit_message_text("starting\\.\\.\\.", parse_mode=ParseMode.MARKDOWN_V2)
            await _start(ctx.application, q.message.chat_id)
            return
    elif d == "eng_stop":
        await _kill()
        msg = "stopped"
    else:
        return

    await q.edit_message_text(msg, parse_mode=ParseMode.MARKDOWN_V2)


async def _push(app, chat_id, msg):
    try:
        await app.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        log.error("push error: %s", e)


async def _start(app, chat_id):
    st.engine = Engine()
    st.on     = True

    orig = st.engine.core.submit

    def patched(uid, asset, impact, snap, **kw):
        pos = orig(uid, asset, impact, snap, **kw)
        if pos:
            side = "🟢 LONG" if pos.side.value == "BUY" else "🔴 SHORT"
            asyncio.create_task(_push(
                app, chat_id,
                f"{side} `{pos.pos_id}`\n"
                f"entry `{_esc(f'{pos.entry:.4f}')}`\n"
                f"sl `{_esc(f'{pos.sl:.4f}')}` tp `{_esc(f'{pos.tp:.4f}')}`"
            ))
        return pos

    st.engine.core.submit = patched
    st.task = asyncio.create_task(_loop(app, chat_id))
    await _push(app, chat_id, "▶ engine started")


async def _loop(app, chat_id):
    tick = 0
    try:
        while st.on:
            await asyncio.sleep(60)
            tick += 1
            if st.engine and tick % 2 == 0:
                b = st.engine.core.book
                m = st.engine.core.mode.value
                await _push(app, chat_id,
                            f"{_icon(m)} `{m}`  open `{b.open_count}`  pnl `{_esc(f'{b.pnl:+.4f}')}`")
    except asyncio.CancelledError:
        pass
    finally:
        await _push(app, chat_id, "■ stopped")


async def _kill():
    st.on = False
    if st.task:
        st.task.cancel()
        try:
            await st.task
        except asyncio.CancelledError:
            pass
        st.task = None
    st.engine = None


def main():
    if not TOKEN:
        raise ValueError("set TELEGRAM_TOKEN in .env")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("positions", cmd_positions))
    app.add_handler(CommandHandler("pnl",       cmd_pnl))
    app.add_handler(CommandHandler("params",    cmd_params))
    app.add_handler(CommandHandler("stop",      cmd_stop))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
