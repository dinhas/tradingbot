import json
import logging
import os
import asyncio
import threading
from telegram import Bot
from telegram.ext import ApplicationBuilder, CommandHandler
from telegram.constants import ParseMode

class TelegramNotifier:
    """
    Handles Telegram notifications and command support.
    Replaces the previous Discord system.
    """
    def __init__(self, config):
        self.config = config
        self.token = config["TELEGRAM_BOT_TOKEN"]
        self.chat_id = config.get("TELEGRAM_CHAT_ID")
        self.logger = logging.getLogger("LiveExecution")
        self.orchestrator = None

        # Initialize Bot for outgoing messages
        self.bot = Bot(token=self.token)
        self.loop = asyncio.new_event_loop()

        # Start the outgoing message loop in a background thread
        self.msg_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.msg_thread.start()

        # Start the command bot in another background thread
        self.cmd_thread = threading.Thread(target=self._run_command_bot, daemon=True)
        self.cmd_thread.start()

    def set_orchestrator(self, orchestrator):
        """Links the orchestrator for command data retrieval."""
        self.orchestrator = orchestrator

    def _start_loop(self):
        """Runs the asyncio loop for the bot instance."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _run_command_bot(self):
        """Runs the Telegram polling loop for commands."""
        # We need a fresh loop for the Application
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        app = ApplicationBuilder().token(self.token).build()

        app.add_handler(CommandHandler("start", self._start_command))
        app.add_handler(CommandHandler("status", self._status_command))
        app.add_handler(CommandHandler("positions", self._positions_command))

        self.logger.info("Telegram command bot listener started.")
        app.run_polling(close_loop=False)

    async def _start_command(self, update, context):
        """Registers the chat ID for notifications."""
        self.chat_id = update.effective_chat.id
        await update.message.reply_text(
            f"✅ **Bot Linked!**\nChat ID `{self.chat_id}` registered for notifications.",
            parse_mode=ParseMode.MARKDOWN
        )
        self.logger.info(f"Telegram Chat ID registered: {self.chat_id}")

    async def _status_command(self, update, context):
        """Replies with account status summary."""
        if not self.orchestrator:
            await update.message.reply_text("❌ Orchestrator not linked yet.")
            return

        state = self.orchestrator.portfolio_state
        balance = state.get('balance', 0)
        equity = state.get('equity', balance)
        peak = state.get('peak_equity', equity)
        drawdown = 1.0 - (equity / peak) if peak > 0 else 0

        msg = (
            "📊 **System Status**\n"
            f"💰 **Balance:** ${balance:,.2f}\n"
            f"📈 **Equity:** ${equity:,.2f}\n"
            f"📉 **Drawdown:** {drawdown:.2%}\n"
            f"🔒 **Positions:** {len(self.orchestrator.active_positions)}/5"
        )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    async def _positions_command(self, update, context):
        """Replies with a list of open positions."""
        if not self.orchestrator:
            await update.message.reply_text("❌ Orchestrator not linked yet.")
            return

        positions = self.orchestrator.active_positions
        if not positions:
            await update.message.reply_text("ℹ️ No active positions.")
            return

        msg = "📂 **Active Positions:**\n"
        for symbol_id, pos_id in positions.items():
            asset = self.orchestrator._get_symbol_name(symbol_id)
            msg += f"• `{asset}` | ID: `{pos_id}`\n"

        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

    def send_message(self, content):
        """Sends a message to the registered chat ID."""
        if not self.chat_id:
            self.logger.warning("Telegram chat_id not set. Notification skipped.")
            return

        asyncio.run_coroutine_threadsafe(
            self.bot.send_message(chat_id=self.chat_id, text=content, parse_mode=ParseMode.MARKDOWN),
            self.loop
        )

    def send_trade_event(self, details):
        """Formats and sends a trade execution alert."""
        symbol = details.get('symbol', 'Unknown')
        action = details.get('action', 'Unknown')
        size = details.get('size', 0)
        
        msg = (
            "🚀 **TRADE EXECUTED**\n"
            f"**Symbol:** `{symbol}`\n"
            f"**Action:** {action}\n"
            f"**Size:** {size}"
        )
        self.send_message(msg)

    def send_error(self, error_msg):
        """Sends an error alert."""
        msg = f"⚠️ **SYSTEM ERROR**\n{error_msg}"
        self.send_message(msg)
