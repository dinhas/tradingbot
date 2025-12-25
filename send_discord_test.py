import os
import asyncio
from dotenv import load_dotenv
from LiveExecution.src.notifications import DiscordNotifier
from twisted.internet import reactor

def main():
    # 1. Load environment variables from .env
    load_dotenv()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    if not webhook_url:
        print("❌ Error: DISCORD_WEBHOOK_URL not found in .env file.")
        return

    # 2. Initialize the notifier
    # We pass a config dict just like the main app does
    config = {
        "DISCORD_WEBHOOK_URL": webhook_url,
        "DISCORD_THREAD_NAME": "Manual Alerts" # Optional: for forum channels
    }
    notifier = DiscordNotifier(config)

    print(f"Sending message to Discord...")

    # 3. Define the send logic
    def send_and_stop():
        d = notifier.send_message("Hello from the TradeGuard script! 🚀")
        
        def success(res):
            print("✅ Message sent successfully!")
            reactor.stop()
            
        def failure(err):
            print(f"❌ Failed to send message: {err}")
            reactor.stop()

        d.addCallback(success)
        d.addErrback(failure)

    # 4. Start the Twisted reactor
    reactor.callWhenRunning(send_and_stop)
    reactor.run()

if __name__ == "__main__":
    main()
