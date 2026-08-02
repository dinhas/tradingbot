import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks, Deferred

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from LiveExecution.src.config import load_config
from LiveExecution.src.logger import setup_logger
from LiveExecution.src.ctrader_client import CTraderClient
from LiveExecution.src.database import DatabaseManager

# Precision per symbol
SYMBOL_DIGITS = {
    'EURUSD': 5,
    'GBPUSD': 5,
    'USDCHF': 5,
    'USDJPY': 3,
    'XAUUSD': 2,
}

SYMBOL_IDS_REVERSE = {
    1: 'EURUSD',
    2: 'GBPUSD',
    41: 'XAUUSD',
    6: 'USDCHF',
    4: 'USDJPY'
}

def get_symbol_name(symbol_id):
    return SYMBOL_IDS_REVERSE.get(symbol_id, f"Symbol_{symbol_id}")

@inlineCallbacks
def run_sltp_verifier():
    logger = logging.getLogger("LiveExecution")
    load_dotenv()
    config = load_config()

    db_path = config.get("DB_PATH", "LiveExecution/data/live_trading.db")
    db = DatabaseManager(db_path)

    client = CTraderClient(config)
    auth_deferred = Deferred()

    def on_auth():
        logger.info("cTrader Client Authenticated successfully.")
        auth_deferred.callback(True)

    client.on_authenticated = on_auth
    client.start()

    logger.info("Connecting to cTrader to inspect open trades...")
    yield auth_deferred

    try:
        open_pos_res = yield client.fetch_open_positions()
        positions = getattr(open_pos_res, 'position', [])

        print("\n" + "=" * 70)
        print(f" SL/TP INTEGRITY CHECK — Found {len(positions)} Open Trade(s)")
        print("=" * 70)

        if not positions:
            print("No open positions found on cTrader account.")
            print("=" * 70 + "\n")
            client.stop()
            reactor.stop()
            return

        repaired_count = 0
        ok_count = 0
        closed_count = 0


        for pos in positions:
            pos_id = pos.positionId
            symbol_id = pos.tradeData.symbolId if hasattr(pos, 'tradeData') else 0
            asset_name = get_symbol_name(symbol_id)
            trade_side = pos.tradeData.tradeSide if hasattr(pos, 'tradeData') else 1 # 1=BUY, 2=SELL
            entry_price = pos.price
            digits = SYMBOL_DIGITS.get(asset_name, 5)
            direction = 1 if trade_side == 1 else -1

            cur_sl = getattr(pos, 'stopLoss', None)
            cur_tp = getattr(pos, 'takeProfit', None)

            has_sl = bool(cur_sl and cur_sl > 0)
            has_tp = bool(cur_tp and cur_tp > 0)

            if has_sl and has_tp:
                print(f"[OK] Position {pos_id} | {asset_name} | Entry: {entry_price:.{digits}f} | SL: {cur_sl:.{digits}f} | TP: {cur_tp:.{digits}f}")
                ok_count += 1
                continue

            print(f"[MISSING SL/TP] Position {pos_id} | {asset_name} | Entry: {entry_price:.{digits}f} | Current SL: {cur_sl} | Current TP: {cur_tp}")

            # Retrieve stored trade from DB
            db_trade = db.get_trade_by_pos_id(pos_id)
            sl_to_set = None
            tp_to_set = None

            if db_trade:
                sl_to_set = db_trade.get('sl')
                tp_to_set = db_trade.get('tp')
                rel_sl = db_trade.get('relative_sl')
                rel_tp = db_trade.get('relative_tp')

                if sl_to_set is None and rel_sl:
                    sl_to_set = entry_price - (direction * rel_sl / 100000.0)
                if tp_to_set is None and rel_tp:
                    tp_to_set = entry_price + (direction * rel_tp / 100000.0)

            # Fallback calculation if DB missing or incomplete: calculate relative to ENTRY PRICE
            if sl_to_set is None or tp_to_set is None:
                # Default 2x / 4x ATR fallback (estimated 15 pips ATR default)
                default_pip = 0.0015 if asset_name == 'USDJPY' else (0.15 if asset_name == 'XAUUSD' else 0.0015)
                sl_dist = 2.0 * default_pip
                tp_dist = 4.0 * default_pip

                step = 10 ** (5 - digits)
                relative_sl = max(int(round(sl_dist * 100000 / step) * step), step)
                relative_tp = max(int(round(tp_dist * 100000 / step) * step), step)

                if sl_to_set is None:
                    sl_to_set = entry_price - (direction * relative_sl / 100000.0)
                if tp_to_set is None:
                    tp_to_set = entry_price + (direction * relative_tp / 100000.0)

            sl_to_set = round(float(sl_to_set), digits)
            tp_to_set = round(float(tp_to_set), digits)

            # Preserve existing SL or TP if already valid
            if has_sl:
                sl_to_set = cur_sl
            if has_tp:
                tp_to_set = cur_tp

            # Attach SL and TP via cTrader API
            try:
                yield client.amend_position_sltp(pos_id, stop_loss=sl_to_set, take_profit=tp_to_set)

                # Log/Update DB
                contract_size = 100 if asset_name == 'XAUUSD' else 100000
                lots = pos.tradeData.volume / (contract_size * 100) if hasattr(pos, 'tradeData') else 0.01
                db.log_trade_opening(pos_id, asset_name, 'BUY' if trade_side == 1 else 'SELL', lots, entry_price, sl=sl_to_set, tp=tp_to_set)

                print(f"  └─► [ATTACHED SUCCESS] Position {pos_id} updated with SL={sl_to_set:.{digits}f}, TP={tp_to_set:.{digits}f}")
                repaired_count += 1
            except Exception as e:
                print(f"  └─► [ATTACH FAILED] {e}. Closing unprotected position {pos_id}...")
                try:
                    vol = pos.tradeData.volume if hasattr(pos, 'tradeData') else 0
                    if vol > 0:
                        yield client.close_position(pos_id, vol)
                        print(f"  └─► [CLOSED SUCCESS] Position {pos_id} ({asset_name}) closed to protect capital.")
                        closed_count += 1
                except Exception as close_err:
                    print(f"  └─► [CLOSE FAILED] Could not close position {pos_id}: {close_err}")

        print("=" * 70)
        print(f" SUMMARY: {ok_count} Protected | {repaired_count} Repaired | {closed_count} Closed | {len(positions)} Total")
        print("=" * 70 + "\n")


    except Exception as e:
        logger.error(f"Error during SL/TP verification: {e}")
    finally:
        client.stop()
        reactor.stop()

def main():
    setup_logger()
    reactor.callWhenRunning(run_sltp_verifier)
    reactor.run()

if __name__ == "__main__":
    main()
