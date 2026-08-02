"""
Test script: Opens a small BTC/USD trade, attempts to attach very close SL/TP,
and captures the cTrader error codes returned.

Usage:
    python scripts/test_sltp_amend.py

Requires .env with CT_* credentials (demo account).
"""
import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from dotenv import load_dotenv
load_dotenv()

import logging
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SLTP_TEST")

APP_ID = os.environ["CT_APP_ID"]
APP_SECRET = os.environ["CT_APP_SECRET"]
ACCOUNT_ID = int(os.environ["CT_ACCOUNT_ID"])
ACCESS_TOKEN = os.environ["CT_ACCESS_TOKEN"]
HOST_TYPE = os.environ.get("CT_HOST_TYPE", "demo")

HOST = EndPoints.PROTOBUF_LIVE_HOST if HOST_TYPE == "live" else EndPoints.PROTOBUF_DEMO_HOST
PORT = EndPoints.PROTOBUF_PORT

ct_client = Client(HOST, PORT, TcpProtocol)
test_pos_id = None
test_volume = None


def check_error(payload, context=""):
    name = getattr(payload.DESCRIPTOR, "name", "") if hasattr(payload, "DESCRIPTOR") else ""
    if name in ("ProtoOAErrorRes", "ProtoErrorRes"):
        code = getattr(payload, "errorCode", "UNKNOWN")
        desc = getattr(payload, "description", "No description")
        log.error(f"  [{context}] ERROR: code={code}, desc={desc}")
        return True, code, desc
    return False, None, None


@inlineCallbacks
def send_proto(req, timeout_sec=30):
    res = yield ct_client.send(req, responseTimeoutInSeconds=timeout_sec)
    return Protobuf.extract(res)


@inlineCallbacks
def run_test():
    global test_pos_id, test_volume

    log.info("=" * 60)
    log.info("SL/TP AMEND ERROR CODE TEST")
    log.info("=" * 60)

    # 1. Auth
    log.info("\n[1] Authenticating...")
    auth_req = ProtoOAApplicationAuthReq()
    auth_req.clientId = APP_ID
    auth_req.clientSecret = APP_SECRET
    auth_payload = yield send_proto(auth_req)
    is_err, code, desc = check_error(auth_payload, "AppAuth")
    if is_err:
        log.critical(f"App auth failed: {code} {desc}")
        reactor.stop()
        return

    acc_req = ProtoOAAccountAuthReq()
    acc_req.ctidTraderAccountId = ACCOUNT_ID
    acc_req.accessToken = ACCESS_TOKEN
    acc_payload = yield send_proto(acc_req)
    is_err, code, desc = check_error(acc_payload, "AccAuth")
    if is_err:
        log.critical(f"Account auth failed: {code} {desc}")
        reactor.stop()
        return
    log.info("  Authenticated OK")

    # 2. Get all symbols via SymbolsListReq
    log.info("\n[2] Fetching available symbols...")
    syms_req = ProtoOASymbolsListReq()
    syms_req.ctidTraderAccountId = ACCOUNT_ID
    syms_req.includeArchivedSymbols = False
    syms_payload = yield send_proto(syms_req, timeout_sec=60)
    is_err, code, desc = check_error(syms_payload, "SymbolsList")
    if is_err:
        log.critical(f"  SymbolsList failed: {code} {desc}")
        reactor.stop()
        return

    symbols = getattr(syms_payload, 'symbol', [])
    log.info(f"  Got {len(symbols)} symbols")

    # Find BTCUSD
    btc_symbol_id = None
    for s in symbols:
        name = str(getattr(s, 'symbolName', '')).upper()
        if 'BTC' in name:
            log.info(f"  Found: id={s.symbolId}, name={s.symbolName}, enabled={s.enabled}")
            if s.enabled:
                btc_symbol_id = s.symbolId
                break

    if btc_symbol_id is None:
        log.critical("  BTCUSD not found or not enabled")
        reactor.stop()
        return

    # Get full symbol details for min/max volume info
    detail_req = ProtoOASymbolByIdReq()
    detail_req.ctidTraderAccountId = ACCOUNT_ID
    detail_req.symbolId.append(btc_symbol_id)
    detail_payload = yield send_proto(detail_req)
    is_err, _, _ = check_error(detail_payload, "SymbolById")
    if not is_err:
        syms = getattr(detail_payload, 'symbol', [])
        if syms:
            s = syms[0]
            log.info(f"  Details: digits={s.digits}, minVol={s.minVolume}, "
                     f"maxVol={s.maxVolume}, stepVol={s.stepVolume}")

    log.info(f"  Using symbolId={btc_symbol_id}")

    # 3. Subscribe to spots
    log.info(f"\n[3] Subscribing to spots...")
    sub_req = ProtoOASubscribeSpotsReq()
    sub_req.ctidTraderAccountId = ACCOUNT_ID
    sub_req.symbolId.append(btc_symbol_id)
    sub_payload = yield send_proto(sub_req)
    is_err, code, desc = check_error(sub_payload, "SubscribeSpots")
    log.info(f"  {'OK' if not is_err else f'Error: {code} {desc}'}")

    # 4. Open a tiny BUY order
    log.info(f"\n[4] Opening BUY market order...")
    for vol in [1, 10, 100]:
        order_req = ProtoOANewOrderReq()
        order_req.ctidTraderAccountId = ACCOUNT_ID
        order_req.symbolId = btc_symbol_id
        order_req.volume = vol
        order_req.tradeSide = ProtoOATradeSide.BUY
        order_req.orderType = ProtoOAOrderType.MARKET

        order_payload = yield send_proto(order_req)
        is_err, code, desc = check_error(order_payload, f"NewOrder(vol={vol})")
        if not is_err and hasattr(order_payload, 'position') and order_payload.position:
            test_pos_id = order_payload.position.positionId
            log.info(f"  Order filled: pos_id={test_pos_id}")
            break
        log.warning(f"  volume={vol} failed: {code} {desc}")
    else:
        log.critical("  All order attempts failed")
        reactor.stop()
        return

    # Get real position data via reconcile
    reconcile_req = ProtoOAReconcileReq()
    reconcile_req.ctidTraderAccountId = ACCOUNT_ID
    reconcile_payload = yield send_proto(reconcile_req)
    entry = None
    test_volume = None
    for p in getattr(reconcile_payload, 'position', []):
        if p.positionId == test_pos_id:
            entry = p.price
            test_volume = p.tradeData.volume if hasattr(p, 'tradeData') else vol
            has_sl = bool(hasattr(p, 'stopLoss') and p.stopLoss and p.stopLoss > 0)
            has_tp = bool(hasattr(p, 'takeProfit') and p.takeProfit and p.takeProfit > 0)
            log.info(f"  Position: entry={entry}, volume={test_volume}")
            log.info(f"  SL attached: {has_sl} (value={getattr(p, 'stopLoss', 0)})")
            log.info(f"  TP attached: {has_tp} (value={getattr(p, 'takeProfit', 0)})")
            break

    if entry is None:
        log.critical(f"  Position {test_pos_id} not found via reconcile")
        reactor.stop()
        return

    # 5. Test SL/TP amend with various distances
    log.info("\n[5] Testing SL/TP amend with various distances...")
    log.info(f"  Entry: {entry}, Position: {test_pos_id}")
    log.info("")

    # For BUY: SL must be BELOW entry, TP must be ABOVE entry
    test_cases = [
        ("SL/TP 0.01 from entry (extreme)",   entry - 0.01,   entry + 0.01),
        ("SL/TP 0.50 from entry",             entry - 0.50,   entry + 0.50),
        ("SL/TP 1.00 from entry",             entry - 1.00,   entry + 1.00),
        ("SL/TP 5.00 from entry",             entry - 5.00,   entry + 5.00),
        ("SL/TP 10.00 from entry",            entry - 10.00,  entry + 10.00),
        ("SL/TP 50.00 from entry",            entry - 50.00,  entry + 50.00),
        ("SL/TP 100.00 from entry",           entry - 100.00, entry + 100.00),
        ("SL/TP 500.00 from entry",           entry - 500.00, entry + 500.00),
        ("SL ABOVE entry (invalid for LONG)", entry + 10.00,  entry + 50.00),
    ]

    for label, sl_price, tp_price in test_cases:
        log.info(f"  Test: {label}")
        log.info(f"    SL={sl_price:.2f}, TP={tp_price:.2f}")

        amend_req = ProtoOAAmendPositionSLTPReq()
        amend_req.ctidTraderAccountId = ACCOUNT_ID
        amend_req.positionId = test_pos_id
        amend_req.stopLoss = float(sl_price)
        amend_req.takeProfit = float(tp_price)

        payload = yield send_proto(amend_req)
        is_err, code, desc = check_error(payload, f"Amend:{label}")
        if is_err:
            log.info(f"    >> REJECTED: [{code}] {desc}")
        else:
            # Check errorCode on execution event
            exec_code = getattr(payload, 'errorCode', '')
            if exec_code:
                log.info(f"    >> REJECTED via executionEvent: [{exec_code}]")
            else:
                log.info(f"    >> ACCEPTED")
            # Reconcile to verify actual SL/TP on position
            r_req = ProtoOAReconcileReq()
            r_req.ctidTraderAccountId = ACCOUNT_ID
            r_payload = yield send_proto(r_req)
            for p in getattr(r_payload, 'position', []):
                if p.positionId == test_pos_id:
                    actual_sl = getattr(p, 'stopLoss', 0)
                    actual_tp = getattr(p, 'takeProfit', 0)
                    log.info(f"    >> Actual on position: SL={actual_sl}, TP={actual_tp}")
                    break

    # 6. Cleanup
    log.info(f"\n[6] Closing test position {test_pos_id}...")
    close_req = ProtoOAClosePositionReq()
    close_req.ctidTraderAccountId = ACCOUNT_ID
    close_req.positionId = test_pos_id
    close_req.volume = int(test_volume)
    close_payload = yield send_proto(close_req)
    is_err, code, desc = check_error(close_payload, "ClosePosition")
    if is_err:
        log.error(f"  Close failed: {code} {desc}")
    else:
        log.info("  Position closed OK")

    log.info("\n" + "=" * 60)
    log.info("TEST COMPLETE")
    log.info("=" * 60)
    reactor.stop()


def on_connected(client):
    log.info("Connected to cTrader. Running test...")
    reactor.callFromThread(run_test)


def on_message(client, message):
    pass


def main():
    ct_client.setConnectedCallback(on_connected)
    ct_client.setMessageReceivedCallback(on_message)
    ct_client.startService()
    log.info(f"Connecting to cTrader ({HOST}:{PORT})...")
    reactor.run()


if __name__ == "__main__":
    main()
