"""
Collecte OFI via IB API — deux mesures complémentaires :

1. OFI quote-based L1 (Cont-Kukanov-Stoikov) : calculé sur les changements de
   prix ET de taille au best bid/ask via reqMktData. Couvre TOUS les symboles
   en continu, sans rotation → utilisable pour le classement cross-sectionnel.
   Colonnes : ofi_l1_*, ofi_l1_norm_* (normalisé par le flux absolu, borné -1..1),
   quote_count_*.

2. Trade imbalance (Lee-Ready) : buy_vol - sell_vol sur les trades classifiés
   via reqTickByTickData. IB limite à ~5 souscriptions simultanées → rotation
   en groupes. Les fenêtres des symboles inactifs (tick_active=0) sont
   incomplètes : NE PAS comparer entre symboles de groupes différents.
   Colonnes : ofi_*, ofi_norm_*, buy_vol_*, sell_vol_*.

Note : reqMktData livre des snapshots agrégés (~250 ms), pas chaque mise à jour
du carnet. À un échantillonnage de 5 s, c'est adéquat ; l'OFI L1 sous-estime
légèrement l'activité intra-snapshot.

Usage :
    python src/app/collect_tick_ofi.py --host 192.168.0.103 --port 7496
    python src/app/collect_tick_ofi.py --host 192.168.0.103 --port 7496 --symbols AAPL,MSFT,NVDA
    python src/app/collect_tick_ofi.py --host 192.168.0.103 --port 7496 --tick-rotation-sec 60
"""

from __future__ import annotations

import argparse
import csv
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "AVGO", "QCOM", "INTC", "MU",
]

WINDOWS: Dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}


@dataclass
class TradeEvent:
    ts: float
    price: float
    size: float
    direction: int  # +1 acheteur, -1 vendeur, 0 inconnu


def _make_contract(symbol: str, exchange: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = exchange
    return c


class TickOFICollector(EWrapper, EClient):

    def __init__(self) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.l1_req_to_symbol: Dict[int, str] = {}
        self.tick_req_to_symbol: Dict[int, str] = {}

        self._bid: Dict[str, float] = {}
        self._ask: Dict[str, float] = {}
        self._bid_size: Dict[str, float] = {}
        self._ask_size: Dict[str, float] = {}
        self._last_price: Dict[str, float] = {}
        self._last_direction: Dict[str, int] = {}
        self._trades: Dict[str, Deque[TradeEvent]] = {}
        # OFI quote-based (Cont-Kukanov-Stoikov, niveau 1)
        # événements (ts, contribution) par symbole
        self._ofi_events: Dict[str, Deque] = {}
        # dernier état traité par côté : (prix, taille)
        self._prev_bid_state: Dict[str, tuple] = {}
        self._prev_ask_state: Dict[str, tuple] = {}

        self._tick_queue: queue.SimpleQueue = queue.SimpleQueue()

        self.error_counts: Dict[str, int] = {}
        self.last_error: Dict[str, str] = {}

    # ── Callbacks IB ─────────────────────────────────────────────────────────

    def nextValidId(self, orderId: int) -> None:
        self.ready.set()

    def error(self, reqId, _errorTime, errorCode, errorString, _advancedOrderRejectJson="") -> None:
        SILENT = {2104, 2106, 2107, 2108, 2158, 2152}
        symbol = (
            self.l1_req_to_symbol.get(reqId)
            or self.tick_req_to_symbol.get(reqId)
            or "GLOBAL"
        )
        with self._lock:
            self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
            self.last_error[symbol] = f"{errorCode} - {errorString}"
        if errorCode not in SILENT:
            print(f"[IB ERROR] reqId={reqId} symbol={symbol} code={errorCode} msg={errorString}")

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        symbol = self.l1_req_to_symbol.get(reqId)
        if not symbol or price <= 0:
            return
        with self._lock:
            if tickType == 1:  # BID
                self._bid[symbol] = price
                self._process_ofi_side_locked(symbol, side="bid")
            elif tickType == 2:  # ASK
                self._ask[symbol] = price
                self._process_ofi_side_locked(symbol, side="ask")

    def tickSize(self, reqId: int, tickType: int, size) -> None:
        symbol = self.l1_req_to_symbol.get(reqId)
        if not symbol:
            return
        size_f = float(size)
        if size_f < 0:
            return
        with self._lock:
            if tickType == 0:  # BID_SIZE
                self._bid_size[symbol] = size_f
                self._process_ofi_side_locked(symbol, side="bid")
            elif tickType == 3:  # ASK_SIZE
                self._ask_size[symbol] = size_f
                self._process_ofi_side_locked(symbol, side="ask")

    def _process_ofi_side_locked(self, symbol: str, side: str) -> None:
        """Calcule la contribution OFI (Cont-Kukanov-Stoikov, L1) du côté mis à jour.

        Contribution bid : +q si le bid monte, -q_prev s'il baisse, delta_q sinon.
        Contribution ask : symétrique (une hausse de taille ask = pression vendeuse).
        """
        if side == "bid":
            price = self._bid.get(symbol, 0.0)
            size = self._bid_size.get(symbol, -1.0)
            if price <= 0 or size < 0:
                return
            prev = self._prev_bid_state.get(symbol)
            self._prev_bid_state[symbol] = (price, size)
            if prev is None:
                return  # premier état : pas de transition exploitable
            prev_price, prev_size = prev
            if price > prev_price:
                contrib = size
            elif price < prev_price:
                contrib = -prev_size
            else:
                contrib = size - prev_size
        else:
            price = self._ask.get(symbol, 0.0)
            size = self._ask_size.get(symbol, -1.0)
            if price <= 0 or size < 0:
                return
            prev = self._prev_ask_state.get(symbol)
            self._prev_ask_state[symbol] = (price, size)
            if prev is None:
                return
            prev_price, prev_size = prev
            if price < prev_price:
                contrib = -size
            elif price > prev_price:
                contrib = prev_size
            else:
                contrib = -(size - prev_size)

        if contrib != 0.0:
            self._ofi_events.setdefault(symbol, deque()).append((time.time(), contrib))

    def tickByTickAllLast(
        self, reqId: int, tickType: int, time_epoch: int,
        price: float, size, tickAttribLast, exchange: str, specialConditions: str
    ) -> None:
        symbol = self.tick_req_to_symbol.get(reqId)
        if not symbol or price <= 0:
            return

        size_f = float(size)

        with self._lock:
            direction = self._classify_direction_locked(price, symbol)
            bid = self._bid.get(symbol, 0.0)
            ask = self._ask.get(symbol, 0.0)
            event = TradeEvent(ts=float(time_epoch), price=price, size=size_f, direction=direction)
            self._trades.setdefault(symbol, deque()).append(event)
            self._last_price[symbol] = price
            if direction != 0:
                self._last_direction[symbol] = direction

        self._tick_queue.put((time_epoch, symbol, price, size_f, direction, bid, ask))

    def _classify_direction_locked(self, price: float, symbol: str) -> int:
        bid = self._bid.get(symbol, 0.0)
        ask = self._ask.get(symbol, 0.0)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            if price > mid + 1e-6:
                return 1
            elif price < mid - 1e-6:
                return -1
        last = self._last_price.get(symbol, price)
        if price > last + 1e-6:
            return 1
        elif price < last - 1e-6:
            return -1
        return self._last_direction.get(symbol, 0)

    # ── Connexion ─────────────────────────────────────────────────────────────

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: int = 12) -> bool:
        self.connect(host, port, client_id)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self.ready.wait(timeout=timeout)

    def stop(self) -> None:
        try:
            for req_id in list(self.tick_req_to_symbol):
                try:
                    self.cancelTickByTickData(req_id)
                except Exception:
                    pass
            for req_id in list(self.l1_req_to_symbol):
                try:
                    self.cancelMktData(req_id)
                except Exception:
                    pass
            time.sleep(0.3)
        finally:
            if self.isConnected():
                self.disconnect()

    # ── Souscriptions ─────────────────────────────────────────────────────────

    def subscribe_l1(self, symbol: str, l1_req_id: int, exchange: str) -> None:
        """Souscrit le flux L1 bid/ask en streaming (pas de limite IB)."""
        with self._lock:
            self.l1_req_to_symbol[l1_req_id] = symbol
            self._trades.setdefault(symbol, deque())
        self.reqMktData(l1_req_id, _make_contract(symbol, exchange), "", False, False, [])

    def subscribe_tick(self, symbol: str, tick_req_id: int, exchange: str) -> None:
        """Souscrit reqTickByTickData AllLast (max 5 simultanés)."""
        with self._lock:
            self.tick_req_to_symbol[tick_req_id] = symbol
        self.reqTickByTickData(tick_req_id, _make_contract(symbol, exchange), "AllLast", 0, False)

    def cancel_tick(self, tick_req_id: int) -> None:
        """Annule une souscription tick-by-tick."""
        with self._lock:
            self.tick_req_to_symbol.pop(tick_req_id, None)
        try:
            self.cancelTickByTickData(tick_req_id)
        except Exception:
            pass

    # ── OFI ───────────────────────────────────────────────────────────────────

    def snapshot(self, symbol: str) -> Dict:
        now = time.time()
        max_window = max(WINDOWS.values())

        with self._lock:
            raw_trades = list(self._trades.get(symbol, []))
            ofi_events = list(self._ofi_events.get(symbol, []))
            bid      = self._bid.get(symbol, 0.0)
            ask      = self._ask.get(symbol, 0.0)
            bid_sz   = self._bid_size.get(symbol, 0.0)
            ask_sz   = self._ask_size.get(symbol, 0.0)
            last_px  = self._last_price.get(symbol, 0.0)
            cutoff = now - max_window
            if symbol in self._trades:
                while self._trades[symbol] and self._trades[symbol][0].ts < cutoff:
                    self._trades[symbol].popleft()
            if symbol in self._ofi_events:
                while self._ofi_events[symbol] and self._ofi_events[symbol][0][0] < cutoff:
                    self._ofi_events[symbol].popleft()

        result: Dict = {
            "bid": bid, "ask": ask,
            "bid_size": bid_sz, "ask_size": ask_sz,
            "spread": (ask - bid) if bid > 0 and ask > 0 else 0.0,
            "last_price": last_px,
        }

        for label, window_sec in WINDOWS.items():
            cutoff_w = now - window_sec

            # ── Trade imbalance (dépend de la rotation tick-by-tick) ──
            w = [t for t in raw_trades if t.ts >= cutoff_w]
            buy_vol  = sum(t.size for t in w if t.direction > 0)
            sell_vol = sum(t.size for t in w if t.direction < 0)
            total    = buy_vol + sell_vol
            ofi      = buy_vol - sell_vol

            # ── OFI quote-based L1 (couvre TOUS les symboles en continu) ──
            ev = [c for (ts, c) in ofi_events if ts >= cutoff_w]
            ofi_l1 = sum(ev)
            abs_flow = sum(abs(c) for c in ev)

            result.update({
                f"buy_vol_{label}":      buy_vol,
                f"sell_vol_{label}":     sell_vol,
                f"ofi_{label}":          ofi,
                f"ofi_norm_{label}":     ofi / total if total > 0 else 0.0,
                f"trade_count_{label}":  len(w),
                f"vwap_{label}":         (
                    sum(t.price * t.size for t in w) / sum(t.size for t in w)
                    if w else last_px
                ),
                f"ofi_l1_{label}":       ofi_l1,
                f"ofi_l1_norm_{label}":  ofi_l1 / abs_flow if abs_flow > 0 else 0.0,
                f"quote_count_{label}":  len(ev),
            })

        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Collecte OFI tick-by-tick IB API")
    parser.add_argument("--host",              type=str,   default="127.0.0.1")
    parser.add_argument("--port",              type=int,   default=None)
    parser.add_argument("--client",            type=int,   default=98)
    parser.add_argument("--symbols",           type=str,   default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--duration-min",      type=int,   default=390)
    parser.add_argument("--sample-sec",        type=float, default=5.0)
    parser.add_argument("--exchange",          type=str,   default="ISLAND")
    parser.add_argument("--tick-group-size",   type=int,   default=5,
                        help="Nb max de souscriptions tick simultanées (défaut 5, limite IB)")
    parser.add_argument("--tick-rotation-sec", type=float, default=60.0,
                        help="Durée en secondes par groupe tick (défaut 60s)")
    parser.add_argument("--output-dir",        type=str,   default="logs/tick_ofi")
    args = parser.parse_args()

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    port = args.port if args.port is not None else 7496

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_path = output_dir / f"tick_ofi_{stamp}.csv"
    raw_path     = output_dir / f"tick_raw_{stamp}.csv"

    # Groupes de rotation tick-by-tick
    gs     = args.tick_group_size
    groups = [symbols[i:i+gs] for i in range(0, len(symbols), gs)]
    use_rotation = len(symbols) > gs
    cycle_sec = len(groups) * args.tick_rotation_sec

    print(f"\n{'='*60}")
    print("TICK OFI COLLECTOR  (reqTickByTickData + Lee-Ready)")
    print(f"{'='*60}")
    print(f"Host/Port      : {args.host}:{port}")
    print(f"Client ID      : {args.client}")
    print(f"Exchange       : {args.exchange}")
    print(f"Symbols ({len(symbols):2d})  : {', '.join(symbols)}")
    if use_rotation:
        print(f"Tick rotation  : {len(groups)} groupes × {args.tick_rotation_sec}s = cycle {cycle_sec:.0f}s")
        for gi, g in enumerate(groups):
            print(f"  Groupe {gi+1}     : {', '.join(g)}")
    else:
        print(f"Tick rotation  : desactivee ({len(symbols)} <= {gs})")
    print(f"Sample OFI     : toutes les {args.sample_sec}s")
    print(f"Duration       : {args.duration_min} min")
    print(f"Output OFI     : {samples_path}")
    print(f"Output ticks   : {raw_path}")
    print(f"{'='*60}\n")

    collector = TickOFICollector()
    if not collector.connect_and_start(args.host, port, args.client):
        print("[ERREUR] Connexion IB impossible (nextValidId non reçu).")
        return 1

    print("[OK] Connecté à IB")

    try:
        # L1 bid/ask pour TOUS les symboles (permanent, pas de limite)
        for i, symbol in enumerate(symbols):
            collector.subscribe_l1(symbol, l1_req_id=1000 + i, exchange=args.exchange)
            time.sleep(0.05)

        # Tick-by-tick seulement pour le premier groupe
        current_group = 0
        for symbol in groups[current_group]:
            i = symbols.index(symbol)
            collector.subscribe_tick(symbol, tick_req_id=2000 + i, exchange=args.exchange)
            time.sleep(0.05)

        active_group_str = ", ".join(groups[current_group])
        print(f"[OK] L1 souscrit pour {len(symbols)} symboles")
        print(f"[OK] Ticks actifs  : {active_group_str}")
        print("[INFO] Warmup 3s...\n")
        time.sleep(3.0)

        tick_rotation_ts = time.time()

        headers = ["ts_et", "symbol", "last_price", "bid", "ask",
                   "bid_size", "ask_size", "spread", "tick_active"]
        for label in WINDOWS:
            headers += [
                f"buy_vol_{label}", f"sell_vol_{label}",
                f"ofi_{label}", f"ofi_norm_{label}",
                f"trade_count_{label}", f"vwap_{label}",
                f"ofi_l1_{label}", f"ofi_l1_norm_{label}", f"quote_count_{label}",
            ]
        raw_headers = ["ts_et", "ts_epoch", "symbol", "price", "size", "direction", "bid", "ask"]

        start_ts      = time.time()
        end_ts        = start_ts + args.duration_min * 60
        loop_idx      = 0
        raw_tick_count = 0

        with samples_path.open("w", newline="", encoding="utf-8") as f_ofi, \
             raw_path.open("w", newline="", encoding="utf-8") as f_raw:

            writer     = csv.writer(f_ofi)
            raw_writer = csv.writer(f_raw)
            writer.writerow(headers)
            raw_writer.writerow(raw_headers)
            print(f"[RUN] OFI agrégé  : {samples_path}")
            print(f"[RUN] Ticks bruts : {raw_path}\n")

            # Ensemble des symboles du groupe actif (pour la colonne tick_active)
            active_set = set(groups[current_group])

            while time.time() < end_ts:
                now = time.time()
                ts_et = datetime.now(ET).isoformat(timespec="milliseconds")

                # ── Rotation tick-by-tick ────────────────────────────────────
                if use_rotation and (now - tick_rotation_ts) >= args.tick_rotation_sec:
                    # Annuler groupe sortant
                    for symbol in groups[current_group]:
                        i = symbols.index(symbol)
                        collector.cancel_tick(tick_req_id=2000 + i)
                    time.sleep(0.1)
                    # Activer groupe entrant
                    current_group = (current_group + 1) % len(groups)
                    for symbol in groups[current_group]:
                        i = symbols.index(symbol)
                        collector.subscribe_tick(symbol, tick_req_id=2000 + i, exchange=args.exchange)
                        time.sleep(0.05)
                    active_set = set(groups[current_group])
                    tick_rotation_ts = now
                    print(f"[ROTATION] → Groupe {current_group+1}/{len(groups)}: {', '.join(groups[current_group])}")

                # ── Snapshot OFI agrégé ──────────────────────────────────────
                for symbol in symbols:
                    snap = collector.snapshot(symbol)
                    tick_active = 1 if symbol in active_set else 0
                    row: List = [
                        ts_et, symbol,
                        f"{snap['last_price']:.4f}",
                        f"{snap['bid']:.4f}",
                        f"{snap['ask']:.4f}",
                        f"{snap['bid_size']:.0f}",
                        f"{snap['ask_size']:.0f}",
                        f"{snap['spread']:.4f}",
                        tick_active,
                    ]
                    for label in WINDOWS:
                        row += [
                            f"{snap[f'buy_vol_{label}']:.2f}",
                            f"{snap[f'sell_vol_{label}']:.2f}",
                            f"{snap[f'ofi_{label}']:.2f}",
                            f"{snap[f'ofi_norm_{label}']:.4f}",
                            snap[f"trade_count_{label}"],
                            f"{snap[f'vwap_{label}']:.4f}",
                            f"{snap[f'ofi_l1_{label}']:.2f}",
                            f"{snap[f'ofi_l1_norm_{label}']:.4f}",
                            snap[f"quote_count_{label}"],
                        ]
                    writer.writerow(row)

                # ── Drain ticks bruts ────────────────────────────────────────
                while not collector._tick_queue.empty():
                    try:
                        ts_ep, sym, px, sz, d, b, a = collector._tick_queue.get_nowait()
                        ts_tick = datetime.fromtimestamp(ts_ep, tz=ET).isoformat(timespec="milliseconds")
                        raw_writer.writerow([ts_tick, ts_ep, sym, f"{px:.4f}", f"{sz:.2f}", d, f"{b:.4f}", f"{a:.4f}"])
                        raw_tick_count += 1
                    except queue.Empty:
                        break

                f_ofi.flush()
                f_raw.flush()
                loop_idx += 1

                if loop_idx % 12 == 0:
                    elapsed = (time.time() - start_ts) / 60
                    previews = []
                    for s in groups[current_group]:
                        snap = collector.snapshot(s)
                        previews.append(
                            f"{s} L1={snap['ofi_l1_norm_1m']:+.2f}"
                            f" trd={snap['ofi_norm_1m']:+.2f} cnt={snap['trade_count_1m']}"
                        )
                    grp_label = f"grp{current_group+1}/{len(groups)}"
                    print(f"[RUN] {elapsed:.1f}min | ticks={raw_tick_count} | {grp_label}: {' | '.join(previews)}")

                time.sleep(args.sample_sec)

        print(f"\n[OK] Collecte terminée.")
        print(f"[OUT] OFI agrégé  : {samples_path}")
        print(f"[OUT] Ticks bruts : {raw_path}  ({raw_tick_count} ticks)")

        print("\nRésumé (dernière snapshot) :")
        for symbol in symbols:
            snap = collector.snapshot(symbol)
            errs = collector.error_counts.get(symbol, 0)
            print(
                f"  {symbol:5s}  last={snap['last_price']:.2f}"
                f"  ofiL1_1m={snap['ofi_l1_norm_1m']:+.3f}"
                f"  ofiL1_5m={snap['ofi_l1_norm_5m']:+.3f}"
                f"  trd_1m={snap['ofi_norm_1m']:+.3f}"
                f"  quotes_1m={snap['quote_count_1m']}"
                f"  err={errs}"
            )
        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur.")
        return 130
    finally:
        collector.stop()


if __name__ == "__main__":
    raise SystemExit(main())
