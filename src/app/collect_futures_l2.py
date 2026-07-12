"""
Collecte L2 (carnet d'ordres) + trades pour futures MES, MNQ, YM via IB API.
Format de sortie: parquet compatible NinjaTrader.

Structure de sortie:
    collectOF/sessions_mes/YYYY-MM-DD/trade_HHMMSS.parquet  -> ts, price, size
    collectOF/sessions_mes/YYYY-MM-DD/quote_HHMMSS.parquet  -> ts, side, price, size

Usage:
    python src/app/collect_futures_l2.py --host 192.168.0.103 --port 7496
    python src/app/collect_futures_l2.py --host 192.168.0.103 --port 7496 --no-ticks
    python src/app/collect_futures_l2.py --host 192.168.0.103 --port 7496 --contract-month 202609
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# Symbole IB -> (exchange, multiplicateur tick, nom session NinjaTrader)
FUTURES: Dict[str, Tuple[str, float, str]] = {
    "MES": ("CME",  5.0, "sessions_mes"),
    "MNQ": ("CME",  2.0, "sessions_mnq"),
    "YM":  ("CBOT", 5.0, "sessions_dow"),
}

WINDOWS: Dict[str, int] = {"1m": 60, "5m": 300, "15m": 900}
CHUNK_SEC = 30  # duree d'un fichier parquet (en secondes, identique NinjaTrader)


def _front_month_simple() -> str:
    """Contrat trimestriel courant (format YYYYMM)."""
    today = date.today()
    for m in [3, 6, 9, 12]:
        if m >= today.month:
            exp = _third_friday(today.year, m)
            if today <= exp:
                return f"{today.year}{m:02d}"
    return f"{today.year + 1}03"


def _third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    first_friday = d + timedelta(days=(4 - d.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


@dataclass
class TradeEvent:
    ts: float
    price: float
    size: float
    direction: int


@dataclass
class L2Book:
    # position (0=best) -> (price, size), garanti par IB
    bids: Dict[int, tuple] = field(default_factory=dict)
    asks: Dict[int, tuple] = field(default_factory=dict)


class FuturesL2Collector(EWrapper, EClient):

    def __init__(self, depth: int = 5) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.depth = depth
        self.ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Mappings reqId
        self.l2_req_to_symbol:   Dict[int, str] = {}
        self.tick_req_to_symbol: Dict[int, str] = {}
        self.l1_req_to_symbol:   Dict[int, str] = {}

        # Carnets L2 (pour status/spread monitoring)
        self._books: Dict[str, L2Book] = {}

        # L1 pour Lee-Ready
        self._bid:            Dict[str, float] = {}
        self._ask:            Dict[str, float] = {}
        self._last_price:     Dict[str, float] = {}
        self._last_direction: Dict[str, int]   = {}

        # Accumulation ticks (pour OFI dans snapshot())
        self._trades: Dict[str, Deque[TradeEvent]] = {}

        # Queues d'evenements bruts -> thread principal pour ecriture parquet
        # Format trade: (symbol, ts_str, price, size)
        # Format quote: (symbol, ts_str, side_str, price, size)
        self._q_trade: queue.SimpleQueue = queue.SimpleQueue()
        self._q_quote: queue.SimpleQueue = queue.SimpleQueue()

        self.error_counts: Dict[str, int] = {}
        self.last_error:   Dict[str, str] = {}

    # ── Callbacks IB ──────────────────────────────────────────────────────────

    def nextValidId(self, orderId: int) -> None:
        self.ready.set()

    def error(self, reqId, _errorTime, errorCode, errorString,
              _advancedOrderRejectJson="") -> None:
        SILENT = {2104, 2106, 2107, 2108, 2158, 2152}
        symbol = (
            self.l2_req_to_symbol.get(reqId)
            or self.tick_req_to_symbol.get(reqId)
            or self.l1_req_to_symbol.get(reqId)
            or "GLOBAL"
        )
        with self._lock:
            self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
            self.last_error[symbol] = f"{errorCode} - {errorString}"
        if errorCode not in SILENT:
            print(f"[IB ERROR] reqId={reqId} symbol={symbol} code={errorCode} msg={errorString}")

    def updateMktDepth(self, reqId: int, position: int, operation: int,
                       side: int, price: float, size: float) -> None:
        self._apply_book_update(reqId, position, operation, side, price, float(size))

    def updateMktDepthL2(self, reqId: int, position: int, marketMaker: str,
                         operation: int, side: int, price: float, size: float,
                         isSmartDepth: bool) -> None:
        self._apply_book_update(reqId, position, operation, side, price, float(size))

    def _apply_book_update(self, reqId: int, position: int, operation: int,
                           side: int, price: float, size: float) -> None:
        symbol = self.l2_req_to_symbol.get(reqId)
        if not symbol:
            return
        with self._lock:
            book = self._books.get(symbol)
            if book is None:
                return
            # side: 1=bid, 0=ask (convention IB)
            target = book.bids if side == 1 else book.asks
            if operation == 2 or size == 0:
                target.pop(position, None)
            else:
                target[position] = (price, float(size))
        # Emettre evenement quote (size > 0 seulement, matching NinjaTrader)
        if size > 0:
            ts = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            side_str = "Bid" if side == 1 else "Ask"
            self._q_quote.put((symbol, ts, side_str, price, int(size)))

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib) -> None:
        symbol = self.l1_req_to_symbol.get(reqId)
        if not symbol or price <= 0:
            return
        with self._lock:
            if tickType == 1:
                self._bid[symbol] = price
            elif tickType == 2:
                self._ask[symbol] = price

    def tickSize(self, reqId: int, tickType: int, size) -> None:
        pass

    def tickByTickAllLast(self, reqId: int, tickType: int, time_epoch: int,
                          price: float, size, tickAttribLast,
                          exchange: str, specialConditions: str) -> None:
        symbol = self.tick_req_to_symbol.get(reqId)
        if not symbol or price <= 0:
            return
        size_i = int(size)
        with self._lock:
            direction = self._classify_direction_locked(price, symbol)
            event = TradeEvent(ts=float(time_epoch), price=price,
                               size=float(size_i), direction=direction)
            self._trades.setdefault(symbol, deque()).append(event)
            self._last_price[symbol] = price
            if direction != 0:
                self._last_direction[symbol] = direction
        # Timestamp exchange (secondes) -> format NinjaTrader avec .000
        ts_str = datetime.fromtimestamp(time_epoch, tz=ET).strftime("%Y-%m-%d %H:%M:%S.000")
        self._q_trade.put((symbol, ts_str, float(price), size_i))

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

    def connect_and_start(self, host: str, port: int, client_id: int,
                          timeout: int = 12) -> bool:
        self.connect(host, port, client_id)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self.ready.wait(timeout=timeout)

    def stop(self) -> None:
        try:
            for req_id in list(self.l2_req_to_symbol):
                try:
                    self.cancelMktDepth(req_id, False)
                except Exception:
                    pass
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

    # ── Souscription ──────────────────────────────────────────────────────────

    def subscribe(self, symbol: str, exchange: str, contract_month: str,
                  l2_req_id: int, tick_req_id: Optional[int],
                  l1_req_id: int) -> None:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "FUT"
        contract.exchange = exchange
        contract.currency = "USD"
        contract.lastTradeDateOrContractMonth = contract_month

        with self._lock:
            self.l2_req_to_symbol[l2_req_id]   = symbol
            self.l1_req_to_symbol[l1_req_id]    = symbol
            if tick_req_id is not None:
                self.tick_req_to_symbol[tick_req_id] = symbol
            self._books[symbol] = L2Book()
            self._trades.setdefault(symbol, deque())

        self.reqMktDepth(l2_req_id, contract, self.depth, False, [])
        self.reqMktData(l1_req_id, contract, "", False, False, [])
        if tick_req_id is not None:
            self.reqTickByTickData(tick_req_id, contract, "AllLast", 0, False)

    # ── Snapshot (utilise uniquement pour le monitoring de statut) ────────────

    def snapshot(self, symbol: str) -> Dict:
        now = time.time()
        max_window = max(WINDOWS.values())

        with self._lock:
            book = self._books.get(symbol, L2Book())
            bids_by_pos = [book.bids[p] for p in sorted(book.bids)]
            asks_by_pos = [book.asks[p] for p in sorted(book.asks)]
            last_px = self._last_price.get(symbol, 0.0)
            raw_trades = list(self._trades.get(symbol, []))
            if symbol in self._trades:
                cutoff = now - max_window
                while self._trades[symbol] and self._trades[symbol][0].ts < cutoff:
                    self._trades[symbol].popleft()

        bid1 = bids_by_pos[0][0] if bids_by_pos else 0.0
        ask1 = asks_by_pos[0][0] if asks_by_pos else 0.0

        result: Dict = {
            "last_price": last_px,
            "bid1": bid1,
            "ask1": ask1,
            "spread": (ask1 - bid1) if bid1 > 0 and ask1 > 0 else 0.0,
        }

        if bid1 > 0 and ask1 > 0 and bids_by_pos and asks_by_pos:
            bsz1 = bids_by_pos[0][1]
            asz1 = asks_by_pos[0][1]
            result["weighted_mid"] = (bid1 * asz1 + ask1 * bsz1) / (bsz1 + asz1)
        else:
            result["weighted_mid"] = last_px

        bid_total = sum(sz for _, sz in bids_by_pos[:self.depth])
        ask_total = sum(sz for _, sz in asks_by_pos[:self.depth])
        total_book = bid_total + ask_total
        result["book_imbalance"] = (bid_total - ask_total) / total_book if total_book > 0 else 0.0

        for label, window_sec in WINDOWS.items():
            cutoff_w = now - window_sec
            w = [t for t in raw_trades if t.ts >= cutoff_w]
            buy_vol  = sum(t.size for t in w if t.direction > 0)
            sell_vol = sum(t.size for t in w if t.direction < 0)
            total    = buy_vol + sell_vol
            ofi      = buy_vol - sell_vol
            result[f"ofi_norm_{label}"] = ofi / total if total > 0 else 0.0
            result[f"trade_count_{label}"] = len(w)

        return result


# ── Ecriture parquet ───────────────────────────────────────────────────────────

def _write_chunk(output_dir: Path, symbol: str,
                 trade_rows: list, quote_rows: list,
                 chunk_label: str, date_str: str) -> Tuple[int, int]:
    """Ecrit les buffers trade et quote en parquet, retourne (nb_trades, nb_quotes)."""
    _, _, session = FUTURES.get(symbol, ("CME", 1.0, f"sessions_{symbol.lower()}"))
    day_dir = output_dir / session / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    n_trades = n_quotes = 0
    if trade_rows:
        df = pd.DataFrame(trade_rows, columns=["ts", "price", "size"])
        df["size"] = df["size"].astype("int64")
        df.to_parquet(day_dir / f"trade_{chunk_label}.parquet", index=False)
        n_trades = len(df)

    if quote_rows:
        df = pd.DataFrame(quote_rows, columns=["ts", "side", "price", "size"])
        df["size"] = df["size"].astype("int64")
        df.to_parquet(day_dir / f"quote_{chunk_label}.parquet", index=False)
        n_quotes = len(df)

    return n_trades, n_quotes


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    front = _front_month_simple()

    parser = argparse.ArgumentParser(description="Collecte L2 futures MES/MNQ/YM -> parquet NinjaTrader")
    parser.add_argument("--host",           type=str,   default="127.0.0.1")
    parser.add_argument("--port",           type=int,   default=7496)
    parser.add_argument("--client",         type=int,   default=97)
    parser.add_argument("--symbols",        type=str,   default="MES,MNQ,YM")
    parser.add_argument("--contract-month", type=str,   default=front,
                        help=f"Mois du contrat YYYYMM (defaut: {front})")
    parser.add_argument("--depth",          type=int,   default=5,
                        help="Niveaux L2 a capturer (defaut: 5)")
    parser.add_argument("--no-ticks",       action="store_true",
                        help="Desactiver reqTickByTickData (si limite compte deja atteinte)")
    parser.add_argument("--duration-min",   type=int,   default=395)
    parser.add_argument("--chunk-sec",      type=float, default=CHUNK_SEC,
                        help=f"Duree d'un fichier parquet en secondes (defaut: {CHUNK_SEC})")
    parser.add_argument("--output-dir",     type=str,   default="collectOF",
                        help="Repertoire de sortie (defaut: collectOF)")
    args = parser.parse_args()

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    output_dir = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("FUTURES L2 COLLECTOR  (format parquet NinjaTrader)")
    print(f"{'='*60}")
    print(f"Host/Port      : {args.host}:{args.port}")
    print(f"Client ID      : {args.client}")
    print(f"Symboles       : {', '.join(symbols)}")
    print(f"Contrat        : {args.contract_month}")
    print(f"Profondeur L2  : {args.depth} niveaux")
    print(f"Chunk parquet  : {args.chunk_sec}s")
    print(f"Duration       : {args.duration_min} min")
    print(f"Output dir     : {output_dir}/")
    for sym in symbols:
        _, _, session = FUTURES.get(sym, ("CME", 1.0, f"sessions_{sym.lower()}"))
        print(f"  {sym} -> {output_dir}/{session}/YYYY-MM-DD/{{trade,quote}}_HHMMSS.parquet")
    print(f"Ticks          : {'DESACTIVES (--no-ticks)' if args.no_ticks else 'actives'}")
    print(f"{'='*60}\n")

    collector = FuturesL2Collector(depth=args.depth)
    if not collector.connect_and_start(args.host, args.port, args.client):
        print("[ERREUR] Connexion IB impossible (nextValidId non recu).")
        return 1

    print("[OK] Connecte a IB")

    try:
        for i, symbol in enumerate(symbols):
            exchange, _, _ = FUTURES.get(symbol, ("CME", 1.0, f"sessions_{symbol.lower()}"))
            collector.subscribe(
                symbol=symbol,
                exchange=exchange,
                contract_month=args.contract_month,
                l2_req_id=1000 + i,
                tick_req_id=None if args.no_ticks else 2000 + i,
                l1_req_id=3000 + i,
            )
            time.sleep(0.2)

        mode = "L2 + L1 (ticks desactives)" if args.no_ticks else "L2 + ticks + L1"
        print(f"[OK] {len(symbols)} futures souscrits ({mode})")
        print("[INFO] Warmup 3s...\n")
        time.sleep(3.0)

        # Buffers par symbole
        trade_rows: Dict[str, list] = {s: [] for s in symbols}
        quote_rows: Dict[str, list] = {s: [] for s in symbols}
        total_trades: Dict[str, int] = {s: 0 for s in symbols}
        total_quotes: Dict[str, int] = {s: 0 for s in symbols}

        start_ts       = time.time()
        end_ts         = start_ts + args.duration_min * 60
        chunk_ts       = datetime.now(ET)
        chunk_label    = chunk_ts.strftime("%H%M%S")
        last_status_ts = start_ts
        chunks_written = 0

        print(f"[RUN] Collecte en cours -> {output_dir}/")
        print(f"[RUN] Rotation toutes les {args.chunk_sec:.0f}s\n")

        while time.time() < end_ts:
            # Drain queue trades
            while True:
                try:
                    sym, ts_str, price, size = collector._q_trade.get_nowait()
                    if sym in trade_rows:
                        trade_rows[sym].append((ts_str, price, size))
                        total_trades[sym] += 1
                except queue.Empty:
                    break

            # Drain queue quotes
            while True:
                try:
                    sym, ts_str, side, price, size = collector._q_quote.get_nowait()
                    if sym in quote_rows:
                        quote_rows[sym].append((ts_str, side, price, size))
                        total_quotes[sym] += 1
                except queue.Empty:
                    break

            # Rotation de chunk
            now_et = datetime.now(ET)
            if (now_et - chunk_ts).total_seconds() >= args.chunk_sec:
                date_str = chunk_ts.strftime("%Y-%m-%d")
                for sym in symbols:
                    _write_chunk(output_dir, sym,
                                 trade_rows[sym], quote_rows[sym],
                                 chunk_label, date_str)
                    trade_rows[sym] = []
                    quote_rows[sym] = []
                chunks_written += 1
                chunk_ts    = now_et
                chunk_label = now_et.strftime("%H%M%S")

            # Statut toutes les 60s
            if time.time() - last_status_ts >= 60:
                elapsed = (time.time() - start_ts) / 60
                print(f"[RUN] {elapsed:.1f}min | chunks={chunks_written}")
                for sym in symbols:
                    snap = collector.snapshot(sym)
                    errs = collector.error_counts.get(sym, 0)
                    print(
                        f"  {sym:4s} bid={snap['bid1']:.2f} ask={snap['ask1']:.2f}"
                        f" spread={snap['spread']:.4f}"
                        f" imb={snap['book_imbalance']:+.2f}"
                        f" ofi1m={snap['ofi_norm_1m']:+.2f}"
                        f" trades={total_trades[sym]} quotes={total_quotes[sym]}"
                        f" err={errs}"
                    )
                last_status_ts = time.time()

            time.sleep(0.2)

        # Flush final
        now_et   = datetime.now(ET)
        date_str = chunk_ts.strftime("%Y-%m-%d")
        for sym in symbols:
            _write_chunk(output_dir, sym,
                         trade_rows[sym], quote_rows[sym],
                         chunk_label, date_str)
        chunks_written += 1

        print(f"\n[OK] Collecte terminee.")
        print(f"[OUT] Repertoire : {output_dir}/")
        print(f"[OUT] Chunks ecrits : {chunks_written}")
        print("\nResume final :")
        for sym in symbols:
            _, _, session = FUTURES.get(sym, ("CME", 1.0, f"sessions_{sym.lower()}"))
            snap = collector.snapshot(sym)
            print(
                f"  {sym:4s}  session={session}"
                f"  last={snap['last_price']:.2f}"
                f"  spread={snap['spread']:.4f}"
                f"  trades={total_trades[sym]}"
                f"  quotes={total_quotes[sym]}"
            )
        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur.")
        # Flush des buffers en cours
        now_et   = datetime.now(ET)
        date_str = chunk_ts.strftime("%Y-%m-%d")
        for sym in symbols:
            _write_chunk(output_dir, sym,
                         trade_rows[sym], quote_rows[sym],
                         chunk_label, date_str)
        return 130
    finally:
        collector.stop()


if __name__ == "__main__":
    raise SystemExit(main())
