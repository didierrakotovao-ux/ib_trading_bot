"""
Collecte L2 (market depth) data-only pour une watchlist NASDAQ.

Aucun ordre n'est envoye. Ce script sert a enregistrer le carnet et un OFI simple
pour analyse intraday (scalping / timing de prise de position).

Exemples:
    python src/app/collect_l2_watchlist.py
    python src/app/collect_l2_watchlist.py --duration-min 90 --client 97
    python src/app/collect_l2_watchlist.py --gateway --exchange ISLAND --levels 5
    python src/app/collect_l2_watchlist.py --symbols AAPL,MSFT,NVDA,AMD
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "TSLA", "AMD", "AVGO", "QCOM", "INTC", "MU",
]


@dataclass
class BookEntry:
    price: float
    size: float


class L2Collector(EWrapper, EClient):
    def __init__(self) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, self)

        self.ready = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.req_to_symbol: Dict[int, str] = {}
        self.symbol_to_req: Dict[str, int] = {}
        self.books: Dict[str, Dict[str, Dict[int, BookEntry]]] = {}
        self.update_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_error: Dict[str, str] = {}

        self._lock = threading.Lock()

    def nextValidId(self, orderId: int) -> None:
        self.ready.set()

    def error(self, reqId, _errorTime, errorCode, errorString, _advancedOrderRejectJson="") -> None:
        # 2104/2106/2158 : connexions data farms OK (informatif)
        # 2152 : avertissement venues non souscrites (IEX reçu quand même)
        SILENT_CODES = {2104, 2106, 2107, 2108, 2158, 2152}
        symbol = self.req_to_symbol.get(reqId, "GLOBAL")
        with self._lock:
            self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
            self.last_error[symbol] = f"{errorCode} - {errorString}"
        if errorCode not in SILENT_CODES:
            print(f"[IB ERROR] reqId={reqId} symbol={symbol} code={errorCode} msg={errorString}")

    def updateMktDepth(self, reqId, position, operation, side, price, size) -> None:
        self._apply_depth_update(reqId, position, operation, side, price, size)

    def updateMktDepthL2(self, reqId, position, marketMaker, operation, side, price, size, isSmartDepth) -> None:
        self._apply_depth_update(reqId, position, operation, side, price, size)

    def _apply_depth_update(self, reqId: int, position: int, operation: int, side: int, price: float, size: float) -> None:
        symbol = self.req_to_symbol.get(reqId)
        if not symbol:
            return

        side_key = "bids" if side == 1 else "asks"

        with self._lock:
            if symbol not in self.books:
                self.books[symbol] = {"bids": {}, "asks": {}}
            side_book = self.books[symbol][side_key]

            if operation == 2:
                side_book.pop(position, None)
            else:
                side_book[position] = BookEntry(price=price, size=float(size))

            self.update_counts[symbol] = self.update_counts.get(symbol, 0) + 1

    def connect_and_start(self, host: str, port: int, client_id: int, timeout: int = 10) -> bool:
        self.connect(host, port, client_id)
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

        if not self.ready.wait(timeout=timeout):
            return False
        return True

    def cancel_symbol(self, symbol: str, smart_depth: bool) -> None:
        with self._lock:
            req_id = self.symbol_to_req.pop(symbol, None)
            if req_id is not None:
                self.req_to_symbol.pop(req_id, None)
                self.books[symbol] = {"bids": {}, "asks": {}}
        if req_id is not None:
            try:
                self.cancelMktDepth(req_id, smart_depth)
            except Exception:
                pass

    def stop(self) -> None:
        try:
            for symbol, req_id in list(self.symbol_to_req.items()):
                try:
                    self.cancelMktDepth(req_id, False)
                except Exception:
                    pass
            time.sleep(0.2)
        finally:
            if self.isConnected():
                self.disconnect()

    def subscribe_symbol(self, symbol: str, req_id: int, levels: int, exchange: str, smart_depth: bool) -> None:
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = exchange

        with self._lock:
            self.req_to_symbol[req_id] = symbol
            self.symbol_to_req[symbol] = req_id
            self.books[symbol] = {"bids": {}, "asks": {}}
            self.update_counts[symbol] = 0
            self.error_counts[symbol] = 0
            self.last_error[symbol] = ""

        self.reqMktDepth(req_id, contract, levels, smart_depth, [])

    def snapshot_symbol(self, symbol: str, levels: int) -> Dict[str, float]:
        with self._lock:
            book = self.books.get(symbol, {"bids": {}, "asks": {}})
            bid_map = dict(book["bids"])
            ask_map = dict(book["asks"])
            updates = self.update_counts.get(symbol, 0)

        bids = [bid_map[p] for p in sorted(bid_map.keys())[:levels]]
        asks = [ask_map[p] for p in sorted(ask_map.keys())[:levels]]

        bid_px_1 = bids[0].price if bids else 0.0
        ask_px_1 = asks[0].price if asks else 0.0
        bid_sz_1 = bids[0].size if bids else 0.0
        ask_sz_1 = asks[0].size if asks else 0.0

        spread = 0.0
        if bid_px_1 > 0 and ask_px_1 > 0 and ask_px_1 >= bid_px_1:
            spread = ask_px_1 - bid_px_1

        bid_depth = sum(x.size for x in bids)
        ask_depth = sum(x.size for x in asks)
        ofi_raw = bid_depth - ask_depth
        denom = bid_depth + ask_depth
        ofi_norm = ofi_raw / denom if denom > 0 else 0.0

        return {
            "bid_px_1": bid_px_1,
            "ask_px_1": ask_px_1,
            "bid_sz_1": bid_sz_1,
            "ask_sz_1": ask_sz_1,
            "spread": spread,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "ofi_raw": ofi_raw,
            "ofi_norm": ofi_norm,
            "updates_total": float(updates),
        }


def parse_symbols(raw: str) -> List[str]:
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return symbols or DEFAULT_SYMBOLS


def build_output_paths(output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        output_dir / f"l2_ofi_samples_{stamp}.csv",
        output_dir / f"l2_ofi_summary_{stamp}.csv",
    )


def build_sqlite_path(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"l2_ofi_samples_{stamp}.db"


def build_parquet_path(output_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"l2_ofi_samples_{stamp}.parquet"


def init_sqlite_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            bid_px_1 REAL NOT NULL,
            bid_sz_1 REAL NOT NULL,
            ask_px_1 REAL NOT NULL,
            ask_sz_1 REAL NOT NULL,
            spread REAL NOT NULL,
            bid_depth REAL NOT NULL,
            ask_depth REAL NOT NULL,
            ofi_raw REAL NOT NULL,
            ofi_norm REAL NOT NULL,
            updates_total INTEGER NOT NULL,
            updates_delta INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_symbol_ts ON samples(symbol, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts)")
    conn.commit()
    return conn


def export_parquet(sqlite_path: Optional[Path], samples_path: Path, parquet_path: Path) -> bool:
    try:
        import pandas as pd
    except Exception as e:
        print(f"[WARN] Export Parquet impossible (pandas indisponible): {e}")
        return False

    try:
        if sqlite_path is not None and sqlite_path.exists():
            with sqlite3.connect(str(sqlite_path)) as conn:
                df = pd.read_sql_query(
                    """
                    SELECT
                        ts, symbol, bid_px_1, bid_sz_1, ask_px_1, ask_sz_1,
                        spread, bid_depth, ask_depth, ofi_raw, ofi_norm,
                        updates_total, updates_delta
                    FROM samples
                    ORDER BY ts, symbol
                    """,
                    conn,
                )
        else:
            df = pd.read_csv(samples_path)

        df.to_parquet(parquet_path, index=False)
        return True
    except Exception as e:
        print(f"[WARN] Export Parquet impossible (pyarrow/fastparquet requis): {e}")
        return False


def print_run_header(args: argparse.Namespace, port: int, symbols: List[str]) -> None:
    n_groups = max(1, (len(symbols) + args.group_size - 1) // args.group_size)
    cycle_sec = n_groups * args.rotation_sec
    print("\n" + "=" * 68)
    print("L2 WATCHLIST COLLECTOR (DATA-ONLY)")
    print("=" * 68)
    print(f"Host/Port      : {args.host}:{port}")
    print(f"Client ID      : {args.client}")
    print(f"Exchange       : {args.exchange}")
    print(f"Smart depth    : {args.smart_depth}")
    print(f"Niveaux L2     : {args.levels}")
    print(f"Sample every   : {args.sample_sec:.2f}s")
    print(f"Duration       : {args.duration_min} min")
    print(f"Symbols ({len(symbols)}): {', '.join(symbols)}")
    if args.smart_depth and len(symbols) > args.group_size:
        print(f"Rotation       : {n_groups} groupes x {args.group_size} symboles, {args.rotation_sec:.1f}s/groupe (cycle={cycle_sec:.0f}s)")
    print("=" * 68 + "\n")


def write_summary(
    summary_path: Path,
    symbols: List[str],
    per_symbol_rows: Dict[str, int],
    spread_acc: Dict[str, float],
    ofi_acc: Dict[str, float],
    updates_first: Dict[str, int],
    updates_last: Dict[str, int],
    elapsed_sec: float,
    collector: L2Collector,
) -> None:
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol",
            "rows",
            "avg_spread",
            "avg_abs_ofi_norm",
            "updates_per_sec",
            "ib_errors",
            "last_error",
        ])

        for s in symbols:
            rows = per_symbol_rows.get(s, 0)
            avg_spread = spread_acc.get(s, 0.0) / rows if rows else 0.0
            avg_abs_ofi = ofi_acc.get(s, 0.0) / rows if rows else 0.0

            delta_updates = max(0, updates_last.get(s, 0) - updates_first.get(s, 0))
            upd_rate = (delta_updates / elapsed_sec) if elapsed_sec > 0 else 0.0

            ib_errors = collector.error_counts.get(s, 0)
            last_error = collector.last_error.get(s, "")

            w.writerow([
                s,
                rows,
                f"{avg_spread:.6f}",
                f"{avg_abs_ofi:.6f}",
                f"{upd_rate:.3f}",
                ib_errors,
                last_error,
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Collecte L2 NASDAQ data-only pour watchlist OFI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Adresse IP de TWS/Gateway (ex: 192.168.0.103)")
    parser.add_argument("--live", action="store_true", help="Live trading ports (7496/4001)")
    parser.add_argument("--gateway", action="store_true", help="Utiliser Gateway (4002/4001) plutot que TWS (7497/7496)")
    parser.add_argument("--port", type=int, default=None, help="Override explicite du port")
    parser.add_argument("--client", type=int, default=97, help="Client ID IB dedie (evite conflit bot live)")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="Liste CSV de symboles")
    parser.add_argument("--duration-min", type=int, default=90, help="Duree de collecte en minutes")
    parser.add_argument("--sample-sec", type=float, default=1.0, help="Frequence d'echantillonnage en secondes")
    parser.add_argument("--levels", type=int, default=5, help="Nombre de niveaux L2 demandes")
    parser.add_argument("--exchange", type=str, default="ISLAND", help="Exchange du contrat (ex: ISLAND, NASDAQ, SMART)")
    parser.add_argument("--smart-depth", action="store_true", help="Activer smart depth (requis pour IEX, limite a 3 souscriptions simultanées)")
    parser.add_argument("--group-size", type=int, default=3, help="Symboles par groupe en mode rotation (defaut: 3, limite IEX avec --smart-depth)")
    parser.add_argument("--rotation-sec", type=float, default=5.0, help="Secondes par groupe avant rotation (defaut: 5.0)")
    parser.add_argument("--output-dir", type=str, default="logs/l2", help="Dossier de sortie CSV")
    parser.add_argument("--save-sqlite", action="store_true", help="Sauvegarde aussi les samples dans SQLite indexe (lecture plus rapide)")
    parser.add_argument("--sqlite-commit-every", type=int, default=30, help="Frequence de commit SQLite en nombre de boucles")
    parser.add_argument("--save-parquet", action="store_true", help="Exporte un fichier Parquet en fin de session")
    args = parser.parse_args()

    if args.duration_min <= 0:
        print("[ERREUR] --duration-min doit etre > 0")
        return 1
    if args.sample_sec <= 0:
        print("[ERREUR] --sample-sec doit etre > 0")
        return 1
    if args.levels <= 0:
        print("[ERREUR] --levels doit etre > 0")
        return 1
    if args.sqlite_commit_every <= 0:
        print("[ERREUR] --sqlite-commit-every doit etre > 0")
        return 1

    symbols = parse_symbols(args.symbols)

    if args.port is not None:
        port = args.port
    elif args.gateway:
        port = 4001 if args.live else 4002
    else:
        port = 7496 if args.live else 7497

    output_dir = Path(args.output_dir)
    samples_path, summary_path = build_output_paths(output_dir)
    parquet_path: Optional[Path] = None
    sqlite_path: Optional[Path] = None
    sqlite_conn: Optional[sqlite3.Connection] = None

    if args.save_sqlite:
        sqlite_path = build_sqlite_path(output_dir)
        sqlite_conn = init_sqlite_db(sqlite_path)

    if args.save_parquet:
        parquet_path = build_parquet_path(output_dir)

    print_run_header(args, port, symbols)

    collector = L2Collector()
    connected = collector.connect_and_start(host=args.host, port=port, client_id=args.client, timeout=12)
    if not connected:
        print("[ERREUR] Impossible de se connecter a IB ou nextValidId non recu.")
        return 1

    print("[OK] Connecte a IB, souscription L2 en cours...")

    # Décide si on utilise la rotation (smart_depth avec plus de symboles que group_size)
    use_rotation = args.smart_depth and len(symbols) > args.group_size

    # Découpe les symboles en groupes
    groups: List[List[str]] = []
    if use_rotation:
        for i in range(0, len(symbols), args.group_size):
            groups.append(symbols[i:i + args.group_size])
    else:
        groups = [symbols]

    headers = [
        "ts", "symbol",
        "bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1",
        "spread", "bid_depth", "ask_depth", "ofi_raw", "ofi_norm",
        "updates_total", "updates_delta",
    ]

    per_symbol_rows: Dict[str, int] = {s: 0 for s in symbols}
    spread_acc: Dict[str, float] = {s: 0.0 for s in symbols}
    ofi_acc: Dict[str, float] = {s: 0.0 for s in symbols}
    updates_prev: Dict[str, int] = {s: 0 for s in symbols}
    updates_first: Dict[str, int] = {}
    updates_last: Dict[str, int] = {s: 0 for s in symbols}

    start_ts = time.time()
    end_ts = start_ts + (args.duration_min * 60)

    req_counter = 2000
    current_group_idx = 0
    active_symbols: Set[str] = set()

    def subscribe_group(group: List[str]) -> None:
        nonlocal req_counter
        for symbol in group:
            collector.subscribe_symbol(
                symbol=symbol,
                req_id=req_counter,
                levels=args.levels,
                exchange=args.exchange,
                smart_depth=args.smart_depth,
            )
            active_symbols.add(symbol)
            req_counter += 1
            time.sleep(0.05)

    def cancel_group(group: List[str]) -> None:
        for symbol in group:
            collector.cancel_symbol(symbol, args.smart_depth)
            active_symbols.discard(symbol)
        time.sleep(0.2)

    try:
        # Souscription du premier groupe
        subscribe_group(groups[current_group_idx])
        time.sleep(1.0)  # Warmup initial

        with samples_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            print(f"[RUN] Ecriture des echantillons dans: {samples_path}")
            if sqlite_path is not None:
                print(f"[RUN] Ecriture SQLite indexee dans: {sqlite_path}")
            if use_rotation:
                print(f"[RUN] Mode rotation: {len(groups)} groupes, {args.rotation_sec:.1f}s/groupe")

            loop_idx = 0
            rotation_start = time.time()

            while time.time() < end_ts:
                now = time.time()

                # Rotation : passer au groupe suivant si le temps est écoulé
                if use_rotation and (now - rotation_start) >= args.rotation_sec:
                    cancel_group(groups[current_group_idx])
                    current_group_idx = (current_group_idx + 1) % len(groups)
                    subscribe_group(groups[current_group_idx])
                    time.sleep(0.5)  # Laisser le temps au nouveau groupe d'arriver
                    rotation_start = time.time()
                    print(f"[ROT] Groupe {current_group_idx + 1}/{len(groups)}: {groups[current_group_idx]}")

                ts_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                sqlite_batch: List[Tuple[object, ...]] = []

                for s in list(active_symbols):
                    snap = collector.snapshot_symbol(s, levels=args.levels)
                    updates_now = int(snap["updates_total"])
                    updates_delta = updates_now - updates_prev[s]
                    updates_prev[s] = updates_now
                    updates_last[s] = updates_now
                    if s not in updates_first:
                        updates_first[s] = updates_now

                    writer.writerow([
                        ts_utc, s,
                        f"{snap['bid_px_1']:.6f}", f"{snap['bid_sz_1']:.2f}",
                        f"{snap['ask_px_1']:.6f}", f"{snap['ask_sz_1']:.2f}",
                        f"{snap['spread']:.6f}",
                        f"{snap['bid_depth']:.2f}", f"{snap['ask_depth']:.2f}",
                        f"{snap['ofi_raw']:.2f}", f"{snap['ofi_norm']:.6f}",
                        updates_now, updates_delta,
                    ])

                    if sqlite_conn is not None:
                        sqlite_batch.append((
                            ts_utc, s,
                            float(snap["bid_px_1"]), float(snap["bid_sz_1"]),
                            float(snap["ask_px_1"]), float(snap["ask_sz_1"]),
                            float(snap["spread"]),
                            float(snap["bid_depth"]), float(snap["ask_depth"]),
                            float(snap["ofi_raw"]), float(snap["ofi_norm"]),
                            int(updates_now), int(updates_delta),
                        ))

                    per_symbol_rows[s] += 1
                    spread_acc[s] += snap["spread"]
                    ofi_acc[s] += abs(snap["ofi_norm"])

                f.flush()

                if sqlite_conn is not None and sqlite_batch:
                    sqlite_conn.executemany(
                        """
                        INSERT INTO samples (
                            ts, symbol, bid_px_1, bid_sz_1, ask_px_1, ask_sz_1, spread,
                            bid_depth, ask_depth, ofi_raw, ofi_norm, updates_total, updates_delta
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_batch,
                    )

                loop_idx += 1
                if sqlite_conn is not None and (loop_idx % args.sqlite_commit_every == 0):
                    sqlite_conn.commit()
                if loop_idx % 30 == 0:
                    elapsed = time.time() - start_ts
                    print(f"[RUN] {elapsed/60:.1f} min ecoulees | samples={loop_idx}")

                time.sleep(args.sample_sec)

        elapsed = max(1e-6, time.time() - start_ts)
        write_summary(
            summary_path=summary_path,
            symbols=symbols,
            per_symbol_rows=per_symbol_rows,
            spread_acc=spread_acc,
            ofi_acc=ofi_acc,
            updates_first=updates_first,
            updates_last=updates_last,
            elapsed_sec=elapsed,
            collector=collector,
        )

        print("\n[OK] Collecte terminee")
        print(f"[OUT] Samples: {samples_path}")
        print(f"[OUT] Summary: {summary_path}\n")
        if sqlite_path is not None:
            if sqlite_conn is not None:
                sqlite_conn.commit()
            print(f"[OUT] SQLite: {sqlite_path}\n")

        if parquet_path is not None:
            if sqlite_conn is not None:
                sqlite_conn.commit()
            if export_parquet(sqlite_path=sqlite_path, samples_path=samples_path, parquet_path=parquet_path):
                print(f"[OUT] Parquet: {parquet_path}\n")

        print("Resume rapide (approx):")
        for s in symbols:
            rows = per_symbol_rows[s]
            avg_spread = spread_acc[s] / rows if rows else 0.0
            avg_abs_ofi = ofi_acc[s] / rows if rows else 0.0
            delta_updates = max(0, updates_last.get(s, 0) - updates_first.get(s, 0))
            upd_rate = delta_updates / elapsed
            print(
                f"  {s:5s} rows={rows:5d} avg_spread={avg_spread:9.6f} "
                f"avg|ofi|={avg_abs_ofi:8.5f} upd/s={upd_rate:7.2f} "
                f"err={collector.error_counts.get(s, 0)}"
            )

        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Interruption utilisateur.")
        return 130
    finally:
        if sqlite_conn is not None:
            sqlite_conn.close()
        collector.stop()


if __name__ == "__main__":
    raise SystemExit(main())
