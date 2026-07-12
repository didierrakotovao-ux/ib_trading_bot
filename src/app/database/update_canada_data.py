"""
Mise à jour quotidienne incrémentale de trading_data_ca.db.
Télécharge les nouvelles barres pour tous les symboles canadiens en lots.

Usage:
    python src/app/database/update_canada_data.py              # Mise à jour quotidienne
    python src/app/database/update_canada_data.py --full       # Re-sync 30 derniers jours
    python src/app/database/update_canada_data.py RY.TO TD.TO  # Symboles spécifiques
    python src/app/database/update_canada_data.py --db other.db # DB alternative
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn_ca


class CanadaDataUpdater:
    """Mise à jour incrémentale des données canadiennes (schéma 'ca' PostgreSQL)."""

    def __init__(self, db_path=None):  # db_path ignoré — connexion via pg_config.py
        pass

    def get_symbols(self) -> list:
        """Récupère tous les symboles en base."""
        conn = get_conn_ca()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        symbols = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        return symbols

    def get_last_global_date(self) -> str:
        """Récupère la dernière date globale en base."""
        conn = get_conn_ca()
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM historical_data")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return str(result[0]) if result and result[0] else None

    def update_batch(self, symbols: list, start_date: str, end_date: str,
                     max_retries: int = 4) -> int:
        """
        Met à jour un lot de symboles via yf.download() batch.
        Retry automatique avec backoff exponentiel si rate-limited.

        Args:
            symbols:     Liste de symboles
            start_date:  Date de début YYYY-MM-DD
            end_date:    Date de fin YYYY-MM-DD
            max_retries: Nombre de tentatives max en cas de rate-limit

        Returns:
            Nombre de barres insérées
        """
        if not symbols:
            return 0

        wait = 30  # secondes d'attente initiale avant retry

        for attempt in range(1, max_retries + 1):
            try:
                # Téléchargement batch
                df = yf.download(
                    symbols,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    threads=False   # threads=False évite les race conditions sur rate-limit
                )

                if df.empty:
                    return 0

                conn = get_conn_ca()
                total_inserted = 0

                # Si un seul symbole, la structure est différente
                if len(symbols) == 1:
                    symbol = symbols[0]
                    inserted = self._insert_single(conn, df, symbol)
                    total_inserted += inserted
                else:
                    # Multi-symboles: colonnes MultiIndex (ticker, field)
                    for symbol in symbols:
                        try:
                            sym_df = df[symbol].copy()
                            if sym_df.empty or sym_df.dropna(how='all').empty:
                                continue
                            inserted = self._insert_single(conn, sym_df, symbol)
                            total_inserted += inserted
                        except (KeyError, Exception):
                            continue

                conn.commit()
                conn.close()
                return total_inserted

            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in
                                    ['too many requests', 'rate limit', '429', 'try after'])
                if is_rate_limit and attempt < max_retries:
                    print(f"  [RATE-LIMIT] Attente {wait}s avant retry "
                          f"(tentative {attempt}/{max_retries})...")
                    time.sleep(wait)
                    wait *= 2   # backoff exponentiel : 30s → 60s → 120s → 240s
                else:
                    if not is_rate_limit:
                        print(f"  [ERROR] Batch {symbols[0]}…{symbols[-1]}: {e}")
                    else:
                        print(f"  [RATE-LIMIT] Lot ignoré après {max_retries} tentatives "
                              f"({symbols[0]}…{symbols[-1]})")
                    return 0

        return 0

    def _insert_single(self, conn, df: pd.DataFrame, symbol: str) -> int:
        """Insère les données d'un symbole en base."""
        # Gérer le MultiIndex résiduel (yfinance 1.4.0 retourne toujours MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            price_types = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
            level0 = set(df.columns.get_level_values(0))
            if level0 & price_types:
                # level 0 = champs OHLCV → garder level 0
                df.columns = df.columns.get_level_values(0)
            else:
                # level 0 = ticker → les champs sont en level 1
                df.columns = df.columns.get_level_values(1)

        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjusted_close'
        })

        # Fix yfinance 1.4.0 : index.name = None (plus 'Date')
        # → forcer le nom avant reset_index() pour que la colonne s'appelle 'date'
        df.index.name = 'date'
        df = df.reset_index()

        if 'date' not in df.columns:
            return 0

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Filtrer données invalides
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                return 0
        df = df.dropna(subset=required)
        df = df[df['volume'] > 0]

        if df.empty:
            return 0

        cur = conn.cursor()
        inserted = 0
        for _, row in df.iterrows():
            try:
                cur.execute("""
                    INSERT INTO historical_data
                    (symbol, date, open, high, low, close, volume, adjusted_close, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, date, source) DO NOTHING
                """, (
                    symbol,
                    row['date'],
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    float(row.get('adjusted_close', row['close'])),
                    'yfinance'
                ))
                if cur.rowcount > 0:
                    inserted += 1
            except Exception:
                pass

        cur.close()
        return inserted

    def run(self, symbols: list = None, full_sync: bool = False,
            batch_size: int = 50):
        """
        Lance la mise à jour.

        Args:
            symbols: Symboles spécifiques (None = tous)
            full_sync: Re-télécharger les 30 derniers jours
            batch_size: Taille des lots pour yf.download()
        """
        if symbols is None:
            symbols = self.get_symbols()

        print(f"[UPDATE] {len(symbols)} symboles à vérifier...")
        print("[UPDATE] Base: PostgreSQL schema 'ca'")

        today    = datetime.now().strftime('%Y-%m-%d')
        end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        if full_sync:
            # Mode full : tous les symboles depuis 30 jours
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"[UPDATE] Mode full: re-sync depuis {start_date}")
            to_update = {sym: start_date for sym in symbols}
        else:
            # Charger TOUTES les dernières dates en 1 requête (plus efficace)
            conn = get_conn_ca()
            cur  = conn.cursor()
            cur.execute("SELECT symbol, MAX(date) FROM historical_data GROUP BY symbol")
            last_dates = {row[0]: str(row[1]) for row in cur.fetchall()}
            cur.close()
            conn.close()

            # Filtrer par symbole : ignorer ceux déjà à jour
            to_update = {}
            default_start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            for sym in symbols:
                ld = last_dates.get(sym)
                if ld and ld >= today:
                    continue  # déjà à jour
                if ld:
                    start = (datetime.strptime(ld, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start = default_start
                if start < end_date:
                    to_update[sym] = start

            already_up = len(symbols) - len(to_update)
            print(f"[UPDATE] {already_up:,} déjà à jour | {len(to_update):,} à télécharger")

        if not to_update:
            print("[UPDATE] Tout est à jour.")
            return

        # Grouper par date de départ pour optimiser les lots
        from collections import defaultdict
        by_start: dict = defaultdict(list)
        for sym, start in to_update.items():
            by_start[start].append(sym)

        total_bars  = 0
        batch_count = 0
        total_batches = sum(
            (len(grp) + batch_size - 1) // batch_size
            for grp in by_start.values()
        )

        for start_date, sym_group in sorted(by_start.items()):
            for i in range(0, len(sym_group), batch_size):
                batch = sym_group[i:i + batch_size]
                batch_count += 1

                bars = self.update_batch(batch, start_date, end_date)
                total_bars += bars

                if bars > 0:
                    print(f"  [{batch_count}/{total_batches}] +{bars} barres "
                          f"({batch[0]}...{batch[-1]}) depuis {start_date}")
                else:
                    print(f"  [{batch_count}/{total_batches}] Aucune nouvelle barre "
                          f"({batch[0]}...{batch[-1]})")

                # Pause anti rate-limit entre les lots (Yahoo Finance : ~2 req/s max)
                if batch_count < total_batches:
                    time.sleep(2)

        print(f"\n[UPDATE] Terminé: {total_bars:,} nouvelles barres ajoutées")


def main():
    parser = argparse.ArgumentParser(
        description="Mise à jour quotidienne des données canadiennes"
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles spécifiques (optionnel)')
    parser.add_argument('--full', '-f', action='store_true',
                        help='Re-sync les 30 derniers jours')
    parser.add_argument('--db', default='trading_data_ca.db',
                        help='Chemin de la base de données (default: trading_data_ca.db)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Taille des lots (default: 50)')

    args = parser.parse_args()

    updater = CanadaDataUpdater(args.db)
    updater.run(
        symbols=args.symbols if args.symbols else None,
        full_sync=args.full,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
