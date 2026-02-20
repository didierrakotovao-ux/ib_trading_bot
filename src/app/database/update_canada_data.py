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
import sqlite3
from datetime import datetime, timedelta
import time
import sys
import os
import argparse


class CanadaDataUpdater:
    """Mise à jour incrémentale des données canadiennes."""

    def __init__(self, db_path: str = "trading_data_ca.db"):
        if os.path.isabs(db_path):
            self.db_path = db_path
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
            self.db_path = os.path.join(project_root, db_path)

    def get_symbols(self) -> list:
        """Récupère tous les symboles en base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols

    def get_last_global_date(self) -> str:
        """Récupère la dernière date globale en base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM historical_data")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else None

    def update_batch(self, symbols: list, start_date: str, end_date: str) -> int:
        """
        Met à jour un lot de symboles via yf.download() batch.

        Args:
            symbols: Liste de symboles
            start_date: Date de début YYYY-MM-DD
            end_date: Date de fin YYYY-MM-DD

        Returns:
            Nombre de barres insérées
        """
        if not symbols:
            return 0

        try:
            # Téléchargement batch (beaucoup plus rapide que un par un)
            df = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=True
            )

            if df.empty:
                return 0

            conn = sqlite3.connect(self.db_path)
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
            print(f"[ERROR] Batch download: {e}")
            return 0

    def _insert_single(self, conn: sqlite3.Connection, df: pd.DataFrame,
                       symbol: str) -> int:
        """Insère les données d'un symbole en base."""
        # Gérer le MultiIndex résiduel
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adjusted_close'
        })

        df = df.reset_index()
        df = df.rename(columns={'Date': 'date'})

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

        cursor = conn.cursor()
        inserted = 0
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO historical_data
                    (symbol, date, open, high, low, close, volume, adjusted_close, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                if cursor.rowcount > 0:
                    inserted += 1
            except Exception:
                pass

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

        print(f"[UPDATE] {len(symbols)} symboles à mettre à jour")
        print(f"[UPDATE] Base: {self.db_path}")

        # Déterminer la date de début
        if full_sync:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            print(f"[UPDATE] Mode full: re-sync depuis {start_date}")
        else:
            last_date = self.get_last_global_date()
            if last_date:
                start_date = (datetime.strptime(last_date, '%Y-%m-%d')).strftime('%Y-%m-%d')
                print(f"[UPDATE] Mode incrémental: depuis {start_date}")
            else:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                print(f"[UPDATE] Pas de données, sync depuis {start_date}")

        end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date >= end_date:
            print("[UPDATE] Données déjà à jour.")
            return

        # Traiter par lots
        total_bars = 0
        n_batches = (len(symbols) + batch_size - 1) // batch_size

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1

            bars = self.update_batch(batch, start_date, end_date)
            total_bars += bars

            if bars > 0:
                print(f"  [{batch_num}/{n_batches}] +{bars} barres "
                      f"({batch[0]}...{batch[-1]})")
            else:
                print(f"  [{batch_num}/{n_batches}] Déjà à jour "
                      f"({batch[0]}...{batch[-1]})")

            # Pause entre les lots pour éviter le rate limiting
            if batch_num < n_batches:
                time.sleep(1)

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
