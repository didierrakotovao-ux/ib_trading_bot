"""
Script de mise à jour incrémentale des données historiques.
Récupère uniquement les nouvelles barres depuis la dernière date en base.

Usage:
    python update_historical_data.py              # Met à jour tous les symboles
    python update_historical_data.py AAPL MSFT    # Met à jour des symboles spécifiques
    python update_historical_data.py --compute    # Met à jour et recalcule les features

À exécuter chaque matin avant le trading.
"""
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


class HistoricalDataUpdater:
    """Met à jour les données historiques de façon incrémentale."""

    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path

    def get_symbols_to_update(self) -> list:
        """Récupère tous les symboles de la base de données."""
        conn = sqlite3.connect(self.db_path, timeout=60)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
        symbols = [row[0] for row in cursor.fetchall()]
        conn.close()
        return symbols

    def get_last_date(self, symbol: str) -> str:
        """Récupère la dernière date en base pour un symbole."""
        conn = sqlite3.connect(self.db_path, timeout=60)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(date) FROM historical_data WHERE symbol = ?
        """, (symbol,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else None

    def update_symbol(self, symbol: str) -> int:
        """
        Met à jour les données pour un symbole.

        Returns:
            Nombre de nouvelles barres ajoutées
        """
        last_date = self.get_last_date(symbol)

        if last_date:
            # Commencer le lendemain de la dernière date
            start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Pas de données, télécharger 2 ans
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        end_date = datetime.now().strftime('%Y-%m-%d')

        # Vérifier si mise à jour nécessaire
        if start_date >= end_date:
            return 0

        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty:
                return 0

            # Gérer le MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Renommer les colonnes
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adjusted_close'
            })

            # Réinitialiser l'index pour avoir 'date' comme colonne
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            df['source'] = 'yfinance'

            # Insérer en base
            conn = sqlite3.connect(self.db_path, timeout=60)
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
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        int(row['volume']),
                        row.get('adjusted_close', row['close']),
                        'yfinance'
                    ))
                    if cursor.rowcount > 0:
                        inserted += 1
                except Exception as e:
                    pass  # Ignorer les doublons

            conn.commit()
            conn.close()

            return inserted

        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            return 0

    def update_all(self, symbols: list = None, verbose: bool = True):
        """
        Met à jour tous les symboles.

        Args:
            symbols: Liste de symboles (optionnel, sinon tous)
            verbose: Afficher les détails
        """
        if symbols is None:
            symbols = self.get_symbols_to_update()

        print(f"[UPDATE] Mise à jour de {len(symbols)} symboles...")

        total_new = 0
        updated_symbols = 0

        for i, symbol in enumerate(symbols):
            new_bars = self.update_symbol(symbol)

            if new_bars > 0:
                total_new += new_bars
                updated_symbols += 1
                if verbose:
                    print(f"  {symbol}: +{new_bars} barres")

            if (i + 1) % 50 == 0:
                print(f"[UPDATE] {i + 1}/{len(symbols)} symboles vérifiés...")

        print(f"[UPDATE] Terminé: {updated_symbols} symboles mis à jour, "
              f"{total_new} nouvelles barres")

        return total_new


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Mise à jour incrémentale des données historiques"
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles spécifiques (optionnel)')
    parser.add_argument('--compute', '-c', action='store_true',
                        help='Recalculer les features après mise à jour')
    parser.add_argument('--db', default='trading_data.db',
                        help='Chemin de la base de données')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Mode silencieux')

    args = parser.parse_args()

    updater = HistoricalDataUpdater(args.db)

    symbols = args.symbols if args.symbols else None
    new_bars = updater.update_all(symbols=symbols, verbose=not args.quiet)

    # Recalculer les features si demandé et s'il y a de nouvelles données
    if args.compute and new_bars > 0:
        print("\n[COMPUTE] Recalcul des features...")
        from compute_features import FeatureComputer
        computer = FeatureComputer(args.db)
        computer.compute_all(symbols=symbols, incremental=True)


if __name__ == "__main__":
    main()
