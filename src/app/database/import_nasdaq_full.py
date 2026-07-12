"""
Import complet des données NASDAQ depuis 2008.
Récupère la liste officielle des symboles NASDAQ et télécharge l'historique complet.

Usage:
    python import_nasdaq_full.py                    # Import complet
    python import_nasdaq_full.py --min-volume 100000  # Filtre volume minimum
    python import_nasdaq_full.py --resume           # Reprendre un import interrompu

Sources des symboles:
- NASDAQ FTP officiel: ftp://ftp.nasdaqtrader.com/symboldirectory/
- DataHub: https://datahub.io/core/nasdaq-listings
"""
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from io import StringIO
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn

# Date de début pour l'historique
START_DATE = "2008-01-01"


class NasdaqImporter:
    """Import complet des données NASDAQ."""

    def __init__(self, db_path=None):  # db_path ignoré — connexion via pg_config.py
        self.symbols = []

    def fetch_nasdaq_symbols(self) -> list:
        """
        Récupère la liste des symboles NASDAQ depuis le FTP officiel.

        Returns:
            Liste des symboles
        """
        print("[FETCH] Récupération de la liste NASDAQ...")

        # URL du FTP NASDAQ (fichier nasdaqlisted.txt)
        nasdaq_url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"

        try:
            response = requests.get(nasdaq_url, timeout=30)
            response.raise_for_status()

            # Parser le fichier (format pipe-delimited)
            lines = response.text.strip().split('\n')
            # Première ligne = header, dernière ligne = timestamp
            header = lines[0].split('|')
            data_lines = lines[1:-1]  # Exclure le footer

            symbols = []
            for line in data_lines:
                parts = line.split('|')
                if len(parts) >= 2:
                    symbol = parts[0].strip()
                    # Exclure les symboles avec caractères spéciaux (warrants, etc.)
                    if symbol and not any(c in symbol for c in [' ', '$', '.', '^']):
                        symbols.append(symbol)

            print(f"[FETCH] {len(symbols)} symboles NASDAQ récupérés")
            return symbols

        except Exception as e:
            print(f"[ERROR] Impossible de récupérer la liste NASDAQ: {e}")
            print("[INFO] Utilisation de la liste de secours...")
            return self._get_fallback_symbols()

    def fetch_other_listed_symbols(self) -> list:
        """
        Récupère aussi les symboles des autres exchanges (NYSE, etc.) listés sur NASDAQ.

        Returns:
            Liste des symboles
        """
        print("[FETCH] Récupération des autres symboles listés...")

        other_url = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

        try:
            response = requests.get(other_url, timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            data_lines = lines[1:-1]

            symbols = []
            for line in data_lines:
                parts = line.split('|')
                if len(parts) >= 1:
                    symbol = parts[0].strip()
                    if symbol and not any(c in symbol for c in [' ', '$', '.', '^']):
                        symbols.append(symbol)

            print(f"[FETCH] {len(symbols)} autres symboles récupérés")
            return symbols

        except Exception as e:
            print(f"[WARN] Impossible de récupérer les autres symboles: {e}")
            return []

    def _get_fallback_symbols(self) -> list:
        """Liste de secours si le FTP ne répond pas."""
        # Top 500 NASDAQ/NYSE par capitalisation (approximatif)
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH',
            'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV',
            'AVGO', 'PEP', 'COST', 'ADBE', 'CRM', 'CSCO', 'ACN', 'NFLX', 'TMO', 'AMD',
            'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS',
            'BAC', 'WFC', 'MS', 'GS', 'C', 'SCHW', 'AXP', 'BLK', 'SPGI', 'CB',
            'LLY', 'PFE', 'ABT', 'DHR', 'AMGN', 'BMY', 'GILD', 'VRTX', 'ISRG', 'REGN',
            'WMT', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG', 'CMG', 'MAR', 'ABNB',
            'CAT', 'BA', 'HON', 'UNP', 'RTX', 'GE', 'LMT', 'UPS', 'DE', 'FDX',
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'PLD', 'AMT', 'CCI', 'EQIX', 'PSA',
            'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'DOW', 'DD', 'PPG',
            # Mid caps populaires
            'PLTR', 'HOOD', 'SOFI', 'RKLB', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
            'COIN', 'MARA', 'RIOT', 'CLSK', 'CIFR', 'WULF', 'IREN', 'APLD', 'HIVE',
            'DKNG', 'PENN', 'CRWD', 'ZS', 'NET', 'DDOG', 'SNOW', 'MDB', 'TEAM',
            'SQ', 'PYPL', 'SHOP', 'ROKU', 'SPOT', 'UBER', 'LYFT', 'DASH', 'ABNB',
        ]

    def get_already_imported(self) -> set:
        """Récupère les symboles déjà présents en base avec données suffisantes."""
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol FROM (
                    SELECT symbol, COUNT(*) as cnt FROM historical_data GROUP BY symbol
                ) sub WHERE cnt >= 1000
            """)
            symbols = {row[0] for row in cur.fetchall()}
            cur.close()
            conn.close()
            return symbols
        except Exception:
            return set()

    def import_symbol(self, symbol: str, start_date: str = START_DATE) -> int:
        """
        Importe les données historiques pour un symbole.

        Args:
            symbol: Symbole à importer
            start_date: Date de début

        Returns:
            Nombre de barres importées
        """
        end_date = datetime.now().strftime('%Y-%m-%d')

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

            # Réinitialiser l'index
            df = df.reset_index()
            df = df.rename(columns={'Date': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['symbol'] = symbol
            df['source'] = 'yfinance'

            # Filtrer les données invalides
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            df = df[df['volume'] > 0]

            if len(df) == 0:
                return 0

            # Insérer en base
            conn = get_conn()
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

            conn.commit()
            cur.close()
            conn.close()

            return inserted

        except Exception as e:
            return 0

    def run(self, include_other: bool = True, resume: bool = False,
            min_volume: int = 0, batch_size: int = 100):
        """
        Lance l'import complet.

        Args:
            include_other: Inclure les symboles non-NASDAQ (NYSE, etc.)
            resume: Reprendre un import interrompu (skip symboles existants)
            min_volume: Volume minimum pour filtrer
            batch_size: Taille des lots pour les pauses
        """
        # Récupérer les symboles
        symbols = self.fetch_nasdaq_symbols()

        if include_other:
            other_symbols = self.fetch_other_listed_symbols()
            symbols = list(set(symbols + other_symbols))

        symbols.sort()
        print(f"[IMPORT] Total: {len(symbols)} symboles à traiter")

        # Si resume, exclure les symboles déjà importés
        if resume:
            already_done = self.get_already_imported()
            symbols = [s for s in symbols if s not in already_done]
            print(f"[IMPORT] {len(symbols)} symboles restants après exclusion")

        # Import
        total_bars = 0
        success = 0
        errors = 0

        for i, symbol in enumerate(symbols):
            bars = self.import_symbol(symbol)

            if bars > 0:
                total_bars += bars
                success += 1
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {bars} barres")
            else:
                errors += 1

            # Pause tous les N symboles pour éviter le rate limiting
            if (i + 1) % batch_size == 0:
                print(f"[IMPORT] {i+1}/{len(symbols)} traités, "
                      f"{total_bars} barres, pause 5s...")
                time.sleep(5)

            # Petite pause entre chaque requête
            time.sleep(0.2)

        print(f"\n[IMPORT] Terminé!")
        print(f"  - Symboles importés: {success}")
        print(f"  - Symboles en erreur: {errors}")
        print(f"  - Total barres: {total_bars:,}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import complet des données NASDAQ depuis 2008"
    )
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Reprendre un import interrompu')
    parser.add_argument('--nasdaq-only', action='store_true',
                        help='Importer seulement NASDAQ (pas NYSE, etc.)')
    parser.add_argument('--min-volume', type=int, default=0,
                        help='Volume minimum moyen')
    parser.add_argument('--db', default='trading_data.db',
                        help='Chemin de la base de données')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Taille des lots pour les pauses')

    args = parser.parse_args()

    importer = NasdaqImporter(args.db)
    importer.run(
        include_other=not args.nasdaq_only,
        resume=args.resume,
        min_volume=args.min_volume,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
