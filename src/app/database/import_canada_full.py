"""
Import complet des données canadiennes (TSX, TSXV, NEO, CSE) depuis 2008.
Récupère la liste des symboles via le screener Yahoo Finance et télécharge l'historique.

Usage:
    python import_canada_full.py                    # Import complet
    python import_canada_full.py --resume           # Reprendre un import interrompu
    python import_canada_full.py --db other.db      # DB alternative
    python import_canada_full.py --batch-size 50    # Lots plus petits
"""
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn_ca, init_ca_schema

# Date de début pour l'historique
START_DATE = "2008-01-01"


class CanadaImporter:
    """Import complet des données canadiennes."""

    def __init__(self, db_path=None):  # db_path ignoré — schéma 'ca' dans PostgreSQL
        self.symbols = []
        self._ensure_table()

    def _ensure_table(self):
        """Crée le schéma 'ca' et la table historical_data si ils n'existent pas."""
        init_ca_schema()
        conn = get_conn_ca()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_data (
                id             SERIAL PRIMARY KEY,
                symbol         TEXT NOT NULL,
                date           DATE NOT NULL,
                open           DOUBLE PRECISION NOT NULL,
                high           DOUBLE PRECISION NOT NULL,
                low            DOUBLE PRECISION NOT NULL,
                close          DOUBLE PRECISION NOT NULL,
                volume         BIGINT NOT NULL,
                adjusted_close DOUBLE PRECISION,
                dividends      DOUBLE PRECISION DEFAULT 0,
                stock_splits   DOUBLE PRECISION DEFAULT 0,
                source         TEXT NOT NULL DEFAULT 'yfinance',
                sma20_volume   DOUBLE PRECISION DEFAULT 0,
                hl_sma20vol    DOUBLE PRECISION DEFAULT 0,
                oc_sma20vol    DOUBLE PRECISION DEFAULT 0,
                macd           DOUBLE PRECISION DEFAULT 0,
                macd_signal    DOUBLE PRECISION DEFAULT 0,
                rsi            DOUBLE PRECISION DEFAULT 0,
                adx            DOUBLE PRECISION DEFAULT 0,
                bb_high        DOUBLE PRECISION DEFAULT 0,
                bb_low         DOUBLE PRECISION DEFAULT 0,
                pct_close      DOUBLE PRECISION DEFAULT 0,
                timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date, source)
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_symbol_date
            ON historical_data(symbol, date DESC)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_date
            ON historical_data(date DESC)
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("[OK] Table historical_data initialisee dans stockca (PostgreSQL)")

    def fetch_canadian_symbols(self) -> list:
        """
        Récupère la liste des symboles canadiens via le screener Yahoo Finance.
        Inclut toutes les actions, ETFs et CDR.

        Returns:
            Liste de dicts {symbol, name}
        """
        print("[FETCH] Récupération des symboles canadiens via Yahoo Finance...")

        try:
            all_quotes = []
            offset = 0
            size = 250

            while True:
                query = yf.EquityQuery('AND', [
                    yf.EquityQuery('EQ', ['region', 'ca']),
                    yf.EquityQuery('GT', ['intradayprice', 0])
                ])
                result = yf.screen(
                    query,
                    sortField='intradaymarketcap',
                    sortAsc=False,
                    size=size,
                    offset=offset
                )
                quotes = result.get('quotes', [])
                if not quotes:
                    break
                all_quotes.extend(quotes)
                offset += size
                total = result.get('total', 0)
                print(f"  [{offset}/{total}] symboles récupérés...")
                if offset >= total:
                    break
                time.sleep(0.5)

            symbols = [
                {'symbol': q.get('symbol', ''), 'name': q.get('shortName', '')}
                for q in all_quotes
            ]

            print(f"[FETCH] {len(symbols)} symboles canadiens récupérés")
            return symbols

        except Exception as e:
            print(f"[ERROR] Screener échoué: {e}")
            print("[INFO] Utilisation de la liste de secours...")
            return self._get_fallback_symbols()

    def _get_fallback_symbols(self) -> list:
        """Liste de secours si le screener ne répond pas."""
        fallback = [
            # Banques
            'RY.TO', 'TD.TO', 'BNS.TO', 'BMO.TO', 'CM.TO', 'NA.TO',
            # Énergie
            'ENB.TO', 'TRP.TO', 'CNQ.TO', 'SU.TO', 'IMO.TO', 'CVE.TO',
            'CPG.TO', 'ARX.TO', 'MEG.TO', 'WCP.TO',
            # Mines
            'ABX.TO', 'FNV.TO', 'AEM.TO', 'K.TO', 'FM.TO', 'TECK.TO',
            'IVN.TO', 'LUN.TO', 'ERO.TO', 'CS.TO',
            # Tech
            'SHOP.TO', 'CSU.TO', 'OTEX.TO', 'LSPD.TO', 'BB.TO',
            # Transport / Industriel
            'CNR.TO', 'CP.TO', 'WCN.TO', 'TFI.TO', 'CAE.TO',
            # Télécoms
            'BCE.TO', 'T.TO', 'RCI-B.TO', 'QBR-B.TO',
            # Consommation
            'ATD.TO', 'L.TO', 'DOL.TO', 'MRU.TO', 'EMP-A.TO',
            # Immobilier
            'REI-UN.TO', 'CAR-UN.TO', 'AP-UN.TO',
            # Cannabis
            'WEED.TO', 'ACB.TO', 'TLRY.TO',
        ]
        return [{'symbol': s, 'name': ''} for s in fallback]

    def get_already_imported(self) -> set:
        """Récupère les symboles déjà présents en base avec données suffisantes."""
        try:
            conn = get_conn_ca()
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol FROM (
                    SELECT symbol, COUNT(*) as cnt FROM historical_data GROUP BY symbol
                ) sub WHERE cnt >= 100
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

            # Insérer en base (schéma 'ca')
            conn = get_conn_ca()
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

    def run(self, resume: bool = False, batch_size: int = 100):
        """
        Lance l'import complet.

        Args:
            resume: Reprendre un import interrompu (skip symboles existants)
            batch_size: Taille des lots pour les pauses
        """
        # Récupérer les symboles
        symbol_list = self.fetch_canadian_symbols()
        symbols = [s['symbol'] for s in symbol_list]

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
                      f"{total_bars:,} barres, pause 5s...")
                time.sleep(5)

            # Petite pause entre chaque requête
            time.sleep(0.2)

        print(f"\n[IMPORT] Terminé!")
        print(f"  - Symboles importés: {success}")
        print(f"  - Symboles en erreur/vides: {errors}")
        print(f"  - Total barres: {total_bars:,}")
        print("  - Base de donnees: PostgreSQL schema 'ca'")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Import complet des données canadiennes (TSX/TSXV/NEO/CSE)"
    )
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Reprendre un import interrompu')
    parser.add_argument('--db', default='trading_data_ca.db',
                        help='Chemin de la base de données (default: trading_data_ca.db)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Taille des lots pour les pauses')

    args = parser.parse_args()

    importer = CanadaImporter(args.db)
    importer.run(
        resume=args.resume,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
