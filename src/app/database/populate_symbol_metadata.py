"""
Peuple la table symbol_metadata avec les informations de secteur/industrie depuis yfinance.
Usage: python populate_symbol_metadata.py
"""
import yfinance as yf
import time
import sys
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn


def populate_metadata(db_path=None, sleep_between: float = 0.5):
    """
    Récupère sector/industry depuis yfinance pour chaque symbole dans historical_data
    et les insère dans symbol_metadata.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
    symbols = [row[0] for row in cur.fetchall()]
    print(f"[INFO] {len(symbols)} symboles à traiter")

    success = 0
    errors = 0

    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector       = info.get('sector', 'Unknown')
            industry     = info.get('industry', 'Unknown')
            company_name = info.get('longName') or info.get('shortName') or ''
            market_cap   = info.get('marketCap', 0)
            exchange     = info.get('exchange', '')
            currency     = info.get('currency', 'USD')

            cur.execute("""
                INSERT INTO symbol_metadata
                (symbol, company_name, sector, industry, market_cap, exchange, currency, last_updated)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector       = EXCLUDED.sector,
                    industry     = EXCLUDED.industry,
                    market_cap   = EXCLUDED.market_cap,
                    exchange     = EXCLUDED.exchange,
                    currency     = EXCLUDED.currency,
                    last_updated = EXCLUDED.last_updated
            """, (
                symbol, company_name, sector, industry,
                market_cap, exchange, currency,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            conn.commit()
            success += 1
            print(f"  [{i+1}/{len(symbols)}] {symbol}: {sector} / {industry}")

        except Exception as e:
            conn.rollback()
            errors += 1
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ERREUR - {e}")

        time.sleep(sleep_between)

    cur.close()
    conn.close()
    print(f"\n[OK] Terminé: {success} succès, {errors} erreurs")


if __name__ == "__main__":
    populate_metadata()
