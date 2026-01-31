"""
Peuple la table symbol_metadata avec les informations de secteur/industrie depuis yfinance.
Usage: python populate_symbol_metadata.py
"""
import sqlite3
import yfinance as yf
import time
from datetime import datetime


def populate_metadata(db_path: str = "trading_data.db", sleep_between: float = 0.5):
    """
    Récupère sector/industry depuis yfinance pour chaque symbole dans historical_data
    et les insère dans symbol_metadata.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Récupérer tous les symboles distincts
    cursor.execute("SELECT DISTINCT symbol FROM historical_data ORDER BY symbol")
    symbols = [row[0] for row in cursor.fetchall()]
    print(f"[INFO] {len(symbols)} symboles à traiter")

    success = 0
    errors = 0

    for i, symbol in enumerate(symbols):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            company_name = info.get('longName') or info.get('shortName') or ''
            market_cap = info.get('marketCap', 0)
            exchange = info.get('exchange', '')
            currency = info.get('currency', 'USD')

            cursor.execute("""
                INSERT OR REPLACE INTO symbol_metadata
                (symbol, company_name, sector, industry, market_cap, exchange, currency, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, company_name, sector, industry,
                market_cap, exchange, currency,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            conn.commit()
            success += 1
            print(f"  [{i+1}/{len(symbols)}] {symbol}: {sector} / {industry}")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(symbols)}] {symbol}: ERREUR - {e}")

        # Rate limiting
        time.sleep(sleep_between)

    conn.close()
    print(f"\n[OK] Terminé: {success} succès, {errors} erreurs")


if __name__ == "__main__":
    populate_metadata()
