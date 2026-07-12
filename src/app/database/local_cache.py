"""
Cache SQLite local des données OHLCV — miroir en lecture de historical_data.

Objectif : supprimer les allers-retours réseau vers PostgreSQL pendant les
backtests (chargements répétés des mêmes fenêtres). PostgreSQL reste la
SOURCE DE VÉRITÉ (live, imports, features) ; le cache est jetable et se
resynchronise tout seul.

Fonctionnement :
  - fichier cache/market_cache.db (SQLite), table historical_data
    (symbol, date, OHLCV, adjusted_close), PK (symbol, date)
  - ensure_fresh() : synchronisation incrémentale depuis PostgreSQL —
    re-télécharge les dates > (watermark local - 5 jours), en INSERT OR
    REPLACE (les 5 jours de recouvrement absorbent les révisions tardives)
  - read_ohlcv() : même requête que le moteur de backtest, servie localement

Limite connue : un symbole nouvellement importé avec tout son historique
passé n'est pas vu par la synchro incrémentale -> rebuild() pour reconstruire.

Usage :
    from src.app.database.local_cache import ensure_fresh, read_ohlcv
    ensure_fresh()
    df = read_ohlcv('2025-01-01', '2025-03-31', min_price=5, min_volume=500000)
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import read_sql

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_PATH = PROJECT_ROOT / 'cache' / 'market_cache.db'
OVERLAP_DAYS = 5  # recouvrement re-téléchargé pour absorber les révisions

_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close',
            'volume', 'adjusted_close']


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol          TEXT NOT NULL,
            date            TEXT NOT NULL,
            open            REAL, high REAL, low REAL, close REAL,
            volume          INTEGER,
            adjusted_close  REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_fresh(verbose: bool = True) -> None:
    """Synchronise le cache avec PostgreSQL (incrémental, quelques secondes)."""
    conn = _connect()
    try:
        row = conn.execute("SELECT MAX(date) FROM historical_data").fetchone()
        watermark = row[0]
        t0 = time.time()
        if watermark is None:
            if verbose:
                print("[CACHE] Cache vide — construction initiale "
                      "(une seule fois, quelques minutes)...")
            df = read_sql("""
                SELECT DISTINCT ON (symbol, date)
                       symbol, date, open, high, low, close, volume, adjusted_close
                FROM historical_data
                WHERE close > 0
                ORDER BY symbol, date, source
            """)
        else:
            since = (pd.Timestamp(watermark)
                     - pd.Timedelta(days=OVERLAP_DAYS)).strftime('%Y-%m-%d')
            df = read_sql("""
                SELECT DISTINCT ON (symbol, date)
                       symbol, date, open, high, low, close, volume, adjusted_close
                FROM historical_data
                WHERE close > 0 AND date >= %s
                ORDER BY symbol, date, source
            """, (since,))
        if df.empty:
            if verbose:
                print("[CACHE] Déjà à jour")
            return
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        conn.executemany(
            "INSERT OR REPLACE INTO historical_data VALUES (?,?,?,?,?,?,?,?)",
            df[_COLUMNS].itertuples(index=False, name=None))
        conn.commit()
        if verbose:
            print(f"[CACHE] {len(df):,} lignes synchronisées "
                  f"en {time.time()-t0:.1f}s -> {CACHE_PATH.name}")
    finally:
        conn.close()


def read_ohlcv(start_date, end_date, min_price: float = 0,
               min_volume: int = 0) -> pd.DataFrame:
    """
    Lecture locale — même sémantique que la requête du moteur de backtest.
    Dates acceptées : str 'YYYY-MM-DD', date ou datetime.
    """
    s = pd.Timestamp(start_date).strftime('%Y-%m-%d')
    e = pd.Timestamp(end_date).strftime('%Y-%m-%d')
    conn = _connect()
    try:
        df = pd.read_sql_query("""
            SELECT symbol, date, open, high, low, close, volume, adjusted_close
            FROM historical_data
            WHERE date BETWEEN ? AND ? AND close >= ? AND volume >= ?
            ORDER BY symbol, date
        """, conn, params=(s, e, min_price, min_volume))
        return df
    finally:
        conn.close()


def rebuild(verbose: bool = True) -> None:
    """Reconstruction complète (après import massif de nouveaux symboles)."""
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
        if verbose:
            print("[CACHE] Cache supprimé — reconstruction...")
    ensure_fresh(verbose=verbose)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Cache SQLite local OHLCV")
    p.add_argument('--rebuild', action='store_true',
                   help='Reconstruction complète du cache')
    a = p.parse_args()
    rebuild() if a.rebuild else ensure_fresh()
