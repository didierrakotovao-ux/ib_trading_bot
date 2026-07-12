"""
Table earnings_dates — dates de publication des résultats financiers.

Source : SEC EDGAR Form 8-K Item 2.02 "Results of Operations".
C'est la vraie date à laquelle le marché apprend les résultats,
plus précise que la date de dépôt du 10-K/10-Q.

Timeline typique :
    Fin du trimestre  (ex: 2026-03-29)
    +2 à +4 semaines  → 8-K Item 2.02 (annonce résultats) ← announcement_date
    +45 à +90 jours   → 10-Q/10-K officiel

Usages :
    - Anti-lookahead précis dans les backtests
    - Calcul du SUE (Standardized Unexpected Earnings)
    - Filtre "pas de position pendant une semaine avant annonce"

Usage :
    from src.app.value.earnings_dates import create_table, load_symbol, get_next_announcement
    create_table()
    dates = load_symbol('AAPL')   # DataFrame avec toutes les annonces
    next_ann = get_next_announcement('AAPL', '2025-10-01')  # prochaine annonce après cette date
"""
import pandas as pd
from typing import Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import get_conn, read_sql


# ---------------------------------------------------------------------------
# Création de la table
# ---------------------------------------------------------------------------

def create_table():
    """Crée la table earnings_dates si elle n'existe pas."""
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS earnings_dates (
            symbol              TEXT    NOT NULL,
            period_end          DATE    NOT NULL,
            announcement_date   DATE    NOT NULL,
            fiscal_period       TEXT,
            fiscal_year         INTEGER,
            source              TEXT    DEFAULT 'edgar_8k',
            last_fetched        TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
            PRIMARY KEY (symbol, period_end)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ed_symbol ON earnings_dates(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ed_ann    ON earnings_dates(announcement_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ed_period ON earnings_dates(period_end)")
    conn.commit()
    cur.close()
    conn.close()
    print("[EARN-DATES] Table earnings_dates prete.")


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_symbol(symbol: str) -> pd.DataFrame:
    """
    Charge toutes les dates d'annonce pour un symbole.

    Returns:
        DataFrame avec colonnes :
            symbol, period_end, announcement_date, fiscal_period, fiscal_year
        Trié par period_end décroissant (plus récent en premier).
        DataFrame vide si aucune donnée.
    """
    return read_sql("""
        SELECT symbol, period_end, announcement_date, fiscal_period, fiscal_year
        FROM earnings_dates
        WHERE symbol = %s
        ORDER BY period_end DESC
    """, (symbol,))


def get_announcement_date(symbol: str, period_end: str) -> Optional[str]:
    """
    Retourne la date d'annonce pour une période donnée, ou None.

    Args:
        symbol     : ticker
        period_end : date de fin de trimestre YYYY-MM-DD

    Returns:
        Date d'annonce str 'YYYY-MM-DD' ou None
    """
    rows = read_sql("""
        SELECT announcement_date
        FROM earnings_dates
        WHERE symbol = %s AND period_end = %s
    """, (symbol, period_end))
    if rows.empty:
        return None
    return str(rows.iloc[0]['announcement_date'])


def get_next_announcement(symbol: str, after_date: str) -> Optional[str]:
    """
    Retourne la prochaine date d'annonce après after_date.
    Utile pour éviter de rentrer en position avant une annonce.

    Args:
        symbol     : ticker
        after_date : date de référence YYYY-MM-DD

    Returns:
        Prochaine announcement_date str 'YYYY-MM-DD' ou None
    """
    rows = read_sql("""
        SELECT MIN(announcement_date) AS next_ann
        FROM earnings_dates
        WHERE symbol = %s AND announcement_date > %s
    """, (symbol, after_date))
    if rows.empty or rows.iloc[0]['next_ann'] is None:
        return None
    return str(rows.iloc[0]['next_ann'])


def get_last_announcement(symbol: str, before_date: str) -> Optional[dict]:
    """
    Retourne la dernière annonce AVANT before_date.
    Utile pour l'anti-lookahead : quelles données étaient disponibles ?

    Args:
        symbol      : ticker
        before_date : date de référence YYYY-MM-DD

    Returns:
        Dict {period_end, announcement_date, fiscal_period, fiscal_year} ou None
    """
    rows = read_sql("""
        SELECT period_end, announcement_date, fiscal_period, fiscal_year
        FROM earnings_dates
        WHERE symbol = %s AND announcement_date <= %s
        ORDER BY announcement_date DESC
        LIMIT 1
    """, (symbol, before_date))
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        'period_end':          str(row['period_end']),
        'announcement_date':   str(row['announcement_date']),
        'fiscal_period':       row['fiscal_period'],
        'fiscal_year':         row['fiscal_year'],
    }


def symbols_with_upcoming_earnings(within_days: int = 7) -> pd.DataFrame:
    """
    Retourne les symboles ayant une annonce dans les N prochains jours.
    Utile pour filtrer les positions avant annonce.

    Returns:
        DataFrame avec colonnes : symbol, announcement_date, fiscal_period
    """
    return read_sql("""
        SELECT symbol, announcement_date, fiscal_period
        FROM earnings_dates
        WHERE announcement_date BETWEEN CURRENT_DATE AND CURRENT_DATE + %s
        ORDER BY announcement_date, symbol
    """, (within_days,))


def get_cache_stats() -> dict:
    """Statistiques de la table."""
    rows = read_sql("""
        SELECT
            COUNT(DISTINCT symbol)         AS nb_symboles,
            COUNT(*)                        AS nb_annonces,
            MIN(announcement_date)          AS plus_ancienne,
            MAX(announcement_date)          AS plus_recente,
            MAX(last_fetched)               AS dernier_fetch
        FROM earnings_dates
    """)
    return rows.iloc[0].to_dict() if not rows.empty else {}


# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

def save_announcements(symbol: str, records: list) -> int:
    """
    Upsert des annonces pour un symbole.

    Args:
        symbol  : ticker
        records : liste de dicts avec clés :
                  period_end, announcement_date, fiscal_period, fiscal_year

    Returns:
        Nombre de lignes insérées/mises à jour
    """
    if not records:
        return 0

    conn = get_conn()
    cur  = conn.cursor()
    n    = 0

    for r in records:
        try:
            cur.execute("""
                INSERT INTO earnings_dates
                    (symbol, period_end, announcement_date,
                     fiscal_period, fiscal_year, source, last_fetched)
                VALUES (%s, %s, %s, %s, %s, 'edgar_8k', NOW())
                ON CONFLICT (symbol, period_end) DO UPDATE SET
                    announcement_date = EXCLUDED.announcement_date,
                    fiscal_period     = EXCLUDED.fiscal_period,
                    fiscal_year       = EXCLUDED.fiscal_year,
                    source            = 'edgar_8k',
                    last_fetched      = NOW()
            """, (
                symbol,
                r['period_end'],
                r['announcement_date'],
                r.get('fiscal_period'),
                r.get('fiscal_year'),
            ))
            n += 1
        except Exception as e:
            print(f"[EARN-DATES] {symbol} {r.get('period_end')}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    create_table()
    stats = get_cache_stats()
    if stats.get('nb_annonces'):
        print(f"earnings_dates : {stats['nb_symboles']:,} symboles | "
              f"{stats['nb_annonces']:,} annonces | "
              f"{stats['plus_ancienne']} -> {stats['plus_recente']}")
    else:
        print("Table vide. Lancer populate_earnings_dates.py")
