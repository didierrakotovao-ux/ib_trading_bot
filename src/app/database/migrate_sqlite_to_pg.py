"""
Migration SQLite -> PostgreSQL.

Transfert toutes les donnees de :
  - trading_data.db     -> PostgreSQL schema public  (tables US)
  - trading_data_ca.db  -> PostgreSQL schema ca       (historical_data CA)

Usage :
    python src/app/database/migrate_sqlite_to_pg.py
    python src/app/database/migrate_sqlite_to_pg.py --dry-run   # compte les lignes seulement
    python src/app/database/migrate_sqlite_to_pg.py --table historical_data  # une seule table
"""
import sqlite3
import sys
import os
import time
import argparse
from datetime import datetime

import psycopg2
import psycopg2.extras

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, PROJECT_ROOT)
from src.app.database.pg_connection import get_conn, get_conn_ca, init_ca_schema

# Chemins SQLite
DB_US = os.path.join(PROJECT_ROOT, 'trading_data.db')
DB_CA = os.path.join(PROJECT_ROOT, 'trading_data_ca.db')

# Taille des lots d'insertion
CHUNK_SIZE = 5000

# Ordre de migration (respecte les dependances FK)
US_TABLES_ORDER = [
    'symbol_metadata',
    'scanner_results',
    'historical_data',
    'technical_indicators',
    'watchlist',
    'computed_features',
    'trading_signals',
    'signal_outcomes',
    'trades',
    'trade_exits',
    'position_stops',
]


def sqlite_row_count(sqlite_conn, table: str) -> int:
    try:
        cur = sqlite_conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]
    except Exception:
        return 0


def sqlite_table_exists(sqlite_conn, table: str) -> bool:
    cur = sqlite_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def get_columns(sqlite_conn, table: str) -> list:
    cur = sqlite_conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def reset_sequence(pg_conn, table: str):
    """Remet la sequence PostgreSQL au MAX(id) de la table pour eviter les conflits futurs."""
    cur = pg_conn.cursor()
    cur.execute(f"""
        SELECT setval(
            pg_get_serial_sequence('{table}', 'id'),
            COALESCE((SELECT MAX(id) FROM {table}), 1)
        )
    """)
    pg_conn.commit()
    cur.close()


def migrate_table(sqlite_conn, pg_conn, table: str, pg_schema: str = 'public',
                  dry_run: bool = False) -> dict:
    """
    Migre une table SQLite vers PostgreSQL.
    Retourne un dict avec les stats (rows_read, rows_inserted, rows_skipped, duration).
    """
    schema_prefix = f"{pg_schema}." if pg_schema != 'public' else ""
    pg_table = f"{schema_prefix}{table}"

    if not sqlite_table_exists(sqlite_conn, table):
        return {'table': table, 'status': 'absent', 'rows_read': 0}

    total_rows = sqlite_row_count(sqlite_conn, table)
    if total_rows == 0:
        print(f"  {table:35} - vide, ignoree")
        return {'table': table, 'status': 'empty', 'rows_read': 0}

    columns = get_columns(sqlite_conn, table)
    col_list = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))

    if table == 'symbol_metadata':
        conflict_clause = "ON CONFLICT (symbol) DO NOTHING"
    elif 'id' in columns:
        conflict_clause = "ON CONFLICT (id) DO NOTHING"
    else:
        conflict_clause = "ON CONFLICT DO NOTHING"

    # execute_values attend un seul %s (remplace tout le bloc VALUES)
    bulk_sql = f"INSERT INTO {pg_table} ({col_list}) VALUES %s {conflict_clause}"
    # fallback ligne par ligne
    single_sql = f"INSERT INTO {pg_table} ({col_list}) VALUES ({placeholders}) {conflict_clause}"

    if dry_run:
        print(f"  {table:35} - {total_rows:>10,} lignes (dry-run)")
        return {'table': table, 'status': 'dry-run', 'rows_read': total_rows}

    t_start = time.time()
    rows_inserted = 0
    rows_skipped  = 0
    offset = 0
    pg_cur = pg_conn.cursor()

    print(f"  {table:35} - {total_rows:>10,} lignes ...", end='', flush=True)

    sqlite_cur = sqlite_conn.cursor()

    while True:
        sqlite_cur.execute(
            f"SELECT {col_list} FROM {table} LIMIT ? OFFSET ?",
            (CHUNK_SIZE, offset)
        )
        rows = sqlite_cur.fetchall()
        if not rows:
            break

        # Convertir les lignes en liste de tuples Python
        batch = [tuple(r) for r in rows]

        try:
            psycopg2.extras.execute_values(pg_cur, bulk_sql, batch, page_size=CHUNK_SIZE)
            pg_conn.commit()
            rows_inserted += pg_cur.rowcount if pg_cur.rowcount >= 0 else len(batch)
        except Exception as e:
            pg_conn.rollback()
            print(f"\n    [WARN] Erreur batch offset={offset}: {e}")
            for row in batch:
                try:
                    pg_cur.execute(single_sql, row)
                    pg_conn.commit()
                    rows_inserted += 1
                except Exception:
                    pg_conn.rollback()
                    rows_skipped += 1

        offset += CHUNK_SIZE
        pct = min(100, int(offset / total_rows * 100))
        print(f"\r  {table:35} - {total_rows:>10,} lignes ... {pct:3}%", end='', flush=True)

    pg_cur.close()
    duration = time.time() - t_start
    print(f"\r  {table:35} - {rows_inserted:>10,} inserees  {rows_skipped:>6,} skippees  [{duration:.1f}s]")

    return {
        'table': table, 'status': 'ok',
        'rows_read': total_rows, 'rows_inserted': rows_inserted,
        'rows_skipped': rows_skipped, 'duration': duration
    }


def run_migration(only_table: str = None, dry_run: bool = False,
                  skip_ca: bool = False, skip_us: bool = False):
    print("=" * 65)
    print(f"MIGRATION SQLite -> PostgreSQL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    if not os.path.exists(DB_US):
        print(f"[ERREUR] Fichier introuvable: {DB_US}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Base US  (trading_data.db -> public schema)
    # ------------------------------------------------------------------
    if skip_us:
        print("\n[1/2] Base US ignoree (--skip-us)")
        results_us = []
    else:
        print(f"\n[1/2] Base US  ({DB_US})")
        sqlite_us = sqlite3.connect(DB_US)
        pg_us = get_conn()
        tables_us = [only_table] if only_table else US_TABLES_ORDER
        results_us = []
        for table in tables_us:
            result = migrate_table(sqlite_us, pg_us, table, pg_schema='public', dry_run=dry_run)
            results_us.append(result)
        if not dry_run:
            print("\n  Remise a zero des sequences...")
            for table in tables_us:
                if sqlite_table_exists(sqlite_us, table) and table != 'symbol_metadata':
                    try:
                        reset_sequence(pg_us, table)
                    except Exception:
                        pass
        sqlite_us.close()
        pg_us.close()

    # ------------------------------------------------------------------
    # 2. Base CA  (trading_data_ca.db -> schema ca)
    # ------------------------------------------------------------------
    if skip_ca:
        print("\n[2/2] Base CA ignoree (--skip-ca)")
        result_ca = None
    elif not only_table or only_table == 'historical_data':
        if os.path.exists(DB_CA):
            print(f"\n[2/2] Base CA  ({DB_CA})")
            init_ca_schema()
            sqlite_ca = sqlite3.connect(DB_CA)
            pg_ca = get_conn_ca()

            result_ca = migrate_table(sqlite_ca, pg_ca, 'historical_data',
                                      pg_schema='public', dry_run=dry_run)

            if not dry_run:
                try:
                    reset_sequence(pg_ca, 'historical_data')
                except Exception:
                    pass

            sqlite_ca.close()
            pg_ca.close()
        else:
            print(f"\n[2/2] {DB_CA} introuvable - ignoree")
            result_ca = None
    else:
        result_ca = None

    # ------------------------------------------------------------------
    # Resume final
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("RESUME")
    print("=" * 65)
    total_inserted = 0
    total_read = 0
    for r in results_us:
        if r.get('rows_read', 0) > 0:
            ins = r.get('rows_inserted', r['rows_read'])
            total_inserted += ins
            total_read += r['rows_read']
    if result_ca and result_ca.get('rows_read', 0) > 0:
        ins = result_ca.get('rows_inserted', result_ca['rows_read'])
        total_inserted += ins
        total_read += result_ca['rows_read']

    print(f"  Total lignes lues    : {total_read:>12,}")
    print(f"  Total lignes inserees: {total_inserted:>12,}")
    print("\nMigration terminee.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migration SQLite -> PostgreSQL')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compter les lignes sans inserer')
    parser.add_argument('--table', type=str, default=None,
                        help='Migrer seulement cette table')
    parser.add_argument('--skip-ca', action='store_true',
                        help='Ignorer la base canadienne (trading_data_ca.db)')
    parser.add_argument('--skip-us', action='store_true',
                        help='Ignorer la base US (trading_data.db), migrer CA seulement')
    args = parser.parse_args()
    run_migration(only_table=args.table, dry_run=args.dry_run,
                  skip_ca=args.skip_ca, skip_us=getattr(args, 'skip_us', False))
