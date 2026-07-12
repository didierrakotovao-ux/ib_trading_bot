"""
Peuplement du cache fondamental depuis SEC EDGAR.

Télécharge les états financiers annuels (10-K) ET trimestriels (10-Q)
pour tous les symboles de historical_data et les sauvegarde dans fundamental_cache.

Avantages vs yfinance :
  - ~10 000 entreprises US vs ~952 avec yfinance
  - Historique depuis 2009 (vs 5 ans)
  - Date de dépôt exacte → anti-lookahead parfait

Taux limite SEC : 10 req/sec.  Avec 1 worker, ~10 symboles/sec.
Pour 11 500 symboles → ~20 min (hors symboles sans CIK ni données).

Usage :
    python populate_edgar_cache.py               # tous les symboles
    python populate_edgar_cache.py AAPL MSFT     # symboles spécifiques
    python populate_edgar_cache.py --workers 4   # parallélisme (max recommandé: 5)
    python populate_edgar_cache.py --force       # re-fetch même si en cache
    python populate_edgar_cache.py --no-quarterly # annuel seulement
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.app.database.pg_connection import read_sql, get_conn
import src.app.value.edgar_client as ec
import src.app.value.fundamental_cache as fc


# ---------------------------------------------------------------------------
# Sauvegarde EDGAR → fundamental_cache
# ---------------------------------------------------------------------------

def save_edgar_records(symbol: str, records: list, conn) -> int:
    """
    Insère/met à jour les records EDGAR dans fundamental_cache.

    Args:
        symbol  : ticker
        records : liste de dicts produits par edgar_client._build_period_records()
        conn    : connexion PostgreSQL ouverte

    Returns:
        Nombre de périodes insérées/mises à jour
    """
    cur     = conn.cursor()
    inserted = 0

    for r in records:
        try:
            cur.execute("""
                INSERT INTO fundamental_cache
                  (symbol, fiscal_year_end, report_available_date, period_type,
                   ebit, total_revenue, gross_profit, net_income, normalized_ebitda,
                   pretax_income, interest_expense,
                   total_assets, current_assets, current_liabilities,
                   long_term_debt, cash_equivalents, net_debt, shares_outstanding,
                   stockholders_equity, total_debt,
                   operating_cash_flow, capex, free_cash_flow,
                   source, last_fetched)
                VALUES (%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,
                        'edgar', NOW())
                ON CONFLICT (symbol, fiscal_year_end, period_type) DO UPDATE SET
                   ebit                  = COALESCE(EXCLUDED.ebit,                  fundamental_cache.ebit),
                   total_revenue         = COALESCE(EXCLUDED.total_revenue,         fundamental_cache.total_revenue),
                   gross_profit          = COALESCE(EXCLUDED.gross_profit,          fundamental_cache.gross_profit),
                   net_income            = COALESCE(EXCLUDED.net_income,            fundamental_cache.net_income),
                   normalized_ebitda     = COALESCE(EXCLUDED.normalized_ebitda,     fundamental_cache.normalized_ebitda),
                   pretax_income         = COALESCE(EXCLUDED.pretax_income,         fundamental_cache.pretax_income),
                   interest_expense      = COALESCE(EXCLUDED.interest_expense,      fundamental_cache.interest_expense),
                   total_assets          = COALESCE(EXCLUDED.total_assets,          fundamental_cache.total_assets),
                   current_assets        = COALESCE(EXCLUDED.current_assets,        fundamental_cache.current_assets),
                   current_liabilities   = COALESCE(EXCLUDED.current_liabilities,   fundamental_cache.current_liabilities),
                   long_term_debt        = COALESCE(EXCLUDED.long_term_debt,        fundamental_cache.long_term_debt),
                   cash_equivalents      = COALESCE(EXCLUDED.cash_equivalents,      fundamental_cache.cash_equivalents),
                   net_debt              = COALESCE(EXCLUDED.net_debt,              fundamental_cache.net_debt),
                   shares_outstanding    = COALESCE(EXCLUDED.shares_outstanding,    fundamental_cache.shares_outstanding),
                   stockholders_equity   = COALESCE(EXCLUDED.stockholders_equity,   fundamental_cache.stockholders_equity),
                   total_debt            = COALESCE(EXCLUDED.total_debt,            fundamental_cache.total_debt),
                   operating_cash_flow   = COALESCE(EXCLUDED.operating_cash_flow,   fundamental_cache.operating_cash_flow),
                   capex                 = COALESCE(EXCLUDED.capex,                 fundamental_cache.capex),
                   free_cash_flow        = COALESCE(EXCLUDED.free_cash_flow,        fundamental_cache.free_cash_flow),
                   report_available_date = EXCLUDED.report_available_date,
                   source                = 'edgar',
                   last_fetched          = NOW()
            """, (
                symbol,
                r['fiscal_year_end'],
                r['report_available_date'],
                r['period_type'],
                r.get('ebit'),
                r.get('total_revenue'),
                r.get('gross_profit'),
                r.get('net_income'),
                r.get('normalized_ebitda'),
                r.get('pretax_income'),
                r.get('interest_expense'),
                r.get('total_assets'),
                r.get('current_assets'),
                r.get('current_liabilities'),
                r.get('long_term_debt'),
                r.get('cash_equivalents'),
                r.get('net_debt'),
                r.get('shares_outstanding'),
                r.get('stockholders_equity'),
                r.get('total_debt'),
                r.get('operating_cash_flow'),
                r.get('capex'),
                r.get('free_cash_flow'),
            ))
            inserted += 1
        except Exception as e:
            print(f"  [EDGAR-SAVE] {symbol} {r.get('fiscal_year_end')}: {e}")

    cur.close()
    return inserted


# ---------------------------------------------------------------------------
# Fetch + save d'un symbole
# ---------------------------------------------------------------------------

def fetch_and_save(symbol: str, include_quarterly: bool = True) -> tuple[str, str]:
    """
    Récupère les données EDGAR et sauvegarde en base.

    Returns:
        (symbol, 'OK:N_annuel+N_trim' | 'NO_CIK' | 'EMPTY' | 'ERR:msg')
    """
    try:
        data = ec.get_fundamentals(symbol)

        if data is None:
            return symbol, 'NO_CIK'

        annual_recs    = data.get('annual_records', [])
        quarterly_recs = data.get('quarterly_records', []) if include_quarterly else []

        if not annual_recs and not quarterly_recs:
            return symbol, 'EMPTY'

        conn = get_conn()
        n_a = save_edgar_records(symbol, annual_recs,    conn) if annual_recs    else 0
        n_q = save_edgar_records(symbol, quarterly_recs, conn) if quarterly_recs else 0
        conn.commit()
        conn.close()

        return symbol, f'OK:{n_a}a+{n_q}q'

    except Exception as e:
        return symbol, f'ERR:{str(e)[:80]}'


# ---------------------------------------------------------------------------
# Peuplement en masse
# ---------------------------------------------------------------------------

def populate(
    symbols: list        = None,
    max_workers: int     = 3,       # Rester bien en-dessous de 10 req/sec
    refresh_days: int    = 30,
    force: bool          = False,
    include_quarterly: bool = True,
    verbose: bool        = True,
):
    """
    Peuple le cache fondamental depuis SEC EDGAR.

    Args:
        symbols          : liste de tickers (None = tous depuis historical_data)
        max_workers      : threads parallèles (max recommandé: 5 pour respecter les limites SEC)
        refresh_days     : ignorer si données récentes de moins de N jours
        force            : re-fetcher même si données récentes
        include_quarterly: inclure les données trimestrielles (10-Q)
        verbose          : afficher les erreurs détaillées
    """
    # S'assurer que la colonne 'source' existe
    _migrate_source_column()

    # Charger le mapping CIK (une seule fois)
    print("[EDGAR] Chargement du mapping ticker → CIK...")
    cik_map = ec.load_cik_map()
    print(f"[EDGAR] {len(cik_map):,} tickers référencés par SEC")

    # Charger la liste de symboles
    if symbols is None:
        df = read_sql('SELECT DISTINCT symbol FROM historical_data ORDER BY symbol')
        symbols = df['symbol'].tolist()
        print(f"[EDGAR] {len(symbols):,} symboles dans historical_data")

    # Filtrer ceux déjà en cache récent (source=edgar, sauf --force)
    if not force:
        cached = _get_edgar_cached_symbols(max_age_days=refresh_days)
        to_fetch = [s for s in symbols if s not in cached]
    else:
        to_fetch = list(symbols)

    # Filtrer les symboles sans CIK connu dès maintenant (évite les appels inutiles)
    no_cik    = [s for s in to_fetch if ec.get_cik(s) is None]
    to_fetch  = [s for s in to_fetch if ec.get_cik(s) is not None]

    skipped   = len(symbols) - len(to_fetch) - len(no_cik)
    print(f"[EDGAR] {skipped:,} déjà en cache | "
          f"{len(no_cik):,} sans CIK SEC | "
          f"{len(to_fetch):,} à télécharger | "
          f"{max_workers} workers")

    if not to_fetch:
        print("[EDGAR] Tout est à jour.")
        return

    # Lancement parallèle
    ok = err = empty = no_cik_rt = 0
    start = time.time()
    errors_shown = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_and_save, s, include_quarterly): s
            for s in to_fetch
        }

        for i, future in enumerate(as_completed(futures), 1):
            symbol, result = future.result()

            if result.startswith('OK'):
                ok += 1
            elif result == 'EMPTY':
                empty += 1
            elif result == 'NO_CIK':
                no_cik_rt += 1
            else:
                err += 1
                if verbose and errors_shown < 15:
                    print(f"  [ERR] {symbol}: {result[4:]}")
                    errors_shown += 1

            # Progression toutes les 100 itérations
            if i % 100 == 0 or i == len(to_fetch):
                elapsed   = time.time() - start
                rate      = i / elapsed if elapsed > 0 else 1
                remaining = (len(to_fetch) - i) / rate
                pct       = i / len(to_fetch) * 100
                print(f"  [{i:5}/{len(to_fetch):5}] {pct:5.1f}%  "
                      f"OK:{ok}  vide:{empty}  err:{err}  "
                      f"({rate:.1f}/s  ~{remaining/60:.0f} min restantes)")

    elapsed = time.time() - start
    print(f"\n[EDGAR] Termine en {elapsed/60:.1f} min : "
          f"{ok} OK | {empty} vides | {err} erreurs")

    # Stats finales
    stats = fc.get_cache_stats()
    if stats.get('nb_symbols'):
        print(f"[EDGAR] Cache total : {stats['nb_symbols']:,} symboles | "
              f"{stats['nb_rows']:,} periodes | "
              f"{stats['oldest_fiscal']} -> {stats['newest_fiscal']}")


# ---------------------------------------------------------------------------
# Migration : ajouter la colonne 'source' si absente
# ---------------------------------------------------------------------------

def _migrate_source_column():
    """Ajoute la colonne 'source' à fundamental_cache si elle n'existe pas."""
    conn = get_conn()
    cur  = conn.cursor()
    try:
        cur.execute("""
            ALTER TABLE fundamental_cache
            ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'yfinance'
        """)
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def _get_edgar_cached_symbols(max_age_days: int = 30) -> set:
    """Retourne les symboles déjà fetchés depuis EDGAR récemment."""
    rows = read_sql(f"""
        SELECT DISTINCT symbol
        FROM fundamental_cache
        WHERE source = 'edgar'
          AND last_fetched > NOW() - INTERVAL '{max_age_days} days'
    """)
    return set(rows['symbol'].tolist()) if not rows.empty else set()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Peuplement du cache fondamental depuis SEC EDGAR'
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles spécifiques (défaut: tous)')
    parser.add_argument('--workers', type=int, default=3,
                        help='Workers parallèles (défaut: 3, max recommandé: 5)')
    parser.add_argument('--refresh', type=int, default=30,
                        help='Re-fetcher si données > N jours (défaut: 30)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetcher même si données récentes')
    parser.add_argument('--no-quarterly', action='store_true',
                        help='Annuel seulement (plus rapide)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Mode silencieux')

    args = parser.parse_args()

    populate(
        symbols          = args.symbols if args.symbols else None,
        max_workers      = args.workers,
        refresh_days     = args.refresh,
        force            = args.force,
        include_quarterly= not args.no_quarterly,
        verbose          = not args.quiet,
    )


if __name__ == '__main__':
    main()
