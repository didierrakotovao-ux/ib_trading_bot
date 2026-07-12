"""
Peuplement du cache fondamental depuis yfinance.

Télécharge les états financiers annuels pour tous les symboles de historical_data
et les sauvegarde dans la table fundamental_cache (PostgreSQL).

Limitation yfinance : ~5 ans d'historique annuel (environ 2020-2025).
Les backtests avant ~2020 ne pourront pas bénéficier des filtres fondamentaux.

Usage :
    python populate_fundamental_cache.py                   # tous les symboles
    python populate_fundamental_cache.py AAPL MSFT         # symboles spécifiques
    python populate_fundamental_cache.py --workers 10      # plus de parallélisme
    python populate_fundamental_cache.py --refresh 90      # re-fetcher si > 90j
    python populate_fundamental_cache.py --force           # tout re-fetcher

Estimation durée (11 500 symboles) :
    8 workers  → ~2-3 heures (recommandé, stable)
    15 workers → ~1-1.5 heure (risque rate-limit yfinance)
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import yfinance as yf
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.app.database.pg_connection import read_sql
import src.app.value.fundamental_cache as fc


# ---------------------------------------------------------------------------
# Fetch d'un seul symbole
# ---------------------------------------------------------------------------

def _safe_fetch(ticker_obj, *attr_names):
    """Essaie plusieurs noms d'attributs yfinance, retourne le premier non-vide."""
    for attr in attr_names:
        try:
            df = getattr(ticker_obj, attr)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return None


def fetch_one(symbol: str, reporting_lag_annual: int = 90,
              reporting_lag_quarterly: int = 45) -> tuple[str, str]:
    """
    Télécharge les états financiers annuels ET trimestriels depuis yfinance.

    Lag par défaut :
        - Annuel    : 90 jours après la fin d'exercice
        - Trimestriel : 45 jours après la fin du trimestre

    Returns:
        (symbol, 'OK:N' | 'EMPTY' | 'ERR:message')
    """
    try:
        t = yf.Ticker(symbol)

        # --- Annuel ---
        income_a   = _safe_fetch(t, 'income_stmt', 'financials')
        balance_a  = _safe_fetch(t, 'balance_sheet')
        cashflow_a = _safe_fetch(t, 'cash_flow', 'cashflow')

        # --- Trimestriel ---
        income_q   = _safe_fetch(t, 'quarterly_income_stmt', 'quarterly_financials')
        balance_q  = _safe_fetch(t, 'quarterly_balance_sheet')
        cashflow_q = _safe_fetch(t, 'quarterly_cash_flow', 'quarterly_cashflow')

        has_annual    = any(df is not None for df in [income_a, balance_a, cashflow_a])
        has_quarterly = any(df is not None for df in [income_q, balance_q, cashflow_q])

        if not has_annual and not has_quarterly:
            return symbol, 'EMPTY'

        n_total = 0

        if has_annual:
            data_a = {
                'info':     {},
                'income':   income_a,
                'balance':  balance_a,
                'cashflow': cashflow_a,
            }
            n_total += fc.save_symbol(symbol, data_a,
                                      reporting_lag=reporting_lag_annual,
                                      period_type='annual')

        if has_quarterly:
            data_q = {
                'info':     {},
                'income':   income_q,
                'balance':  balance_q,
                'cashflow': cashflow_q,
            }
            n_total += fc.save_symbol(symbol, data_q,
                                      reporting_lag=reporting_lag_quarterly,
                                      period_type='quarterly')

        return symbol, f'OK:{n_total}'

    except Exception as e:
        return symbol, f'ERR:{str(e)[:80]}'


# ---------------------------------------------------------------------------
# Peuplement en masse
# ---------------------------------------------------------------------------

def populate(
    symbols: list   = None,
    max_workers: int = 8,
    refresh_days: int = 30,
    reporting_lag: int = 90,
    force: bool  = False,
    verbose: bool = True,
):
    """
    Peuple le cache fondamental pour une liste de symboles.

    Args:
        symbols       : liste de tickers (None = tous depuis historical_data)
        max_workers   : nombre de threads parallèles
        refresh_days  : ignorer si données plus récentes que N jours
        reporting_lag : délai publication rapports (jours) — anti-lookahead
        force         : re-fetcher même si données récentes
        verbose       : afficher le détail
    """
    # Créer la table si besoin
    fc.create_table()

    # Charger la liste de symboles depuis la base si non fournie
    if symbols is None:
        df = read_sql('SELECT DISTINCT symbol FROM historical_data ORDER BY symbol')
        symbols = df['symbol'].tolist()
        print(f"[FUND] {len(symbols):,} symboles dans historical_data")

    # Filtrer ceux déjà en cache récent (sauf --force)
    if not force:
        cached = fc.get_cached_symbols(max_age_days=refresh_days)
        to_fetch = [s for s in symbols if s not in cached]
    else:
        to_fetch = list(symbols)

    skipped = len(symbols) - len(to_fetch)
    print(f"[FUND] {skipped:,} déjà en cache ({refresh_days}j) | "
          f"{len(to_fetch):,} à télécharger | "
          f"{max_workers} workers | lag={reporting_lag}j")

    if not to_fetch:
        print("[FUND] Tout est à jour.")
        return

    # Lancement parallèle
    ok = err = empty = 0
    start = time.time()
    errors_shown = 0

    reporting_lag_q = max(30, reporting_lag // 2)  # 45j par défaut si lag=90

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, s, reporting_lag, reporting_lag_q): s
            for s in to_fetch
        }

        for i, future in enumerate(as_completed(futures), 1):
            symbol, result = future.result()

            if result.startswith('OK'):
                ok += 1
            elif result == 'EMPTY':
                empty += 1
            else:
                err += 1
                if verbose and errors_shown < 20:
                    print(f"  [ERR] {symbol}: {result[4:]}")
                    errors_shown += 1

            # Progression toutes les 100 itérations
            if i % 100 == 0 or i == len(to_fetch):
                elapsed  = time.time() - start
                rate     = i / elapsed if elapsed > 0 else 1
                remaining = (len(to_fetch) - i) / rate
                pct      = i / len(to_fetch) * 100
                print(f"  [{i:5}/{len(to_fetch):5}] {pct:5.1f}%  "
                      f"OK:{ok}  vide:{empty}  err:{err}  "
                      f"({rate:.1f}/s, ~{remaining/60:.0f} min restantes)")

    elapsed = time.time() - start
    print(f"\n[FUND] Terminé en {elapsed/60:.1f} min : "
          f"{ok} OK | {empty} vides | {err} erreurs")

    # Statistiques finales
    stats = fc.get_cache_stats()
    if stats.get('nb_symbols'):
        print(f"[FUND] Cache total : {stats['nb_symbols']:,} symboles | "
              f"{stats['nb_rows']:,} exercices | "
              f"exercices: {stats['oldest_fiscal']} -> {stats['newest_fiscal']}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Peuplement du cache fondamental depuis yfinance'
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles spécifiques (défaut: tous)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Workers parallèles (défaut: 8)')
    parser.add_argument('--refresh', type=int, default=30,
                        help='Re-fetcher si données > N jours (défaut: 30)')
    parser.add_argument('--lag', type=int, default=90,
                        help='Délai publication rapports en jours (défaut: 90)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetcher même si données récentes')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Mode silencieux')

    args = parser.parse_args()

    populate(
        symbols      = args.symbols if args.symbols else None,
        max_workers  = args.workers,
        refresh_days = args.refresh,
        reporting_lag= args.lag,
        force        = args.force,
        verbose      = not args.quiet,
    )


if __name__ == '__main__':
    main()
