"""
Peuplement de la table earnings_dates depuis SEC EDGAR.

Télécharge les Form 8-K "Item 2.02 – Results of Operations" pour chaque
symbole et en déduit les dates d'annonce des résultats trimestriels/annuels.

Logique de matching 8-K → période :
    Pour chaque 8-K Item 2.02 déposé à la date D :
    1. Chercher dans fundamental_cache (période trimestrielle) la fin de trimestre
       qui précède D de 10 à 90 jours.
    2. Sinon, utiliser la fin de trimestre calendaire la plus proche
       (31 mars, 30 juin, 30 sep, 31 déc).
    3. Le fiscal_period (Q1/Q2/Q3/Q4/FY) est déduit de la séquence.

L'API EDGAR retourne les 1000 dépôts les plus récents dans 'filings.recent'.
Pour l'historique complet, on pagine via 'filings.files'.

Usage :
    python populate_earnings_dates.py               # tous les symboles EDGAR
    python populate_earnings_dates.py AAPL MSFT     # symboles spécifiques
    python populate_earnings_dates.py --workers 4   # parallélisme (max: 5)
    python populate_earnings_dates.py --force       # re-fetch même si en cache
    python populate_earnings_dates.py --recent      # seulement les 1000 derniers dépôts
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import argparse
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.app.database.pg_connection import read_sql, get_conn
import src.app.value.edgar_client   as ec
import src.app.value.earnings_dates as ed


# ---------------------------------------------------------------------------
# Matching 8-K date → période fiscale
# ---------------------------------------------------------------------------

# Fins de trimestres calendaires standards
_STD_QUARTER_ENDS = [
    (3, 31), (6, 30), (9, 30), (12, 31)
]


def _std_quarter_end(ann_date: date) -> date:
    """Retourne la fin de trimestre calendaire qui précède ann_date de 10-90 jours."""
    for delta_days in range(10, 91):
        candidate = ann_date - timedelta(days=delta_days)
        for month, day in _STD_QUARTER_ENDS:
            try:
                qe = date(candidate.year, month, day)
                if qe == candidate:
                    return qe
            except ValueError:
                pass
    # Fallback : fin de trimestre la plus proche avant ann_date
    year = ann_date.year
    candidates = []
    for y in [year - 1, year]:
        for month, day in _STD_QUARTER_ENDS:
            try:
                candidates.append(date(y, month, day))
            except ValueError:
                pass
    past = [c for c in candidates if c < ann_date]
    return max(past) if past else ann_date - timedelta(days=30)


def _match_period(ann_date: date, known_ends: list) -> date:
    """
    Fait correspondre une date d'annonce à la meilleure fin de période.

    Args:
        ann_date   : date du 8-K
        known_ends : liste de dates de fin de période depuis fundamental_cache

    Returns:
        Date de fin de période (period_end)
    """
    ann_d = ann_date if isinstance(ann_date, date) else date.fromisoformat(str(ann_date))

    # 1. Chercher dans les périodes connues (depuis fundamental_cache)
    if known_ends:
        candidates = [
            e for e in known_ends
            if timedelta(days=10) <= (ann_d - e) <= timedelta(days=120)
        ]
        if candidates:
            return max(candidates)  # le plus proche avant l'annonce

    # 2. Fallback : trimestre calendaire standard
    return _std_quarter_end(ann_d)


def _infer_fiscal_period(period_end: date, all_periods: list) -> tuple:
    """
    Déduit le fiscal_period (Q1/Q2/Q3/Q4/FY) et le fiscal_year.

    Args:
        period_end  : date de fin de période
        all_periods : toutes les fins de période pour ce symbole (triées)

    Returns:
        (fiscal_period str, fiscal_year int)
    """
    if not all_periods:
        return None, period_end.year

    # Trouver le FY (fin d'exercice annuel) le plus proche ≥ period_end
    annual_ends = sorted(all_periods)

    # Exercice courant = premier FY end >= period_end
    fy_end = None
    for e in annual_ends:
        if e >= period_end:
            fy_end = e
            break
    if fy_end is None:
        fy_end = annual_ends[-1]

    # Compter combien de trimestres précèdent period_end dans cet exercice
    # (exercice commence ~12 mois avant fy_end)
    fy_start = fy_end - timedelta(days=370)
    quarters_in_fy = sorted([
        e for e in all_periods
        if fy_start <= e <= fy_end
    ])

    if not quarters_in_fy:
        return 'Q?', fy_end.year

    idx = quarters_in_fy.index(period_end) if period_end in quarters_in_fy else -1

    if period_end == fy_end:
        fp = 'FY'
    elif idx == 0:
        fp = 'Q1'
    elif idx == 1:
        fp = 'Q2'
    elif idx == 2:
        fp = 'Q3'
    elif idx == 3:
        fp = 'Q4'
    else:
        fp = 'Q?'

    return fp, fy_end.year


# ---------------------------------------------------------------------------
# Fetch 8-K pour un symbole
# ---------------------------------------------------------------------------

def _fetch_8k_dates(cik: str, recent_only: bool = False) -> list:
    """
    Récupère toutes les dates de 8-K Item 2.02 depuis EDGAR pour un CIK.

    Returns:
        Liste de date objects (dates d'annonce)
    """
    results = []

    def _process_filings(filings_block: dict):
        forms  = filings_block.get('form', [])
        dates  = filings_block.get('filingDate', [])
        items  = filings_block.get('items', [])
        for form, filing_date, item in zip(forms, dates, items):
            if form == '8-K' and '2.02' in str(item):
                try:
                    results.append(date.fromisoformat(filing_date))
                except ValueError:
                    pass

    # Dépôts récents
    url  = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = ec._http_get(url)
    if not data:
        return []

    recent = data.get('filings', {}).get('recent', {})
    _process_filings(recent)

    # Pages supplémentaires (historique complet)
    if not recent_only:
        for extra_file in data.get('filings', {}).get('files', []):
            try:
                page = ec._http_get(
                    f"https://data.sec.gov/submissions/{extra_file['name']}"
                )
                if page:
                    _process_filings(page)
            except Exception:
                pass

    return sorted(set(results))


def fetch_one(symbol: str, recent_only: bool = False) -> tuple:
    """
    Récupère et sauvegarde les dates d'annonce 8-K pour un symbole.

    Returns:
        (symbol, 'OK:N' | 'NO_CIK' | 'EMPTY' | 'ERR:msg')
    """
    try:
        cik = ec.get_cik(symbol)
        if cik is None:
            return symbol, 'NO_CIK'

        ann_dates = _fetch_8k_dates(cik, recent_only=recent_only)
        if not ann_dates:
            return symbol, 'EMPTY'

        # Charger les périodes connues depuis fundamental_cache
        rows = read_sql("""
            SELECT DISTINCT fiscal_year_end
            FROM fundamental_cache
            WHERE symbol = %s AND period_type = 'quarterly'
            ORDER BY fiscal_year_end
        """, (symbol,))

        known_ends = []
        if not rows.empty:
            known_ends = [
                date.fromisoformat(str(d))
                for d in rows['fiscal_year_end'].tolist()
                if d is not None
            ]

        # Charger les fins d'exercice annuels (pour déduire le fiscal_period)
        rows_a = read_sql("""
            SELECT DISTINCT fiscal_year_end
            FROM fundamental_cache
            WHERE symbol = %s AND period_type = 'annual'
            ORDER BY fiscal_year_end
        """, (symbol,))

        annual_ends = []
        if not rows_a.empty:
            annual_ends = [
                date.fromisoformat(str(d))
                for d in rows_a['fiscal_year_end'].tolist()
                if d is not None
            ]

        all_known = sorted(set(known_ends + annual_ends))

        # Construire les records
        records = []
        for ann_d in ann_dates:
            period_end = _match_period(ann_d, known_ends)
            fp, fy     = _infer_fiscal_period(period_end, all_known)
            records.append({
                'period_end':        period_end.isoformat(),
                'announcement_date': ann_d.isoformat(),
                'fiscal_period':     fp,
                'fiscal_year':       fy,
            })

        n = ed.save_announcements(symbol, records)
        return symbol, f'OK:{n}'

    except Exception as e:
        return symbol, f'ERR:{str(e)[:80]}'


# ---------------------------------------------------------------------------
# Peuplement en masse
# ---------------------------------------------------------------------------

def populate(
    symbols: list       = None,
    max_workers: int    = 3,
    refresh_days: int   = 30,
    force: bool         = False,
    recent_only: bool   = False,
    verbose: bool       = True,
):
    """
    Peuple earnings_dates pour tous les symboles ayant des données EDGAR.

    Args:
        symbols      : tickers (None = tous ceux dans fundamental_cache edgar)
        max_workers  : threads parallèles (max recommandé: 5)
        refresh_days : ignorer si fetchés il y a moins de N jours
        force        : re-fetcher même si récents
        recent_only  : seulement les 1000 dépôts les plus récents (plus rapide)
        verbose      : afficher les erreurs détaillées
    """
    ed.create_table()

    # Charger la liste : tous les symboles ayant des données EDGAR
    if symbols is None:
        df = read_sql("""
            SELECT DISTINCT symbol
            FROM fundamental_cache
            WHERE source = 'edgar'
            ORDER BY symbol
        """)
        symbols = df['symbol'].tolist()
        print(f"[EARN] {len(symbols):,} symboles avec donnees EDGAR")

    # Filtrer ceux déjà en cache récent
    if not force:
        cached = _get_cached_symbols(max_age_days=refresh_days)
        to_fetch = [s for s in symbols if s not in cached]
    else:
        to_fetch = list(symbols)

    # Filtrer ceux sans CIK
    no_cik   = [s for s in to_fetch if ec.get_cik(s) is None]
    to_fetch = [s for s in to_fetch if ec.get_cik(s) is not None]

    skipped = len(symbols) - len(to_fetch) - len(no_cik)
    print(f"[EARN] {skipped:,} deja en cache | "
          f"{len(no_cik):,} sans CIK | "
          f"{len(to_fetch):,} a telecharger | "
          f"{max_workers} workers"
          + (" (recent only)" if recent_only else ""))

    if not to_fetch:
        print("[EARN] Tout est a jour.")
        return

    ok = err = empty = 0
    start = time.time()
    errors_shown = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, s, recent_only): s
            for s in to_fetch
        }

        for i, future in enumerate(as_completed(futures), 1):
            symbol, result = future.result()

            if result.startswith('OK'):
                ok += 1
            elif result == 'EMPTY':
                empty += 1
            elif result == 'NO_CIK':
                pass
            else:
                err += 1
                if verbose and errors_shown < 15:
                    print(f"  [ERR] {symbol}: {result[4:]}")
                    errors_shown += 1

            if i % 100 == 0 or i == len(to_fetch):
                elapsed   = time.time() - start
                rate      = i / elapsed if elapsed > 0 else 1
                remaining = (len(to_fetch) - i) / rate
                pct       = i / len(to_fetch) * 100
                print(f"  [{i:5}/{len(to_fetch):5}] {pct:5.1f}%  "
                      f"OK:{ok}  vide:{empty}  err:{err}  "
                      f"({rate:.1f}/s  ~{remaining/60:.0f} min restantes)")

    elapsed = time.time() - start
    print(f"\n[EARN] Termine en {elapsed/60:.1f} min : "
          f"{ok} OK | {empty} vides | {err} erreurs")

    stats = ed.get_cache_stats()
    if stats.get('nb_annonces'):
        print(f"[EARN] Cache : {stats['nb_symboles']:,} symboles | "
              f"{stats['nb_annonces']:,} annonces | "
              f"{stats['plus_ancienne']} -> {stats['plus_recente']}")


def _get_cached_symbols(max_age_days: int = 30) -> set:
    rows = read_sql(f"""
        SELECT DISTINCT symbol FROM earnings_dates
        WHERE last_fetched > NOW() - INTERVAL '{max_age_days} days'
    """)
    return set(rows['symbol'].tolist()) if not rows.empty else set()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Peuplement earnings_dates depuis SEC EDGAR 8-K'
    )
    parser.add_argument('symbols', nargs='*',
                        help='Symboles specifiques (defaut: tous)')
    parser.add_argument('--workers', type=int, default=3,
                        help='Workers paralleles (defaut: 3, max: 5)')
    parser.add_argument('--refresh', type=int, default=30,
                        help='Re-fetcher si > N jours (defaut: 30)')
    parser.add_argument('--force', action='store_true',
                        help='Re-fetcher meme si recents')
    parser.add_argument('--recent', action='store_true',
                        help='Seulement les 1000 depots recents (plus rapide)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Mode silencieux')

    args = parser.parse_args()

    populate(
        symbols     = args.symbols if args.symbols else None,
        max_workers = args.workers,
        refresh_days= args.refresh,
        force       = args.force,
        recent_only = args.recent,
        verbose     = not args.quiet,
    )


if __name__ == '__main__':
    main()
