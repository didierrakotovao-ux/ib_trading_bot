"""
Top 10% EBIT/TEV à une date donnée.

Charge les prix de clôture depuis historical_data pour calculer la market cap
historique (anti-lookahead correct : price * shares_outstanding à trade_date).

Usage :
    python value_top10.py                           # aujourd'hui
    python value_top10.py --date 2023-01-01         # date historique
    python value_top10.py --date 2022-01-01 --pct 5 # top 5%
    python value_top10.py --date 2023-01-01 --n 50  # top 50 absolus
"""
import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datetime import date, timedelta
import pandas as pd

from src.app.database.pg_connection import read_sql
from src.app.value.fundamental_filters import FundamentalFilters


def _load_prices(trade_date: date) -> dict:
    """
    Charge le dernier prix de clôture disponible pour chaque symbole
    dans les 5 jours précédant trade_date (pour les jours fériés/weekends).
    """
    df = read_sql("""
        SELECT DISTINCT ON (symbol) symbol, close
        FROM historical_data
        WHERE date <= %s AND date >= %s
        ORDER BY symbol, date DESC
    """, (trade_date, trade_date - timedelta(days=5)))
    if df.empty:
        return {}
    return dict(zip(df['symbol'], df['close']))


def get_top_value(trade_date: date, top_pct: float = 0.10,
                  top_n: int = None, workers: int = 8) -> pd.DataFrame:
    """
    Retourne les symboles avec le meilleur EBIT/TEV à trade_date.

    Args:
        trade_date : date de la décision (anti-lookahead appliqué)
        top_pct    : fraction à retenir (ex: 0.10 = top 10%)
        top_n      : nombre absolu à retenir (prioritaire sur top_pct si fourni)
        workers    : threads pour le calcul parallèle

    Returns:
        DataFrame trié par EBIT/TEV décroissant :
            symbol, ebit_tev_pct, rank, pct_rank
    """
    # Tous les symboles avec données EDGAR
    df_syms = read_sql("""
        SELECT DISTINCT symbol
        FROM fundamental_cache
        WHERE source = 'edgar'
        ORDER BY symbol
    """)
    if df_syms.empty:
        print("[VALUE] Aucun symbole dans fundamental_cache.")
        return pd.DataFrame()

    symbols = df_syms['symbol'].tolist()
    print(f"[VALUE] {len(symbols):,} symboles EDGAR | date={trade_date}")

    # Prix de clôture à trade_date (pour calculer market cap historique)
    prices = _load_prices(trade_date)
    print(f"[VALUE] {len(prices):,} prix de cloture trouves pour {trade_date}")

    # FundamentalFilters en mode backtest (pas de yfinance = pas de lookahead)
    ff = FundamentalFilters(backtest_mode=True, max_workers=workers)

    # Pré-charger tous les fondamentaux en parallèle
    print(f"[VALUE] Chargement fondamentaux ({workers} workers)...")
    ff._prefetch_parallel(symbols)

    # Calculer EBIT/TEV pour chaque symbole avec son prix à trade_date
    print(f"[VALUE] Calcul EBIT/TEV...")
    results = {}
    for sym in symbols:
        price = prices.get(sym)   # None si pas de données de prix
        r = ff.calc_ebit_tev(sym, trade_date=trade_date, current_price=price)
        results[sym] = r

    # Construire le DataFrame (seulement les symboles avec un ratio calculé)
    rows = [{'symbol': s, 'ebit_tev': r}
            for s, r in results.items() if r is not None]
    if not rows:
        print("[VALUE] Aucune donnée EBIT/TEV disponible pour cette date.")
        return pd.DataFrame()

    df = (pd.DataFrame(rows)
            .sort_values('ebit_tev', ascending=False)
            .reset_index(drop=True))
    df['rank']         = df.index + 1
    df['pct_rank']     = df['rank'] / len(df)
    df['ebit_tev_pct'] = (df['ebit_tev'] * 100).round(2)

    # Retenir top N ou top pct
    if top_n is not None:
        df_top = df.head(top_n).copy()
    else:
        n_keep = max(1, int(len(df) * top_pct))
        df_top = df.head(n_keep).copy()

    return df_top[['symbol', 'ebit_tev_pct', 'rank', 'pct_rank']]


def main():
    parser = argparse.ArgumentParser(description='Top X%% EBIT/TEV a une date donnee')
    parser.add_argument('--date',    type=str,   default=None,
                        help="Date YYYY-MM-DD (defaut: aujourd'hui)")
    parser.add_argument('--pct',     type=float, default=10.0,
                        help='Pourcentage a retenir (defaut: 10)')
    parser.add_argument('--n',       type=int,   default=None,
                        help='Nombre absolu a retenir (prioritaire sur --pct)')
    parser.add_argument('--workers', type=int,   default=8,
                        help='Threads paralleles (defaut: 8)')
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()

    df = get_top_value(
        trade_date = trade_date,
        top_pct    = args.pct / 100.0,
        top_n      = args.n,
        workers    = args.workers,
    )

    if df.empty:
        return

    label = f"top {args.n}" if args.n else f"top {args.pct}%"
    total = df['rank'].max()

    print(f"\n{'='*58}")
    print(f"  EBIT/TEV  {label.upper()}  au  {trade_date}")
    print(f"{'='*58}")
    print(f"{'#':>5}  {'Symbole':<10}  {'EBIT/TEV':>10}  {'Centile':>8}")
    print(f"{'-'*58}")
    for _, row in df.iterrows():
        print(f"{int(row['rank']):>5}  {row['symbol']:<10}  "
              f"{row['ebit_tev_pct']:>8.2f}%  "
              f"{row['pct_rank']:>7.1%}")
    print(f"{'='*58}")
    print(f"  {len(df)} symboles retenus sur {total} avec ratio calculé")
    print(f"{'='*58}")


if __name__ == '__main__':
    main()
