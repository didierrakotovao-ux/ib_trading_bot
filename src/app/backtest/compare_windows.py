"""
Tableau comparatif des backtests accumulés en base (stratégie x fenêtre).

Grâce à l'effacement par période (clear_backtest_trades), chaque re-run ne
remplace que sa propre fenêtre : la table trades accumule les résultats de
toutes les combinaisons stratégie x période.

Usage: python src/app/backtest/compare_windows.py
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pandas as pd
from src.app.database.pg_connection import read_sql


def main():
    df = read_sql("""
        SELECT strategy_name,
               backtest_start_date AS debut,
               backtest_end_date   AS fin,
               COUNT(*)                                   AS trades,
               COUNT(*) FILTER (WHERE pnl_net > 0)        AS gagnants,
               ROUND(SUM(pnl_net)::numeric, 0)            AS pnl_total,
               ROUND(AVG(pnl_net)::numeric, 0)            AS pnl_moyen,
               ROUND(MAX(pnl_net)::numeric, 0)            AS max_win,
               ROUND(MIN(pnl_net)::numeric, 0)            AS max_loss,
               ROUND(AVG(bars_held)::numeric, 1)          AS barres_moy
        FROM trades
        WHERE trade_mode = 'backtest' AND backtest_start_date IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 2, 1
    """)
    if df.empty:
        print("Aucun backtest en base.")
        return

    df['win_rate'] = (df['gagnants'] / df['trades'] * 100).round(0)
    cols = ['strategy_name', 'debut', 'fin', 'trades', 'win_rate',
            'pnl_total', 'pnl_moyen', 'max_win', 'max_loss', 'barres_moy']
    print("=" * 100)
    print("BACKTESTS EN BASE — stratégie x fenêtre")
    print("=" * 100)
    print(df[cols].to_string(index=False))

    # Vue portefeuille : somme des stratégies par fenêtre
    print("\nVue portefeuille (somme des stratégies par fenêtre):")
    port = df.groupby(['debut', 'fin']).agg(
        strategies=('strategy_name', 'nunique'),
        trades=('trades', 'sum'),
        pnl_total=('pnl_total', 'sum')).reset_index()
    print(port.to_string(index=False))


if __name__ == "__main__":
    main()
