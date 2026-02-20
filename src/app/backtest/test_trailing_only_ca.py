"""
Backtest Trailing Stop pour actions canadiennes (trading_data_ca.db).

Usage:
    python test_trailing_only_ca.py                          # Période par défaut
    python test_trailing_only_ca.py --start 2023-01-01 --end 2024-12-31
    python test_trailing_only_ca.py --scoring ml             # Modèle basique
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
import argparse
from engine import BacktestEngine
from trailing_only_wrapper import TrailingOnlyBTWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Trailing Stop - Actions canadiennes"
    )
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='Date de début YYYY-MM-DD (default: 2024-01-01)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='Date de fin YYYY-MM-DD (default: 2024-12-31)')
    parser.add_argument('--scoring', choices=['smooth_ml', 'ml'], default='smooth_ml',
                        help='Type de scoring (default: smooth_ml)')
    parser.add_argument('--trailing', type=float, default=5.0,
                        help='Trailing stop %% (default: 5.0)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='Capital initial (default: 100000)')
    parser.add_argument('--max-stocks', type=int, default=5,
                        help='Nombre max de positions (default: 5)')
    parser.add_argument('--min-price', type=float, default=2.0,
                        help='Prix minimum (default: 2.0)')
    parser.add_argument('--min-volume', type=int, default=100000,
                        help='Volume minimum (default: 100000)')
    parser.add_argument('--max-symbols', type=int, default=None,
                        help='Limiter le nombre de symboles (default: tous)')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')

    print("=" * 60)
    print("BACKTEST CANADIEN - TRAILING STOP ONLY")
    print("=" * 60)
    print(f"Periode: {start_date.date()} -> {end_date.date()}")
    print(f"Scoring: {args.scoring}")
    print(f"Trailing Stop: {args.trailing}%")
    print(f"Capital: {args.capital:,.0f}$")
    print(f"Filtres: price >= ${args.min_price}, volume >= {args.min_volume:,}")
    print(f"DB: trading_data_ca.db")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=TrailingOnlyBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=args.capital,
        scoring_type=args.scoring,
        db_path="trading_data_ca.db",
        min_price=args.min_price,
        min_volume=args.min_volume,
        max_symbols=args.max_symbols,
    )

    results = engine.run()

    print()
    print("=" * 60)
    print("RESULTATS")
    print("=" * 60)
    print(f"Final Value: {results['final_value']:,.2f}$")
    print(f"PnL: {results['pnl']:,.2f}$")
    print(f"Rendement: {results['pnl']/args.capital*100:+.1f}%")
    print(f"Sharpe: {results['sharpe']}")
    print(f"Drawdown: {results['drawdown']}")


if __name__ == "__main__":
    main()
