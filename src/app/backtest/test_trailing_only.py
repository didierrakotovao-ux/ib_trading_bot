"""
Script de test pour le wrapper TrailingOnly (sans Take Profit).
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from engine import BacktestEngine
from trailing_only_wrapper import TrailingOnlyBTWrapper

if __name__ == "__main__":
    # Période de backtest
    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 3, 30)

    print("=" * 60)
    print("BACKTEST - TRAILING STOP ONLY (pas de TP)")
    print("=" * 60)
    print(f"Période: {start_date.date()} -> {end_date.date()}")
    print(f"Trailing Stop: 5%")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=TrailingOnlyBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000
    )

    results = engine.run()

    print()
    print("=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"Final Value: {results['final_value']:,.2f}")
    print(f"PnL: {results['pnl']:,.2f}")
    print(f"Sharpe: {results['sharpe']}")
    print(f"Drawdown: {results['drawdown']}")
    print(f"Trades: {results['trades']}")
