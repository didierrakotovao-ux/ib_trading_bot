"""
Script de test pour le wrapper ExhaustionStop.
Compare l'approche "attendre l'essoufflement" vs trailing stop immédiat.
"""
import sys
sys.path.insert(0, '.')

from datetime import datetime
from engine import BacktestEngine
from exhaustion_stop_wrapper import ExhaustionStopBTWrapper

if __name__ == "__main__":
    # Période de backtest
    start_date = datetime(2020, 2, 18)
    end_date = datetime(2020, 3, 27)

    print("=" * 60)
    print("BACKTEST - EXHAUSTION STOP (stop après essoufflement)")
    print("=" * 60)
    print(f"Période: {start_date.date()} -> {end_date.date()}")
    print()
    print("Paramètres:")
    print("  - Trailing Stop: 5% (activé après essoufflement)")
    print("  - Seuil essoufflement: 2 signaux minimum")
    print("  - Signaux surveillés: MACD, RSI, Volume, ADX, Momentum")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=ExhaustionStopBTWrapper,
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
    print()
    print("Consultez le fichier trading_journal_exhaustion_*.csv pour le détail des trades")
    print("et diagnostique_exhaustion_*.txt pour les logs de détection.")
