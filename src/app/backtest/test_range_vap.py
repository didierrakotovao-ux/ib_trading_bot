"""
Script de test pour RangeVapStrategy (Volume Profile + filtre ADX).

Options :
  --period      : sideways_2015 | sideways_2018 | bull_2023_2024 | covid_crash |
                   covid_recovery | bear_2022 | full_2020_2024
  --start/--end : dates libres YYYY-MM-DD
  --max_stocks  : nombre max de positions simultanées (défaut: 5)
"""
import sys
import os
import argparse
sys.path.insert(0, '.')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datetime import datetime
from engine import BacktestEngine
from range_vap_bt_wrapper import RangeVapBTWrapper

PERIODS = {
    "sideways_2015":   (datetime(2015, 2,  1), datetime(2016, 2,  1)),  # S&P 500 range-bound (choc Chine/pétrole)
    "sideways_2018":   (datetime(2018, 3,  1), datetime(2018, 10, 1)),  # Range avant le crash de déc. 2018
    "bull_2023_2024":  (datetime(2023, 1,  1), datetime(2024, 12, 31)),
    "covid_crash":     (datetime(2020, 2, 18), datetime(2020, 3,  25)),
    "covid_recovery":  (datetime(2020, 4,  1), datetime(2021, 12, 31)),
    "bear_2022":       (datetime(2022, 1,  1), datetime(2022, 12, 31)),
    "full_2020_2024":  (datetime(2020, 1,  1), datetime(2024, 12, 31)),
}


def main():
    parser = argparse.ArgumentParser(description="Backtest AdDivergenceStrategy")
    parser.add_argument('--period', choices=list(PERIODS.keys()), help='Période prédéfinie')
    parser.add_argument('--start', type=str, help='Date début YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='Date fin YYYY-MM-DD')
    parser.add_argument('--max_stocks', type=int, default=5, help='Nombre max de positions (défaut: 5)')
    args = parser.parse_args()

    if args.period:
        start_date, end_date = PERIODS[args.period]
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date   = datetime.strptime(args.end,   '%Y-%m-%d')
    else:
        start_date, end_date = PERIODS["sideways_2015"]

    print("=" * 60)
    print("BACKTEST - RangeVapStrategy (Volume Profile + filtre ADX)")
    print("=" * 60)
    print(f"Période       : {start_date.date()} -> {end_date.date()}")
    print(f"Max positions : {args.max_stocks}")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=RangeVapBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000
    )

    # RangeVapBTWrapper a une signature de params différente des wrappers
    # momentum (pas de scoring_type/use_sue_filter/db_path) -> on bypass run().
    def _patched_run():
        engine._configure_broker()
        engine._load_data()
        engine._add_analyzers()
        engine.cerebro.addstrategy(
            RangeVapBTWrapper,
            start_date=engine.start_date,
            end_date=engine.end_date,
            dataframes=engine.dataframes,
            max_stocks=args.max_stocks,
        )
        results_bt = engine.cerebro.run()
        strat = results_bt[0]
        result_dict = {
            "final_value": engine.cerebro.broker.getvalue(),
            "pnl":         engine.cerebro.broker.getvalue() - engine.initial_cash,
            "trades":      engine._convert_to_serializable(strat.analyzers.trades.get_analysis()),
            "sharpe":      engine._convert_to_serializable(strat.analyzers.sharpe.get_analysis()),
            "drawdown":    engine._convert_to_serializable(strat.analyzers.drawdown.get_analysis()),
            "start_date":  engine.start_date.strftime("%Y-%m-%d"),
            "end_date":    engine.end_date.strftime("%Y-%m-%d"),
            "initial_cash": engine.initial_cash,
        }
        engine._save_results_json(result_dict)
        return result_dict

    results = _patched_run()

    print()
    print("=" * 60)
    print("RÉSULTATS")
    print("=" * 60)
    print(f"Valeur finale : {results['final_value']:>12,.2f}$")
    print(f"PnL           : {results['pnl']:>+12,.2f}$")
    print(f"PnL %         : {results['pnl'] / 100_000 * 100:>+11.1f}%")

    sharpe = results['sharpe'].get('sharperatio')
    if sharpe:
        print(f"Sharpe        : {sharpe:>12.3f}")

    dd = results['drawdown']
    if dd:
        max_dd = dd.get('max', {}).get('drawdown', 0)
        print(f"Max Drawdown  : {max_dd:>11.1f}%")

    t = results['trades']
    total = t.get('total', {}).get('total', 0)
    won   = t.get('won',   {}).get('total', 0)
    lost  = t.get('lost',  {}).get('total', 0)
    if total:
        print(f"Trades        : {total:>12} (W:{won} / L:{lost})")
        print(f"Win Rate      : {won/total*100:>11.1f}%")
    else:
        print("Trades        :            0 (aucune entrée déclenchée)")
    print("=" * 60)


if __name__ == "__main__":
    main()
