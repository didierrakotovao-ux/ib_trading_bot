"""
Script de test pour le wrapper TrailingOnly (sans Take Profit).

Options :
  --period      : bull_2023_2024 | covid_crash | covid_recovery | bear_2022 | full_2020_2024
  --start/--end : dates libres YYYY-MM-DD
  --trailing    : % de trailing stop (défaut: 5)
  --cooldown    : jours d'interdiction de réentrée après stop-out (défaut: 30, 0=désactivé)
"""
import sys
import os
import argparse
sys.path.insert(0, '.')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datetime import datetime
from engine import BacktestEngine
from trailing_only_wrapper import TrailingOnlyBTWrapper

PERIODS = {
    "bull_2023_2024":  (datetime(2023, 1,  1), datetime(2024, 12, 31)),
    "covid_crash":     (datetime(2020, 2, 18), datetime(2020, 3,  25)),
    "covid_recovery":  (datetime(2020, 4,  1), datetime(2021, 12, 31)),
    "bear_2022":       (datetime(2022, 1,  1), datetime(2022, 12, 31)),
    "full_2020_2024":  (datetime(2020, 1,  1), datetime(2024, 12, 31)),
}


def main():
    parser = argparse.ArgumentParser(description="Backtest TrailingOnly")
    parser.add_argument('--period', choices=list(PERIODS.keys()), help='Période prédéfinie')
    parser.add_argument('--start', type=str, help='Date début YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='Date fin YYYY-MM-DD')
    parser.add_argument('--use_fondamental_data', action='store_true', help='Utiliser les données fondamentales')
    parser.add_argument('--trailing', type=float, default=None,
                        help='Trailing stop %% (défaut: 8 pour smooth_ml, 5 pour wyckoff_ml)')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Jours sans réentrée après stop-out (défaut: 30, 0=désactivé)')
    parser.add_argument('--scoring', choices=['smooth_ml', 'wyckoff_ml'], default='smooth_ml',
                        help='Type de scoring (défaut: smooth_ml)')
    parser.add_argument('--smooth-model', type=str, default=None,
                        help='Chemin du modèle smooth (optionnel)')
    parser.add_argument('--wyckoff-model', type=str, default=None,
                        help='Chemin du modèle wyckoff (optionnel)')
    args = parser.parse_args()



    if args.period:
        start_date, end_date = PERIODS[args.period]
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date   = datetime.strptime(args.end,   '%Y-%m-%d')
    else:
        start_date = datetime(2025, 4, 22)
        end_date   = datetime(2025, 10, 7)

    trailing_value = args.trailing if args.trailing is not None else (
        8.0 if args.scoring == 'smooth_ml' else 5.0
    )
    cooldown_desc = f"{args.cooldown}j" if args.cooldown > 0 else "désactivé"

    print("=" * 60)
    print("BACKTEST - TRAILING STOP ONLY (pas de TP)")
    print("=" * 60)
    print(f"Période       : {start_date.date()} -> {end_date.date()}")
    print(f"Scoring       : {args.scoring}")
    print(f"Trailing Stop : {trailing_value}%")
    print(f"Cooldown      : {cooldown_desc}")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=TrailingOnlyBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000,
        scoring_type=args.scoring,
        smooth_model_path=args.smooth_model,
        wyckoff_model_path=args.wyckoff_model,
    )

    # Patch pour injecter trailing_percent et cooldown_days
    def _patched_run():
        engine._configure_broker()
        engine._load_data()
        engine._add_analyzers()
        engine.cerebro.addstrategy(
            TrailingOnlyBTWrapper,
            start_date=engine.start_date,
            end_date=engine.end_date,
            dataframes=engine.dataframes,
            scoring_type=engine.scoring_type,
            smooth_model_path=engine.smooth_model_path,
            wyckoff_model_path=engine.wyckoff_model_path,
            use_sue_filter=engine.use_sue_filter,
            sue_threshold=engine.sue_threshold,
            db_path=engine.db_path,
            trailing_percent=trailing_value,
            cooldown_days=args.cooldown,
            use_fondamental_data=args.use_fondamental_data,
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
    print("=" * 60)


if __name__ == "__main__":
    main()
