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
from typing import Optional
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
    parser.add_argument('--compare-all', action='store_true',
                        help='Lance smooth_ml + wyckoff_ml + regime_switch sur la même période')
    parser.add_argument('--cooldown', type=int, default=30,
                        help='Jours sans réentrée après stop-out (défaut: 30, 0=désactivé)')
    parser.add_argument('--scoring', choices=['smooth_ml', 'wyckoff_ml', 'regime_switch'], default='smooth_ml',
                        help='Type de scoring (défaut: smooth_ml)')
    parser.add_argument('--smooth-model', type=str, default=None,
                        help='Chemin du modèle smooth (optionnel)')
    parser.add_argument('--wyckoff-model', type=str, default=None,
                        help='Chemin du modèle wyckoff (optionnel)')
    # Seuils par défaut 18/27 : la grille de sensibilité (2026-07-12) montre
    # une surface plate (principe robuste) avec un léger avantage aux bandes
    # larges — moins de bascules. 20/25 était la cellule la plus faible.
    parser.add_argument('--vix-risk-on', type=float, default=18.0,
                        help='Seuil VIX pour retourner en mode Momentum (regime_switch)')
    parser.add_argument('--vix-risk-off', type=float, default=27.0,
                        help='Seuil VIX pour basculer en mode Wyckoff (regime_switch)')
    parser.add_argument('--switch-mom-trailing', type=float, default=8.0,
                        help='Trailing %% utilisé par Momentum en mode regime_switch')
    # 8.0 : DOIT rester aligné sur les labels du modèle wyckoff
    # (triple-barrière trailing 8% — stop_config.json). Un trailing 5%
    # exécute une autre stratégie que celle que le modèle a apprise.
    parser.add_argument('--switch-wy-trailing', type=float, default=8.0,
                        help='Trailing %% utilisé par Wyckoff en mode regime_switch')
    args = parser.parse_args()



    if args.period:
        start_date, end_date = PERIODS[args.period]
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date   = datetime.strptime(args.end,   '%Y-%m-%d')
    else:
        start_date = datetime(2025, 4, 22)
        end_date   = datetime(2025, 10, 7)

    cooldown_desc = f"{args.cooldown}j" if args.cooldown > 0 else "désactivé"

    def _run_single_mode(scoring_mode: str, trailing_value: Optional[float]):
        print("=" * 60)
        print("BACKTEST - TRAILING STOP ONLY (pas de TP)")
        print("=" * 60)
        print(f"Période       : {start_date.date()} -> {end_date.date()}")
        print(f"Scoring       : {scoring_mode}")
        if scoring_mode == 'regime_switch':
            print(f"VIX Switch    : risk_on<={args.vix_risk_on:.1f} | risk_off>={args.vix_risk_off:.1f}")
            print(f"Trailing      : momentum={args.switch_mom_trailing}% | wyckoff={args.switch_wy_trailing}%")
        else:
            print(f"Trailing Stop : {trailing_value}%")
        print(f"Cooldown      : {cooldown_desc}")
        print("=" * 60)

        engine = BacktestEngine(
            strategy_cls=TrailingOnlyBTWrapper,
            start_date=start_date,
            end_date=end_date,
            initial_cash=100000,
            scoring_type=scoring_mode,
            smooth_model_path=args.smooth_model,
            wyckoff_model_path=args.wyckoff_model,
        )

        engine._configure_broker()
        engine._load_data()
        engine._add_analyzers()
        engine.cerebro.addstrategy(
            TrailingOnlyBTWrapper,
            start_date=engine.start_date,
            end_date=engine.end_date,
            dataframes=engine.dataframes,
            scoring_type=scoring_mode,
            smooth_model_path=engine.smooth_model_path,
            wyckoff_model_path=engine.wyckoff_model_path,
            use_sue_filter=engine.use_sue_filter,
            sue_threshold=engine.sue_threshold,
            db_path=engine.db_path,
            trailing_percent=trailing_value if trailing_value is not None else 8.0,
            regime_risk_on=args.vix_risk_on,
            regime_risk_off=args.vix_risk_off,
            regime_momentum_trailing=args.switch_mom_trailing,
            regime_wyckoff_trailing=args.switch_wy_trailing,
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
            "scoring": scoring_mode,
        }
        engine._save_results_json(result_dict)
        return result_dict

    def _print_results(results):
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
        won = t.get('won', {}).get('total', 0)
        lost = t.get('lost', {}).get('total', 0)
        if total:
            print(f"Trades        : {total:>12} (W:{won} / L:{lost})")
            print(f"Win Rate      : {won/total*100:>11.1f}%")
        print("=" * 60)

    if args.compare_all:
        runs = [
            ('smooth_ml', args.trailing if args.trailing is not None else 8.0),
            # 8.0 : aligné sur les labels du modèle wyckoff (trailing 8%)
            ('wyckoff_ml', args.trailing if args.trailing is not None else 8.0),
            ('regime_switch', None),
        ]
        all_results = []
        for scoring_mode, trailing_value in runs:
            results = _run_single_mode(scoring_mode, trailing_value)
            _print_results(results)
            all_results.append(results)

        print()
        print("=" * 72)
        print("COMPARATIF DES 3 MODES")
        print("=" * 72)
        print(f"{'Mode':<16}{'PnL $':>14}{'PnL %':>10}{'Sharpe':>10}{'MaxDD%':>10}{'Trades':>10}")
        for r in all_results:
            mode = r.get('scoring', 'n/a')
            pnl = r.get('pnl', 0.0)
            pnl_pct = pnl / 100000 * 100
            sharpe = r.get('sharpe', {}).get('sharperatio')
            dd = r.get('drawdown', {}).get('max', {}).get('drawdown', 0.0)
            trades = r.get('trades', {}).get('total', {}).get('total', 0)
            sharpe_str = f"{sharpe:.3f}" if sharpe is not None else "n/a"
            print(f"{mode:<16}{pnl:>+14,.2f}{pnl_pct:>+10.1f}{sharpe_str:>10}{dd:>10.1f}{trades:>10}")
        print("=" * 72)
    else:
        # 8.0 pour tous les modes : aligné sur stop_config.json et les labels
        # triple-barrière des deux modèles
        trailing_value = args.trailing if args.trailing is not None else 8.0
        results = _run_single_mode(args.scoring, trailing_value)
        _print_results(results)


if __name__ == "__main__":
    main()
