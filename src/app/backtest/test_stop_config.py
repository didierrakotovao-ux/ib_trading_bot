"""
Backtest avec la configuration de stops depuis stop_config.json.

Modifiez stop_config.json pour tester différentes combinaisons :
  - protection.type  : "fixed" ou "trailing"
  - protection.pct   : % de recul
  - profit.type      : "fixed" ou "dynamic"
  - profit.fixed_pct : % de hausse (si fixed)
  - profit.dynamic_atr_mult : multiplicateur ATR (si dynamic)

Usage :
    cd src/app/backtest
    python test_stop_config.py
    python test_stop_config.py --period bull_2023_2024
    python test_stop_config.py --start 2022-01-01 --end 2022-12-31
"""
import sys
import os
import argparse
sys.path.insert(0, '.')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datetime import datetime
from engine import BacktestEngine
from stop_config_bt_wrapper import StopConfigBTWrapper
from src.app.stop_manager import StopConfig

# ---------------------------------------------------------------------------
# Périodes prédéfinies (mêmes que compare_models.py)
# ---------------------------------------------------------------------------
PERIODS = {
    "bull_2023_2024":  (datetime(2023, 1,  1), datetime(2024, 12, 31)),
    "covid_crash":     (datetime(2020, 2, 18), datetime(2020, 3,  25)),
    "covid_recovery":  (datetime(2020, 4,  1), datetime(2021, 12, 31)),
    "bear_2022":       (datetime(2022, 1,  1), datetime(2022, 12, 31)),
    "full_2020_2024":  (datetime(2020, 1,  1), datetime(2024, 12, 31)),
}


def main():
    parser = argparse.ArgumentParser(
        description="Backtest avec configuration de stops (stop_config.json)"
    )
    parser.add_argument(
        '--period', choices=list(PERIODS.keys()),
        help='Période prédéfinie'
    )
    parser.add_argument('--start', type=str, help='Date début YYYY-MM-DD')
    parser.add_argument('--end',   type=str, help='Date fin   YYYY-MM-DD')
    parser.add_argument(
        '--config', type=str,
        default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'stop_config.json'),
        help='Chemin vers stop_config.json (défaut : racine du projet)'
    )
    args = parser.parse_args()

    # Résoudre les dates
    if args.period:
        start_date, end_date = PERIODS[args.period]
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date   = datetime.strptime(args.end,   '%Y-%m-%d')
    else:
        # Défaut : année en cours
        start_date = datetime(2024, 1, 1)
        end_date   = datetime(2024, 12, 31)

    # Charger la config
    config_path = os.path.abspath(args.config)
    stop_cfg = StopConfig.from_json(config_path)

    # Affichage
    prot_desc = f"{stop_cfg.protection_type.upper()} -{stop_cfg.protection_pct}%"
    prof_desc = (f"FIXED +{stop_cfg.profit_fixed_pct}%"
                 if stop_cfg.profit_type == "fixed"
                 else f"DYNAMIC {stop_cfg.profit_atr_mult}×ATR({stop_cfg.profit_atr_period})")

    print("=" * 60)
    print("BACKTEST — STOP CONFIG")
    print("=" * 60)
    print(f"Période    : {start_date.date()} → {end_date.date()}")
    print(f"Protection : {prot_desc}")
    print(f"Profit     : {prof_desc}")
    print(f"Config     : {config_path}")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=StopConfigBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100_000,
    )

    # Passer la config au wrapper via addstrategy kwargs
    # (BacktestEngine.run() appelle addstrategy — on surcharge ici)
    engine._stop_config = stop_cfg  # sera récupéré ci-dessous

    # Patch minimal : injecter 'config' dans les kwargs de addstrategy
    _original_run = engine.run

    def _patched_run():
        engine._configure_broker()
        engine._load_data()
        engine._add_analyzers()
        engine.cerebro.addstrategy(
            StopConfigBTWrapper,
            start_date=engine.start_date,
            end_date=engine.end_date,
            dataframes=engine.dataframes,
            scoring_type=engine.scoring_type,
            use_sue_filter=engine.use_sue_filter,
            sue_threshold=engine.sue_threshold,
            db_path=engine.db_path,
            config=stop_cfg,          # ← paramètre supplémentaire
        )
        import backtrader as bt
        results = engine.cerebro.run()
        strat = results[0]
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
    print("RÉSULTATS BACKTRADER")
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
