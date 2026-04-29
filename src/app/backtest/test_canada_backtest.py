"""
Backtest du pipeline momentum sur actions canadiennes (TSX/TSXV/NEO).

Valide la qualité du filtre CanadianScreener sur données historiques.
Configuration des stops : stop_config.json (identique au backtest US).

Usage :
    cd src/app/backtest
    python test_canada_backtest.py
    python test_canada_backtest.py --period bull_2023_2024
    python test_canada_backtest.py --period tsx_boom
    python test_canada_backtest.py --start 2022-01-01 --end 2022-12-31
    python test_canada_backtest.py --period full_2020_2024 --scoring smooth_ml
    python test_canada_backtest.py --period full_2020_2024 --min-volume 50000
"""
import sys
import os
import argparse
sys.path.insert(0, '.')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import backtrader as bt
from datetime import datetime
from engine import BacktestEngine
from canada_bt_wrapper import CanadaBTWrapper
from src.app.stop_manager import StopConfig

# ---------------------------------------------------------------------------
# Périodes prédéfinies
# ---------------------------------------------------------------------------
PERIODS = {
    # Mêmes périodes que US pour comparaison directe
    "bull_2023_2024":  (datetime(2023, 1,  1), datetime(2024, 12, 31)),
    "covid_crash":     (datetime(2020, 2, 18), datetime(2020, 3,  25)),
    "covid_recovery":  (datetime(2020, 4,  1), datetime(2021, 12, 31)),
    "bear_2022":       (datetime(2022, 1,  1), datetime(2022, 12, 31)),
    "full_2020_2024":  (datetime(2020, 1,  1), datetime(2024, 12, 31)),
    # Période spécifique TSX — boom matières premières / crypto-mining 2021
    "tsx_boom":        (datetime(2021, 1,  1), datetime(2021, 12, 31)),
}


def main():
    parser = argparse.ArgumentParser(
        description="Backtest pipeline momentum — actions canadiennes"
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
        help='Chemin vers stop_config.json'
    )
    parser.add_argument(
        '--scoring', choices=['smooth_ml', 'ml'], default='smooth_ml',
        help='Type de scoring ML (default: smooth_ml)'
    )
    parser.add_argument(
        '--min-volume', type=int, default=100000,
        help='Volume moyen minimum (default: 100000, marché CA plus illiquide que US)'
    )
    parser.add_argument(
        '--min-price', type=float, default=2.0,
        help='Prix minimum en CAD (default: 2.0)'
    )
    parser.add_argument(
        '--capital', type=int, default=100000,
        help='Capital initial (default: 100000)'
    )
    args = parser.parse_args()

    # Résoudre les dates
    if args.period:
        start_date, end_date = PERIODS[args.period]
    elif args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date   = datetime.strptime(args.end,   '%Y-%m-%d')
    else:
        start_date = datetime(2023, 1, 1)
        end_date   = datetime(2024, 12, 31)

    config_path = os.path.abspath(args.config)
    stop_cfg = StopConfig.from_json(config_path)

    prot_desc = f"{stop_cfg.protection_type.upper()} -{stop_cfg.protection_pct}%"
    prof_desc = (f"FIXED +{stop_cfg.profit_fixed_pct}%"
                 if stop_cfg.profit_type == "fixed"
                 else f"DYNAMIC {stop_cfg.profit_atr_mult}×ATR({stop_cfg.profit_atr_period})")

    print("=" * 60)
    print("BACKTEST — ACTIONS CANADIENNES (TSX/TSXV/NEO)")
    print("=" * 60)
    print(f"Période    : {start_date.date()} → {end_date.date()}")
    print(f"DB         : trading_data_ca.db")
    print(f"Scoring    : {args.scoring}")
    print(f"Min price  : {args.min_price:.2f}$ CAD")
    print(f"Min volume : {args.min_volume:,}")
    print(f"Protection : {prot_desc}")
    print(f"Profit     : {prof_desc}")
    print(f"Capital    : {args.capital:,}$")
    print("=" * 60)

    engine = BacktestEngine(
        strategy_cls=CanadaBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=args.capital,
        scoring_type=args.scoring,
        db_path="trading_data_ca.db",
        min_price=args.min_price,
        min_volume=args.min_volume,
    )

    _original_run = engine.run

    def _patched_run():
        engine._configure_broker()
        engine._load_data()
        engine._add_analyzers()
        engine.cerebro.addstrategy(
            CanadaBTWrapper,
            start_date=engine.start_date,
            end_date=engine.end_date,
            dataframes=engine.dataframes,
            scoring_type=engine.scoring_type,
            db_path=engine.db_path,
            config=stop_cfg,
            smooth_model_path="models/smooth_ml_ca.pkl",
        )
        results = engine.cerebro.run()
        strat = results[0]
        result_dict = {
            "final_value":  engine.cerebro.broker.getvalue(),
            "pnl":          engine.cerebro.broker.getvalue() - engine.initial_cash,
            "trades":       engine._convert_to_serializable(strat.analyzers.trades.get_analysis()),
            "sharpe":       engine._convert_to_serializable(strat.analyzers.sharpe.get_analysis()),
            "drawdown":     engine._convert_to_serializable(strat.analyzers.drawdown.get_analysis()),
            "start_date":   engine.start_date.strftime("%Y-%m-%d"),
            "end_date":     engine.end_date.strftime("%Y-%m-%d"),
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
    print(f"PnL %         : {results['pnl'] / args.capital * 100:>+11.1f}%")

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
