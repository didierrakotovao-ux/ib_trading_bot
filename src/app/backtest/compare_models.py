"""
Script de comparaison des différents modèles ML via backtest.

Compare:
1. Smooth Momentum (modèle actuel)
2. Earnings Momentum (avec SUE/CAR3)
3. Earnings comme filtre uniquement (SUE > 0)

Usage:
    python compare_models.py
    python compare_models.py --period bull      # Période haussière 2023-2024
    python compare_models.py --period crash     # COVID crash
    python compare_models.py --period recovery  # Post-COVID recovery
    python compare_models.py --all              # Toutes les périodes
"""
import sys
import os
sys.path.insert(0, '.')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datetime import datetime
import json
import pandas as pd
from engine import BacktestEngine
from trailing_only_wrapper import TrailingOnlyBTWrapper


# Périodes de test
PERIODS = {
    'bull_2023_2024': {
        'name': 'Bull Market 2023-2024',
        'start': datetime(2023, 10, 31),
        'end': datetime(2024, 7, 31),
    },
    'covid_crash': {
        'name': 'COVID Crash',
        'start': datetime(2020, 2, 18),
        'end': datetime(2020, 3, 25),
    },
    'covid_recovery': {
        'name': 'COVID Recovery',
        'start': datetime(2020, 3, 18),
        'end': datetime(2020, 12, 31),
    },
    'bear_2022': {
        'name': 'Bear Market 2022',
        'start': datetime(2022, 1, 1),
        'end': datetime(2022, 10, 31),
    },
    'full_2020_2024': {
        'name': 'Full Period 2020-2024',
        'start': datetime(2020, 1, 1),
        'end': datetime(2024, 12, 31),
    },
}

# Configurations des modèles à tester
MODELS = {
    'smooth_momentum': {
        'name': 'Smooth Momentum',
        'scoring_type': 'smooth_ml',
        'use_earnings_filter': False,
    },
    'earnings_momentum': {
        'name': 'Earnings Momentum',
        'scoring_type': 'earnings_ml',
        'use_earnings_filter': False,
    },
    'smooth_with_sue_filter': {
        'name': 'Smooth + SUE Filter',
        'scoring_type': 'smooth_ml',
        'use_earnings_filter': True,
        'sue_threshold': 0,  # SUE > 0
    },
}


def run_backtest(period_key: str, model_key: str) -> dict:
    """
    Lance un backtest pour une période et un modèle donnés.

    Returns:
        Dict avec les résultats
    """
    period = PERIODS[period_key]
    model = MODELS[model_key]

    print(f"\n{'='*60}")
    print(f"BACKTEST: {model['name']} | {period['name']}")
    print(f"Période: {period['start'].date()} -> {period['end'].date()}")
    print(f"Scoring type: {model['scoring_type']}")
    print(f"{'='*60}")

    try:
        engine = BacktestEngine(
            strategy_cls=TrailingOnlyBTWrapper,
            start_date=period['start'],
            end_date=period['end'],
            initial_cash=100000,
            scoring_type=model['scoring_type'],
            use_sue_filter=model.get('use_earnings_filter', False),
            sue_threshold=model.get('sue_threshold', 0.0)
        )

        results = engine.run()

        # Extraire les métriques clés
        trades_info = results.get('trades', {})
        total_trades = trades_info.get('total', {}).get('total', 0)
        won = trades_info.get('won', {}).get('total', 0)
        lost = trades_info.get('lost', {}).get('total', 0)

        win_rate = (won / total_trades * 100) if total_trades > 0 else 0

        sharpe = results.get('sharpe', {}).get('sharperatio', None)
        if sharpe is None:
            sharpe = 0

        drawdown = results.get('drawdown', {})
        max_dd = drawdown.get('max', {}).get('drawdown', 0)

        metrics = {
            'period': period['name'],
            'model': model['name'],
            'final_value': results['final_value'],
            'pnl': results['pnl'],
            'pnl_pct': (results['pnl'] / 100000) * 100,
            'total_trades': total_trades,
            'won': won,
            'lost': lost,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
        }

        print(f"\nRésultats:")
        print(f"  PnL: {metrics['pnl']:+,.2f}$ ({metrics['pnl_pct']:+.2f}%)")
        print(f"  Trades: {total_trades} (W:{won} / L:{lost})")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Sharpe: {sharpe:.2f}" if sharpe else "  Sharpe: N/A")
        print(f"  Max Drawdown: {max_dd:.2f}%")

        return metrics

    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'period': period['name'],
            'model': model['name'],
            'error': str(e)
        }


def run_comparison(periods: list = None, models: list = None):
    """
    Lance les backtests comparatifs.

    Args:
        periods: Liste des clés de périodes (None = toutes)
        models: Liste des clés de modèles (None = tous)
    """
    if periods is None:
        periods = list(PERIODS.keys())
    if models is None:
        models = list(MODELS.keys())

    results = []

    for period_key in periods:
        for model_key in models:
            metrics = run_backtest(period_key, model_key)
            results.append(metrics)

    # Créer un DataFrame récapitulatif
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("RÉCAPITULATIF COMPARATIF")
    print("="*80)

    # Afficher par période
    for period_key in periods:
        period_name = PERIODS[period_key]['name']
        print(f"\n{period_name}:")
        print("-" * 60)

        period_df = df[df['period'] == period_name]

        if 'error' not in period_df.columns or period_df['error'].isna().all():
            summary = period_df[['model', 'pnl', 'pnl_pct', 'total_trades',
                                 'win_rate', 'sharpe', 'max_drawdown']]
            print(summary.to_string(index=False))
        else:
            print(period_df[['model', 'error']].to_string(index=False))

    # Sauvegarder les résultats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"backtest_comparison_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[OK] Résultats sauvegardés: {output_file}")

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comparaison des modèles ML")
    parser.add_argument('--period', type=str, choices=list(PERIODS.keys()),
                        help='Période spécifique à tester')
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()),
                        help='Modèle spécifique à tester')
    parser.add_argument('--all', action='store_true',
                        help='Tester toutes les combinaisons')

    args = parser.parse_args()

    periods = [args.period] if args.period else None
    models = [args.model] if args.model else None

    if args.all:
        periods = None
        models = None

    # Par défaut, tester seulement la période bull avec tous les modèles
    if periods is None and not args.all:
        periods = ['bull_2023_2024']

    run_comparison(periods=periods, models=models)


if __name__ == "__main__":
    main()
