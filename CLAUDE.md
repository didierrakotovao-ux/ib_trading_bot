# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Système de trading algorithmique pour Interactive Brokers avec scoring ML et backtesting. Le code et les commentaires sont principalement en français.

## Commands

### Setup
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Data Import
```bash
python src/app/database/import_yahoo_bulk.py
python src/app/database/import_canada_full.py
python src/app/database/import_nasdaq_full.py
python src/app/database/update_historical_data.py
```

### Live Trading (requires IB Gateway/TWS on port 4002)
```bash
python src/app/trading.py
```

### Backtesting
```bash
cd src/app/backtest
python test_trailing_only.py      # Trailing stop strategy
python test_exhaustion_stop.py    # Exhaustion-based exit
python compare_journals.py <file1.csv> <file2.csv>  # Compare results
python debug_entries.py           # Debug entry decisions
```

## Architecture

### Core Components

```
Trading (trading.py)
    ├── MarketDataProvider (IB API wrapper + yfinance fallback)
    ├── Strategies (momentum.py, addivergence.py)
    │   └── MLScoring (ml_scoring.py) → models/*.pkl
    └── PositionManager (position lifecycle)

Backtest (backtest/)
    ├── BacktestEngine (engine.py) → Backtrader
    └── Wrappers (trailing_only_wrapper.py, exhaustion_stop_wrapper.py)
        └── Outputs: trading_journal_*.csv, diagnostique_*.txt
```

### Strategy Interface
Toutes les stratégies héritent de `Strategy` et implémentent:
- `get_symbols(trade_date)` → liste des symboles à trader
- `get_order_params()` → paramètres d'ordres IB
- `scanner_filters()` → filtres de screener IB

### ML Scoring Pipeline
Features calculées dans `ml_scoring.py._create_features()`:
- Indicateurs techniques: RSI, MACD, ADX, Bollinger Bands
- Métriques de volume: hl_sma20vol, oc_sma20vol, volume_ratio
- Returns: 5/10/20 jours
- Volatilité et position vs 52-week high

Score 0-100, seuil d'entrée: 65

### Backtest Wrappers
Les wrappers Backtrader (`*_bt_wrapper.py`) adaptent les stratégies métier:
- Utilisent `MarketDataMock` pour simuler les données
- Génèrent des journaux CSV horodatés
- Écrivent des logs de diagnostic

## Key Configuration Values

| Parameter | Value | Location |
|-----------|-------|----------|
| IB Port | 4002 | market_data_provider.py |
| Score Threshold | 65 | strategies/*.py |
| Max Stocks | 5 | strategies/*.py |
| Lookback Days | 350 | strategies/*.py |
| Trailing Stop | 5% | wrappers |
| Capital | 100,000$ | backtest/engine.py |

## Database

SQLite `trading_data.db` avec table principale `historical_data`:
- Colonnes: symbol, date, open, high, low, close, volume, adjusted_close
- Filtres par défaut: price 5-1000$, volume 500k+

## Import Patterns

Le projet utilise `sys.path.append()` pour les imports:
```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_scoring import MLScoring
```
