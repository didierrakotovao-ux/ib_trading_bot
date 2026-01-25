# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Système de trading algorithmique pour Interactive Brokers avec scoring ML et backtesting. Le code et les commentaires sont principalement en français.

**Key Stats:**
- ~4,740 lines of Python code across 30 modules
- 1 pre-trained XGBoost model (momentum_model.pkl)
- SQLite database with 9 tables
- 144 tracked symbols (US mega-caps and large-caps)

## Directory Structure

```
ib_trading_bot/
├── src/app/
│   ├── trading.py                    # Main orchestration class
│   ├── position_manager.py           # Position lifecycle management
│   ├── strategies/
│   │   ├── strategy.py               # Base strategy interface
│   │   ├── momentum.py               # ML-based momentum strategy (active)
│   │   └── addivergence.py           # A/D divergence strategy
│   ├── ml/
│   │   ├── ml_scoring.py             # ML model scoring engine
│   │   ├── ml_momentum_predictor.py  # Model training (XGBoost)
│   │   ├── scoring.py                # Base scoring interface
│   │   ├── addivergencescoring.py    # A/D divergence scoring
│   │   └── momentum_scoring.py
│   ├── database/
│   │   ├── db_manager.py             # SQLite database manager
│   │   ├── trade_journal.py          # Trade logging
│   │   └── import_yahoo_bulk.py      # Bulk data import
│   ├── screener/providers/
│   │   └── market_data_provider.py   # IB API wrapper + yfinance fallback
│   └── backtest/
│       ├── engine.py                 # Backtrader engine
│       ├── trailing_only_wrapper.py  # Trailing stop strategy
│       ├── exhaustion_stop_wrapper.py # Exhaustion-based exit
│       ├── market_data_mock.py       # Mock data provider for backtest
│       ├── order_translator.py       # IB order to Backtrader converter
│       ├── test_trailing_only.py     # Trailing stop test runner
│       ├── test_exhaustion_stop.py   # Exhaustion stop test runner
│       ├── compare_journals.py       # Compare backtest results
│       ├── debug_entries.py          # Debug entry decisions
│       └── backtest_dashboard.py     # Results visualization
├── models/
│   └── momentum_model.pkl            # Trained XGBoost model
├── trading_data.db                   # SQLite database
├── scanner_params.xml                # IB Scanner configuration
├── requirements.txt
└── CLAUDE.md
```

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Data Import
```bash
python src/app/database/import_yahoo_bulk.py
```

### Live Trading (requires IB Gateway/TWS)
```bash
python src/app/trading.py
```

### Backtesting
```bash
cd src/app/backtest
python test_trailing_only.py                    # Trailing stop strategy
python test_exhaustion_stop.py                  # Exhaustion-based exit
python compare_journals.py <file1.csv> <file2.csv>  # Compare results
python debug_entries.py                         # Debug entry decisions
```

## Architecture

### Core Components

```
Trading (trading.py)
    ├── MarketDataProvider (IB API wrapper + yfinance fallback)
    ├── Strategies (momentum.py, addivergence.py)
    │   └── MLScoring (ml_scoring.py) → models/*.pkl
    ├── PositionManager (position lifecycle)
    └── TradeJournal (database logging)

Backtest (backtest/)
    ├── BacktestEngine (engine.py) → Backtrader
    ├── MarketDataMock (cached data provider)
    └── Wrappers (trailing_only_wrapper.py, exhaustion_stop_wrapper.py)
        └── Outputs: trading_journal_*.csv, diagnostique_*.txt
```

### Strategy Interface

Toutes les stratégies héritent de `Strategy` (strategy.py) et implémentent:
- `get_symbols(trade_date)` → liste des symboles à trader
- `get_order_params()` → paramètres d'ordres IB
- `scanner_filters()` → filtres de screener IB

### Trading Strategies

**MomentumStrategy (active):**
- ML-based prediction (5% gain in 20 days target)
- Scanner: IB MOST_ACTIVE on NASDAQ
- Entry: LMT @ close × 1.005 (0.5% buffer)
- Exit: 5% trailing stop (GTC)

**AdDivergenceStrategy (inactive):**
- A/D divergence technical trading
- Max 60% capital exposure
- Exit: 5% trailing stop + 10% take profit bracket

## ML Scoring Pipeline

### Features (18 total)

Calculées dans `ml_scoring.py._create_features()`:

**Technical Indicators:**
- RSI (14-period)
- MACD & MACD Signal (12/26/9)
- ADX (14-period, trend strength)
- Bollinger Bands (20-period, 2 std dev)

**Volume Metrics:**
- `hl_sma20vol`: (High-Low) / SMA20(Volume)
- `oc_sma20vol`: (Open-Close) / SMA20(Volume)
- `volume_ratio`: Volume / SMA20(Volume)

**Momentum Metrics:**
- `return_5d`, `return_10d`, `return_20d`
- `volatility_10d` (10-day return std dev)
- `high_52w_pct` (close vs 52-week high)

**Derived Features:**
- `macd_hist`: MACD - Signal
- `bb_position`: Normalized Bollinger position
- `rsi_momentum`: RSI - 50
- `trend_strength`: ADX × sign(MACD)
- `pct_close`: Daily % change

### Score Interpretation
- Score 65+: BUY signal (entry threshold)
- Score 60-64: Watch
- Score 40-59: HOLD
- Score <40: AVOID

## Wyckoff Effort/Result Analysis

Intégration des principes Wyckoff pour analyser la relation effort (volume) vs résultat (prix).

### Wyckoff Features (5 indicateurs)

| Feature | Description |
|---------|-------------|
| `effort_result_ratio` | Volume normalisé / spread normalisé. Élevé = absorption (accumulation/distribution) |
| `volume_spread_analysis` | Spread × volume × direction. Mesure la force du mouvement |
| `wyckoff_accumulation` | Détecte les phases d'absorption (fort volume, range étroit) |
| `effort_result_divergence` | Divergence entre direction prix et volume sur 5 jours |
| `smart_money_flow` | Flux cumulatif basé sur position du close dans le range |

### Phases Wyckoff Détectées

| Phase | Description | Action |
|-------|-------------|--------|
| ACCUMULATION | Smart money achète discrètement | Préparer achat |
| DISTRIBUTION | Smart money vend discrètement | Éviter/sortir |
| MARKUP | Tendance haussière confirmée | Acheter/tenir |
| MARKDOWN | Tendance baissière confirmée | Éviter |
| RANGING | Pas de direction claire | Attendre |

### Utilisation

```python
# Analyse Wyckoff seule
wyckoff = scoring.get_wyckoff_analysis(df)
print(wyckoff['phase'])  # ACCUMULATION, DISTRIBUTION, etc.

# Analyse combinée ML + Wyckoff (60% ML, 40% Wyckoff)
combined = scoring.get_combined_analysis(df)
print(combined['combined_score'])  # 0-100
print(combined['confidence'])  # HIGH si ML et Wyckoff concordent
```

### Configuration Stratégie

```python
# Activer Wyckoff dans MomentumStrategy
strategy = MomentumStrategy(
    market_data=provider,
    capital=100000,
    use_wyckoff=True,      # Activer analyse Wyckoff
    wyckoff_weight=0.4     # 40% du score combiné
)
```

## Trade Lifecycle

```
1. SCANNER PHASE
   └─ IB MOST_ACTIVE scanner → ~200 stocks

2. SCORING PHASE
   ├─ Fetch 350 days historical data
   ├─ MLScoring.score() → 0-100
   ├─ Filter: score >= 65
   └─ Select top 5 by score

3. ORDER GENERATION
   ├─ Entry: LMT @ close × 1.005
   └─ Stop: TRAIL 5% (child order, GTC)

4. EXECUTION
   ├─ Validate capital available
   ├─ Check no duplicate position
   └─ Place parent + child orders

5. POSITION MANAGEMENT
   ├─ PositionManager tracks state
   └─ Stop trigger → close + journal log
```

## Key Configuration Values

| Parameter | Value | Location |
|-----------|-------|----------|
| IB Port (Paper TWS) | 7497 | trading.py |
| IB Port (Live TWS) | 7496 | trading.py |
| IB Port (Paper Gateway) | 4002 | trading.py |
| IB Port (Live Gateway) | 4001 | trading.py |
| Score Threshold | 65 | strategies/momentum.py |
| Max Positions | 5 | trading.py |
| Capital per Position | $20,000 | trading.py (1/5 of total) |
| Total Capital | $100,000 | trading.py |
| Lookback Days | 350 | strategies/momentum.py |
| Trailing Stop | 5% | strategies/momentum.py |
| Commission Rate | 0.1% | trading.py, wrappers |
| Price Filter | $5-$1,000 | scanner_filters() |
| Volume Filter | 500k+ | scanner_filters() |

## Database Schema

SQLite `trading_data.db` avec 9 tables:

### Main Tables

**historical_data** - OHLCV + indicators
```sql
symbol, date, open, high, low, close, volume, adjusted_close,
sma20_volume, hl_sma20vol, oc_sma20vol, macd, macd_signal,
rsi, adx, bb_high, bb_low, pct_close
-- Unique: (symbol, date, source)
```

**trades** - Trade journal
```sql
trade_mode,         -- backtest/paper/live
strategy_name,
symbol, date_entree, prix_entree, quantite,
date_sortie, prix_sortie, cause_sortie,
pnl_brut, commission, pnl_net,
bars_held, score_entree, exhaustion_signals
```

**trading_signals** - Entry/exit signals
```sql
signal_type,        -- BUY/SELL/HOLD/ACCUMULATION/DISTRIBUTION/WATCH
strategy, price, confidence, target_price, stop_loss
```

**signal_outcomes** - ML labels
```sql
price_5d, price_10d, price_20d, max_gain, max_loss, roi
```

### Other Tables
- `technical_indicators` - Pre-calculated indicators
- `scanner_results` - Screener history
- `watchlist` - Symbol watchlist
- `symbol_metadata` - Company info

## Backtest Framework

### Wrappers

**TrailingOnlyBTWrapper:**
- Pure 5% trailing stop (no take profit)
- Mode: Let profits run

**ExhaustionStopBTWrapper:**
- Trailing stop activates after exhaustion detection
- Exhaustion signals (2+ required):
  - MACD histogram decreasing (3-day)
  - RSI in extreme zone (>70)
  - Volume ratio decreasing (5-day)
  - ADX decreasing (3-day)
- Minimum 3 bars before checking

### Outputs
- `trading_journal_*.csv` - Trade log
- `diagnostique_*.txt` - Diagnostic info

## Import Patterns

Le projet utilise `sys.path.append()` pour les imports cross-module:
```python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_scoring import MLScoring
```

## Code Conventions

### Language
- Code comments and variables: French
- Function/class names: English
- Documentation: French preferred

### Database Access
```python
conn = sqlite3.connect('trading_data.db', check_same_thread=False)
conn.row_factory = sqlite3.Row  # Dict-like access
```

### Error Handling
- Graceful fallback for missing libraries (pandas_ta)
- yfinance fallback when IB unavailable

## Testing Utilities

| File | Purpose |
|------|---------|
| `test_trailing_only.py` | Backtest with trailing stop |
| `test_exhaustion_stop.py` | Backtest with exhaustion detection |
| `debug_entries.py` | Analyze entry decision logic |
| `compare_journals.py` | Compare two backtest CSV results |
| `backtest_dashboard.py` | Visualize backtest performance |

## Dependencies

```
ibapi                 # Interactive Brokers API
yfinance              # Historical data download
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computing
scikit-learn>=1.3.0   # ML utilities (StandardScaler)
xgboost>=2.0.0        # XGBoost classifier
backtrader            # Backtesting framework
matplotlib>=3.7.0     # Visualization
seaborn>=0.12.0       # Statistical visualization
joblib>=1.3.0         # Model serialization
```

## AI Assistant Guidelines

### When Modifying Code
1. Preserve French comments and variable names
2. Follow existing import patterns with `sys.path.append()`
3. Use consistent commission rate (0.1%) in all calculations
4. Maintain SQLite `check_same_thread=False` for threading
5. Keep strategy interface consistent (inherit from `Strategy`)

### Common Tasks
- **Add new strategy**: Create in `strategies/`, inherit from `Strategy`
- **Modify ML features**: Edit `ml_scoring.py._create_features()`
- **Change trading params**: Update in `trading.py` or strategy files
- **Add backtest wrapper**: Create in `backtest/`, follow existing pattern

### Files to Avoid Modifying
- `models/momentum_model.pkl` - Trained model file
- `trading_data.db` - Production database
- `scanner_params.xml` - IB scanner config (use strategy methods instead)
