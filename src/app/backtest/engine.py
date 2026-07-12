import os
import backtrader as bt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from addivergence_bt_wrapper import AdDivergenceBTWrapper
from market_data_mock import MarketDataMock
import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.database.pg_connection import read_sql


class BacktestEngine:
    """
    Moteur de backtest générique basé sur Backtrader
    """

    def __init__(
        self,
        strategy_cls,
        start_date,
        end_date,
        initial_cash=100000,
        commission=0.001,
        stake=100,
        lookback_days=350,  # Jours de données historiques pour le scoring
        scoring_type="smooth_ml",
        smooth_model_path=None,  # Chemin du modèle smooth_ml (None = modèle US par défaut)
        wyckoff_model_path=None,  # Chemin du modèle wyckoff_ml
        use_sue_filter=False,  # Filtre SUE (Novy-Marx)
        sue_threshold=0.0,  # Seuil SUE
        max_symbols=None,  # Limiter le nombre de symboles (None = tous)
        db_path=None,  # Chemin DB (None = trading_data.db)
        min_price=5.0,  # Prix minimum
        min_volume=500000  # Volume minimum
    ):
        self.strategy_cls = strategy_cls
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.stake = stake
        self.lookback_days = lookback_days
        self.scoring_type = scoring_type
        self.smooth_model_path = smooth_model_path
        self.wyckoff_model_path = wyckoff_model_path
        self.use_sue_filter = use_sue_filter
        self.sue_threshold = sue_threshold
        self.max_symbols = max_symbols
        self.min_price = min_price
        self.min_volume = min_volume
        self.market_data = MarketDataMock(None)

        # Résoudre le chemin de la BD
        if db_path:
            if os.path.isabs(db_path):
                self.db_path = db_path
            else:
                project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
                self.db_path = os.path.abspath(os.path.join(project_root, db_path))
        else:
            self.db_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'trading_data.db')
            )

        # stdstats=False: pas d'observateurs par défaut
        # runonce=False: permet l'exécution même si les données ne sont pas synchronisées
        self.cerebro = bt.Cerebro(stdstats=False, runonce=False)

    # ------------------------------------------------------------------
    def _load_data(self):
        self.dataframes = {}
        # Charger les données depuis (start_date - lookback_days) pour avoir assez d'historique pour le scoring
        data_start_date = self.start_date - timedelta(days=self.lookback_days + 100)  # +100 pour marge (252 jours trading = ~365 calendaires)

        query = """
            SELECT symbol, date, open, high, low, close, volume, adjusted_close
            FROM historical_data
            WHERE date BETWEEN %s AND %s
            AND close >= %s
            AND volume >= %s
            ORDER BY symbol, date
        """
        df = read_sql(
            query,
            (data_start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'),
             self.min_price, self.min_volume)
        )
        if df.empty:
            print("[INFO] Aucun symbole ne passe le screener.")
            return

        print(f"[DATA] Chargement: {data_start_date.date()} -> {self.end_date.date()}")
        print(f"[DATA] {len(df)} lignes, {df['symbol'].nunique()} symboles")

        # Limiter le nombre de symboles si spécifié
        unique_symbols = df['symbol'].unique()
        if self.max_symbols and len(unique_symbols) > self.max_symbols:
            unique_symbols = unique_symbols[:self.max_symbols]
            df = df[df['symbol'].isin(unique_symbols)]
            print(f"[DATA] Limité à {self.max_symbols} symboles pour le test")

        # Stocker tous les DataFrames pour le cache
        # Backtrader attend une première barre sur tous les flux ajoutés avant
        # d'appeler next() : un symbole dont la 1ère ligne disponible tombe
        # après start_date (ex: penny stock qui ne passe le filtre prix/volume
        # qu'un seul jour, tard dans la période) bloque next() pour TOUT le
        # cerebro jusqu'à cette date -> on l'exclut.
        start_ts = pd.Timestamp(self.start_date)
        symbols_added = 0
        symbols_skipped = 0
        for symbol, group in df.groupby("symbol"):
            group = group.copy()
            group['date'] = pd.to_datetime(group['date'])
            group.set_index('date', inplace=True)

            if group.index.min() > start_ts:
                symbols_skipped += 1
                continue

            self.dataframes[symbol] = group

            # Ajouter chaque symbole à Backtrader
            data = bt.feeds.PandasData(
                dataname=group,
                name=symbol
            )
            self.cerebro.adddata(data)
            symbols_added += 1

        print(f"[DATA] {symbols_added} symboles ajoutés à Backtrader "
              f"({symbols_skipped} exclus: données débutant après {self.start_date.date()})")
    # ------------------------------------------------------------------
    def _configure_broker(self):
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)

        self.cerebro.addsizer(
            bt.sizers.FixedSize,
            stake=self.stake
        )

    # ------------------------------------------------------------------
    def _add_analyzers(self):
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")

    # ------------------------------------------------------------------
    def run(self):
        self._configure_broker()
        self._load_data()
        self._add_analyzers()

        self.cerebro.addstrategy(
            self.strategy_cls,
            start_date=self.start_date,
            end_date=self.end_date,
            dataframes=self.dataframes,  # Passer les DataFrames complets pour le cache
            scoring_type=self.scoring_type,
            smooth_model_path=self.smooth_model_path,
            wyckoff_model_path=self.wyckoff_model_path,
            use_sue_filter=self.use_sue_filter,
            sue_threshold=self.sue_threshold,
            db_path=self.db_path
        )

        results = self.cerebro.run()
        strat = results[0]

        result_dict = {
            "final_value": self.cerebro.broker.getvalue(),
            "pnl": self.cerebro.broker.getvalue() - self.initial_cash,
            "trades": self._convert_to_serializable(strat.analyzers.trades.get_analysis()),
            "sharpe": self._convert_to_serializable(strat.analyzers.sharpe.get_analysis()),
            "drawdown": self._convert_to_serializable(strat.analyzers.drawdown.get_analysis()),
            "returns": self._convert_to_serializable(strat.analyzers.returns.get_analysis()),
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_cash": self.initial_cash,
        }

        # Sauvegarder les résultats en JSON pour le dashboard
        self._save_results_json(result_dict)

        return result_dict

    def _convert_to_serializable(self, obj):
        """Convertit les objets numpy/OrderedDict en types Python natifs."""
        import numpy as np
        from collections import OrderedDict

        if isinstance(obj, (OrderedDict, dict)):
            # Convertir les clés datetime en strings
            return {
                (k.strftime("%Y-%m-%d") if hasattr(k, 'strftime') else str(k)):
                self._convert_to_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'strftime'):  # datetime objects
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return obj

    def _save_results_json(self, results):
        """Sauvegarde les résultats dans un fichier JSON."""
        import json
        resultats_dir = os.path.join(os.path.dirname(__file__), 'resultats')
        os.makedirs(resultats_dir, exist_ok=True)
        filename = os.path.join(resultats_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[RESULTS] Sauvegardé: {filename}")
    
    def plot(self):
        try:
            self.cerebro.plot()
        except (IndexError, Exception) as e:
            print(f"[WARN] Impossible de générer le graphique: {e}") 

if __name__ == "__main__":
    # Exemple d'utilisation
    start_date=datetime(2025, 5, 1)
    end_date=datetime(2025, 9, 30)

    engine = BacktestEngine(
        strategy_cls=AdDivergenceBTWrapper,
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000
    )
    results = engine.run()

    # engine.plot()

    print("Final Value:", results["final_value"])
    print("PnL:", results["pnl"])
    print("Sharpe:", results["sharpe"])
    print("Drawdown:", results["drawdown"])
    print("Trades:", results["trades"])