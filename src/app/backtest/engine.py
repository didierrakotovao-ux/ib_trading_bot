import backtrader as bt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from addivergence_bt_wrapper import AdDivergenceBTWrapper
from market_data_mock import MarketDataMock
import sqlite3


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
        lookback_days=350  # Jours de données historiques pour le scoring
    ):
        self.strategy_cls = strategy_cls
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.stake = stake
        self.lookback_days = lookback_days
        self.market_data = MarketDataMock(None)

        # stdstats=False: pas d'observateurs par défaut
        # runonce=False: permet l'exécution même si les données ne sont pas synchronisées
        self.cerebro = bt.Cerebro(stdstats=False, runonce=False)

    # ------------------------------------------------------------------
    def _load_data(self):
        self.dataframes = {}
        conn = sqlite3.connect("trading_data.db")

        # Charger les données depuis (start_date - lookback_days) pour avoir assez d'historique pour le scoring
        data_start_date = self.start_date - timedelta(days=self.lookback_days + 50)  # +50 pour marge de sécurité

        query = """
            SELECT symbol, date, open, high, low, close, volume, adjusted_close
            FROM historical_data
            WHERE date BETWEEN ? AND ?
            AND close >= 5.0
            AND close <= 500.0
            AND volume >= 500000
            ORDER BY symbol, date
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(data_start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
        )
        conn.close()
        if df.empty:
            print("[INFO] Aucun symbole ne passe le screener.")
            return

        print(f"[DATA] Chargement: {data_start_date.date()} -> {self.end_date.date()}")
        print(f"[DATA] {len(df)} lignes, {df['symbol'].nunique()} symboles")

        # Stocker tous les DataFrames pour le cache
        symbols_added = 0
        for symbol, group in df.groupby("symbol"):
            group = group.copy()
            group['date'] = pd.to_datetime(group['date'])
            group.set_index('date', inplace=True)
            self.dataframes[symbol] = group

            # Ajouter chaque symbole à Backtrader
            data = bt.feeds.PandasData(
                dataname=group,
                name=symbol
            )
            self.cerebro.adddata(data)
            symbols_added += 1

        print(f"[DATA] {symbols_added} symboles ajoutés à Backtrader")
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
            dataframes=self.dataframes  # Passer les DataFrames complets pour le cache
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
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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