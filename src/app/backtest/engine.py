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
        symbols,
        start_date,
        end_date,
        initial_cash=100000,
        commission=0.001,
        stake=100
    ):
        self.strategy_cls = strategy_cls
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.commission = commission
        self.stake = stake
        self.market_data = MarketDataMock(None)

        self.cerebro = bt.Cerebro(stdstats=False)

    # ------------------------------------------------------------------
    def _load_data(self):
        self.dataframes = {}
        conn = sqlite3.connect("trading_data.db")
        query = """
            SELECT symbol, date, open, high, low, close, volume, adjusted_close
            FROM historical_data
            WHERE date BETWEEN ? AND ?
            AND close >= 5.0
            AND close <= 500.0
            AND volume >= 500000
            ORDER BY symbol, date ASC
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
        )
        conn.close()
        if df.empty:
            print("[INFO] Aucun symbole ne passe le screener.")
            return

        # On groupe par symbole et on injecte chaque DataFrame dans Backtrader
        for symbol, group in df.groupby("symbol"):
            group['date'] = pd.to_datetime(group['date'])
            group.set_index('date', inplace=True)
            self.dataframes[symbol] = group
            data = bt.feeds.PandasData(
                dataname=group,
                name=symbol
            )
            self.cerebro.adddata(data)
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
            end_date=self.end_date
        )

        results = self.cerebro.run()
        strat = results[0]

        return {
            "final_value": self.cerebro.broker.getvalue(),
            "pnl": self.cerebro.broker.getvalue() - self.initial_cash,
            "trades": strat.analyzers.trades.get_analysis(),
            "sharpe": strat.analyzers.sharpe.get_analysis(),
            "drawdown": strat.analyzers.drawdown.get_analysis(),
            "returns": strat.analyzers.returns.get_analysis(),
        }
    
    def plot(self):
        self.cerebro.plot() 

if __name__ == "__main__":
    # Exemple d'utilisation
    start_date=datetime(2025, 8, 1)
    end_date=datetime(2025, 9, 30)

    engine = BacktestEngine(
        strategy_cls=AdDivergenceBTWrapper,
        symbols=[
            'AVGO', 'PEP', 'COST', 'ADBE', 'CRM', 'CSCO', 'ACN', 'NFLX', 'TMO', 'AMD',
            'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS',
            # Finance
            'BAC', 'WFC', 'MS', 'GS', 'C', 'SCHW', 'AXP', 'BLK', 'SPGI', 'CB',
            'MMC', 'PGR', 'TFC', 'USB', 'PNC', 'CME', 'MCO', 'AON', 'ICE', 'AFL',
            # Healthcare
            'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BMY', 'GILD', 'VRTX', 'CVS', 'ISRG', 'CI', 'MDT', 'REGN', 'HUM', 'ZTS',        
        ],
        start_date=start_date,
        end_date=end_date,
        initial_cash=100000
    )
    results = engine.run()

    engine.plot()

    print("Final Value:", results["final_value"])
    print("PnL:", results["pnl"])
    print("Sharpe:", results["sharpe"])
    print("Drawdown:", results["drawdown"])
    print("Trades:", results["trades"])