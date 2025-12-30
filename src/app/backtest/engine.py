import backtrader as bt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from addivergence_bt_wrapper import AdDivergenceBTWrapper
from market_data_mock import MarketDataMock


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
        print(f"Loading data from {start_date} to {self.end_date}...")
        self.dataframes = {}
        for symbol in self.symbols:
            print(f"Loading data for {symbol} from {start_date} to {self.end_date}...")
            df = self.market_data.get_historical_data(
                symbol,
                self.start_date,
                self.end_date,
                interval="1d")
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                self.dataframes[symbol] = df
                data = bt.feeds.PandasData(
                    dataname=df,
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
    start_date=datetime(2025, 9, 1)
    end_date=datetime(2025, 9, 30)

    engine = BacktestEngine(
        strategy_cls=AdDivergenceBTWrapper,
        symbols=[
            'INTC', 'QCOM', 'TXN', 'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS',
            'BAC', 'WFC', 'MS', 'GS', 'C', 'SCHW', 'AXP', 'BLK', 'SPGI', 'CB',        
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