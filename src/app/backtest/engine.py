import backtrader as bt
import pandas as pd

class BacktestEngine:
    def __init__(self, strategy_cls, data: pd.DataFrame, cash=100000):
        self.strategy_cls = strategy_cls
        self.data = data
        self.cash = cash
        self.cerebro = bt.Cerebro()

    def run(self):
        datafeed = bt.feeds.PandasData(dataname=self.data)
        self.cerebro.adddata(datafeed)
        self.cerebro.addstrategy(self.strategy_cls)
        self.cerebro.broker.setcash(self.cash)
        result = self.cerebro.run()
        return result

    def plot(self):
        self.cerebro.plot()
