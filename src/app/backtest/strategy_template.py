import backtrader as bt

class StrategyTemplate(bt.Strategy):
    params = dict(
        stop_loss=0.05,  # 5% stop loss
        take_profit=0.10 # 10% take profit
    )

    def __init__(self):
        self.order = None

    def next(self):
        if not self.position:
            # Exemple d'entrée simple
            self.order = self.buy()
        else:
            # Exemple de sortie simple
            if self.data.close[0] >= self.position.price * (1 + self.p.take_profit):
                self.close()
            elif self.data.close[0] <= self.position.price * (1 - self.p.stop_loss):
                self.close()
