from app.strategies.strategy import Strategy

class AdDivergenceStrategy(Strategy):
    name = "AdDivergenceStrategy"
    def entry_signal(self, data, i) -> bool:
        pass
    def exit_signal(self, data, i, trade) -> bool:
        pass
    def scanner_filters(self):
        pass




