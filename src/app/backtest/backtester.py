class Backtester:
    def __init__(self, screener, strategy, trading, historical_data):
        self.screener = screener
        self.strategy = strategy
        self.trading = trading
        self.historical_data = historical_data
        self.results = []

    def run(self):
        """
        Boucle principale du backtest :
        1. Met à jour le screener et la stratégie
        2. Applique le scoring
        3. Prend les décisions d’entrée/sortie via la stratégie
        4. Passe les ordres à la classe Trading
        5. Met à jour les positions et collecte les résultats
        """
        pass

    def report(self):
        """
        Génère un rapport de performance
        """
        pass
