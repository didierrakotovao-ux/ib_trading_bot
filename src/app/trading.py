from screener.providers.market_data_provider import MarketDataProvider
from strategies.addivergence import AdDivergenceStrategy


class Trading:
    """
        Classe centrale de l'application, orchestre les éléments de trading via un provider de données de marché.
        1-recevoir les critères de recherche de la stratégie
        2-execute les screeners via le market data provider
        3-transmets les données de marché à la stratégie pour décision
        4-place les ordres pour les symboles sélectionnés
        5-gère les positions ouvertes et le suivi des ordres
        6-écrit le journal de performance
    """
    def __init__(self):
        self.market_data_provider = MarketDataProvider(port=7497)
        self.strategies = [AdDivergenceStrategy()]
        self.orders = []
        self.positions = {}

    def place_order(self, order_contract):
        """
        Reçoit un contrat d'ordre généré par la stratégie et le transmet au provider via placeOrder.
        """
        result = self.market_data_provider.placeOrder(order_contract)
        self.orders.append(order_contract)
        return result

    def update_orders(self):
        """
        Met à jour le statut des ordres (remplis, annulés, etc.)
        """
        pass

    def get_positions(self):
        """
        Retourne les positions courantes
        """
        return self.positions

    def close_position(self, symbol):
        """
        Ferme une position existante
        """
        pass

    def init_trade(self):
        """ 
            1- Obtient le critere de scanner de la stratégie
            2- Exécute le scanner via le market data provider
            3- Retourne la liste des symboles trouvés et passe à la stratégie pour décision de trading.
        """ 
        try:
            symbolList: list = []
            symbolToTrade: list = []
            self.market_data_provider.connect()
            for strategy in self.strategies:
                print(f"  Strategie: {strategy.name}...")
                scan_sub = strategy.scanner_filters()
                # symbols = self.market_data_provider.get_scanner_results(scan_sub, max_results=200)
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Mocked for testing
                strategy.set_symbols_to_analyse(symbols)
                for symbol in strategy.get_symbols():
                    symbolToTrade.append(symbol)
                symbolList.append(symbols)

            
            for symbol in symbolList:
                print(f"  Stocks trouvés: {symbol}")

            for symbol in symbolToTrade:
                print(f"  Stocks a trader: {symbol}")

            self.market_data_provider.disconnect()
            return symbolList
            
        except Exception as e:
            self.market_data_provider.disconnect()
            print(f"Erreur lors du trading: {e}")

if __name__ == "__main__":
    trading = Trading()
    trading.init_trade()
        