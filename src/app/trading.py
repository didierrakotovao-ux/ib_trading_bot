
from screener.providers.market_data_provider import MarketDataProvider
from strategies.addivergence import AdDivergenceStrategy
from position_manager import PositionManager


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
        self.market_data_provider = MarketDataProvider(port=7497, client_id=1)
        self.strategies = [AdDivergenceStrategy(client_id=2)]
        self.orders = []
        self.position_manager = PositionManager()
        self.order_callbacks = []  # Liste de callbacks à appeler sur exécution d'ordre

    def register_order_callback(self, callback):
        """
        Permet d'enregistrer une fonction callback à appeler lors de l'exécution d'un ordre.
        La fonction doit accepter (symbol, order, status, fill_price, fill_qty, fill_time)
        """
        self.order_callbacks.append(callback)

    def on_order_executed(self, symbol, order, status, fill_price, fill_qty, fill_time):
        """
        Appelle tous les callbacks enregistrés lors de l'exécution d'un ordre.
        Ajoute la position dans le PositionManager uniquement si l'ordre d'entrée (BUY) est exécuté.
        """
        # Ajout automatique de la position uniquement si l'ordre d'entrée est exécuté
        if order.action == "BUY" and status.lower() == "filled":
            self.position_manager.open_position(
                symbol=symbol,
                qty=fill_qty,
                entry_price=fill_price,
                entry_time=fill_time
            )
        for cb in self.order_callbacks:
            cb(symbol, order, status, fill_price, fill_qty, fill_time)

    def place_order(self, contract, order):
        """
        Reçoit un contrat et un ordre généré par la stratégie et les transmet au provider via placeOrder.
        Appelle le callback sur exécution d'ordre (à compléter selon le retour du provider).
        """
        result = self.market_data_provider.placeOrder(contract, order)
        self.orders.append((contract, order))
        # Exemple d'appel du callback (à adapter selon le retour réel du provider)
        # self.on_order_executed(symbol, order, status, fill_price, fill_qty, fill_time)
        return result

    def update_orders(self):
        """
        Met à jour le statut des ordres (remplis, annulés, etc.)
        """
        pass

    def get_positions(self):
        """
        Retourne les positions courantes (ouvertes)
        """
        return self.position_manager.get_open_positions()

    def close_position(self, symbol, exit_price, exit_time=None):
        """
        Ferme une position existante et met à jour le P&L
        """
        return self.position_manager.close_position(symbol, exit_price, exit_time)

    def init_trade(self):
        """
        1- Obtient le critere de scanner de la stratégie
        2- Exécute le scanner via le market data provider
        3- Retourne la liste des symboles trouvés et passe à la stratégie pour décision de trading.
        4- Ouvre les positions dans le PositionManager uniquement lors de l'exécution effective de l'ordre d'entrée (voir on_order_executed).
        """
        try:
            symbolList: list = []
            symbolToTrade: list = []
            self.market_data_provider.connect()
            for strategy in self.strategies:
                print(f"  Strategie: {strategy.name}...")
                scan_sub = strategy.scanner_filters()
                # symbols = self.market_data_provider.get_scanner_results(scan_sub, max_results=200)

                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'INTC', 'AMD', 'NFLX']

                strategy.set_symbols_to_analyse(symbols)
                for symbol in strategy.get_symbols():
                    symbolToTrade.append(symbol)
                for orders in strategy.get_order_params():
                    contrat = self.market_data_provider.create_contract(orders['symbol'])
                    if not self.market_data_provider.is_connected():
                        self.market_data_provider.connect()
                    self.place_order(contrat, orders['entry_order'])
                    self.place_order(contrat, orders['stop_order'])
                    self.place_order(contrat, orders['take_profit_order'])
            for symbol in symbolToTrade:
                print(f"  Stocks a trader: {symbol}")

            return symbolList

        except Exception as e:
            self.market_data_provider.disconnect()
            print(f"Erreur lors du trading: {e}")

if __name__ == "__main__":
    trading = Trading()
    trading.init_trade()
        