from datetime import datetime, timedelta
from strategies.momentum import MomentumStrategy
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
        # , AdDivergenceStrategy(self.market_data_provider)
        self.strategies = [MomentumStrategy(self.market_data_provider)]
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
        symbol = getattr(contract, 'symbol', None)
        if order.action == "BUY" and symbol and self.position_manager.has_open_position(symbol):
            print(f"[Trading] Position déjà ouverte sur {symbol}, ordre ignoré.")
            return None
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
            trade_date = datetime.now()

            symbolList: list = []
            symbolToTrade: list = []
            # Connexion gérée par le main, pas de double connect ici
            for strategy in self.strategies:
                print(f"  Strategie: {strategy.name}...")
                scan_sub = strategy.scanner_filters()
                symbols = self.market_data_provider.get_scanner_results(scan_sub, max_results=200)
                strategy.set_symbols_to_analyse(symbols)
                
                for symbol in strategy.get_symbols(trade_date):
                    symbolToTrade.append(symbol)
                for orders in strategy.get_order_params():
                    contrat = self.market_data_provider.create_contract(orders['symbol'])
                    self.place_order(contrat, orders['entry_order'])
                    self.place_order(contrat, orders['stop_order'])
                    if 'take_profit_order' in orders:
                        self.place_order(contrat, orders['take_profit_order'])

            for symbol in symbolToTrade:
                print(f"  Stocks a trader: {symbol}")

            return symbolList

        except Exception as e:
            # Disconnect géré par le main dans finally
            print(f"Erreur lors du trading: {e}")
            import traceback
            traceback.print_exc()

    def sync_positions_with_ib(self):
        """
        Synchronise le PositionManager avec les positions réelles récupérées via l'API IB.
        """
        # Vérification stricte de la connexion
        if not self.market_data_provider.is_connected():
            print("[DEBUG] MarketDataProvider non connecté, tentative de reconnexion...")
            # Recréation d'une nouvelle instance pour une reconnexion propre
            from screener.providers.market_data_provider import MarketDataProvider
            self.market_data_provider = MarketDataProvider(port=self.market_data_provider.port, client_id=self.market_data_provider.client_id)
            if not self.market_data_provider.connect():
                print("[DEBUG] Impossible de se connecter à IB, synchronisation annulée.")
                return
        ib_positions = self.market_data_provider.get_ib_positions()
        print(f"Positions IB récupérées: {ib_positions}")
        for pos in ib_positions:
            if not self.position_manager.has_open_position(pos['symbol']):
                self.position_manager.open_position(
                    symbol=pos['symbol'],
                    qty=pos['qty'],
                    entry_price=pos['avg_cost'],
                    entry_time=None,
                    position_type="long" if pos['qty'] > 0 else "short"
                )
        print(f"Positions synchronisées depuis IB: {[p.symbol for p in self.position_manager.get_open_positions()]}" )

    def close(self):
        """
        Déconnecte proprement le MarketDataProvider (IB) et arrête le thread associé.
        """
        if self.market_data_provider.is_connected():
            self.market_data_provider.disconnect()
        print("Déconnexion propre IB terminée.")



if __name__ == "__main__":
    client_id = 11  # À ajuster si besoin
    trading = Trading()
    try:
        print("[TRADING] Connexion à IB...")
        connected = trading.market_data_provider.connect()
        print(f"[TRADING] Connecté: {connected}")
        if not connected:
            print("[TRADING] Échec de connexion, arrêt du script.")
            exit(1)
        trading.sync_positions_with_ib()
        trading.init_trade()
        pass
    except KeyboardInterrupt:
        print("Arrêt demandé par l'utilisateur (Ctrl+C)")        
    finally:
        print("[TRADING] Déconnexion propre...")
        trading.market_data_provider.disconnect()
        print("[TRADING] Déconnecté.")
