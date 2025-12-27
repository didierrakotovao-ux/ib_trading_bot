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
    def __init__(self, market_data_provider):
        self.market_data_provider = market_data_provider  # Instance du provider (ex: IBKR, simulateur, etc.)
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
