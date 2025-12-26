class Strategy:
    name: str

    def scanner_filters(self):
        pass

    def entry_signal(self, data, i) -> bool:
        """Signal d’entrée à l’index i"""
        pass

    def exit_signal(self, data, i, trade) -> bool:
        """Signal de sortie"""
        pass
