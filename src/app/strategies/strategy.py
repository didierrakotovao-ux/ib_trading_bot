from ibapi.scanner import ScannerSubscription

class Strategy:
    name: str

    def scanner_filters(self) -> ScannerSubscription:
        pass

    def entry_signal(self, data, i) -> bool:
        """Signal d’entrée à l’index i"""
        pass

    def exit_signal(self, data, i, trade) -> bool:
        """Signal de sortie"""
        pass
    def get_symbols(self) -> list:
        """Retourne la liste des symboles à trader"""
        pass
    def set_symbols_to_analyse(self, symbols: list):
        """Définit la liste des symboles analysés"""
        pass


