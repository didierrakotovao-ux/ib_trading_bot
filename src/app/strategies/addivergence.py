from app.strategies.strategy import Strategy
from ibapi.scanner import ScannerSubscription

class AdDivergenceStrategy(Strategy):
    name = "AdDivergenceStrategy"
    """
        Exemple de configuration de scanner IB pour cette stratégie :        
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = scan_type
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 500.0
        scan_sub.aboveVolume = 500_000
        scan_sub.marketCapAbove = 10_000_000_000
    """
    """ 
        
        la création du contrat d'ordre (entrée et sortie) 
        et la fourniture des données à scorer seront implémentées ici.
    """
    def entry_signal(self, data, i) -> bool:
        pass
    def exit_signal(self, data, i, trade) -> bool:
        pass

    def scanner_filters(self) -> ScannerSubscription:
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "HOT_BY_VOLUME"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 500.0
        scan_sub.aboveVolume = 500_000
        scan_sub.marketCapAbove = 10_000_000_000
        return scan_sub




