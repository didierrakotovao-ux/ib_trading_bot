from datetime import datetime, timedelta
from ml.addivergencescoring import AdDivergenceScoring
from screener.providers.market_data_provider import MarketDataProvider
from strategies.strategy import Strategy
from ibapi.scanner import ScannerSubscription

class AdDivergenceStrategy(Strategy):
    name = "AdDivergenceStrategy"
    symbolsToAnalyse = []
    symbolsToTrade = []
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
    def __init__(self):
        self.scoring = AdDivergenceScoring()
        self.market_data = MarketDataProvider(port=7497)
        self.lookback_days = 90
        self.score_threshold = 60   

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
    
    def get_symbols(self) -> list: # type: ignore
        """Retourne la liste des symboles à trader"""
        self.market_data.connect()
        for symbol in self.symbolsToAnalyse:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            data = self.market_data.get_historical_data(symbol, start_date, end_date, interval="1d")
            if data is not None:
                score =self.scoring.score(data)
                if score >= self.score_threshold:
                    self.symbolsToTrade.append(symbol)
        self.market_data.disconnect()
        return self.symbolsToTrade

    def set_symbols_to_analyse(self, symbols: list):
        """Définit la liste des symboles analysés"""
        self.symbolsToAnalyse= symbols






