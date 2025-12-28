from datetime import datetime, timedelta
from ml.addivergencescoring import AdDivergenceScoring
from screener.providers.market_data_provider import MarketDataProvider
from strategies.strategy import Strategy
from ibapi.scanner import ScannerSubscription
from ibapi.order import Order

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
    def __init__(self, market_data: MarketDataProvider, capital=10000, max_stocks=5):
        self.scoring = AdDivergenceScoring()
        self.market_data = market_data
        self.lookback_days = 350
        self.score_threshold = 60
        self.capital = capital  # Montant total dédié à la stratégie
        self.max_stocks = max_stocks  # Nombre max de stocks à trader

    def entry_signal(self, data, i) -> bool:
        pass
    def exit_signal(self, data, i, trade) -> bool:
        pass

    def scanner_filters(self) -> ScannerSubscription:
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "MOST_ACTIVE"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 500.0
        scan_sub.aboveVolume = 500_000
        scan_sub.marketCapAbove = 10_000_000_000
        return scan_sub
    
    def get_symbols(self) -> list: # type: ignore
        """Retourne la liste des symboles à trader (max_stocks)"""
        if not self.market_data.is_connected():
            self.market_data.connect()
        scored_symbols = []
        for symbol in self.symbolsToAnalyse:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            data = self.market_data.get_historical_data(symbol, start_date, end_date, interval="1d")
            if data is not None:
                score = self.scoring.score(data)
                if score >= self.score_threshold:
                    scored_symbols.append((symbol, score, data))
        # Trier par score décroissant et ne garder que max_stocks
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]
        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}  # stocke les dataframes pour chaque symbole
        self.market_data.disconnect()
        return self.symbolsToTrade

    def get_order_params(self):
        """
        Génère les paramètres d'ordre pour chaque symbole sélectionné.
        Lie le stop loss et le take profit à l'ordre d'entrée (parent/child orders IB).
        Retourne une liste de dicts {symbol, entry_order, stop_order, take_profit_order}.
        Ne place les ordres que pendant les heures d'ouverture du marché US (9h30-16h00 US/Eastern, jours ouvrés).
        """
        import pytz
        from datetime import datetime, time as dtime
        us_eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(us_eastern)
        is_weekday = now_et.weekday() < 5
        market_open = dtime(9, 30)
        market_close = dtime(16, 0)
        if not (is_weekday and market_open <= now_et.time() <= market_close):
            print("⏳ Marché fermé : les ordres ne seront pas générés/envoyés.")
            return []

        if not hasattr(self, 'symbolsToTrade') or not hasattr(self, 'symbolsData'):
            raise Exception("Appeler get_symbols() avant get_order_params()")
        n = len(self.symbolsToTrade)
        if n == 0:
            return []
        capital_per_stock = self.capital / n
        order_params = []
        order_id_gen = iter(range(100000, 200000))
        self.market_data.nextValidId(next(order_id_gen))
        for symbol in self.symbolsToTrade:
            df = self.symbolsData[symbol]
            last_close = df['close'].iloc[-1]
            qty = int(capital_per_stock / last_close)
            parent_id = next(order_id_gen)

            # Parent: ordre d'entrée
            entryorder = Order()
            entryorder.action = "BUY"
            entryorder.orderType = "STP LMT"
            entryorder.totalQuantity = qty
            entryorder.lmtPrice = round(last_close * 0.99, 2)  # Limite à 1% sous le close
            entryorder.auxPrice = round(last_close * 1.01, 2)  # Stop à 0.5% sous le close
            entryorder.transmit = False
            entryorder.eTradeOnly = False
            entryorder.firmQuoteOnly = False
            entryorder.orderId = parent_id
            entryorder.tif = "GTC"  # Good Till Cancelled

            # Child 1: Stop loss suiveur
            slorder = Order()
            slorder.action = "SELL"
            slorder.orderType = "TRAIL"
            slorder.auxPrice = round(last_close * 0.90, 2)  # Trailing stop à 10% sous le prix
            slorder.totalQuantity = qty
            slorder.parentId = parent_id
            slorder.transmit = False
            slorder.eTradeOnly = False
            slorder.firmQuoteOnly = False
            slorder.orderId = parent_id + 1
            slorder.trailingPercent = 10.0  # Trailing de 10%
            slorder.tif = "GTC"  # Good Till Cancelled

            # Child 2: Take profit
            tporder = Order()
            tporder.action = "SELL"
            tporder.orderType = "LMT"
            tporder.lmtPrice = round(last_close * 1.20, 2)  # Take profit à 20% au-dessus du close
            tporder.totalQuantity = qty
            tporder.parentId = parent_id
            tporder.transmit = True  # Le dernier transmet l'ensemble
            tporder.eTradeOnly = False
            tporder.firmQuoteOnly = False
            tporder.orderId = parent_id + 2
            tporder.tif = "GTC"  # Good Till Cancelled
                            
            order_params.append({
                'symbol': symbol,
                'entry_order': entryorder,
                'stop_order': slorder,
                'take_profit_order': tporder
            })
            parent_id += 10  # Incrément pour éviter collision d'IDs
        return order_params

    def set_symbols_to_analyse(self, symbols: list):
        """Définit la liste des symboles analysés"""
        self.symbolsToAnalyse= symbols






