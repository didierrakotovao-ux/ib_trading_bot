from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.bollinger_rsi_scoring import BollingerRsiScoring
from src.app.screener.providers.market_data_provider import MarketDataProvider
from src.app.strategies.strategy import Strategy
from ibapi.scanner import ScannerSubscription
from ibapi.order import Order


class RangeBollingerStrategy(Strategy):
    """
    Stratégie de retour à la moyenne pour marché de côté : achat près de la
    bande basse de Bollinger avec RSI survendu en reprise (BollingerRsiScoring).
    Sortie plus serrée qu'une stratégie momentum, car on vise un retour vers
    la moyenne plutôt qu'une poursuite de tendance.
    """
    name = "RangeBollingerStrategy"
    symbolsToAnalyse = []
    symbolsToTrade = []

    def __init__(self, market_data: MarketDataProvider, capital=10000, max_stocks=5):
        self.scoring = BollingerRsiScoring()
        self.market_data = market_data
        self.lookback_days = 350
        self.score_threshold = 65
        self.capital = capital  # Montant total dédié à la stratégie
        self.max_stocks = max_stocks  # Nombre max de stocks à trader
        self.max_exposure = 0.6  # Limiter l'exposition à 60% du capital (réduit le drawdown)

    def scanner_filters(self) -> ScannerSubscription:
        """
        Exemple de configuration de scanner IB pour cette stratégie :
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "TOP_PERC_LOSERS"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 1000.0
        scan_sub.aboveVolume = 500_000
        """
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "TOP_PERC_LOSERS"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 1000.0
        scan_sub.aboveVolume = 500_000
        return scan_sub

    def get_symbols(self, trade_date) -> list:  # type: ignore
        """Retourne la liste des symboles à trader (max_stocks)."""
        scored_symbols = []
        for symbol in self.symbolsToAnalyse:
            start_date = trade_date - timedelta(days=self.lookback_days)
            data = self.market_data.get_historical_data(symbol, start_date, trade_date, interval="1d")
            if data is not None:
                score = self.scoring.score(data)
                if score >= self.score_threshold:
                    scored_symbols.append((symbol, score, data))
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]
        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}
        return self.symbolsToTrade

    def get_order_params(self):
        """
        Génère les paramètres d'ordre pour chaque symbole sélectionné.
        Stop loss et take profit plus serrés qu'en momentum (mean reversion =
        mouvements visés plus courts). Lie le stop loss et le take profit à
        l'ordre d'entrée (parent/child orders IB).
        """
        if not hasattr(self, 'symbolsToTrade') or not hasattr(self, 'symbolsData'):
            raise Exception("Appeler get_symbols() avant get_order_params()")
        n = len(self.symbolsToTrade)
        if n == 0:
            return []
        capital_per_stock = (self.capital * self.max_exposure) / self.max_stocks
        order_params = []
        if hasattr(self.market_data, '_next_req_id'):
            start_id = self.market_data._next_req_id
        else:
            start_id = 100000
        order_id_gen = iter(range(start_id, start_id + 10000))
        for symbol in self.symbolsToTrade:
            df = self.symbolsData[symbol]
            last_close = df['close'].iloc[-1]
            qty = int(capital_per_stock / last_close)

            if qty <= 0:
                print(f"[ORDER PARAMS] Capital insuffisant pour {symbol} (close={last_close}, capital_per_stock={capital_per_stock})")
                continue

            print(f"[ORDER PARAMS] Préparation des ordres pour {symbol} avec close={last_close} et capital par stock={capital_per_stock} et quantité={qty}  ")
            parent_id = next(order_id_gen)
            stop_id = next(order_id_gen)
            tp_id = next(order_id_gen)

            entry_price = round(last_close * 1.005, 2)  # 0.5% au-dessus du close pour assurer le fill
            entryorder = Order()
            entryorder.action = "BUY"
            entryorder.orderType = "LMT"
            entryorder.lmtPrice = entry_price
            entryorder.totalQuantity = qty
            entryorder.transmit = False
            entryorder.eTradeOnly = False
            entryorder.firmQuoteOnly = False
            entryorder.orderId = parent_id
            entryorder.tif = "GTC"

            slorder = Order()
            slorder.action = "SELL"
            slorder.orderType = "TRAIL"
            slorder.totalQuantity = qty
            slorder.parentId = parent_id
            slorder.transmit = False
            slorder.eTradeOnly = False
            slorder.firmQuoteOnly = False
            slorder.orderId = stop_id
            slorder.trailingPercent = 4.0  # Trailing de 4% (mean reversion = mouvement plus court qu'en momentum)
            slorder.tif = "GTC"

            tporder = Order()
            tporder.action = "SELL"
            tporder.orderType = "LMT"
            tporder.lmtPrice = round(last_close * 1.07, 2)  # Take profit à 7% au-dessus du close
            tporder.totalQuantity = qty
            tporder.parentId = parent_id
            tporder.transmit = True
            tporder.eTradeOnly = False
            tporder.firmQuoteOnly = False
            tporder.orderId = tp_id
            tporder.tif = "GTC"

            order_params.append({
                "symbol": symbol,
                "entry_order": entryorder,
                "stop_order": slorder,
                "take_profit_order": tporder,
            })
            if hasattr(self.market_data, '_next_req_id'):
                self.market_data._next_req_id += 3  # Incrémente l'ID pour les prochains ordres

        return order_params

    def set_symbols_to_analyse(self, symbols: list):
        self.symbolsToAnalyse = symbols
