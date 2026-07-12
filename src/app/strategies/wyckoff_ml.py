from datetime import timedelta
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ibapi.order import Order
from ibapi.scanner import ScannerSubscription

from src.app.database.db_manager import DatabaseManager
from src.app.ml.ml_wyckoff_scoring import WyckoffMLScoring
from src.app.screener.providers.market_data_provider import MarketDataProvider
from src.app.strategies.strategy import Strategy


class WyckoffMLStrategy(Strategy):
    """
    Strategie dediee au pipeline Wyckoff ML.
    Contrairement a MomentumStrategy, elle ne fait pas de pre-filtres 12-1/FIP
    et selectionne directement les meilleurs scores Wyckoff.
    """

    name = "WyckoffMLStrategy"
    symbolsToAnalyse = []
    symbolsToTrade = []

    def __init__(
        self,
        market_data: MarketDataProvider,
        capital=10000,
        max_stocks=5,
        use_trailing_stop=False,
        trailing_percent=5.0,
        scoring_type="wyckoff_ml",
        db_path="trading_data.db",
        wyckoff_model_path=None,
        score_threshold=55,
    ):
        if scoring_type != "wyckoff_ml":
            raise ValueError(
                f"scoring_type={scoring_type!r} non supporte pour WyckoffMLStrategy. "
                "Valeur attendue: 'wyckoff_ml'."
            )

        model_path = wyckoff_model_path or "models/wyckoff_model.pkl"
        self.scoring = WyckoffMLScoring(model_path=model_path, db_path=db_path)

        self.scoring_type = scoring_type
        self.market_data = market_data
        self.lookback_days = 400
        self.score_threshold = score_threshold
        self.capital = capital
        self.max_stocks = max_stocks
        self.use_trailing_stop = use_trailing_stop
        self.trailing_percent = trailing_percent

        self.db_manager = DatabaseManager(db_path)

    def scanner_filters(self) -> ScannerSubscription:
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "MOST_ACTIVE"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 1000.0
        scan_sub.aboveVolume = 500_000
        return scan_sub

    def get_symbols(self, trade_date) -> list:  # type: ignore
        scored_symbols = []
        _cache_preloaded = getattr(self.market_data, '_preloaded', False)

        for symbol in self.symbolsToAnalyse:
            start_date = trade_date - timedelta(days=self.lookback_days)
            data = None

            if _cache_preloaded:
                data = self.market_data.get_historical_data(
                    symbol, start_date, trade_date, interval="1d"
                )

            if data is None or len(data) < 60:
                data = self.db_manager.get_historical_data(symbol, start_date, trade_date)

            if data is None or len(data) < 60:
                data = self.market_data.get_historical_data(
                    symbol, start_date, trade_date, interval="1d"
                )

            if data is None or len(data) < 60:
                continue

            score = self.scoring.score(data, symbol=symbol)
            if score >= self.score_threshold:
                scored_symbols.append((symbol, score, data))

        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]

        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}

        print(
            f"[WYCKOFF] {trade_date} | {len(self.symbolsToTrade)} selectionnes "
            f"(seuil={self.score_threshold}, top={self.max_stocks})"
            + (f": {self.symbolsToTrade}" if self.symbolsToTrade else "")
        )
        return self.symbolsToTrade

    def get_order_params(self):
        if not hasattr(self, 'symbolsToTrade') or not hasattr(self, 'symbolsData'):
            raise Exception("Appeler get_symbols() avant get_order_params()")

        if len(self.symbolsToTrade) == 0:
            return []

        capital_per_stock = self.capital / self.max_stocks
        order_params = []

        for symbol in self.symbolsToTrade:
            df = self.symbolsData[symbol]
            last_close = df['close'].iloc[-1]
            qty = int(capital_per_stock / last_close)

            if qty <= 0:
                continue

            entry_price = round(last_close * 1.005, 2)
            entryorder = Order()
            entryorder.action = "BUY"
            entryorder.orderType = "LMT"
            entryorder.lmtPrice = entry_price
            entryorder.totalQuantity = qty
            entryorder.eTradeOnly = False
            entryorder.firmQuoteOnly = False
            entryorder.tif = "DAY"

            order_dict = {
                'symbol': symbol,
                'entry_order': entryorder
            }

            if self.use_trailing_stop:
                entryorder.transmit = False

                slorder = Order()
                slorder.action = "SELL"
                slorder.orderType = "TRAIL"
                slorder.totalQuantity = qty
                slorder.transmit = True
                slorder.eTradeOnly = False
                slorder.firmQuoteOnly = False
                slorder.trailingPercent = self.trailing_percent
                slorder.tif = "GTC"

                order_dict['stop_order'] = slorder
            else:
                entryorder.transmit = True

            order_params.append(order_dict)

        return order_params

    def set_symbols_to_analyse(self, symbols: list):
        self.symbolsToAnalyse = symbols
