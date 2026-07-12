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
from src.app.strategies.earnings_filter import EarningsFilter


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
        # Seuil calibré par balayage EV out-of-time (2025-04 -> 2026-07,
        # modèle avec contexte marché) : +1.58%/trade à 60 vs +0.32% à 55
        score_threshold=60,
        use_earnings_filter=True,
        earnings_blackout_days=14,
        post_earnings_days=5,
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

        # Filtre earnings symétrique : pas d'entrée dans [annonce-14j, annonce+5j]
        # (validé sur 3 backtests : 14/14 trades bloqués étaient perdants)
        self.use_earnings_filter = use_earnings_filter
        self.earnings_filter = EarningsFilter(
            blackout_days=earnings_blackout_days,
            post_earnings_days=post_earnings_days) \
            if use_earnings_filter else None

        self.db_manager = DatabaseManager(db_path)

        # Scores pré-calculés (backtest uniquement) : injectés par le wrapper
        self.precomputed = None

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

        # Filtre earnings en amont du scoring (une requête pour tout le lot)
        symbols = self.symbolsToAnalyse
        if self.use_earnings_filter and symbols:
            self.earnings_filter.preload(symbols, trade_date)
            n_before = len(symbols)
            symbols = [s for s in symbols
                       if not self.earnings_filter.is_in_blackout(s, trade_date)]
            if n_before != len(symbols):
                print(f"[WYCKOFF] {trade_date} | {n_before - len(symbols)} symboles "
                      f"en blackout earnings écartés")

        # Mode backtest avec scores pré-calculés : lookups O(1), données
        # chargées uniquement pour les symboles sélectionnés
        if self.precomputed is not None:
            scored = [(s, self.precomputed.score(s, trade_date)) for s in symbols]
            scored = [(s, sc) for s, sc in scored if sc >= self.score_threshold]
            scored.sort(key=lambda x: x[1], reverse=True)
            start_date = trade_date - timedelta(days=self.lookback_days)
            self.symbolsToTrade, self.symbolsData, self.symbolsScores = [], {}, {}
            for symbol, score in scored[:self.max_stocks]:
                data = self.market_data.get_historical_data(
                    symbol, start_date, trade_date, interval="1d")
                if data is None or len(data) < 60:
                    data = self.db_manager.get_historical_data(
                        symbol, start_date, trade_date)
                if data is None or len(data) < 60:
                    continue
                self.symbolsToTrade.append(symbol)
                self.symbolsData[symbol] = data
                self.symbolsScores[symbol] = score
            print(f"[WYCKOFF] {trade_date} | {len(self.symbolsToTrade)} selectionnes "
                  f"(precalcule, seuil={self.score_threshold})"
                  + (f": {self.symbolsToTrade}" if self.symbolsToTrade else ""))
            return self.symbolsToTrade

        for symbol in symbols:
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
        # Scores d'entrée, lus par le wrapper de backtest (score_entree)
        self.symbolsScores = {s[0]: s[1] for s in selected}

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
