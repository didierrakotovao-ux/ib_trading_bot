from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_scoring import MLScoring
from src.app.ml.addivergencescoring import AdDivergenceScoring
from src.app.screener.providers.market_data_provider import MarketDataProvider
from src.app.strategies.strategy import Strategy
from ibapi.scanner import ScannerSubscription
from ibapi.order import Order

class MomentumStrategy(Strategy):
    name = "MomentumStrategy"
    symbolsToAnalyse = []
    symbolsToTrade = []
    """
        Exemple de configuration de scanner IB pour cette stratégie :        
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = scan_type
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 1000.0
        scan_sub.aboveVolume = 500_000
        scan_sub.marketCapAbove = 10_000_000_000
    """
    """ 
        la création du contrat d'ordre (entrée et sortie) 
        et la fourniture des données à scorer seront implémentées ici.
    """
    def __init__(self, market_data: MarketDataProvider, capital=10000, max_stocks=5,
                 use_trailing_stop=False, trailing_percent=5.0):
        self.scoring = MLScoring(model_path="models/momentum_model.pkl")
        self.market_data = market_data
        self.lookback_days = 350
        self.score_threshold = 65
        self.capital = capital  # Montant total dédié à la stratégie
        self.max_stocks = max_stocks  # Nombre max de stocks à trader
        self.use_trailing_stop = use_trailing_stop  # Si True, place un trailing stop automatique
        self.trailing_percent = trailing_percent  # Pourcentage du trailing stop

    def scanner_filters(self) -> ScannerSubscription:
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = "STK.NASDAQ"
        scan_sub.scanCode = "MOST_ACTIVE"
        scan_sub.abovePrice = 5.0
        scan_sub.belowPrice = 1000.0
        scan_sub.aboveVolume = 500_000
        # scan_sub.marketCapAbove = 10_000_000_000
        return scan_sub
    
    def get_symbols(self, trade_date) -> list: # type: ignore
        """Retourne la liste des symboles à trader (max_stocks), en excluant les ETFs à effet de levier."""
        scored_symbols = []
        for symbol in self.symbolsToAnalyse:
            start_date = trade_date - timedelta(days=self.lookback_days)
            data = self.market_data.get_historical_data(symbol, start_date, trade_date, interval="1d")
            if data is not None:
                score = self.scoring.score(data)
                if score >= self.score_threshold:
                    scored_symbols.append((symbol, score, data))
        # Trier par score décroissant et ne garder que max_stocks
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]
        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}  # stocke les dataframes pour chaque symbole
        return self.symbolsToTrade

    def get_order_params(self):
        """
        Génère les paramètres d'ordre pour chaque symbole sélectionné.
        Lie le stop loss et le take profit à l'ordre d'entrée (parent/child orders IB).
        Retourne une liste de dicts {symbol, entry_order, stop_order, take_profit_order}.
        Ne place les ordres que pendant les heures d'ouverture du marché US (9h30-16h00 US/Eastern, jours ouvrés).
        """
        if not hasattr(self, 'symbolsToTrade') or not hasattr(self, 'symbolsData'):
            raise Exception("Appeler get_symbols() avant get_order_params()")
        n = len(self.symbolsToTrade)
        if n == 0:
            return []
        # Chaque position = max 1/5 du capital (max_stocks positions)
        capital_per_stock = self.capital / self.max_stocks
        order_params = []
        # S'assurer que la connexion est prête et récupérer le vrai nextValidId
        if hasattr(self.market_data, '_next_req_id'):
            start_id = self.market_data._next_req_id
        else:
            # Valeur de secours si jamais _next_req_id n'existe pas
            start_id = 100000
        order_id_gen = iter(range(start_id, start_id + 10000))
        for symbol in self.symbolsToTrade:
            df = self.symbolsData[symbol]
            last_close = df['close'].iloc[-1]
            qty = int(capital_per_stock / last_close)

            # Vérifier si le capital est suffisant
            if qty <= 0:
                print(f"[ORDER PARAMS] Capital insuffisant pour {symbol} (close={last_close}, capital_per_stock={capital_per_stock})")
                continue

            print(f"[ORDER PARAMS] Préparation des ordres pour {symbol} avec close={last_close} et capital par stock={capital_per_stock} et quantité={qty}  ")
            parent_id = next(order_id_gen)
            stop_id = next(order_id_gen)

            # Ordre d'entrée (Limit order avec prix légèrement au-dessus pour exécution rapide)
            entry_price = round(last_close * 1.005, 2)  # 0.5% au-dessus du close pour assurer le fill
            entryorder = Order()
            entryorder.action = "BUY"
            entryorder.orderType = "LMT"
            entryorder.lmtPrice = entry_price
            entryorder.totalQuantity = qty
            entryorder.eTradeOnly = False
            entryorder.firmQuoteOnly = False
            entryorder.orderId = parent_id
            entryorder.tif = "GTC"  # Good Till Cancelled

            # Construire le dict de retour
            order_dict = {
                'symbol': symbol,
                'entry_order': entryorder
            }

            if self.use_trailing_stop:
                # Trailing stop activé - bracket order
                entryorder.transmit = False  # Attendre le child

                slorder = Order()
                slorder.action = "SELL"
                slorder.orderType = "TRAIL"
                slorder.totalQuantity = qty
                slorder.parentId = parent_id  # Lié à l'ordre d'entrée
                slorder.transmit = True  # Transmet le bracket complet
                slorder.eTradeOnly = False
                slorder.firmQuoteOnly = False
                slorder.orderId = stop_id
                slorder.trailingPercent = self.trailing_percent
                slorder.tif = "GTC"

                order_dict['stop_order'] = slorder
                print(f"[ORDER PARAMS] Trailing stop {self.trailing_percent}% activé pour {symbol}")
            else:
                # Pas de trailing stop - ordre simple
                entryorder.transmit = True  # Transmettre immédiatement
                print(f"[ORDER PARAMS] Pas de trailing stop pour {symbol} (gestion manuelle)")

            order_params.append(order_dict)
            # Note: _next_req_id est incrémenté automatiquement par placeOrder
        return order_params

    def set_symbols_to_analyse(self, symbols: list):
        """Définit la liste des symboles analysés"""
        self.symbolsToAnalyse= symbols
