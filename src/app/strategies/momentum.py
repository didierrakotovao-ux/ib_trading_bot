from datetime import datetime, timedelta
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_scoring import MLScoring
from src.app.ml.ml_smooth_scoring import SmoothMLScoring
from src.app.ml.ml_earnings_scoring import EarningsMLScoring
from src.app.ml.addivergencescoring import AdDivergenceScoring
from src.app.ml.earnings_features import EarningsFeatures
from src.app.strategies.momentum_filters import MomentumFilters
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
                 use_trailing_stop=False, trailing_percent=5.0,
                 scoring_type="ml",
                 momentum_top_pct=0.3, fip_threshold=0.0,
                 use_seasonal_threshold=True,
                 use_sue_filter=False, sue_threshold=0.0):
        if scoring_type == "smooth_ml":
            self.scoring = SmoothMLScoring(model_path="models/smooth_momentum_model.pkl")
        elif scoring_type == "earnings_ml":
            self.scoring = EarningsMLScoring(model_path="models/earnings_momentum_model.pkl")
        else:
            self.scoring = MLScoring(model_path="models/momentum_model.pkl")
        self.scoring_type = scoring_type
        self.market_data = market_data
        self.lookback_days = 400  # ~252 jours de trading + marge pour les features
        self.score_threshold = 65  # Seuil par défaut (peut être overridé par saisonnalité)
        self.capital = capital
        self.max_stocks = max_stocks
        self.use_trailing_stop = use_trailing_stop
        self.trailing_percent = trailing_percent

        # Filtres Gray & Vogel
        self.momentum_top_pct = momentum_top_pct  # Top 30% par momentum 12-1
        self.fip_threshold = fip_threshold          # FIP < 0 = bon momentum smooth
        self.use_seasonal_threshold = use_seasonal_threshold  # Seuil dynamique par mois

        # Filtre SUE (Novy-Marx)
        self.use_sue_filter = use_sue_filter  # Activer le filtre SUE
        self.sue_threshold = sue_threshold     # SUE > sue_threshold (défaut: 0)
        if use_sue_filter:
            self.earnings_features = EarningsFeatures()
            print(f"[PIPELINE] Filtre SUE activé (seuil > {sue_threshold})")

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
        """
        Pipeline de sélection basé sur Quantitative Momentum (Gray & Vogel):
        1. Récupérer les données historiques
        2. Pré-filtre: Momentum 12-1 mois (top 30%)
        3. Pré-filtre: Frog-in-the-Pan FIP < 0
        4. Scoring ML sur les candidats restants
        5. Seuil dynamique selon la saisonnalité
        6. Top max_stocks par score
        """
        # Étape 1: Récupérer les données pour tous les symboles
        all_candidates = []
        nan_count = 0
        for symbol in self.symbolsToAnalyse:
            start_date = trade_date - timedelta(days=self.lookback_days)
            data = self.market_data.get_historical_data(symbol, start_date, trade_date, interval="1d")
            if data is not None and len(data) >= 60:
                momentum_12_1 = MomentumFilters.calc_momentum_12_1(data)
                if np.isnan(momentum_12_1):
                    nan_count += 1
                all_candidates.append((symbol, momentum_12_1, data))

        valid_count = len(all_candidates) - nan_count
        if all_candidates:
            sample = all_candidates[0]
            print(f"[PIPELINE] {len(all_candidates)} symboles ({valid_count} valides, {nan_count} NaN momentum)")
            print(f"[PIPELINE] Exemple: {sample[0]} -> {len(sample[2])} barres, mom_12_1={sample[1]}")
        else:
            print(f"[PIPELINE] 0 symboles avec données suffisantes")

        # Étape 2: Filtre momentum 12-1 (top 30%)
        momentum_filtered = MomentumFilters.filter_top_momentum(
            all_candidates, self.momentum_top_pct)
        print(f"[PIPELINE] {len(momentum_filtered)} symboles après filtre momentum 12-1 (top {self.momentum_top_pct*100:.0f}%)")

        # Étape 3: Filtre FIP
        fip_filtered = []
        for symbol, mom, data in momentum_filtered:
            fip = MomentumFilters.calc_fip(data)
            if not np.isnan(fip) and fip < self.fip_threshold:
                fip_filtered.append((symbol, mom, fip, data))
                print(f"  [FIP] {symbol}: mom_12_1={mom:.2%}, FIP={fip:.4f} [OK]")
            else:
                fip_val = f"{fip:.4f}" if not np.isnan(fip) else "N/A"
                print(f"  [FIP] {symbol}: mom_12_1={mom:.2%}, FIP={fip_val} [X]")

        print(f"[PIPELINE] {len(fip_filtered)} symboles après filtre FIP (< {self.fip_threshold})")

        # Étape 3b: Filtre SUE optionnel (Novy-Marx)
        if self.use_sue_filter:
            sue_filtered = []
            for symbol, mom, fip, data in fip_filtered:
                try:
                    earnings = self.earnings_features.get_latest_earnings_features(symbol)
                    sue = earnings.get('sue')
                    if sue is not None and sue > self.sue_threshold:
                        sue_filtered.append((symbol, mom, fip, data))
                        print(f"  [SUE] {symbol}: SUE={sue:.2f}% > {self.sue_threshold} [OK]")
                    else:
                        sue_val = f"{sue:.2f}%" if sue is not None else "N/A"
                        print(f"  [SUE] {symbol}: SUE={sue_val} [X]")
                except Exception as e:
                    print(f"  [SUE] {symbol}: Erreur ({e}) [X]")
            print(f"[PIPELINE] {len(sue_filtered)} symboles après filtre SUE (> {self.sue_threshold})")
            fip_filtered = sue_filtered

        # Étape 4: Scoring ML sur les candidats restants
        # Déterminer le seuil selon la saisonnalité
        if self.use_seasonal_threshold:
            month = trade_date.month if hasattr(trade_date, 'month') else trade_date.month
            threshold = MomentumFilters.get_score_threshold(month)
            print(f"[PIPELINE] Seuil saisonnier: {threshold} (mois {month})")
        else:
            threshold = self.score_threshold

        scored_symbols = []
        for symbol, mom, fip, data in fip_filtered:
            if isinstance(self.scoring, SmoothMLScoring):
                score = self.scoring.score(data, symbol=symbol)
            else:
                score = self.scoring.score(data)

            if score >= threshold:
                scored_symbols.append((symbol, score, data))
                print(f"  [SCORE] {symbol}: score={score} >= {threshold} [OK]")
            else:
                print(f"  [SCORE] {symbol}: score={score} < {threshold} [X]")

        print(f"[PIPELINE] {len(scored_symbols)} symboles après scoring ML (seuil={threshold})")

        # Étape 5: Trier par score décroissant, garder top max_stocks
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]
        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}
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
        for symbol in self.symbolsToTrade:
            df = self.symbolsData[symbol]
            last_close = df['close'].iloc[-1]
            qty = int(capital_per_stock / last_close)

            # Vérifier si le capital est suffisant
            if qty <= 0:
                print(f"[ORDER PARAMS] Capital insuffisant pour {symbol} (close={last_close}, capital_per_stock={capital_per_stock})")
                continue

            print(f"[ORDER PARAMS] Préparation des ordres pour {symbol} avec close={last_close} et capital par stock={capital_per_stock} et quantité={qty}  ")

            # Ordre d'entrée (Limit order avec prix légèrement au-dessus pour exécution rapide)
            # Note: les orderId sont assignés par le MarketDataProvider lors du placeOrder
            entry_price = round(last_close * 1.005, 2)  # 0.5% au-dessus du close pour assurer le fill
            entryorder = Order()
            entryorder.action = "BUY"
            entryorder.orderType = "LMT"
            entryorder.lmtPrice = entry_price
            entryorder.totalQuantity = qty
            entryorder.eTradeOnly = False
            entryorder.firmQuoteOnly = False
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
                slorder.transmit = True  # Transmet le bracket complet
                slorder.eTradeOnly = False
                slorder.firmQuoteOnly = False
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
