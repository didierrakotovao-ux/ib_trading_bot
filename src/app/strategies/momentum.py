from datetime import datetime, timedelta
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_smooth_scoring import SmoothMLScoring
from src.app.strategies.momentum_filters import MomentumFilters
from src.app.strategies.earnings_filter import EarningsFilter
from src.app.screener.providers.market_data_provider import MarketDataProvider
from src.app.database.db_manager import DatabaseManager
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
                 scoring_type="smooth_ml",
                 momentum_top_pct=0.3, fip_threshold=0.0,
                 use_seasonal_threshold=False,
                 use_earnings_filter=True, earnings_blackout_days=14,
                 post_earnings_days=5,
                 use_sue_filter=False, sue_threshold=0.0,
                 db_path="trading_data.db",
                 smooth_model_path=None):
        if scoring_type != "smooth_ml":
            raise ValueError(
                f"scoring_type={scoring_type!r} n'est plus supporté — "
                "seul 'smooth_ml' (smooth_momentum_model.pkl) est utilisé en production."
            )
        _model_path = smooth_model_path or "models/smooth_momentum_model.pkl"
        self.scoring = SmoothMLScoring(model_path=_model_path, db_path=db_path)
        self.scoring_type = scoring_type
        self.market_data = market_data
        self.lookback_days = 400  # ~252 jours de trading + marge pour les features
        # Seuil sur l'ÉCHELLE CALIBRÉE (probabilités isotoniques, modèle
        # 2018-2026 à 100 arbres). Balayage out-of-time 2025-04→2026-06 :
        #   >=50 : précision 49.7% | EV +2.77% | ~3 signaux/j (> capacité)
        #   >=55 : précision 54.5% | EV +3.49% | ~10 signaux/mois = capacité
        # Avec 5 slots et ~2 semaines de détention, 55 gagne 5 pts de
        # précision et +0.7% d'EV sans coût de capacité. Le test d'ère
        # 2010-2017 (utilisateur) confirme la structure : 56% à >=50.
        # NB: les anciens seuils 60-65 vivaient sur l'échelle non calibrée.
        self.score_threshold = 55
        self.capital = capital
        self.max_stocks = max_stocks
        self.use_trailing_stop = use_trailing_stop
        self.trailing_percent = trailing_percent

        # Filtres Gray & Vogel
        self.momentum_top_pct = momentum_top_pct  # Top 30% par momentum 12-1
        self.fip_threshold = fip_threshold          # FIP < 0 = bon momentum smooth
        # Seuil saisonnier (window dressing, Gray & Vogel) : désactivé par
        # défaut depuis que month_sin/month_cos sont DANS le modèle — la
        # saisonnalité est apprise, un seuil externe la compterait deux fois
        # (et la table de get_score_threshold est calibrée sur l'ancienne
        # échelle de score)
        self.use_seasonal_threshold = use_seasonal_threshold

        # Filtre earnings : pas d'entrée si annonce de résultats proche
        # (les gaps post-earnings traversent le trailing stop)
        self.use_earnings_filter = use_earnings_filter
        self.earnings_blackout_days = earnings_blackout_days
        self.post_earnings_days = post_earnings_days
        self.earnings_filter = EarningsFilter(
            blackout_days=earnings_blackout_days,
            post_earnings_days=post_earnings_days) \
            if use_earnings_filter else None

        # Filtre SUE (Novy-Marx) — retiré, EarningsFeatures n'est plus chargé
        if use_sue_filter:
            raise ValueError("use_sue_filter=True n'est plus supporté (EarningsFeatures retiré).")
        self.use_sue_filter = use_sue_filter
        self.sue_threshold = sue_threshold

        # DB locale pour les données historiques (priorité sur yfinance)
        self.db_manager = DatabaseManager(db_path)

        # Scores pré-calculés (backtest uniquement) : injectés par le wrapper
        # via score_precompute.py — get_symbols passe alors en mode lookup
        # au lieu de recalculer features+score par symbole et par jour
        self.precomputed = None

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
        # Mode backtest avec scores pré-calculés : pipeline en lookups O(1)
        if self.precomputed is not None:
            return self._get_symbols_precomputed(trade_date)

        # Étape 1: Récupérer les données pour tous les symboles
        all_candidates = []
        nan_count = 0
        # Déterminer si market_data a un cache préchargé (mode backtest)
        _cache_preloaded = getattr(self.market_data, '_preloaded', False)
        for symbol in self.symbolsToAnalyse:
            start_date = trade_date - timedelta(days=self.lookback_days)
            data = None
            source = "none"
            # Mode backtest : utiliser le cache en mémoire en priorité (évite les requêtes DB par symbole)
            if _cache_preloaded:
                data = self.market_data.get_historical_data(symbol, start_date, trade_date, interval="1d")
                source = "cache"
            # Mode live ou cache manquant : requête DB puis fallback yfinance
            if data is None or len(data) < 60:
                data = self.db_manager.get_historical_data(symbol, start_date, trade_date)
                source = "db"
            if data is None or len(data) < 60:
                data = self.market_data.get_historical_data(symbol, start_date, trade_date, interval="1d")
                source = "yfinance"
            if data is not None and len(data) >= 60:
                momentum_12_1 = MomentumFilters.calc_momentum_12_1(data)
                if np.isnan(momentum_12_1):
                    nan_count += 1
                all_candidates.append((symbol, momentum_12_1, data))

        valid_count = len(all_candidates) - nan_count
        if all_candidates:
            print(f"[PIPELINE] {trade_date} | {len(all_candidates)} symboles ({valid_count} valides, {nan_count} NaN momentum)")
        else:
            print(f"[PIPELINE] {trade_date} | 0 symboles avec données suffisantes")

        # Étape 2: Filtre momentum 12-1 (top 30%)
        momentum_filtered = MomentumFilters.filter_top_momentum(
            all_candidates, self.momentum_top_pct)
        print(f"[PIPELINE] {trade_date} | {len(momentum_filtered)} après momentum top {self.momentum_top_pct*100:.0f}%")

        # Étape 3: Filtre FIP
        fip_filtered = []
        for symbol, mom, data in momentum_filtered:
            fip = MomentumFilters.calc_fip(data)
            if not np.isnan(fip) and fip < self.fip_threshold:
                fip_filtered.append((symbol, mom, fip, data))

        print(f"[PIPELINE] {trade_date} | {len(fip_filtered)} après filtre FIP (< {self.fip_threshold})")

        # Étape 3b: Filtre earnings — pas d'entrée si annonce de résultats
        # dans les earnings_blackout_days jours (gap risk au travers du stop)
        if self.use_earnings_filter and fip_filtered:
            self.earnings_filter.preload([t[0] for t in fip_filtered], trade_date)
            fip_filtered = [
                t for t in fip_filtered
                if not self.earnings_filter.is_in_blackout(t[0], trade_date)
            ]
            print(f"[PIPELINE] {trade_date} | {len(fip_filtered)} après filtre earnings "
                  f"(blackout {self.earnings_blackout_days}j)")

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
            score = self.scoring.score(data, symbol=symbol)

            if score >= threshold:
                scored_symbols.append((symbol, score, data))

        print(f"[PIPELINE] {trade_date} | {len(scored_symbols)} après scoring ML (seuil={threshold})"
              + (f": {[s[0] for s in scored_symbols]}" if scored_symbols else ""))

        # Étape 5: Trier par score décroissant, garder top max_stocks
        scored_symbols.sort(key=lambda x: x[1], reverse=True)
        selected = scored_symbols[:self.max_stocks]
        self.symbolsToTrade = [s[0] for s in selected]
        self.symbolsData = {s[0]: s[2] for s in selected}
        # Scores d'entrée, exposés pour la journalisation (score_entree)
        self.symbolsScores = {s[0]: s[1] for s in selected}
        return self.symbolsToTrade

    def _get_symbols_precomputed(self, trade_date) -> list:
        """
        Même pipeline que get_symbols (momentum 12-1 -> FIP -> earnings ->
        score -> top N) mais servi par les lookups pré-calculés — les données
        ne sont chargées que pour les symboles finalement sélectionnés.
        """
        pre = self.precomputed

        # Étape 2: momentum 12-1 (mêmes sémantiques que filter_top_momentum)
        candidates = []
        for symbol in self.symbolsToAnalyse:
            mom = pre.momentum_12_1(symbol, trade_date)
            if not np.isnan(mom):
                candidates.append((symbol, mom))
        candidates.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(candidates) * self.momentum_top_pct)) \
            if candidates else 0
        momentum_filtered = candidates[:n_keep]
        print(f"[PIPELINE] {trade_date} | {len(momentum_filtered)} après momentum "
              f"top {self.momentum_top_pct*100:.0f}% (précalculé)")

        # Étape 3: FIP
        fip_filtered = [
            (s, m) for s, m in momentum_filtered
            if not np.isnan(pre.fip(s, trade_date))
            and pre.fip(s, trade_date) < self.fip_threshold
        ]
        print(f"[PIPELINE] {trade_date} | {len(fip_filtered)} après filtre FIP")

        # Étape 3b: earnings
        if self.use_earnings_filter and fip_filtered:
            self.earnings_filter.preload([s for s, _ in fip_filtered], trade_date)
            fip_filtered = [
                (s, m) for s, m in fip_filtered
                if not self.earnings_filter.is_in_blackout(s, trade_date)
            ]
            print(f"[PIPELINE] {trade_date} | {len(fip_filtered)} après filtre earnings")

        # Étape 4: seuil de score
        if self.use_seasonal_threshold:
            threshold = MomentumFilters.get_score_threshold(trade_date.month)
        else:
            threshold = self.score_threshold
        scored = [(s, pre.score(s, trade_date)) for s, _ in fip_filtered]
        scored = [(s, sc) for s, sc in scored if sc >= threshold]
        print(f"[PIPELINE] {trade_date} | {len(scored)} après scoring (seuil={threshold})"
              + (f": {[s for s, _ in scored]}" if scored else ""))

        # Étape 5: top max_stocks, puis données pour les seuls sélectionnés
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = scored[:self.max_stocks]
        start_date = trade_date - timedelta(days=self.lookback_days)
        self.symbolsToTrade, self.symbolsData, self.symbolsScores = [], {}, {}
        for symbol, score in selected:
            data = self.market_data.get_historical_data(
                symbol, start_date, trade_date, interval="1d")
            if data is None or len(data) < 60:
                data = self.db_manager.get_historical_data(symbol, start_date, trade_date)
            if data is None or len(data) < 60:
                continue
            self.symbolsToTrade.append(symbol)
            self.symbolsData[symbol] = data
            self.symbolsScores[symbol] = score
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
            entryorder.tif = "DAY"  # Good Till Cancelled

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
