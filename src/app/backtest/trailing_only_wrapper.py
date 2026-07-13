"""
Wrapper Backtrader avec uniquement Trailing Stop (pas de Take Profit).
La position reste ouverte tant que le trailing stop n'est pas touché.
"""
import backtrader as bt
from datetime import date, datetime, timedelta
from bisect import bisect_right

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.strategies.momentum import MomentumStrategy
from src.app.strategies.wyckoff_ml import WyckoffMLStrategy
from src.app.strategies.addivergence import AdDivergenceStrategy
from src.app.value.fundamental_filters import FundamentalFilters
from src.app.database.trade_journal import TradeJournal, TradeMode
from src.app.database.pg_connection import read_sql
from market_data_mock import MarketDataMock


class TrailingOnlyBTWrapper(bt.Strategy):
    """
    Stratégie avec trailing stop uniquement (pas de TP).
    Laisse courir les gains tant que le stop n'est pas touché.
    """
    params = dict(
        capital=100000,
        max_stocks=5,
        trailing_percent=5.0,  # Trailing stop à 5%
        cooldown_days=30,       # Jours d'interdiction de réentrée après stop-out (0 = désactivé)
        dataframes=None,  # DataFrames pré-chargés depuis SQLite
        scoring_type="smooth_ml",
        smooth_model_path=None,  # Chemin du modèle smooth_ml (None = modèle US par défaut)
        wyckoff_model_path=None,  # Chemin du modèle wyckoff_ml
        # Seuils 18/27 : zone stable de la grille de sensibilité 2026-07-12
        # (surface plate — bandes larges légèrement meilleures que 20/25)
        regime_vix_symbol='^VIX',
        regime_risk_on=18.0,
        regime_risk_off=27.0,
        regime_momentum_trailing=8.0,
        # 8.0 : aligné sur les labels du modèle wyckoff (triple-barrière
        # trailing 8%, stop_config.json) — 5% exécuterait une autre stratégie
        # que celle apprise
        regime_wyckoff_trailing=8.0,
        use_sue_filter=False,  # Filtre SUE (Novy-Marx)
        sue_threshold=0.0,  # Seuil SUE (SUE > sue_threshold)
        db_path=None  # Chemin DB (None = trading_data.db)
    )

    def __init__(self, start_date, end_date, use_fondamental_data=False, dataframes=None):
        self.start_date = start_date
        self.end_date = end_date
        self.use_fondamental_data = use_fondamental_data

        # MarketData mock avec cache des DataFrames
        self.market_data = MarketDataMock(self.datas)

        # Injecter les DataFrames pré-chargés dans le cache
        if self.p.dataframes:
            self._preload_cache(self.p.dataframes)

        # Résoudre db_path pour MomentumStrategy
        db_path = "trading_data.db"
        if self.p.db_path:
            db_path = os.path.basename(self.p.db_path)

        # Stratégie métier
        self.is_regime_switch = self.p.scoring_type == "regime_switch"
        self.active_regime = None  # "momentum" | "wyckoff"
        self._vix_dates = []
        self._vix_values = {}

        if self.is_regime_switch:
            self.strategy_momentum = MomentumStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=True,
                trailing_percent=self.p.regime_momentum_trailing,
                scoring_type="smooth_ml",
                smooth_model_path=self.p.smooth_model_path,
                use_sue_filter=self.p.use_sue_filter,
                sue_threshold=self.p.sue_threshold,
                db_path=db_path,
            )
            self.strategy_wyckoff = WyckoffMLStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=True,
                trailing_percent=self.p.regime_wyckoff_trailing,
                scoring_type="wyckoff_ml",
                wyckoff_model_path=self.p.wyckoff_model_path,
                db_path=db_path,
            )
            self.strategy = None
            self._load_vix_series()
        elif self.p.scoring_type == "wyckoff_ml":
            self.strategy = WyckoffMLStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=True,
                trailing_percent=self.p.trailing_percent,
                scoring_type=self.p.scoring_type,
                wyckoff_model_path=self.p.wyckoff_model_path,
                db_path=db_path,
            )
        else:
            self.strategy = MomentumStrategy(
                market_data=self.market_data,
                capital=self.p.capital,
                max_stocks=self.p.max_stocks,
                use_trailing_stop=True,
                trailing_percent=self.p.trailing_percent,
                scoring_type=self.p.scoring_type,
                smooth_model_path=self.p.smooth_model_path,
                use_sue_filter=self.p.use_sue_filter,
                sue_threshold=self.p.sue_threshold,
                db_path=db_path,
            )

        # Pré-calcul vectorisé des scores (une fois pour toute la fenêtre,
        # au lieu de recalculer features+score par symbole et par jour) —
        # gain ~50-100x sur la durée du backtest, chemin live inchangé
        if self.p.dataframes:
            try:
                from score_precompute import precompute_for_strategy
                if self.is_regime_switch:
                    self.strategy_momentum.precomputed = precompute_for_strategy(
                        scoring_type="smooth_ml",
                        dataframes=self.p.dataframes,
                        model_path=self.p.smooth_model_path,
                    )
                    self.strategy_wyckoff.precomputed = precompute_for_strategy(
                        scoring_type="wyckoff_ml",
                        dataframes=self.p.dataframes,
                        model_path=self.p.wyckoff_model_path,
                    )
                else:
                    self.strategy.precomputed = precompute_for_strategy(
                        scoring_type=self.p.scoring_type,
                        dataframes=self.p.dataframes,
                        model_path=(self.p.wyckoff_model_path
                                    if self.p.scoring_type == "wyckoff_ml"
                                    else self.p.smooth_model_path),
                    )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[PRECOMPUTE][WARN] Pré-calcul indisponible ({e}) — "
                      f"scoring par jour classique")

        self.orders_by_symbol = {}
        self.pending_order_bundles = {}
        self.last_entry_prices = {}
        self.pending_stops = {}
        self.stop_exit_dates = {}   # symbol → date de dernier stop-out

        # Filtres fondamentaux (si activés)
        if self.use_fondamental_data:
            self.fundamental = FundamentalFilters(backtest_mode=True, max_workers=8)
            self._fund_symbols_cache = None  # résultat du filtre mis en cache
            self._fund_cache_ym      = None  # (année, mois) du dernier calcul

        # Journal de trading en BD
        self.trade_entries = {}
        if self.p.scoring_type == "wyckoff_ml":
            model_tag = "Wyckoff"
        elif self.is_regime_switch:
            model_tag = "RegimeSwitch"
        else:
            model_tag = "Smooth"
        if self.use_fondamental_data:
            self.strategy_name = f"TrailingOnly_{model_tag}_WithFundamental"
        else:
            self.strategy_name = f"TrailingOnly_{model_tag}"
           
        self.trade_journal = TradeJournal("backtest_journal.db")
        # Effacer les trades backtest existants pour cette stratégie SUR CETTE
        # PÉRIODE uniquement — les runs d'autres fenêtres sont conservés
        self.trade_journal.clear_backtest_trades(
            self.strategy_name,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date)

        # Préparation du fichier de diagnostic
        diag_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        resultats_dir = os.path.join(os.path.dirname(__file__), 'resultats')
        os.makedirs(resultats_dir, exist_ok=True)
        self.diag_filename = os.path.join(resultats_dir, f"diagnostique_trailing_only_{diag_date}.txt")
        with open(self.diag_filename, "w") as f:
            f.write(f"--- Début du diagnostic TrailingOnly ({diag_date}) ---\n")

    def _load_vix_series(self):
        """Charge la série VIX pour piloter le switch de régime."""
        try:
            start = (self.start_date - timedelta(days=40)).strftime('%Y-%m-%d')
            end = self.end_date.strftime('%Y-%m-%d')
            df = read_sql(
                """
                SELECT date, close
                FROM historical_data
                WHERE symbol = %s
                  AND date BETWEEN %s AND %s
                ORDER BY date
                """,
                (self.p.regime_vix_symbol, start, end)
            )
            if df is None or df.empty:
                print(f"[REGIME][WARN] Série {self.p.regime_vix_symbol} introuvable."
                      " Fallback momentum.")
                return

            for _, row in df.iterrows():
                d = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                self._vix_dates.append(d)
                self._vix_values[d] = float(row['close'])

            print(f"[REGIME] Série {self.p.regime_vix_symbol} chargée: {len(self._vix_dates)} points")
        except Exception as e:
            print(f"[REGIME][WARN] Impossible de charger le VIX ({e})")

    def _get_latest_vix(self, current_date):
        if not self._vix_dates:
            return None
        idx = bisect_right(self._vix_dates, current_date) - 1
        if idx < 0:
            return None
        d = self._vix_dates[idx]
        return self._vix_values.get(d)

    def _resolve_active_strategy(self, current_date):
        if not self.is_regime_switch:
            return self.strategy, self.p.scoring_type

        vix = self._get_latest_vix(current_date)
        if vix is None:
            if self.active_regime is None:
                self.active_regime = "momentum"
            return (
                self.strategy_momentum if self.active_regime == "momentum" else self.strategy_wyckoff,
                self.active_regime,
            )

        previous = self.active_regime
        if self.active_regime is None:
            self.active_regime = "wyckoff" if vix >= self.p.regime_risk_off else "momentum"
        elif self.active_regime == "momentum" and vix >= self.p.regime_risk_off:
            self.active_regime = "wyckoff"
        elif self.active_regime == "wyckoff" and vix <= self.p.regime_risk_on:
            self.active_regime = "momentum"

        if previous != self.active_regime:
            self.log_diag(
                f"[REGIME] {current_date} switch {previous} -> {self.active_regime} "
                f"(VIX={vix:.2f}, on<={self.p.regime_risk_on}, off>={self.p.regime_risk_off})"
            )

        active_strategy = self.strategy_momentum if self.active_regime == "momentum" else self.strategy_wyckoff
        return active_strategy, self.active_regime

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price, exit_date, exit_price, cause, pnl_brut, bars_held=0, score=None):
        """Enregistre un trade dans le journal BD."""
        # Calculer les commissions (0.1% à l'achat + 0.1% à la vente)
        commission_rate = 0.001
        commission_entree = entry_price * qty * commission_rate
        commission_sortie = exit_price * qty * commission_rate
        total_commission = commission_entree + commission_sortie
        pnl_net = pnl_brut - total_commission

        self.trade_journal.log_trade(
            trade_mode=TradeMode.BACKTEST,
            strategy_name=self.strategy_name,
            symbol=symbol,
            date_entree=entry_date,
            prix_entree=entry_price,
            quantite=qty,
            date_sortie=exit_date,
            prix_sortie=exit_price,
            cause_sortie=cause,
            pnl_brut=round(pnl_brut, 2),
            commission=round(total_commission, 2),
            pnl_net=round(pnl_net, 2),
            bars_held=bars_held,
            score_entree=score,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date
        )

    def _preload_cache(self, dataframes):
        """Pré-charge les DataFrames dans le cache du MarketDataMock."""
        import pandas as pd
        for symbol, df in dataframes.items():
            df_reset = df.reset_index()
            df_reset.columns = [col.lower() for col in df_reset.columns]
            if 'date' not in df_reset.columns and 'index' in df_reset.columns:
                df_reset = df_reset.rename(columns={'index': 'date'})
            df_reset['date'] = pd.to_datetime(df_reset['date'])
            self.market_data._data_cache[symbol] = df_reset
        self.market_data._preloaded = True
        print(f"[CACHE] {len(dataframes)} symboles pré-chargés dans le cache")

    def log_diag(self, message):
        with open(self.diag_filename, "a") as f:
            f.write(message + "\n")

    def notify_order(self, order):
        symbol = order.data._name if hasattr(order, 'data') and order.data else 'N/A'
        print(f"[NOTIFY] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}")
        self.log_diag(f"[notify_order] status={order.getstatusname()}, symbol={symbol}, isbuy={order.isbuy()}")

        if order.status in [order.Completed]:
            if order.isbuy():
                order_type = 'ENTRY'
                self.last_entry_prices[symbol] = order.executed.price
                # Stocker les infos d'entrée pour le journal
                self.trade_entries[symbol] = {
                    'date_entree': order.data.datetime.datetime(0),
                    'prix_entree': order.executed.price,
                    'quantite': abs(order.executed.size),
                    'bar_entree': len(self),
                    # Score ML à l'entrée (None si la stratégie ne l'expose pas)
                    'score': self.pending_order_bundles.get(symbol, {}).get('score'),
                    'entry_regime': self.pending_order_bundles.get(symbol, {}).get('regime'),
                }

                # Placer le trailing stop uniquement
                data = self._get_data(symbol)
                if data is not None:
                    # Trailing stop (suit le prix à la hausse, perte max = trailing_percent)
                    stop_order = self.sell(
                        data=data,
                        size=abs(order.executed.size),
                        exectype=bt.Order.StopTrail,
                        trailpercent=self.p.trailing_percent / 100.0
                    )

                    self.pending_stops[symbol] = stop_order
                    self.log_diag(f"[BT] Trailing stop {self.p.trailing_percent}% placé pour {symbol}")

            elif order.issell():
                self.orders_by_symbol[symbol] = None
                if symbol not in self.last_entry_prices:
                    self.log_diag(f"[WARNING] Stop orphelin pour {symbol}")
                    if symbol in self.pending_stops:
                        self.pending_stops[symbol] = None
                    return

                last_entry_price = self.last_entry_prices[symbol]
                pnl = (order.executed.price - last_entry_price) * abs(order.executed.size)
                order_type = 'TRAILING_STOP'

                # Nettoyer le pending stop
                if symbol in self.pending_stops:
                    self.pending_stops[symbol] = None

                # Calculer bars_held et écrire dans le journal
                bars_held = 0
                if symbol in self.trade_entries:
                    entry_info = self.trade_entries[symbol]
                    bars_held = len(self) - entry_info.get('bar_entree', len(self))
                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=entry_info['date_entree'],
                        qty=entry_info['quantite'],
                        entry_price=entry_info['prix_entree'],
                        exit_date=order.data.datetime.datetime(0),
                        exit_price=order.executed.price,
                        cause=order_type,
                        pnl_brut=pnl,
                        bars_held=bars_held,
                        score=entry_info.get('score')
                    )
                    del self.trade_entries[symbol]

                # Enregistrer la date de stop-out pour le cooldown
                if self.p.cooldown_days > 0:
                    self.stop_exit_dates[symbol] = order.data.datetime.date(0)

                # Nettoyage complet de tous les dictionnaires pour permettre une réentrée
                if symbol in self.pending_stops:
                    del self.pending_stops[symbol]
                if symbol in self.last_entry_prices:
                    del self.last_entry_prices[symbol]
                if symbol in self.orders_by_symbol:
                    del self.orders_by_symbol[symbol]
                if symbol in self.pending_order_bundles:
                    del self.pending_order_bundles[symbol]

                cooldown_msg = (f", cooldown jusqu'au {self.stop_exit_dates[symbol] + timedelta(days=self.p.cooldown_days)}"
                                if self.p.cooldown_days > 0 else "")
                self.log_diag(f"[CLEANUP] Position fermée pour {symbol} via {order_type}, PnL: {pnl:.2f}, Bars: {bars_held}{cooldown_msg}")

            self.pending_order_bundles[symbol] = None

    def prenext(self):
        """Appelé avant que tous les data feeds soient alignés - on trade quand même."""
        self.next()

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        bar_num = len(self)
        total_bars = len(self.datas[0])

        # Ne trader que dans la période spécifiée
        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return

        # Log seulement les jours de trading (pas tous les jours)
        if bar_num % 20 == 0 or bar_num == total_bars:
            print(f"[NEXT] Date={current_date}, Bar={bar_num}/{total_bars}")
        self.log_diag(f"Analyse pour la date {current_date}...")
        self.log_diag(f"[DIAG] Cash: {self.broker.get_cash():.2f} | Value: {self.broker.get_value():.2f}")

        # Optimisation: ne pas chercher de nouvelles positions si on a déjà max_stocks
        current_positions = sum(1 for d in self.datas if self.getposition(d).size != 0)
        if current_positions >= self.p.max_stocks:
            return

        symbols = [d._name for d in self.datas if not str(d._name).startswith('^')]

        if self.use_fondamental_data:
            ym = (current_date.year, current_date.month)

            # Recalculer une fois par mois (les fondamentaux changent trimestriellement)
            if self._fund_cache_ym != ym:
                all_syms = [d._name for d in self.datas if not str(d._name).startswith('^')]
                prices   = {d._name: d.close[0] for d in self.datas}

                # Pré-charger tous les fondamentaux en parallèle (cache 24h ensuite)
                self.fundamental._prefetch_parallel(all_syms)

                # 1. Piotroski > 6
                pio_syms, _ = self.fundamental.filter_piotroski(
                    all_syms, min_score=6, trade_date=current_date)

                # 2. EBIT/TEV top 10% parmi les symboles Piotroski OK
                ratios  = {s: self.fundamental.calc_ebit_tev(
                               s, trade_date=current_date,
                               current_price=prices.get(s))
                           for s in pio_syms}
                valid   = {s: r for s, r in ratios.items() if r is not None}
                n_keep  = max(1, int(len(valid) * 0.10))
                top_ev  = [s for s, _ in
                           sorted(valid.items(), key=lambda x: x[1], reverse=True)[:n_keep]]

                self._fund_symbols_cache = top_ev 
                self._fund_cache_ym = ym
                print(f"[FUND] {current_date}: {len(top_ev)} top EBIT/TEV")

            symbols = self._fund_symbols_cache
            
        active_strategy, active_mode = self._resolve_active_strategy(current_date)
        active_strategy.set_symbols_to_analyse(symbols)
        symbols_to_trade = active_strategy.get_symbols(current_date)
        self.log_diag(f"[REGIME] Mode actif: {active_mode}")
        self.log_diag(f"{current_date} Symboles sélectionnés: {symbols_to_trade}")

        # FILTRER les symboles déjà en position AVANT de générer les ordres
        symbols_available = []
        for symbol in symbols_to_trade:
            data = self._get_data(symbol)
            if not data:
                continue
                
            pos = self.getposition(data)
            
            # Vérifier si position déjà ouverte ou ordres en attente
            has_position = pos.size != 0
            has_pending_order = symbol in self.orders_by_symbol and self.orders_by_symbol[symbol] is not None
            has_pending_stop = symbol in self.pending_stops and self.pending_stops[symbol] is not None
            
            if has_position or has_pending_order or has_pending_stop:
                self.log_diag(f"[FILTER] {symbol} ignoré (position={has_position}, order={has_pending_order}, stop={has_pending_stop})")
                continue

            # Vérifier le cooldown après stop-out
            if self.p.cooldown_days > 0 and symbol in self.stop_exit_dates:
                days_since_stop = (current_date - self.stop_exit_dates[symbol]).days
                if days_since_stop < self.p.cooldown_days:
                    self.log_diag(f"[COOLDOWN] {symbol} ignoré (stop il y a {days_since_stop}j, cooldown={self.p.cooldown_days}j)")
                    continue

            symbols_available.append(symbol)
        
        if not symbols_available:
            self.log_diag(f"[FILTER] Aucun symbole disponible pour trader ce jour")
            return
        
        # Mettre à jour la liste des symboles à trader (APRÈS filtrage)
        active_strategy.symbolsToTrade = symbols_available
        self.log_diag(f"[FILTER] Symboles disponibles après filtrage: {symbols_available}")

        # Mettre à jour le capital avec la valeur actuelle du portefeuille
        current_capital = self.broker.getvalue()
        active_strategy.capital = current_capital
        self.log_diag(f"[CAPITAL] Valeur portefeuille actuelle: {current_capital:,.2f}$")

        # Générer les ordres UNIQUEMENT pour les symboles disponibles
        order_bundles = active_strategy.get_order_params()

        print(f"[DEBUG] Nombre de bundles à traiter: {len(order_bundles)}")
        print(f"[DEBUG] Data feeds disponibles: {[d._name for d in self.datas[:10]]}...")

        # Sizing dynamique base sur le cash disponible pour limiter les rejets Margin.
        reserved_cash = float(self.broker.get_cash())
        planned_entries = 0
        cash_buffer_mult = 1.002  # petit coussin pour commissions/slippage

        for bundle in order_bundles:
            slots_left = self.p.max_stocks - current_positions - planned_entries
            if slots_left <= 0:
                break

            symbol = bundle["symbol"]
            data = self._get_data(symbol)
            if not data:
                print(f"[ERROR] Data non trouvée pour {symbol}")
                self.log_diag(f"[ERROR] Data non trouvée pour {symbol}")
                continue

            # Double vérification (normalement déjà filtré, mais par sécurité)
            pos = self.getposition(data)
            if pos.size != 0:
                print(f"[WARNING] {symbol} a position, skip")
                self.log_diag(f"[WARNING] {symbol} a une position mais n'aurait pas dû passer le filtre, skip")
                continue

            # Ajuster la quantité en fonction du cash restant et des slots restants.
            requested_qty = int(bundle["entry_order"].totalQuantity)
            entry_price_ref = float(getattr(bundle["entry_order"], "lmtPrice", data.close[0] * 1.005))
            budget_per_slot = reserved_cash / max(slots_left, 1)
            affordable_qty = int(budget_per_slot / max(entry_price_ref * cash_buffer_mult, 0.01))
            qty = min(requested_qty, affordable_qty)

            if qty <= 0:
                self.log_diag(
                    f"[CASH] {symbol} ignoré (cash réservé={reserved_cash:.2f}, "
                    f"slots_left={slots_left}, prix_ref={entry_price_ref:.2f})"
                )
                continue

            cash = self.broker.get_cash()
            price = data.close[0]
            print(
                f"[DEBUG] Tentative achat {symbol}: qty={qty} (req={requested_qty}), "
                f"price={price:.2f}, cash={cash:.2f}, reserved={reserved_cash:.2f}"
            )

            try:
                entry_order = self.buy(data=data, size=qty)
                print(f"[DEBUG] buy() returned: {entry_order}")

                if entry_order is not None:
                    print(f"[BT] Entry order placé pour {symbol}, qty={qty}, ref={entry_order.ref}")
                    self.log_diag(f"[BT] Entry order placé pour {symbol}, qty={qty}")
                    self.orders_by_symbol[symbol] = {"entry": entry_order}
                    self.pending_order_bundles[symbol] = {
                        "bundle": bundle,
                        "regime": active_mode,
                        "score": getattr(active_strategy, 'symbolsScores', {}).get(symbol),
                    }
                    reserved_cash -= qty * entry_price_ref * cash_buffer_mult
                    planned_entries += 1
                else:
                    print(f"[ERROR] Échec placement ordre pour {symbol} (buy returned None)")
                    self.log_diag(f"[ERROR] Échec de placement d'ordre pour {symbol}")
            except Exception as e:
                print(f"[EXCEPTION] buy() failed for {symbol}: {e}")
                import traceback
                traceback.print_exc()

    def _get_data(self, symbol):
        for d in self.datas:
            if d._name == symbol:
                return d
        return None

    def notify_trade(self, trade):
        if trade.isclosed:
            symbol = trade.data._name
            self.orders_by_symbol.pop(symbol, None)

    def stop(self):
        """Appelé à la fin du backtest - ferme toutes les positions ouvertes et affiche le résumé."""

        # Fermer toutes les positions ouvertes à la fin du backtest
        print("\n[INFO] Clôture des positions ouvertes en fin de backtest...")
        closed_count = 0
        missing_entries = 0

        # Debug: afficher l'état des trackers
        print(f"[DEBUG] trade_entries: {list(self.trade_entries.keys())}")
        print(f"[DEBUG] pending_stops: {list(self.pending_stops.keys())}")

        for data in self.datas:
            symbol = data._name
            pos = self.getposition(data)

            # Vérifier toute position non nulle (long ou short)
            if pos.size != 0:
                exit_price = data.close[0]
                exit_date = data.datetime.datetime(0)
                qty = abs(pos.size)

                print(f"[DEBUG] Position ouverte trouvée: {symbol}, size={pos.size}, price={pos.price:.2f}")

                if symbol in self.trade_entries:
                    entry_info = self.trade_entries[symbol]
                    pnl_brut = (exit_price - entry_info['prix_entree']) * entry_info['quantite']
                    bars_held = len(self) - entry_info.get('bar_entree', len(self))

                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=entry_info['date_entree'],
                        qty=entry_info['quantite'],
                        entry_price=entry_info['prix_entree'],
                        exit_date=exit_date,
                        exit_price=exit_price,
                        cause='END_OF_BACKTEST',
                        pnl_brut=pnl_brut,
                        bars_held=bars_held,
                        score=entry_info.get('score')
                    )
                    print(f"  {symbol}: Fermé à {exit_price:.2f} (Entrée: {entry_info['prix_entree']:.2f}, PnL: {pnl_brut:+.2f}$, Bars: {bars_held})")
                    closed_count += 1
                else:
                    # Position sans trace d'entrée - utiliser le prix moyen de Backtrader
                    avg_price = pos.price if pos.price > 0 else exit_price
                    # PnL correct selon le type de position
                    if pos.size > 0:  # LONG
                        pnl_brut = (exit_price - avg_price) * qty
                    else:  # SHORT (ne devrait pas arriver mais calculer correctement)
                        pnl_brut = (avg_price - exit_price) * qty

                    position_type = "LONG" if pos.size > 0 else "SHORT"
                    print(f"  [WARNING] {symbol}: Position {position_type} sans entry info (size={pos.size}, avg_price={avg_price:.2f})")
                    print(f"            Fermé à {exit_price:.2f} (PnL estimé: {pnl_brut:+.2f}$)")

                    # Logger avec les infos disponibles de Backtrader
                    self._log_trade_journal(
                        symbol=symbol,
                        entry_date=exit_date - timedelta(days=5),  # Date estimée
                        qty=qty,
                        entry_price=avg_price,
                        exit_date=exit_date,
                        exit_price=exit_price,
                        cause='END_OF_BACKTEST_NO_ENTRY_INFO',
                        pnl_brut=pnl_brut,
                        bars_held=0
                    )
                    closed_count += 1
                    missing_entries += 1

                # Annuler le trailing stop en attente
                if symbol in self.pending_stops and self.pending_stops[symbol]:
                    stop_order = self.pending_stops[symbol]
                    if stop_order.status not in [bt.Order.Completed, bt.Order.Canceled]:
                        self.cancel(stop_order)

        print(f"\n[INFO] Positions fermées: {closed_count} (dont {missing_entries} sans entry info)")
        
        print("\n" + "=" * 60)
        print("RÉSUMÉ BD - TrailingOnly")
        print("=" * 60)
        summary = self.trade_journal.get_performance_summary(
            trade_mode=TradeMode.BACKTEST,
            strategy_name=self.strategy_name,
            backtest_start_date=self.start_date,
            backtest_end_date=self.end_date
        )
        if "error" not in summary:
            print(f"Trades: {summary['total_trades']} (W:{summary['winning_trades']} / L:{summary['losing_trades']})")
            print(f"Win Rate: {summary['win_rate']:.1f}%")
            print(f"PnL Net Total: {summary['total_pnl_net']:,.2f}$")
            print(f"PnL Net Moyen: {summary['avg_pnl_net']:,.2f}$")
            print(f"Max Win: {summary['max_win']:,.2f}$ | Max Loss: {summary['max_loss']:,.2f}$")
        else:
            print(summary.get("error", "Erreur inconnue"))

        print("=" * 60)

        # Fermer la connexion BD
        self.trade_journal.close()
