"""
Wrapper Backtrader avec uniquement Trailing Stop (pas de Take Profit).
La position reste ouverte tant que le trailing stop n'est pas touché.
"""
import backtrader as bt
from datetime import date, datetime, timedelta

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.strategies.momentum import MomentumStrategy
from src.app.strategies.addivergence import AdDivergenceStrategy
from src.app.database.trade_journal import TradeJournal, TradeMode
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
        dataframes=None,  # DataFrames pré-chargés depuis SQLite
        scoring_type="smooth_ml",  # Type de scoring: "ml", "smooth_ml", "earnings_ml"
        use_sue_filter=False,  # Filtre SUE (Novy-Marx)
        sue_threshold=0.0  # Seuil SUE (SUE > sue_threshold)
    )

    def __init__(self, start_date, end_date, dataframes=None):
        self.start_date = start_date
        self.end_date = end_date

        # MarketData mock avec cache des DataFrames
        self.market_data = MarketDataMock(self.datas)

        # Injecter les DataFrames pré-chargés dans le cache
        if self.p.dataframes:
            self._preload_cache(self.p.dataframes)

        # Stratégie métier
        self.strategy = MomentumStrategy(
            market_data=self.market_data,
            capital=self.p.capital,
            max_stocks=self.p.max_stocks,
            use_trailing_stop=True,
            trailing_percent=self.p.trailing_percent,
            scoring_type=self.p.scoring_type,
            use_sue_filter=self.p.use_sue_filter,
            sue_threshold=self.p.sue_threshold
        )

        self.orders_by_symbol = {}
        self.pending_order_bundles = {}
        self.last_entry_prices = {}
        self.pending_stops = {}

        # Journal de trading en BD
        self.trade_entries = {}
        self.strategy_name = "TrailingOnly"
        self.trade_journal = TradeJournal()
        # Effacer les trades backtest existants pour cette stratégie
        self.trade_journal.clear_backtest_trades(self.strategy_name)

        # Préparation du fichier de diagnostic
        diag_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diag_filename = f"diagnostique_trailing_only_{diag_date}.txt"
        with open(self.diag_filename, "w") as f:
            f.write(f"--- Début du diagnostic TrailingOnly ({diag_date}) ---\n")

    def _log_trade_journal(self, symbol, entry_date, qty, entry_price, exit_date, exit_price, cause, pnl_brut, bars_held=0):
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
                    'bar_entree': len(self)
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
                        bars_held=bars_held
                    )
                    del self.trade_entries[symbol]

                # Nettoyage complet de tous les dictionnaires pour permettre une réentrée
                if symbol in self.pending_stops:
                    del self.pending_stops[symbol]
                if symbol in self.last_entry_prices:
                    del self.last_entry_prices[symbol]
                if symbol in self.orders_by_symbol:
                    del self.orders_by_symbol[symbol]
                if symbol in self.pending_order_bundles:
                    del self.pending_order_bundles[symbol]
                    
                self.log_diag(f"[CLEANUP] Position fermée pour {symbol} via {order_type}, PnL: {pnl:.2f}, Bars: {bars_held}, tous les trackers nettoyés")

            self.pending_order_bundles[symbol] = None

    def next(self):
        current_date = self.datas[0].datetime.date(0)

        # Ne trader que dans la période spécifiée
        if current_date < self.start_date.date() or current_date > self.end_date.date():
            return

        self.log_diag(f"Analyse pour la date {current_date}...")
        self.log_diag(f"[DIAG] Cash: {self.broker.get_cash():.2f} | Value: {self.broker.get_value():.2f}")

        symbols = [d._name for d in self.datas]
        self.strategy.set_symbols_to_analyse(symbols)
        symbols_to_trade = self.strategy.get_symbols(current_date)
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
                
            symbols_available.append(symbol)
        
        if not symbols_available:
            self.log_diag(f"[FILTER] Aucun symbole disponible pour trader ce jour")
            return
        
        # Mettre à jour la liste des symboles à trader (APRÈS filtrage)
        self.strategy.symbolsToTrade = symbols_available
        self.log_diag(f"[FILTER] Symboles disponibles après filtrage: {symbols_available}")
        
        # Générer les ordres UNIQUEMENT pour les symboles disponibles
        order_bundles = self.strategy.get_order_params()

        print(f"[DEBUG] Nombre de bundles à traiter: {len(order_bundles)}")
        print(f"[DEBUG] Data feeds disponibles: {[d._name for d in self.datas[:10]]}...")

        for bundle in order_bundles:
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

            # Placer l'ordre d'entrée (market order)
            qty = bundle["entry_order"].totalQuantity
            cash = self.broker.get_cash()
            price = data.close[0]
            print(f"[DEBUG] Tentative achat {symbol}: qty={qty}, price={price:.2f}, cash={cash:.2f}")

            entry_order = self.buy(data=data, size=qty)

            if entry_order is not None:
                print(f"[BT] Entry order placé pour {symbol}, qty={qty}")
                self.log_diag(f"[BT] Entry order placé pour {symbol}, qty={qty}")
                self.orders_by_symbol[symbol] = {"entry": entry_order}
                self.pending_order_bundles[symbol] = bundle
            else:
                print(f"[ERROR] Échec placement ordre pour {symbol} (buy returned None)")
                self.log_diag(f"[ERROR] Échec de placement d'ordre pour {symbol}")

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
                        bars_held=bars_held
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
            strategy_name=self.strategy_name
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
