"""
Provider de données utilisant l'API Interactive Brokers.
Wrapper autour de votre code existant pour l'adapter à l'interface DataProvider.
"""
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.scanner import ScannerSubscription
from ibapi.execution import ExecutionFilter
from collections import defaultdict
import threading
import time
import yfinance as yf

class MarketDataProvider(EWrapper, EClient):
    """Provider utilisant Interactive Brokers API pour récupérer les stocks et yfinance pour les historiques."""
    scan_sub: Optional[ScannerSubscription] = None
    
    def __init__(self, host: str = "127.0.0.1", port: int = 4001, client_id: int = 5):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        self._connected = False
        self._next_req_id = 10000
        self._thread = None
        self._ready = False  # Synchronisation sur nextValidId
        
        # Pour le scanner
        self.scanner_results = []
        self.scanner_done = False
        
        # Pour les données historiques
        self.history_buf = defaultdict(list)
        self.req_map = {}
        self.history_done = {}

        # Pour le suivi des ordres et callbacks
        self.order_callbacks = []  # Callbacks à appeler sur exécution
        self.placed_orders = {}  # orderId -> {contract, order} pour tracking

        # Pour la récupération des exécutions historiques
        self.executions = []
        self.executions_done = False

        # Pour la récupération du cash disponible
        self._account_cash = None
        self._account_cash_done = False

        # Pour la détection des erreurs d'ordre (fallback)
        self._last_order_error = None

    # --- Gestion des positions IB ---
    def req_ib_positions(self):
        """Demande les positions en cours à IB, attend que la connexion soit vraiment prête."""
        self._ib_positions = []
        self._ib_positions_done = False
        if not self.wait_until_ready():
            print("[DEBUG] Impossible de récupérer les positions : IB non prêt.")
            return []
        self.reqPositions()
        # Attendre la fin de la récupération (positionEnd)
        timeout = 10
        start = time.time()
        while not self._ib_positions_done and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self._ib_positions

    def position(self, account, contract, position, avgCost):
        """Callback IB pour chaque position ouverte."""
        print(f"[DEBUG] position callback: {contract.symbol} qty={position} avgCost={avgCost} account={account}")
        # On ne garde que les actions US (STK)
        if contract.secType == 'STK' and position != 0:
            self._ib_positions.append({
                'symbol': contract.symbol,
                'qty': position,
                'avg_cost': avgCost,
                'account': account
            })

    def positionEnd(self):
        print("[DEBUG] positionEnd callback called")
        self._ib_positions_done = True

    # --- Ordres ouverts ---
    def req_open_orders(self, timeout: float = 5.0) -> list:
        """Retourne les ordres ouverts dans IB (pour détecter les SELL déjà actifs)."""
        self._open_orders = []
        self._open_orders_done = False
        self.reqOpenOrders()
        start = time.time()
        while not self._open_orders_done and (time.time() - start) < timeout:
            time.sleep(0.1)
        return list(self._open_orders)

    def openOrder(self, orderId, contract, order, orderState):
        """Callback IB pour chaque ordre ouvert."""
        if not hasattr(self, '_open_orders'):
            self._open_orders = []
        self._open_orders.append({
            'orderId': orderId,
            'symbol': contract.symbol,
            'action': order.action,
            'qty': order.totalQuantity,
            'orderType': order.orderType,
            'status': orderState.status,
        })

    def openOrderEnd(self):
        """Callback IB signalant la fin de la liste des ordres ouverts."""
        if not hasattr(self, '_open_orders_done'):
            self._open_orders_done = False
        self._open_orders_done = True

    def req_account_cash(self, timeout: float = 10.0) -> float:
        """Retourne le cash disponible (AvailableFunds) depuis IB. Retourne -1 si indisponible."""
        self._account_cash = None
        self._account_cash_done = False
        req_id = self._next_req_id
        self._next_req_id += 1
        self.reqAccountSummary(req_id, "All", "AvailableFunds")
        start = time.time()
        while not self._account_cash_done and (time.time() - start) < timeout:
            time.sleep(0.1)
        self.cancelAccountSummary(req_id)
        if self._account_cash is None:
            print("[ACCOUNT] Cash non disponible depuis IB")
            return -1.0
        print(f"[ACCOUNT] Cash disponible IB: {self._account_cash:,.2f}$")
        return self._account_cash

    def accountSummary(self, reqId, account, tag, value, currency):
        """Callback IB pour le résumé de compte."""
        if tag == "AvailableFunds":
            try:
                self._account_cash = float(value)
            except ValueError:
                pass

    def accountSummaryEnd(self, reqId):
        """Callback IB signalant la fin du résumé de compte."""
        self._account_cash_done = True

    def get_ib_positions(self):
        """Récupère la liste des positions ouvertes via l'API IB."""
        if not self.is_connected():
            print("[DEBUG] Connexion IB non active, tentative de connexion...")
            self.connect()
            # Attendre que la connexion soit bien établie
            timeout = 10
            start = time.time()
            while not self.is_connected() and (time.time() - start) < timeout:
                time.sleep(0.1)
            if not self.is_connected():
                print("[DEBUG] Impossible d'établir la connexion IB pour get_ib_positions")
                return []
        # Attendre que l'API soit vraiment prête (nextValidId)
        if not self.wait_until_ready():
            return []
        return self.req_ib_positions()
    
    def connect(self) -> bool:
        """Établit la connexion avec IB Gateway/TWS."""
        try:
            
            EClient.connect(self, self.host, self.port, self.client_id)
            # Lancer le thread de communication
            self._thread = threading.Thread(target=self.run, daemon=True)
            self._thread.start()
            
            # Attendre la connexion
            timeout = 10
            start = time.time()
            while not self.isConnected() and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if self.isConnected():
                self._connected = True
                print(f"[OK] Connecté à IB ({self.host}:{self.port})")
                return True
            else:
                print(f"[ERREUR] Timeout de connexion à IB")
                return False
                
        except Exception as e:
            print(f"[ERREUR] Erreur de connexion IB: {e}")
            return False
    
    def disconnect(self):
        """Ferme la connexion IB."""
        if self.isConnected():
            EClient.disconnect(self)
        self._connected = False
        print("Deconnecte d'IB")
    
    def is_connected(self) -> bool:
        return self._connected and self.isConnected() # type: ignore
    
    def nextValidId(self, orderId: int):
        """Callback IB après connexion."""
        super().nextValidId(orderId)
        self._next_req_id = orderId
        self._ready = True
        print(f"IB connecté, nextValidId = {orderId}")
    
    def wait_until_ready(self, timeout=10):
        start = time.time()
        while not self._ready and (time.time() - start) < timeout:
            time.sleep(0.1)
        if not self._ready:
            print("[DEBUG] IB API pas prête (nextValidId non reçu)")
        return self._ready

    def get_scanner_results(
        self,
        scan_sub: ScannerSubscription, 
        max_results: int = 50
    ) -> List[Dict]:
        """
        Lance un scanner IB et retourne les résultats.
        
        Args:
            scan_type: Code de scan IB ("HOT_BY_VOLUME", "TOP_PERC_GAIN", etc.)
            filters: Dict avec clés optionnelles:
                - location: "STK.NASDAQ", "STK.NYSE", "STK.US", etc.
                - price_min, price_max
                - volume_min
                - market_cap_min
            max_results: Nombre max de résultats
        """
        if not self.is_connected():
            print("[ERREUR] Non connecté à IB")
            self.connect()
            return []
        self.scan_sub = scan_sub
        # Reset
        self.scanner_results = []
        self.scanner_done = False
        req_id = self._next_req_id
        self._next_req_id += 1
        print(f"🔍 Lancement scanner IB: {self.scan_sub.scanCode } sur {self.scan_sub.locationCode}")
        # Protection supplémentaire : vérifier la connexion juste avant le scanner
        if not self.is_connected():
            print("[DEBUG] Connexion IB perdue juste avant reqScannerSubscription !")
            return []
        try:
            print("[DEBUG] Avant reqScannerSubscription, connexion active.")
            self.reqScannerSubscription(req_id, self.scan_sub, [], [])
            # Attendre les résultats
            timeout = 30
            start = time.time()
            while not self.scanner_done and (time.time() - start) < timeout:
                time.sleep(0.2)
                if self.scanner_results is not None and len(self.scanner_results) >= max_results:
                    break
            # Annuler la souscription
            self.cancelScannerSubscription(req_id)
            results = self.scanner_results[:max_results]
            print(f"[OK] Scanner terminé: {len(results)} résultats")
            return results
        except Exception as e:
            print(f"[ERREUR] Erreur scanner: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques via yfinance.
        
        Args:
            symbol: Symbole (ex: "AAPL")
            start_date: Date de début
            end_date: Date de fin
            interval: "1d", "1h", "5m", etc.
            
        Returns:
            DataFrame avec colonnes standardisées: date, open, high, low, close, volume
        """
        leverage_keywords = [
            'leveraged', 'ultra', '2x', '3x', 'bull', 'bear', 'proshares', 'direxion', 'x2', 'x3', 'triple', 'double'
        ]
        try:
            ticker = yf.Ticker(symbol)
            df = None
            for attempt in range(3):
                try:
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=False  # Garder les prix non ajustés
                    )
                    break
                except Exception as retry_e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # backoff: 1s, 2s
                    else:
                        raise retry_e
            if df is None:
                return None
            
            if df.empty:
                print(f"⚠️  Aucune donnée pour {symbol}")
                return None

            try:
                info = ticker.info or {}
                name = (info.get('longName') or info.get('shortName') or '').lower()
                summary = (info.get('summaryDetail') or '').lower() if isinstance(info.get('summaryDetail'), str) else ''
                # Exclure si mot-clé trouvé dans le nom ou le résumé
                if any(kw in name for kw in leverage_keywords) or any(kw in summary for kw in leverage_keywords):
                    print(f"[FILTRAGE ETF LEVERAGE] {symbol} exclu: {name}")
                    return None
            except Exception:
                # ticker.info peut échouer (rate-limit, timeout) - on continue sans filtrage
                pass
            
            # Standardiser les colonnes
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # Renommer pour uniformiser
            column_mapping = {
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Ne garder que les colonnes nécessaires
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            print(f"[OK] {symbol}: {len(df)} barres récupérées")
            return df
            
        except Exception as e:
            print(f"[ERREUR] Erreur lors de la récupération de {symbol}: {e}")
            return None
    
    def error(self, reqId, errorCode, errorString):
        """Callback IB pour les erreurs."""
        print(f"[IB ERROR] reqId={reqId} code={errorCode} msg={errorString}")
        # Capturer les erreurs d'ordre pour le mécanisme de fallback
        if errorCode >= 400 and reqId > 0:
            self._last_order_error = {'req_id': reqId, 'code': errorCode, 'msg': errorString}

    def register_order_callback(self, callback):
        """
        Enregistre un callback à appeler lors de l'exécution d'un ordre.
        Le callback doit accepter: (symbol, action, status, fill_price, fill_qty, fill_time)
        """
        self.order_callbacks.append(callback)

    def placeOrder(self, contract:Contract, order:Order, parent_order_id:int = None):
        """
        Override pour loguer les ordres placés et les tracker.
        Les orderId sont toujours assignés par le provider via _next_req_id.
        Si parent_order_id est fourni, l'ordre est un child (bracket).
        """
        order_id = self._next_req_id
        order.orderId = order_id

        # Si c'est un child order (trailing stop), lier au parent
        if parent_order_id is not None:
            order.parentId = parent_order_id

        print(f"[IB ORDER] Placing order for {contract.symbol} {order.action} {order.totalQuantity} @ {order.orderType} (orderId={order_id})")

        # Tracker l'ordre pour pouvoir l'identifier dans les callbacks
        self.placed_orders[order_id] = {
            'contract': contract,
            'order': order,
            'symbol': contract.symbol,
            'action': order.action
        }

        super().placeOrder(order_id, contract, order)
        self._next_req_id += 1
        return order_id

    def cancelOrder(self, order_id: int):
        """Annule un ordre actif par son orderId."""
        print(f"[IB ORDER] Cancelling order orderId={order_id}")
        super().cancelOrder(order_id, "")

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
                    permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice=0.0):
        """Callback IB pour le statut des ordres."""
        print(f"[ORDER STATUS] orderId={orderId} status={status} filled={filled} remaining={remaining} avgFillPrice={avgFillPrice}")
        # Stocker le statut pour utilisation ultérieure
        if not hasattr(self, 'order_statuses'):
            self.order_statuses = {}
        self.order_statuses[orderId] = {
            'status': status,
            'filled': filled,
            'remaining': remaining,
            'avgFillPrice': avgFillPrice,
            'parentId': parentId
        }

    def execDetails(self, reqId, contract, execution):
        """Callback IB pour les détails d'exécution."""
        print(f"[EXEC DETAILS] orderId={execution.orderId} symbol={contract.symbol} side={execution.side} "
              f"shares={execution.shares} price={execution.price} time={execution.time}")

        # Stocker les exécutions pour utilisation ultérieure
        if not hasattr(self, 'executions'):
            self.executions = []
        self.executions.append({
            'orderId': execution.orderId,
            'symbol': contract.symbol,
            'side': execution.side,
            'shares': execution.shares,
            'price': execution.price,
            'time': execution.time,
            'execId': execution.execId
        })

        # Notifier les callbacks enregistrés
        # Convertir side IB (BOT/SLD) en action (BUY/SELL)
        action = "BUY" if execution.side == "BOT" else "SELL"

        # Parser le temps d'exécution
        try:
            fill_time = datetime.strptime(execution.time, "%Y%m%d %H:%M:%S")
        except:
            fill_time = datetime.now()

        for callback in self.order_callbacks:
            try:
                callback(
                    symbol=contract.symbol,
                    action=action,
                    status="filled",
                    fill_price=execution.price,
                    fill_qty=execution.shares,
                    fill_time=fill_time
                )
            except Exception as e:
                print(f"[ERROR] Callback error: {e}")

    def execDetailsEnd(self, reqId):
        """Callback IB signalant la fin de la liste des exécutions."""
        print(f"[EXEC DETAILS END] reqId={reqId}, {len(self.executions)} exécutions reçues")
        self.executions_done = True

    def req_executions(self, client_id: int = None, since_days: int = 7) -> list:
        """
        Demande l'historique des exécutions à IB.

        Args:
            client_id: Filtrer par client ID (optionnel)
            since_days: Nombre de jours en arrière pour récupérer les exécutions (défaut: 7)

        Retourne la liste des exécutions.
        """
        # Attendre que l'API soit prête avant de faire la requête
        if not self.wait_until_ready():
            print("[WARNING] IB API non prête pour req_executions")
            return []

        self.executions = []
        self.executions_done = False

        # Créer un filtre avec date de début pour récupérer les exécutions multi-jours
        exec_filter = ExecutionFilter()
        if client_id:
            exec_filter.clientId = client_id

        # Setter la date de début au format IB "yyyymmdd-hh:mm:ss"
        start_date = datetime.now() - timedelta(days=since_days)
        exec_filter.time = start_date.strftime("%Y%m%d-%H:%M:%S")

        req_id = self._next_req_id
        self._next_req_id += 1

        print(f"[IB] Demande des exécutions (reqId={req_id})...")
        self.reqExecutions(req_id, exec_filter)

        # Attendre la réponse
        timeout = 10
        start = time.time()
        while not self.executions_done and (time.time() - start) < timeout:
            time.sleep(0.1)

        if not self.executions_done:
            print("[WARNING] Timeout en attendant les exécutions")

        return self.executions

    def create_contract(self, ticker: str, sec_type: str = 'STK', exchange: str = 'SMART', currency: str = 'USD') -> Contract:
        """Crée un contrat IB pour un ETF"""
        contract = Contract()
        contract.symbol = ticker
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    # --- Prix en temps réel ---
    def tickPrice(self, reqId, tickType, price, attrib):
        """Callback IB pour les prix en temps réel."""
        # tickType 4 = LAST, 1 = BID, 2 = ASK
        if tickType in (1, 2, 4) and price > 0:
            self._current_prices[reqId] = price
            if tickType == 4:  # LAST price received, we're done
                self._price_received[reqId] = True

    def get_current_price(self, symbol: str, timeout: float = 5.0) -> float:
        """
        Récupère le prix actuel d'un symbole via IB.
        Retourne le dernier prix tradé, ou 0 si échec.
        """
        if not self.is_connected():
            print(f"[WARNING] Non connecté à IB pour get_current_price({symbol})")
            return 0.0

        if not hasattr(self, '_current_prices'):
            self._current_prices = {}
            self._price_received = {}

        contract = self.create_contract(symbol)
        req_id = self._next_req_id
        self._next_req_id += 1

        self._current_prices[req_id] = 0.0
        self._price_received[req_id] = False

        # Demander snapshot — reqMktDataType(3) active les données différées (15 min, gratuit)
        # si la souscription temps réel n'est pas disponible
        self.reqMktDataType(3)
        self.reqMktData(req_id, contract, "", True, False, [])

        # Attendre le prix
        start = time.time()
        while not self._price_received.get(req_id, False) and (time.time() - start) < timeout:
            time.sleep(0.1)

        price = self._current_prices.get(req_id, 0.0)

        # Annuler la souscription
        self.cancelMktData(req_id)

        # Cleanup
        if req_id in self._current_prices:
            del self._current_prices[req_id]
        if req_id in self._price_received:
            del self._price_received[req_id]

        return price
    
    def force_close(self):
        """
        Force la fermeture de la connexion IB et du thread associé.
        """
        import sys
        try:
            self.disconnect()
            if self._thread and self._thread.is_alive():
                print("[DEBUG] Attente de l'arrêt du thread IB...")
                self._thread.join(timeout=2)
                if self._thread.is_alive():
                    print("[DEBUG] Le thread IB ne s'arrête pas, forçage de l'arrêt du process.")
                    sys.exit(0)
        except Exception as e:
            print(f"[DEBUG] Erreur lors de force_close: {e}")
            sys.exit(1)

    def scannerData(self, reqId, rank, contract, distance, benchmark, projection, legsStr):
        symbol = getattr(contract, "symbol", None)
        if not symbol or symbol == "N/A":
            symbol = getattr(contract, "localSymbol", None)
        if not symbol or symbol == "N/A":
            # Fallback : parser la chaîne contract si possible
            contract_str = str(contract)
            # Format attendu : id,symbol,STK,...
            parts = contract_str.split(",")
            if len(parts) > 1:
                symbol = parts[1]
            else:
                symbol = "N/A"
        self.scanner_results.append(symbol)

    def scannerDataEnd(self, reqId):
        print(f"[DEBUG] scannerDataEnd: reqId={reqId}, total results={len(self.scanner_results)}")
        self.scanner_done = True



