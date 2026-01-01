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
            
            EClient.connect(self, self.host, self.port, self._next_req_id)
            self._next_req_id += 1
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
        print("🔌 Déconnecté d'IB")
    
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
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False  # Garder les prix non ajustés
            )
            
            if df.empty:
                print(f"⚠️  Aucune donnée pour {symbol}")
                return None
            
            info = ticker.info
            name = (info.get('longName') or info.get('shortName') or '').lower()
            summary = (info.get('summaryDetail') or '').lower() if isinstance(info.get('summaryDetail'), str) else ''
                # Exclure si mot-clé trouvé dans le nom ou le résumé
            if any(kw in name for kw in leverage_keywords) or any(kw in summary for kw in leverage_keywords):
                print(f"[FILTRAGE ETF LEVERAGE] {symbol} exclu: {name}")
                return None
            
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

    def placeOrder(self, contract:Contract, order:Order):
        """Override pour loguer les ordres placés."""
        print(f"[IB ORDER] Placing order for {contract.symbol} {order.action} {order.totalQuantity} @ {order.orderType}")
        super().placeOrder(order.orderId if hasattr(order, 'orderId') else self._next_req_id, contract, order)
        self._next_req_id += 1

    def create_contract(self, ticker: str, sec_type: str = 'STK', exchange: str = 'SMART', currency: str = 'USD') -> Contract:
        """Crée un contrat IB pour un ETF"""
        contract = Contract()
        contract.symbol = ticker
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract
    
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



