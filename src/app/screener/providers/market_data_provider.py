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

try:
    import yfinance as yf # type: ignore
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance n'est pas installé. Installez-le avec: pip install yfinance")


class MarketDataProvider(EWrapper, EClient):
    """Provider utilisant Interactive Brokers API pour récupérer les stocks et yfinance pour les historiques."""
    scan_sub: Optional[ScannerSubscription] = None
    
    def __init__(self, host: str = "127.0.0.1", port: int = 4001, client_id: int = 1):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self._connected = False
        self._next_req_id = 1
        self._thread = None
        
        # Pour le scanner
        self.scanner_results = []
        self.scanner_done = False
        
        # Pour les données historiques
        self.history_buf = defaultdict(list)
        self.req_map = {}
        self.history_done = {}
    
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
                print(f"✅ Connecté à IB ({self.host}:{self.port})")
                return True
            else:
                print(f"❌ Timeout de connexion à IB")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de connexion IB: {e}")
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
        print(f"IB connecté, nextValidId = {orderId}")
    
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
            print("❌ Non connecté à IB")
            return []
        
        self.scan_sub = scan_sub
        # Reset
        self.scanner_results = []
        self.scanner_done = False
        
        req_id = self._next_req_id
        self._next_req_id += 1
        
        print(f"🔍 Lancement scanner IB: {self.scan_sub.scanCode } sur {self.scan_sub.locationCode}")
        
        try:
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
            print(f"✅ Scanner terminé: {len(results)} résultats")
            return results
            
        except Exception as e:
            print(f"❌ Erreur scanner: {e}")
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
        if not self.is_connected():
            print("❌ Provider non connecté")
            return None
        
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
            
            print(f"✅ {symbol}: {len(df)} barres récupérées")
            return df
            
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de {symbol}: {e}")
            return None
    
    def error(self, reqId, errorCode, errorString):
        """Callback IB pour les erreurs."""
        # Filtrer les messages informatifs
        if errorCode in (2104, 2106, 2158):
            return
        
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
    
    def create_trailing_stop_order(self, quantity, trailing_percent, parent_order_id=0):
        """Créer un ordre trailing stop"""
        order = Order()
        order.action = "SELL"  # Trailing stop est toujours une vente
        order.orderType = "TRAIL"
        order.totalQuantity = quantity
        order.trailingPercent = trailing_percent  # Pourcentage de trailing
        order.transmit = True
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        
        # Lier au parent order si spécifié
        if parent_order_id > 0:
            order.parentId = parent_order_id
            order.transmit = False
        
        return order
    
    def create_limit_order(self, action: str, quantity: int, limit_price: float, parent_order_id=0):
        """Créer un ordre limite"""
        order = Order()
        order.action = action  # "BUY" ou "SELL"
        order.orderType = "LMT"
        order.totalQuantity = quantity
        order.lmtPrice = limit_price    
        order.transmit = True
        order.eTradeOnly = False            



