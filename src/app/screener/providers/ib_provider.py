"""
Provider de données utilisant l'API Interactive Brokers.
Wrapper autour de votre code existant pour l'adapter à l'interface DataProvider.
"""
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
from .base_provider import DataProvider

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.scanner import ScannerSubscription
from collections import defaultdict
import threading
import time


class IBDataProvider(EWrapper, EClient, DataProvider):
    """Provider utilisant Interactive Brokers API."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 4001, client_id: int = 1):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        
        self.host = host
        self.port = port
        self.client_id = client_id
        
        self._connected = False
        self._next_req_id = 10000
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
        self._next_req_id = max(self._next_req_id, orderId)
        print(f"IB connecté, nextValidId = {orderId}")
    
    def get_scanner_results(
        self, 
        scan_type: str,
        filters: Optional[Dict] = None,
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
        
        # Configuration par défaut
        filters = filters or {}
        
        scan_sub = ScannerSubscription()
        scan_sub.instrument = "STK"
        scan_sub.locationCode = filters.get('location', 'STK.NASDAQ')
        scan_sub.scanCode = scan_type
        scan_sub.abovePrice = filters.get('price_min', 2.0)
        scan_sub.belowPrice = filters.get('price_max', 10000.0)
        scan_sub.aboveVolume = filters.get('volume_min', 500_000)
        
        if 'market_cap_min' in filters:
            scan_sub.marketCapAbove = filters['market_cap_min']
        
        # Reset
        self.scanner_results = []
        self.scanner_done = False
        
        req_id = self._next_req_id
        self._next_req_id += 1
        
        print(f"🔍 Lancement scanner IB: {scan_type} sur {scan_sub.locationCode}")
        
        try:
            self.reqScannerSubscription(req_id, scan_sub, [], [])
            
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
    
    def scannerData(self, reqId, rank, contractDetails, distance, benchmark, projection, legsStr):
        """Callback IB pour chaque résultat du scanner."""
        symbol = contractDetails.contract.symbol
        exchange = getattr(contractDetails.contract, "exchange", "SMART") or "SMART"
        
        self.scanner_results.append({
            "symbol": symbol,
            "exchange": exchange,
            "rank": rank,
            "distance": distance,
            "benchmark": benchmark,
            "projection": projection
        })
    
    def scannerDataEnd(self, reqId):
        """Callback IB quand le scanner est terminé."""
        self.scanner_done = True
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques via IB.
        
        Note: IB utilise une duration relative, pas des dates absolues.
        """
        if not self.is_connected():
            print("❌ Non connecté à IB")
            return None
        
        # Calculer la duration
        delta = end_date - start_date
        days = delta.days
        
        if days <= 30:
            duration = f"{days} D"
        elif days <= 365:
            duration = f"{days // 30} M"
        else:
            duration = f"{days // 365} Y"
        
        # Mapper l'intervalle
        bar_size_map = {
            "1d": "1 day",
            "1h": "1 hour",
            "5m": "5 mins",
            "1m": "1 min"
        }
        bar_size = bar_size_map.get(interval, "1 day")
        
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        
        req_id = self._next_req_id
        self._next_req_id += 1
        
        self.history_buf[req_id] = []
        self.history_done[req_id] = False
        self.req_map[req_id] = symbol
        
        try:
            self.reqHistoricalData(
                req_id, contract, "", duration, bar_size, "TRADES", 1, 1, False, []
            )
            
            # Attendre la réponse
            timeout = 30
            start = time.time()
            while not self.history_done.get(req_id) and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            bars = self.history_buf.get(req_id, [])
            
            if not bars:
                print(f"⚠️  Aucune donnée pour {symbol}")
                return None
            
            df = pd.DataFrame(bars)
            df['date'] = pd.to_datetime(df['date'])
            
            print(f"✅ {symbol}: {len(df)} barres récupérées")
            return df
            
        except Exception as e:
            print(f"❌ Erreur historique {symbol}: {e}")
            return None
        finally:
            # Cleanup
            self.history_buf.pop(req_id, None)
            self.history_done.pop(req_id, None)
            self.req_map.pop(req_id, None)
    
    def historicalData(self, reqId, bar):
        """Callback IB pour chaque barre historique."""
        self.history_buf[reqId].append({
            'date': bar.date,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })
    
    def historicalDataEnd(self, reqId, start, end):
        """Callback IB quand les données historiques sont complètes."""
        self.history_done[reqId] = True
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel via IB (snapshot).
        Note: Nécessite un abonnement aux données temps réel.
        """
        # Pour simplifier, on prend la dernière clôture des données historiques
        df = self.get_historical_data(
            symbol,
            datetime.now() - timedelta(days=2),
            datetime.now(),
            "1d"
        )
        
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
        
        return None
    
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        """Callback IB pour les erreurs."""
        # Filtrer les messages informatifs
        if errorCode in (2104, 2106, 2158):
            return
        
        print(f"[IB ERROR] reqId={reqId} code={errorCode} msg={errorString}")
