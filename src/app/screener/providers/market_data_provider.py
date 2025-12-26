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
