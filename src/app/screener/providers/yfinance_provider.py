"""
Provider de données utilisant yfinance (gratuit, pas de compte requis).
"""
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

try:
    import yfinance as yf # type: ignore
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance n'est pas installé. Installez-le avec: pip install yfinance")


class YFinanceDataProvider():
    """Provider utilisant yfinance pour récupérer les données gratuitement."""
    
    def __init__(self):
        self._connected = False
        
    def connect(self) -> bool:
        """yfinance ne nécessite pas de connexion."""
        if not YFINANCE_AVAILABLE:
            print("❌ yfinance n'est pas disponible")
            return False
        self._connected = True
        print("✅ YFinance provider initialisé")
        return True
    
    def disconnect(self):
        """Rien à déconnecter avec yfinance."""
        self._connected = False
    
    def is_connected(self) -> bool:
        return self._connected and YFINANCE_AVAILABLE
    
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
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Récupère le dernier prix connu via yfinance."""
        if not self.is_connected():
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Essayer différents champs selon disponibilité
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            if price:
                return float(price)
            
            # Fallback: dernière clôture des données historiques
            df = self.get_historical_data(
                symbol,
                datetime.now() - timedelta(days=5),
                datetime.now()
            )
            if df is not None and not df.empty:
                return float(df['close'].iloc[-1])
            
            return None
            
        except Exception as e:
            print(f"❌ Erreur get_current_price pour {symbol}: {e}")
            return None
