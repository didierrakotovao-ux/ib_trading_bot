"""
Classe abstraite pour les providers de données.
Permet de switcher facilement entre différentes sources (IB, yfinance, Alpha Vantage, etc.)
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


class DataProvider(ABC):
    """Interface abstraite pour tous les providers de données."""
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Établit la connexion avec la source de données.
        
        Returns:
            bool: True si connexion réussie, False sinon
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Ferme la connexion avec la source de données."""
        pass
    
    @abstractmethod
    def get_scanner_results(
        self, 
        scan_type: str,
        filters: Optional[Dict] = None,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Récupère les résultats d'un scanner de marché.
        
        Args:
            scan_type: Type de scan (ex: "HOT_BY_VOLUME", "TOP_PERC_GAIN")
            filters: Filtres optionnels (price_min, price_max, volume_min, market_cap_min, etc.)
            max_results: Nombre maximum de résultats
            
        Returns:
            List[Dict]: Liste de dictionnaires avec symbole, exchange, rank, etc.
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Récupère les données historiques pour un symbole.
        
        Args:
            symbol: Symbole de l'action (ex: "AAPL")
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle des données ("1d", "1h", "5m", etc.)
            
        Returns:
            DataFrame avec colonnes: date, open, high, low, close, volume
            None si erreur ou pas de données
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole.
        
        Args:
            symbol: Symbole de l'action
            
        Returns:
            float: Prix actuel ou None si erreur
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Vérifie si le provider est connecté.
        
        Returns:
            bool: True si connecté
        """
        return False
