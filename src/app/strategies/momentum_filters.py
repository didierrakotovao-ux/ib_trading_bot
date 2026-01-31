"""
Filtres de pré-sélection basés sur "Quantitative Momentum" (Gray & Vogel).

Pipeline:
1. Momentum 12-1: Rendement sur 12 mois excluant le dernier mois
2. Frog-in-the-Pan (FIP): Qualité du momentum (smooth vs erratique)
3. Saisonnalité: Seuil de score dynamique selon le mois
"""
import pandas as pd
import numpy as np


class MomentumFilters:
    """Filtres de pré-sélection pour la stratégie momentum."""

    @staticmethod
    def calc_momentum_12_1(df: pd.DataFrame) -> float:
        """
        Calcule le rendement momentum 12-1 mois.
        Rendement des 12 derniers mois en excluant le dernier mois.

        Pourquoi exclure le dernier mois: effet de reversal à court terme.
        Les actions qui ont monté le dernier mois ont tendance à corriger.

        Args:
            df: DataFrame avec colonne 'close', trié par date croissante

        Returns:
            Rendement 12-1 en décimal (ex: 0.25 = +25%), ou NaN si pas assez de données
        """
        if len(df) < 252:
            return np.nan

        # close[-252] = prix il y a 12 mois
        # close[-21] = prix il y a 1 mois
        close = df['close'].values
        price_12m_ago = close[-252]
        price_1m_ago = close[-21]

        if price_12m_ago <= 0:
            return np.nan

        return (price_1m_ago - price_12m_ago) / price_12m_ago

    @staticmethod
    def calc_fip(df: pd.DataFrame, flat_threshold: float = 0.005) -> float:
        """
        Calcule le score Frog-in-the-Pan (FIP).

        FIP = %flatdays * (%negdays - %posdays)

        - FIP négatif = bon (momentum smooth, plus de jours positifs)
        - FIP positif = mauvais (momentum erratique, gros jumps)

        Un FIP très négatif indique un stock qui monte progressivement,
        jour après jour, sans à-coups (grenouille dans la casserole).

        Args:
            df: DataFrame avec colonne 'close', trié par date croissante
            flat_threshold: Seuil de variation pour considérer un jour comme plat (0.5%)

        Returns:
            Score FIP (négatif = bon), ou NaN si pas assez de données
        """
        if len(df) < 252:
            return np.nan

        # Prendre les 252 derniers jours
        close = df['close'].values[-252:]

        # Rendements journaliers
        daily_returns = np.diff(close) / close[:-1]
        n_days = len(daily_returns)

        # Classifier les jours
        pos_days = np.sum(daily_returns > flat_threshold)
        neg_days = np.sum(daily_returns < -flat_threshold)
        flat_days = n_days - pos_days - neg_days

        # Pourcentages
        pct_pos = pos_days / n_days
        pct_neg = neg_days / n_days
        pct_flat = flat_days / n_days

        # FIP = %flat * (%neg - %pos)
        fip = pct_flat * (pct_neg - pct_pos)

        return fip

    @staticmethod
    def get_score_threshold(month: int) -> int:
        """
        Retourne le seuil de score ML dynamique selon le mois.

        Basé sur la saisonnalité du momentum:
        - Mois forts: seuil plus bas (plus de trades)
        - Mois faibles: seuil plus haut (plus sélectif)

        Args:
            month: Numéro du mois (1-12)

        Returns:
            Seuil de score (0-100)
        """
        # Mois faibles pour le momentum (janvier, avril)
        if month in (1, 4):
            return 80

        # Mois forts (novembre, décembre, février, mars)
        if month in (11, 12, 2, 3):
            return 55

        # Mois neutres
        return 65

    @staticmethod
    def filter_top_momentum(candidates: list, top_pct: float = 0.3) -> list:
        """
        Garde le top N% des candidats par momentum 12-1.

        Args:
            candidates: Liste de tuples (symbol, momentum_12_1, data)
            top_pct: Pourcentage à garder (0.3 = top 30%)

        Returns:
            Liste filtrée des meilleurs candidats
        """
        # Exclure les NaN
        valid = [(s, m, d) for s, m, d in candidates if not np.isnan(m)]

        if not valid:
            return []

        # Trier par momentum décroissant
        valid.sort(key=lambda x: x[1], reverse=True)

        # Garder le top N%
        n_keep = max(1, int(len(valid) * top_pct))
        return valid[:n_keep]
