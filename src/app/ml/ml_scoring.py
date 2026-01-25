"""
ML Scoring - Scoring basé sur le modèle ML de prédiction momentum.
Prédit la probabilité qu'un stock augmente de >5% dans les 20 prochains jours.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Optional
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.scoring import Scoring


class MLScoring(Scoring):
    """
    Scoring utilisant le modèle ML entraîné pour prédire les hausses >5%.
    """
    name = "MLScoring"
    df: pd.DataFrame

    # Features requises par le modèle (originales)
    FEATURE_COLUMNS = [
        'hl_sma20vol', 'oc_sma20vol', 'macd', 'macd_signal', 'rsi', 'adx',
        'bb_high', 'bb_low', 'pct_close', 'macd_hist', 'bb_position',
        'rsi_momentum', 'volume_ratio', 'trend_strength', 'return_5d',
        'return_10d', 'return_20d', 'volatility_10d', 'high_52w_pct',
    ]

    # Features Wyckoff simplifiées - basées sur série temporelle de volume
    WYCKOFF_COLUMNS = [
        'volume_above_sma20',        # Boolean: volume > SMA20
        'volume_streak',             # Nombre de jours consécutifs avec volume > SMA20
        'volume_consistency_10d',    # Coefficient de variation du volume sur 10 jours
        'sustained_accumulation',    # Score d'accumulation soutenue (0-1)
    ]

    def __init__(self, model_path: str = "models/momentum_model.pkl"):
        """
        Initialise le scoring ML.

        Args:
            model_path: Chemin vers le modèle sauvegardé
        """
        self.df = None
        self.model = None
        self.scaler = None
        self.model_loaded = False
        self.model_path = Path(model_path)

        # Essayer de charger le modèle
        self._load_model()

    def _load_model(self):
        """Charge le modèle ML."""
        try:
            if self.model_path.exists():
                data = joblib.load(self.model_path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.model_loaded = True
                print(f"[OK] Modèle ML chargé: {self.model_path}")
            else:
                print(f"[WARN] Modèle ML non trouvé: {self.model_path}")
                print("       Entraînez-le avec: python src/app/ml/ml_momentum_predictor.py")
        except Exception as e:
            print(f"[ERROR] Erreur chargement modèle: {e}")

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les features nécessaires pour le modèle.

        Args:
            df: DataFrame avec données OHLCV

        Returns:
            DataFrame enrichi avec les features
        """
        df = df.copy()

        # Normaliser les noms de colonnes
        df.columns = df.columns.str.lower()

        # --- Features de base (si pas déjà présentes) ---

        # SMA20 du volume
        if 'sma20_volume' not in df.columns:
            df['sma20_volume'] = df['volume'].rolling(window=20).mean()

        # hl_sma20vol et oc_sma20vol
        if 'hl_sma20vol' not in df.columns:
            df['hl_sma20vol'] = (df['high'] - df['low']) / (df['sma20_volume'] + 1e-6)

        if 'oc_sma20vol' not in df.columns:
            df['oc_sma20vol'] = (df['open'] - df['close']) / (df['sma20_volume'] + 1e-6)

        # MACD
        if 'macd' not in df.columns:
            try:
                import pandas_ta as ta
                macd_df = ta.macd(df['close'], fast=12, slow=26)
                df['macd'] = macd_df['MACD_12_26_9']
                df['macd_signal'] = macd_df['MACDs_12_26_9']
            except:
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                df['macd'] = ema12 - ema26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # RSI
        if 'rsi' not in df.columns:
            try:
                import pandas_ta as ta
                df['rsi'] = ta.rsi(df['close'], length=14)
            except:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-6)
                df['rsi'] = 100 - (100 / (1 + rs))

        # ADX
        if 'adx' not in df.columns:
            try:
                import pandas_ta as ta
                adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                df['adx'] = adx_df['ADX_14']
            except:
                df['adx'] = 25  # Valeur par défaut

        # Bollinger Bands
        if 'bb_high' not in df.columns:
            sma20 = df['close'].rolling(20).mean()
            std20 = df['close'].rolling(20).std()
            df['bb_high'] = sma20 + 2 * std20
            df['bb_low'] = sma20 - 2 * std20

        # pct_close
        if 'pct_close' not in df.columns:
            df['pct_close'] = df['close'].pct_change()

        # --- Features calculées ---

        # MACD Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Position dans les Bandes de Bollinger
        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(
            bb_range > 0,
            (df['close'] - df['bb_low']) / bb_range,
            0.5
        )

        # RSI centré
        df['rsi_momentum'] = df['rsi'] - 50

        # Volume ratio
        df['volume_ratio'] = df['volume'] / (df['sma20_volume'] + 1e-6)

        # Trend strength
        df['trend_strength'] = df['adx'] * np.sign(df['macd'])

        # Rendements passés
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # Volatilité
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()

        # Position vs plus haut 52 semaines
        high_52w = df['high'].rolling(252, min_periods=50).max()
        df['high_52w_pct'] = df['close'] / (high_52w + 1e-6)

        # --- Features Wyckoff Effort/Result ---
        df = self._create_wyckoff_features(df)

        return df

    def _create_wyckoff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les features Wyckoff simplifiées basées sur série temporelle de volume.

        Principe: volume constant au-dessus de SMA20 pendant N jours = accumulation institutionnelle

        Args:
            df: DataFrame avec données OHLCV et indicateurs de base

        Returns:
            DataFrame enrichi avec features Wyckoff
        """
        df = df.copy()

        # --- 1. Volume au-dessus de SMA20 (booléen) ---
        df['volume_above_sma20'] = (df['volume'] > df['sma20_volume']).astype(int)

        # --- 2. Streak de jours consécutifs avec volume > SMA20 ---
        # Calcule le nombre de jours consécutifs où le volume est au-dessus de SMA20
        def calculate_streak(series):
            """Calcule la série de streaks consécutifs."""
            streak = pd.Series(0, index=series.index)
            current_streak = 0
            for i in range(len(series)):
                if series.iloc[i] == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                streak.iloc[i] = current_streak
            return streak

        df['volume_streak'] = calculate_streak(df['volume_above_sma20'])

        # --- 3. Consistance du volume sur 10 jours ---
        # Coefficient de variation (CV) = std / mean
        # CV faible = volume constant (institutionnel)
        # CV élevé = volume erratique (retail, spéculatif)
        volume_mean_10d = df['volume'].rolling(10).mean()
        volume_std_10d = df['volume'].rolling(10).std()
        df['volume_consistency_10d'] = volume_std_10d / (volume_mean_10d + 1e-6)

        # --- 4. Score d'accumulation soutenue ---
        # Combinaison: streak élevé + volume constant = forte probabilité d'accumulation
        # Score de 0 à 1
        #   - streak >= 10 jours : contribution max
        #   - CV < 0.3 (30%) : volume très constant
        streak_score = np.minimum(df['volume_streak'] / 10.0, 1.0)  # Max à 10 jours
        consistency_score = np.maximum(1.0 - df['volume_consistency_10d'], 0.0)  # Plus CV est bas, plus le score est haut
        consistency_score = np.minimum(consistency_score / 0.7, 1.0)  # Normaliser

        df['sustained_accumulation'] = streak_score * consistency_score

        return df

    def score(self, df: pd.DataFrame) -> int:
        """
        Score un DataFrame avec le modèle ML.

        Args:
            df: DataFrame avec données OHLCV (minimum 60 jours)

        Returns:
            Score de 0 à 100 (probabilité de hausse >5%)
        """
        if not self.model_loaded:
            print("[WARN] Modèle ML non chargé, retourne 0")
            return 0

        try:
            # Créer les features
            df = self._create_features(df)
            self.df = df

            # Vérifier qu'on a assez de données
            if len(df) < 60:
                print(f"[WARN] Pas assez de données ({len(df)} < 60)")
                return 0

            # Prendre la dernière ligne avec toutes les features
            df_clean = df.dropna(subset=self.FEATURE_COLUMNS)

            if df_clean.empty:
                print("[WARN] Pas de données valides après nettoyage")
                return 0

            # Extraire les features de la dernière ligne
            X = df_clean[self.FEATURE_COLUMNS].iloc[-1:].values

            # Normaliser
            X_scaled = self.scaler.transform(X)

            # Prédire la probabilité
            probability = self.model.predict_proba(X_scaled)[0, 1]

            # Convertir en score 0-100
            score = int(probability * 100)

            return max(0, min(score, 100))

        except Exception as e:
            print(f"[ERREUR MLScoring] {e}")
            return 0

    def get_prediction_details(self, df: pd.DataFrame) -> dict:
        """
        Retourne les détails de la prédiction.

        Args:
            df: DataFrame avec données OHLCV

        Returns:
            Dictionnaire avec probabilité, signal, et features clés
        """
        score = self.score(df)
        probability = score / 100.0

        # Signal basé sur la probabilité
        if probability >= 0.6:
            signal = "BUY"
        elif probability >= 0.4:
            signal = "HOLD"
        else:
            signal = "AVOID"

        # Features clés
        details = {
            'score': score,
            'probability': probability,
            'signal': signal,
        }

        if self.df is not None and not self.df.empty:
            last = self.df.iloc[-1]
            details['features'] = {
                'volatility_10d': float(last.get('volatility_10d', 0)),
                'high_52w_pct': float(last.get('high_52w_pct', 0)),
                'rsi': float(last.get('rsi', 0)),
                'adx': float(last.get('adx', 0)),
                'macd_hist': float(last.get('macd_hist', 0)),
            }

        return details

    def get_wyckoff_analysis(self, df: pd.DataFrame) -> dict:
        """
        Analyse Wyckoff simplifiée basée sur série temporelle de volume.

        Principe: volume constant au-dessus de SMA20 pendant N jours = accumulation institutionnelle

        Args:
            df: DataFrame avec données OHLCV (minimum 60 jours)

        Returns:
            Dictionnaire avec analyse Wyckoff complète
        """
        try:
            # Créer les features si pas déjà fait
            df = self._create_features(df)

            if len(df) < 60:
                return {'error': 'Pas assez de données', 'wyckoff_score': 0}

            df_clean = df.dropna(subset=self.WYCKOFF_COLUMNS)
            if df_clean.empty:
                return {'error': 'Données Wyckoff invalides', 'wyckoff_score': 0}

            last = df_clean.iloc[-1]

            # Extraire les métriques Wyckoff simplifiées
            volume_above = int(last.get('volume_above_sma20', 0))
            streak = int(last.get('volume_streak', 0))
            consistency = float(last.get('volume_consistency_10d', 1.0))
            sustained_acc = float(last.get('sustained_accumulation', 0))

            # --- Interprétation du streak ---
            if streak >= 10:
                streak_interpretation = "STRONG_ACCUMULATION"
                streak_quality = "10+ jours de volume soutenu - accumulation institutionnelle probable"
            elif streak >= 7:
                streak_interpretation = "MODERATE_ACCUMULATION"
                streak_quality = "7-9 jours de volume soutenu - intérêt institutionnel"
            elif streak >= 5:
                streak_interpretation = "BUILDING"
                streak_quality = "5-6 jours de volume soutenu - début d'accumulation"
            elif streak >= 3:
                streak_interpretation = "EARLY_SIGNAL"
                streak_quality = "3-4 jours de volume soutenu - signal précoce"
            else:
                streak_interpretation = "NO_SIGNAL"
                streak_quality = "Pas de streak significatif"

            # --- Interprétation de la consistance ---
            # CV < 0.3 = très constant, CV > 0.5 = erratique
            if consistency < 0.25:
                consistency_interpretation = "VERY_CONSISTENT"
                consistency_quality = "Volume très stable - comportement institutionnel"
            elif consistency < 0.35:
                consistency_interpretation = "CONSISTENT"
                consistency_quality = "Volume stable - accumulation ordonnée"
            elif consistency < 0.50:
                consistency_interpretation = "MODERATE"
                consistency_quality = "Volume modérément variable"
            else:
                consistency_interpretation = "ERRATIC"
                consistency_quality = "Volume erratique - spéculation retail probable"

            # --- Phase Detection ---
            # Basée sur streak + consistance + tendance prix
            price_trend_20d = float(df_clean['close'].iloc[-1] / df_clean['close'].iloc[-20] - 1) if len(df_clean) >= 20 else 0

            if streak >= 7 and consistency < 0.4 and price_trend_20d > 0.02:
                phase = "MARKUP"
                phase_description = "Hausse confirmée avec volume institutionnel soutenu"
            elif streak >= 7 and consistency < 0.4 and price_trend_20d < -0.02:
                phase = "MARKDOWN"
                phase_description = "Baisse avec volume soutenu - distribution ou capitulation"
            elif streak >= 5 and consistency < 0.4 and abs(price_trend_20d) <= 0.02:
                phase = "ACCUMULATION"
                phase_description = "Volume soutenu en range - accumulation probable"
            elif streak < 3 and consistency > 0.5:
                phase = "DISTRIBUTION"
                phase_description = "Volume erratique - possible distribution"
            else:
                phase = "RANGING"
                phase_description = "Pas de signal clair"

            # --- Score Wyckoff Simplifié (0-100) ---
            wyckoff_score = 50  # Base neutre

            # Contribution du streak (max +30 points)
            streak_contribution = min(streak * 3, 30)
            wyckoff_score += streak_contribution

            # Contribution de la consistance (max +20 points)
            # Plus CV est bas, plus on ajoute de points
            if consistency < 0.5:
                consistency_contribution = (0.5 - consistency) * 40  # 0 à 20 points
                wyckoff_score += consistency_contribution
            else:
                wyckoff_score -= 10  # Pénalité pour volume erratique

            # Bonus si aujourd'hui volume > SMA20
            if volume_above:
                wyckoff_score += 5

            # Ajustement basé sur tendance prix
            if price_trend_20d > 0.05:
                wyckoff_score += 5  # Tendance haussière
            elif price_trend_20d < -0.05:
                wyckoff_score -= 10  # Tendance baissière

            wyckoff_score = max(0, min(100, int(wyckoff_score)))

            # --- Signal final ---
            if wyckoff_score >= 70:
                wyckoff_signal = "BUY"
            elif wyckoff_score >= 55:
                wyckoff_signal = "HOLD"
            elif wyckoff_score >= 40:
                wyckoff_signal = "CAUTION"
            else:
                wyckoff_signal = "AVOID"

            return {
                'wyckoff_score': wyckoff_score,
                'wyckoff_signal': wyckoff_signal,
                'phase': phase,
                'phase_description': phase_description,
                'metrics': {
                    'volume_streak': streak,
                    'streak_interpretation': streak_interpretation,
                    'streak_quality': streak_quality,
                    'volume_consistency_10d': round(consistency, 3),
                    'consistency_interpretation': consistency_interpretation,
                    'consistency_quality': consistency_quality,
                    'volume_above_sma20_today': bool(volume_above),
                    'sustained_accumulation_score': round(sustained_acc, 2),
                    'price_trend_20d': round(price_trend_20d * 100, 2),  # En pourcentage
                },
            }

        except Exception as e:
            return {'error': str(e), 'wyckoff_score': 0}

    def get_combined_analysis(self, df: pd.DataFrame) -> dict:
        """
        Combine l'analyse ML et l'analyse Wyckoff pour une vue complète.

        Args:
            df: DataFrame avec données OHLCV

        Returns:
            Dictionnaire avec scores ML et Wyckoff combinés
        """
        ml_details = self.get_prediction_details(df)
        wyckoff_analysis = self.get_wyckoff_analysis(df)

        ml_score = ml_details.get('score', 0)
        wyckoff_score = wyckoff_analysis.get('wyckoff_score', 50)

        # Score combiné: 60% ML, 40% Wyckoff
        combined_score = int(ml_score * 0.6 + wyckoff_score * 0.4)

        # Signal combiné
        if combined_score >= 65 and wyckoff_analysis.get('wyckoff_signal') != 'AVOID':
            combined_signal = "STRONG_BUY"
        elif combined_score >= 60:
            combined_signal = "BUY"
        elif combined_score >= 45:
            combined_signal = "HOLD"
        elif combined_score >= 35:
            combined_signal = "CAUTION"
        else:
            combined_signal = "AVOID"

        # Confiance: accord entre ML et Wyckoff
        ml_bullish = ml_score >= 60
        wyckoff_bullish = wyckoff_score >= 55
        if ml_bullish == wyckoff_bullish:
            confidence = "HIGH"
            confidence_note = "ML et Wyckoff en accord"
        else:
            confidence = "MEDIUM"
            confidence_note = "Divergence ML/Wyckoff - prudence recommandée"

        return {
            'combined_score': combined_score,
            'combined_signal': combined_signal,
            'confidence': confidence,
            'confidence_note': confidence_note,
            'ml_analysis': {
                'score': ml_score,
                'signal': ml_details.get('signal', 'N/A'),
            },
            'wyckoff_analysis': {
                'score': wyckoff_score,
                'signal': wyckoff_analysis.get('wyckoff_signal', 'N/A'),
                'phase': wyckoff_analysis.get('phase', 'N/A'),
            },
            'wyckoff_details': wyckoff_analysis,
        }


# Test standalone
if __name__ == "__main__":
    import sqlite3

    # Charger des données de test
    db_path = "trading_data.db"
    conn = sqlite3.connect(db_path)

    symbols_to_test = ['AAPL', 'NVDA', 'GOOGL', 'QCOM', 'TXN', 'INTU', 'TFC', 'USB', 'PNC',
                       'ABBV', 'MRK', 'PFE', 'ORLY', 'AZO', 'YUM', 'RTX', 'GE', 'LMT',
                       'EOG', 'MPC', 'PSX', 'EXC', 'SRE', 'PLD', 'NEM', 'FCX', 'NUE']

    scoring = MLScoring(model_path="models/momentum_model.pkl")

    print("\n" + "=" * 70)
    print("TEST MLScoring + Analyse Wyckoff Volume Streak")
    print("=" * 70)

    for symbol in symbols_to_test:
        query = f"""
            SELECT date, open, high, low, close, volume,
                   sma20_volume, hl_sma20vol, oc_sma20vol,
                   macd, macd_signal, rsi, adx, bb_high, bb_low, pct_close
            FROM historical_data
            WHERE symbol = '{symbol}'
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)

        if not df.empty:
            # Analyse combinée ML + Wyckoff
            combined = scoring.get_combined_analysis(df)

            print(f"\n{symbol}:")
            print(f"  ML Score: {combined['ml_analysis']['score']} -> {combined['ml_analysis']['signal']}")
            print(f"  Wyckoff Score: {combined['wyckoff_analysis']['score']} -> {combined['wyckoff_analysis']['signal']}")
            print(f"  Phase: {combined['wyckoff_analysis']['phase']}")
            print(f"  Combined: {combined['combined_score']} -> {combined['combined_signal']}")
            print(f"  Confidence: {combined['confidence']}")

            # Détails Wyckoff Volume Streak
            if 'wyckoff_details' in combined and 'metrics' in combined['wyckoff_details']:
                m = combined['wyckoff_details']['metrics']
                print(f"  Volume Streak: {m.get('volume_streak', 0)} jours ({m.get('streak_interpretation', 'N/A')})")
                print(f"  Consistency: {m.get('volume_consistency_10d', 0):.3f} ({m.get('consistency_interpretation', 'N/A')})")
                print(f"  Price Trend 20d: {m.get('price_trend_20d', 0):.2f}%")
        else:
            print(f"{symbol}: Pas de données")

    conn.close()
