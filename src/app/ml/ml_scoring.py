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

    # Features Wyckoff effort/result (nouvelles - pas utilisées par le modèle ML existant)
    WYCKOFF_COLUMNS = [
        'effort_result_ratio',       # Efficacité du mouvement prix vs volume
        'volume_spread_analysis',    # Spread vs volume (VSA classique)
        'wyckoff_accumulation',      # Détection d'absorption (fort vol, range étroit)
        'effort_result_divergence',  # Divergence effort/résultat sur 5 jours
        'smart_money_flow',          # Flux cumulatif de pression acheteuse
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
        Calcule les features Wyckoff basées sur le principe effort vs résultat.

        Principe Wyckoff:
        - Effort = Volume (l'énergie mise dans le mouvement)
        - Result = Variation de prix (le résultat de cet effort)
        - Divergence effort/résultat = signal de retournement potentiel

        Args:
            df: DataFrame avec données OHLCV et indicateurs de base

        Returns:
            DataFrame enrichi avec features Wyckoff
        """
        df = df.copy()

        # --- 1. Effort/Result Ratio ---
        # Mesure l'efficacité: grand mouvement de prix avec peu de volume = efficace
        # Petit mouvement avec beaucoup de volume = épuisement
        spread = df['high'] - df['low']  # Range du jour (résultat)
        avg_spread = spread.rolling(20).mean()
        spread_normalized = spread / (avg_spread + 1e-6)

        # Effort/Result: si ratio élevé, le volume produit peu de mouvement (absorption)
        # Si ratio faible, peu de volume produit grand mouvement (mouvement efficace)
        df['effort_result_ratio'] = df['volume_ratio'] / (spread_normalized + 1e-6)

        # --- 2. Volume Spread Analysis (VSA) ---
        # Analyse classique Wyckoff/VSA: compare le spread au volume
        # - Wide spread + high volume = mouvement significatif (continuation)
        # - Narrow spread + high volume = absorption (accumulation/distribution)
        # - Wide spread + low volume = mouvement faible (potentiel piège)
        # Score VSA: spread_normalized * volume_ratio_direction
        price_direction = np.sign(df['close'] - df['open'])
        df['volume_spread_analysis'] = spread_normalized * df['volume_ratio'] * price_direction

        # --- 3. Wyckoff Accumulation Detection ---
        # Détecte les phases d'absorption: fort volume avec range étroit
        # Indique que l'offre est absorbée sans faire bouger le prix
        atr_14 = spread.rolling(14).mean()  # Proxy pour ATR
        is_narrow_range = spread < (atr_14 * 0.7)  # Range < 70% de la moyenne
        is_high_volume = df['volume_ratio'] > 1.2  # Volume > 120% de la moyenne

        # Score d'accumulation: 0 à 1
        # Plus le volume est élevé avec un range étroit, plus le score est élevé
        accumulation_score = np.where(
            is_narrow_range,
            np.minimum(df['volume_ratio'] / 2, 1.0),  # Cap à 1.0
            0.0
        )
        df['wyckoff_accumulation'] = accumulation_score

        # --- 4. Effort/Result Divergence (multi-jours) ---
        # Compare la direction du mouvement prix vs la tendance du volume sur 5 jours
        # Divergence = volume augmente mais prix stagne (ou vice versa)
        price_change_5d = df['close'].pct_change(5)
        volume_change_5d = df['volume'].pct_change(5)

        # Normaliser les changements
        price_dir = np.sign(price_change_5d)
        volume_dir = np.sign(volume_change_5d)

        # Divergence: -1 (bearish), 0 (neutre), +1 (bullish)
        # Volume monte + prix monte = confirmation bullish (+1)
        # Volume monte + prix baisse = absorption bearish (potentiel spring, +0.5)
        # Volume baisse + prix monte = mouvement faible (bearish, -0.5)
        # Volume baisse + prix baisse = fin de distribution (-1)
        df['effort_result_divergence'] = np.where(
            (volume_dir > 0) & (price_dir > 0), 1.0,      # Bullish confirmation
            np.where(
                (volume_dir > 0) & (price_dir < 0), 0.5,  # Potential spring/accumulation
                np.where(
                    (volume_dir < 0) & (price_dir > 0), -0.5,  # Weak rally
                    np.where(
                        (volume_dir < 0) & (price_dir < 0), -1.0,  # Distribution ending
                        0.0  # Neutre
                    )
                )
            )
        )

        # --- 5. Smart Money Flow ---
        # Indicateur cumulatif de pression acheteuse basé sur position du close dans le range
        # Inspiré du Chaikin Money Flow mais simplifié
        # Close près du high avec fort volume = smart money achète
        # Close près du low avec fort volume = smart money vend
        clv = np.where(
            spread > 0,
            ((df['close'] - df['low']) - (df['high'] - df['close'])) / spread,
            0.0
        )  # Close Location Value: -1 (close=low) à +1 (close=high)

        # Money Flow = CLV * Volume, puis moyenne sur 10 jours
        money_flow = clv * df['volume']
        df['smart_money_flow'] = money_flow.rolling(10).sum() / (df['volume'].rolling(10).sum() + 1e-6)

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
        Analyse Wyckoff effort vs résultat.

        Retourne une analyse détaillée des signaux Wyckoff:
        - Effort/Result ratio et interprétation
        - Phase détectée (accumulation, distribution, markup, markdown)
        - Qualité du mouvement actuel
        - Signal composite Wyckoff

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
            last_5 = df_clean.tail(5)

            # Extraire les métriques Wyckoff
            effort_result = float(last.get('effort_result_ratio', 1.0))
            vsa = float(last.get('volume_spread_analysis', 0))
            accumulation = float(last.get('wyckoff_accumulation', 0))
            divergence = float(last.get('effort_result_divergence', 0))
            smart_money = float(last.get('smart_money_flow', 0))

            # --- Interprétation ---

            # 1. Effort/Result Interpretation
            if effort_result > 2.0:
                effort_interpretation = "ABSORPTION"  # Fort volume, peu de mouvement
                effort_quality = "Accumulation/Distribution probable"
            elif effort_result < 0.5:
                effort_interpretation = "EFFICIENT"  # Mouvement efficace
                effort_quality = "Mouvement fort avec peu de résistance"
            else:
                effort_interpretation = "NORMAL"
                effort_quality = "Mouvement proportionnel au volume"

            # 2. VSA Interpretation
            if vsa > 1.5:
                vsa_interpretation = "BULLISH_EXPANSION"
                vsa_quality = "Fort mouvement haussier confirmé par volume"
            elif vsa < -1.5:
                vsa_interpretation = "BEARISH_EXPANSION"
                vsa_quality = "Fort mouvement baissier confirmé par volume"
            elif abs(vsa) < 0.3:
                vsa_interpretation = "CONSOLIDATION"
                vsa_quality = "Mouvement faible, consolidation"
            else:
                vsa_interpretation = "MODERATE"
                vsa_quality = "Mouvement modéré"

            # 3. Phase Detection (basée sur moyenne 5 jours)
            avg_accumulation = float(last_5['wyckoff_accumulation'].mean())
            avg_divergence = float(last_5['effort_result_divergence'].mean())
            avg_smart_money = float(last_5['smart_money_flow'].mean())

            if avg_accumulation > 0.3 and avg_smart_money > 0.2:
                phase = "ACCUMULATION"
                phase_description = "Phase d'accumulation: smart money achète discrètement"
            elif avg_accumulation > 0.3 and avg_smart_money < -0.2:
                phase = "DISTRIBUTION"
                phase_description = "Phase de distribution: smart money vend discrètement"
            elif avg_divergence > 0.5 and avg_smart_money > 0:
                phase = "MARKUP"
                phase_description = "Phase de hausse: tendance haussière confirmée"
            elif avg_divergence < -0.5 and avg_smart_money < 0:
                phase = "MARKDOWN"
                phase_description = "Phase de baisse: tendance baissière confirmée"
            else:
                phase = "RANGING"
                phase_description = "Phase de range: pas de direction claire"

            # 4. Signal Composite Wyckoff (0-100)
            # Pondération des facteurs pour un score bullish
            wyckoff_score = 50  # Neutre par défaut

            # Smart money flow (+/- 20 points)
            wyckoff_score += smart_money * 20

            # Divergence effort/résultat (+/- 15 points)
            wyckoff_score += divergence * 15

            # VSA (+/- 10 points)
            if vsa > 0:
                wyckoff_score += min(vsa * 5, 10)
            else:
                wyckoff_score += max(vsa * 5, -10)

            # Bonus accumulation avec smart money positif (+10 points)
            if accumulation > 0.5 and smart_money > 0:
                wyckoff_score += 10

            # Malus absorption avec smart money négatif (-10 points)
            if accumulation > 0.5 and smart_money < 0:
                wyckoff_score -= 10

            wyckoff_score = max(0, min(100, int(wyckoff_score)))

            # Signal final
            if wyckoff_score >= 65:
                wyckoff_signal = "BUY"
            elif wyckoff_score >= 50:
                wyckoff_signal = "HOLD"
            elif wyckoff_score >= 35:
                wyckoff_signal = "CAUTION"
            else:
                wyckoff_signal = "AVOID"

            return {
                'wyckoff_score': wyckoff_score,
                'wyckoff_signal': wyckoff_signal,
                'phase': phase,
                'phase_description': phase_description,
                'metrics': {
                    'effort_result_ratio': round(effort_result, 2),
                    'effort_interpretation': effort_interpretation,
                    'effort_quality': effort_quality,
                    'volume_spread_analysis': round(vsa, 2),
                    'vsa_interpretation': vsa_interpretation,
                    'vsa_quality': vsa_quality,
                    'wyckoff_accumulation': round(accumulation, 2),
                    'effort_result_divergence': round(divergence, 2),
                    'smart_money_flow': round(smart_money, 3),
                },
                'averages_5d': {
                    'accumulation': round(avg_accumulation, 2),
                    'divergence': round(avg_divergence, 2),
                    'smart_money': round(avg_smart_money, 3),
                }
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
    print("TEST MLScoring + Analyse Wyckoff Effort/Result")
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
            print(f"  Phase Wyckoff: {combined['wyckoff_analysis']['phase']}")
            print(f"  Score Combiné: {combined['combined_score']} -> {combined['combined_signal']}")
            print(f"  Confiance: {combined['confidence']} ({combined['confidence_note']})")

            # Détails Wyckoff
            if 'wyckoff_details' in combined and 'metrics' in combined['wyckoff_details']:
                metrics = combined['wyckoff_details']['metrics']
                print(f"  Métriques Wyckoff:")
                print(f"    - Effort/Result: {metrics['effort_result_ratio']} ({metrics['effort_interpretation']})")
                print(f"    - VSA: {metrics['volume_spread_analysis']} ({metrics['vsa_interpretation']})")
                print(f"    - Smart Money Flow: {metrics['smart_money_flow']}")
        else:
            print(f"{symbol}: Pas de données")

    conn.close()
