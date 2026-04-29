"""
Entraînement du modèle smooth_ml pour actions canadiennes (TSX/TSXV/NEO).
Utilise trading_data_ca.db et sauvegarde models/smooth_ml_ca.pkl.

Usage:
    python src/app/ml/train_smooth_ml_ca.py
    python src/app/ml/train_smooth_ml_ca.py --min-date 2016-01-01
"""
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.app.ml.ml_smooth_momentum_predictor import MLSmoothMomentumPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Entraîne smooth_ml sur données canadiennes"
    )
    parser.add_argument(
        '--min-date', default='2015-01-01',
        help='Date minimum pour les données d\'entraînement (default: 2015-01-01)'
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    db_path = os.path.join(project_root, 'trading_data_ca.db')
    model_path = os.path.join(project_root, 'models', 'smooth_ml_ca.pkl')

    predictor = MLSmoothMomentumPredictor(
        db_path=db_path,
        model_path=model_path,
    )

    metrics = predictor.run_full_training(min_date=args.min_date)

    print(f"\nROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
