"""
Entraînement du modèle smooth_ml pour actions canadiennes (TSX/TSXV/NEO).
Utilise trading_data_ca.db et sauvegarde models/smooth_ml_ca.pkl.

Usage:
    python src/app/ml/train_smooth_ml_ca.py
    python src/app/ml/train_smooth_ml_ca.py --min-date 2016-01-01
    python src/app/ml/train_smooth_ml_ca.py --date-min 2016-01-01 --date-max 2025-12-31
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
    parser.add_argument(
        '--date-min', default=None,
        help='Date de début (alias moderne). Prioritaire si fourni.'
    )
    parser.add_argument(
        '--date-max', default=None,
        help='Date de fin des données d\'entraînement (YYYY-MM-DD)'
    )
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    db_path = os.path.join(project_root, 'trading_data_ca.db')
    model_path = os.path.join(project_root, 'models', 'smooth_ml_ca.pkl')

    predictor = MLSmoothMomentumPredictor(
        db_path=db_path,
        model_path=model_path,
        market="ca",   # base stockca — sans ce paramètre, l'entraînement
                       # porterait silencieusement sur les données US
    )

    date_min = args.date_min or args.min_date
    metrics = predictor.run_full_training(
        date_min=date_min,
        date_max=args.date_max,
    )

    print(f"\nROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    recs = metrics.get('threshold_recommendations', {})
    for label in ('precision_50', 'precision_66'):
        rec = recs.get(label)
        if rec is None:
            print(f"{label}: aucun seuil disponible")
            continue
        print(
            f"{label}: seuil={int(rec['threshold']*100)} "
            f"precision={rec['precision']*100:.1f}% "
            f"EV/trade={rec['ev_per_trade']*100:+.2f}% "
            f"entrees/jour={rec['entries_per_day']:.1f}"
        )


if __name__ == "__main__":
    main()
