"""
Script de débogage pour comprendre pourquoi les entrées diffèrent entre Trailing et Exhaustion.
Trace les décisions d'entrée jour par jour.
"""
import sys
sys.path.insert(0, '.')

import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Importer le scoring
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from src.app.ml.ml_scoring import MLScoring


def analyze_entries():
    """Analyse les scores et décisions d'entrée pour chaque jour."""

    # Charger les deux journaux
    trailing_file = None
    exhaustion_file = None

    # Trouver les fichiers les plus récents
    import glob
    trailing_files = sorted(glob.glob("trading_journal_trailing_*.csv"), reverse=True)
    exhaustion_files = sorted(glob.glob("trading_journal_exhaustion_*.csv"), reverse=True)

    if trailing_files:
        trailing_file = trailing_files[0]
    if exhaustion_files:
        exhaustion_file = exhaustion_files[0]

    if not trailing_file or not exhaustion_file:
        print("Fichiers journaux non trouvés!")
        return

    print(f"Trailing: {trailing_file}")
    print(f"Exhaustion: {exhaustion_file}")

    df_trailing = pd.read_csv(trailing_file)
    df_exhaustion = pd.read_csv(exhaustion_file)

    df_trailing['date_entree'] = pd.to_datetime(df_trailing['date_entree'])
    df_exhaustion['date_entree'] = pd.to_datetime(df_exhaustion['date_entree'])

    # Trouver les entrées uniques à chaque journal
    trailing_entries = set(df_trailing['symbole'] + '_' + df_trailing['date_entree'].dt.strftime('%Y-%m-%d'))
    exhaustion_entries = set(df_exhaustion['symbole'] + '_' + df_exhaustion['date_entree'].dt.strftime('%Y-%m-%d'))

    only_trailing = trailing_entries - exhaustion_entries
    only_exhaustion = exhaustion_entries - trailing_entries

    print("\n" + "=" * 70)
    print("ENTRÉES UNIQUES À TRAILING")
    print("=" * 70)
    for entry in sorted(only_trailing):
        symbol, date = entry.rsplit('_', 1)
        print(f"  {symbol} @ {date}")

    print("\n" + "=" * 70)
    print("ENTRÉES UNIQUES À EXHAUSTION")
    print("=" * 70)
    for entry in sorted(only_exhaustion):
        symbol, date = entry.rsplit('_', 1)
        print(f"  {symbol} @ {date}")

    # Analyser les positions ouvertes à chaque date clé
    print("\n" + "=" * 70)
    print("ANALYSE DES POSITIONS OUVERTES")
    print("=" * 70)

    # Reconstruire l'état des positions jour par jour
    def get_open_positions(df, target_date):
        """Retourne les positions ouvertes à une date donnée."""
        df_copy = df.copy()
        df_copy['date_sortie'] = pd.to_datetime(df_copy['date_sortie'])

        # Positions entrées avant ou à target_date et sorties après target_date
        open_pos = df_copy[
            (df_copy['date_entree'].dt.date <= target_date) &
            (df_copy['date_sortie'].dt.date > target_date)
        ]
        return list(open_pos['symbole'].values)

    # Dates clés à analyser (dates des entrées uniques)
    key_dates = set()
    for entry in only_trailing | only_exhaustion:
        _, date = entry.rsplit('_', 1)
        key_dates.add(datetime.strptime(date, '%Y-%m-%d').date())

    # Ajouter les jours précédents
    all_dates = set()
    for d in key_dates:
        all_dates.add(d)
        all_dates.add(d - timedelta(days=1))

    for date in sorted(all_dates):
        open_trailing = get_open_positions(df_trailing, date)
        open_exhaustion = get_open_positions(df_exhaustion, date)

        if open_trailing != open_exhaustion or date in key_dates:
            print(f"\n{date}:")
            print(f"  Trailing ({len(open_trailing)}): {open_trailing}")
            print(f"  Exhaustion ({len(open_exhaustion)}): {open_exhaustion}")

            # Vérifier si on atteint max_stocks (5)
            if len(open_trailing) >= 5:
                print(f"  -> Trailing: MAX STOCKS ATTEINT, pas de nouvelle entrée possible")
            if len(open_exhaustion) >= 5:
                print(f"  -> Exhaustion: MAX STOCKS ATTEINT, pas de nouvelle entrée possible")


def analyze_scores_for_date(target_date_str):
    """Analyse les scores de tous les symboles pour une date donnée."""

    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')

    # Charger les données
    conn = sqlite3.connect("trading_data.db")
    lookback_days = 350
    data_start = target_date - timedelta(days=lookback_days + 50)

    query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM historical_data
        WHERE date BETWEEN ? AND ?
        ORDER BY symbol, date
    """
    df = pd.read_sql_query(query, conn, params=(
        data_start.strftime('%Y-%m-%d'),
        target_date.strftime('%Y-%m-%d')
    ))
    conn.close()

    if df.empty:
        print("Pas de données")
        return

    # Scorer chaque symbole
    scoring = MLScoring(model_path="models/momentum_model.pkl")

    print(f"\n{'='*70}")
    print(f"SCORES AU {target_date_str} (seuil = 65)")
    print(f"{'='*70}")

    scores = []
    for symbol, group in df.groupby('symbol'):
        group = group.copy()
        group['date'] = pd.to_datetime(group['date'])
        group.set_index('date', inplace=True)

        score = scoring.score(group)
        if score >= 65:
            scores.append((symbol, score))

    scores.sort(key=lambda x: -x[1])

    print(f"\nSymboles éligibles (score >= 65):")
    for symbol, score in scores[:10]:  # Top 10
        print(f"  {symbol}: {score}")

    print(f"\nTotal éligibles: {len(scores)}")
    print(f"Top 5 (max_stocks): {[s[0] for s in scores[:5]]}")


if __name__ == "__main__":
    analyze_entries()

    # Analyser des dates spécifiques
    print("\n" + "=" * 70)
    print("ANALYSE DES SCORES POUR LES DATES CLÉS")
    print("=" * 70)

    # Dates où Exhaustion a des entrées uniques
    key_dates = ['2025-02-12', '2025-02-21', '2025-02-25', '2025-03-04', '2025-03-05', '2025-03-10', '2025-03-12']

    for date in key_dates:
        analyze_scores_for_date(date)
