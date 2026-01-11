"""
Script de comparaison de deux journaux de trading.
Identifie les différences d'entrées, sorties et performances.
"""
import pandas as pd
import sys
from pathlib import Path


def load_journal(filepath):
    """Charge un journal CSV."""
    df = pd.read_csv(filepath)
    df['date_entree'] = pd.to_datetime(df['date_entree'])
    df['date_sortie'] = pd.to_datetime(df['date_sortie'])
    return df


def compare_journals(file1, file2, name1="Journal 1", name2="Journal 2"):
    """Compare deux journaux de trading."""

    print("=" * 70)
    print("COMPARAISON DES JOURNAUX DE TRADING")
    print("=" * 70)
    print(f"  {name1}: {Path(file1).name}")
    print(f"  {name2}: {Path(file2).name}")
    print("=" * 70)

    df1 = load_journal(file1)
    df2 = load_journal(file2)

    # --- Stats globales ---
    print("\n## STATISTIQUES GLOBALES\n")
    print(f"{'Métrique':<25} {name1:>20} {name2:>20}")
    print("-" * 65)
    print(f"{'Nombre de trades':<25} {len(df1):>20} {len(df2):>20}")
    print(f"{'PnL net total':<25} {df1['pnl_net'].sum():>20,.2f} {df2['pnl_net'].sum():>20,.2f}")
    print(f"{'PnL net moyen':<25} {df1['pnl_net'].mean():>20,.2f} {df2['pnl_net'].mean():>20,.2f}")

    wins1 = (df1['pnl_net'] > 0).sum()
    wins2 = (df2['pnl_net'] > 0).sum()
    print(f"{'Trades gagnants':<25} {wins1:>20} {wins2:>20}")
    print(f"{'Win rate':<25} {wins1/len(df1)*100:>19.1f}% {wins2/len(df2)*100:>19.1f}%")

    # --- Créer une clé unique pour chaque trade ---
    df1['key'] = df1['symbole'] + '_' + df1['date_entree'].dt.strftime('%Y-%m-%d')
    df2['key'] = df2['symbole'] + '_' + df2['date_entree'].dt.strftime('%Y-%m-%d')

    keys1 = set(df1['key'])
    keys2 = set(df2['key'])

    common = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1

    # --- Trades uniques à chaque journal ---
    print(f"\n## TRADES UNIQUES\n")
    print(f"Trades communs: {len(common)}")
    print(f"Uniquement dans {name1}: {len(only_in_1)}")
    print(f"Uniquement dans {name2}: {len(only_in_2)}")

    if only_in_1:
        print(f"\n### Trades uniquement dans {name1}:\n")
        df_only1 = df1[df1['key'].isin(only_in_1)].sort_values('date_entree')
        for _, row in df_only1.iterrows():
            pnl_str = f"+{row['pnl_net']:.2f}" if row['pnl_net'] > 0 else f"{row['pnl_net']:.2f}"
            print(f"  {row['symbole']:<6} | Entrée: {row['date_entree'].strftime('%Y-%m-%d')} @ {row['prix_entree']:.2f} | "
                  f"Sortie: {row['date_sortie'].strftime('%Y-%m-%d')} @ {row['prix_sortie']:.2f} | PnL: {pnl_str}")

    if only_in_2:
        print(f"\n### Trades uniquement dans {name2}:\n")
        df_only2 = df2[df2['key'].isin(only_in_2)].sort_values('date_entree')
        for _, row in df_only2.iterrows():
            pnl_str = f"+{row['pnl_net']:.2f}" if row['pnl_net'] > 0 else f"{row['pnl_net']:.2f}"
            print(f"  {row['symbole']:<6} | Entrée: {row['date_entree'].strftime('%Y-%m-%d')} @ {row['prix_entree']:.2f} | "
                  f"Sortie: {row['date_sortie'].strftime('%Y-%m-%d')} @ {row['prix_sortie']:.2f} | PnL: {pnl_str}")

    # --- Comparer les trades communs ---
    if common:
        print(f"\n## DIFFÉRENCES SUR TRADES COMMUNS\n")

        differences = []
        for key in common:
            t1 = df1[df1['key'] == key].iloc[0]
            t2 = df2[df2['key'] == key].iloc[0]

            diffs = []

            # Comparer les prix d'entrée
            if abs(t1['prix_entree'] - t2['prix_entree']) > 0.01:
                diffs.append(f"Prix entrée: {t1['prix_entree']:.2f} vs {t2['prix_entree']:.2f}")

            # Comparer les dates de sortie
            if t1['date_sortie'] != t2['date_sortie']:
                diffs.append(f"Date sortie: {t1['date_sortie'].strftime('%Y-%m-%d')} vs {t2['date_sortie'].strftime('%Y-%m-%d')}")

            # Comparer les prix de sortie
            if abs(t1['prix_sortie'] - t2['prix_sortie']) > 0.01:
                diffs.append(f"Prix sortie: {t1['prix_sortie']:.2f} vs {t2['prix_sortie']:.2f}")

            # Comparer les causes de sortie
            if t1['cause_sortie'] != t2['cause_sortie']:
                diffs.append(f"Cause: {t1['cause_sortie']} vs {t2['cause_sortie']}")

            # Comparer les PnL
            pnl_diff = t2['pnl_net'] - t1['pnl_net']
            if abs(pnl_diff) > 1:
                diffs.append(f"PnL: {t1['pnl_net']:.2f} vs {t2['pnl_net']:.2f} (diff: {pnl_diff:+.2f})")

            if diffs:
                differences.append({
                    'symbole': t1['symbole'],
                    'date_entree': t1['date_entree'],
                    'diffs': diffs,
                    'pnl_diff': pnl_diff
                })

        if differences:
            print(f"Trades avec différences: {len(differences)}/{len(common)}\n")
            for d in sorted(differences, key=lambda x: x['date_entree']):
                print(f"  {d['symbole']:<6} | Entrée: {d['date_entree'].strftime('%Y-%m-%d')}")
                for diff in d['diffs']:
                    print(f"         -> {diff}")
                print()
        else:
            print("Aucune différence significative sur les trades communs.")

    # --- Impact sur la performance ---
    print("\n## IMPACT SUR LA PERFORMANCE\n")

    # PnL des trades uniques
    if only_in_1:
        pnl_only1 = df1[df1['key'].isin(only_in_1)]['pnl_net'].sum()
        print(f"PnL des trades uniques à {name1}: {pnl_only1:+,.2f}")

    if only_in_2:
        pnl_only2 = df2[df2['key'].isin(only_in_2)]['pnl_net'].sum()
        print(f"PnL des trades uniques à {name2}: {pnl_only2:+,.2f}")

    # Différence totale
    diff_total = df2['pnl_net'].sum() - df1['pnl_net'].sum()
    print(f"\nDifférence totale de PnL: {diff_total:+,.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        name1 = sys.argv[3] if len(sys.argv) > 3 else "Trailing"
        name2 = sys.argv[4] if len(sys.argv) > 4 else "Exhaustion"
    else:
        # Fichiers par défaut - à modifier selon vos fichiers
        file1 = "trading_journal_trailing_20260109_162441.csv"
        file2 = "trading_journal_exhaustion_20260110_233110.csv"
        name1 = "Trailing"
        name2 = "Exhaustion"

    compare_journals(file1, file2, name1, name2)
