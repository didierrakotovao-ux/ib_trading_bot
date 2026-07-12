"""
Consolide les fichiers parquet 30s en fichiers journaliers.

Usage:
    python src/app/consolidate_ofi.py                        # tout collectOF
    python src/app/consolidate_ofi.py --session sessions_mes # une session
    python src/app/consolidate_ofi.py --date 2026-06-18      # un jour specifique
    python src/app/consolidate_ofi.py --dry-run              # voir sans ecrire
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def consolidate_day(day_dir: Path, dry_run: bool = False) -> dict:
    """Consolide tous les chunks 30s d'un jour en deux fichiers journaliers."""
    trade_files = sorted(day_dir.glob("trade_*.parquet"))
    quote_files = sorted(day_dir.glob("quote_*.parquet"))

    result = {"trades": 0, "quotes": 0, "skipped": False}

    out_trade = day_dir / "trades.parquet"
    out_quote = day_dir / "quotes.parquet"

    # Ne pas recraser si deja consolide et aucun nouveau chunk
    if out_trade.exists() and out_quote.exists():
        n_chunks = len(trade_files)
        if n_chunks == 0:
            result["skipped"] = True
            return result

    if trade_files:
        df = pd.concat([pd.read_parquet(f) for f in trade_files], ignore_index=True)
        df = df.sort_values("ts").reset_index(drop=True)
        result["trades"] = len(df)
        if not dry_run:
            df.to_parquet(out_trade, index=False)

    if quote_files:
        df = pd.concat([pd.read_parquet(f) for f in quote_files], ignore_index=True)
        df = df.sort_values("ts").reset_index(drop=True)
        result["quotes"] = len(df)
        if not dry_run:
            df.to_parquet(out_quote, index=False)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolide les chunks 30s en fichiers journaliers")
    parser.add_argument("--root",    type=str, default="collectOF")
    parser.add_argument("--session", type=str, default=None,
                        help="Ex: sessions_mes, sessions_mnq, sessions_dow")
    parser.add_argument("--date",    type=str, default=None,
                        help="Ex: 2026-06-18")
    parser.add_argument("--dry-run", action="store_true",
                        help="Afficher sans ecrire")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERREUR] Repertoire introuvable: {root}")
        return 1

    sessions = sorted(root.iterdir()) if args.session is None else [root / args.session]
    sessions = [s for s in sessions if s.is_dir()]

    total_trades = total_quotes = 0
    days_done = 0

    for session in sessions:
        dates = sorted(session.iterdir()) if args.date is None else [session / args.date]
        dates = [d for d in dates if d.is_dir()]

        for day_dir in dates:
            r = consolidate_day(day_dir, dry_run=args.dry_run)
            if r["skipped"]:
                continue
            tag = "[DRY]" if args.dry_run else "[OK] "
            print(f"{tag} {session.name}/{day_dir.name}  trades={r['trades']:,}  quotes={r['quotes']:,}")
            total_trades += r["trades"]
            total_quotes += r["quotes"]
            days_done += 1

    print(f"\nTotal: {days_done} jours  {total_trades:,} trades  {total_quotes:,} quotes")
    if args.dry_run:
        print("(dry-run: aucun fichier ecrit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
