"""
Scheduler quotidien de trading.

Automatise les tâches suivantes chaque jour de bourse :
  - 09h30 ET : prise de position (trading.py → init_trade)
  - 16h15 ET : synchronisation du journal depuis IB (live_journal.py sync)
  - 16h30 ET : mise à jour des données historiques US
  - 16h45 ET : mise à jour des données historiques Canada

Usage :
    python src/app/daily_scheduler.py
    python src/app/daily_scheduler.py --paper    # Paper trading (défaut)
    python src/app/daily_scheduler.py --live     # Live trading (port 7496)
    python src/app/daily_scheduler.py --dry-run  # Affiche le planning sans exécuter

Logs écrits dans : logs/scheduler_YYYY-MM-DD.log
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, date

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Chemins (relatifs à la racine du projet)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
APP_DIR = os.path.join(PROJECT_ROOT, 'src', 'app')

SCRIPTS = {
    'trading':        os.path.join(APP_DIR, 'trading.py'),
    'stop_monitor':   os.path.join(APP_DIR, 'stop_manager.py'),
    'journal_sync':   os.path.join(APP_DIR, 'live_journal.py'),
    'update_us':      os.path.join(APP_DIR, 'database', 'update_historical_data.py'),
    'update_canada':  os.path.join(APP_DIR, 'database', 'update_canada_data.py'),
}

# Répertoire de travail par tâche (trading.py exige d'être lancé depuis src/app)
WORKING_DIRS = {
    'trading':        APP_DIR,
    'stop_monitor':   APP_DIR,
    'journal_sync':   APP_DIR,
    'update_us':      PROJECT_ROOT,
    'update_canada':  PROJECT_ROOT,
}

# Planning horaire (heure locale de la machine, format HH:MM)
SCHEDULE = [
    # (heure, tâche, description, background)
    # background=True : lancé en arrière-plan (Popen), le scheduler continue immédiatement
    # background=False : bloquant (run), le scheduler attend la fin avant de continuer
    ("09:20", "trading",       "Prise de position",                          False),
    ("10:00", "stop_monitor",  "Surveillance des stops (boucle jusqu'à 16h)", True),
    # journal_sync, update_us, update_canada gérés par tâches Windows séparées
]

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"scheduler_{date.today().strftime('%Y-%m-%d')}.log")

    logger = logging.getLogger('scheduler')
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Fichier
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Exécution des tâches
# ---------------------------------------------------------------------------

def run_task(task: str, port: int, dry_run: bool, logger: logging.Logger,
             background: bool = False) -> bool:
    """
    Lance le script associé à la tâche.
    background=True : Popen (non bloquant), le scheduler continue immédiatement.
    background=False : run (bloquant), attend la fin du script.
    Retourne True si lancement réussi.
    """
    script = SCRIPTS.get(task)
    if not script or not os.path.exists(script):
        logger.error(f"Script introuvable pour la tâche '{task}': {script}")
        return False

    # Construction de la commande
    cmd = [sys.executable, script]

    if task == 'trading':
        if port == 7496:
            cmd += ['--live']
        logger.info(f"Port IB utilisé: {port} ({'paper' if port in (7497, 4002) else 'live'})")

    elif task == 'stop_monitor':
        cmd += ['--port', str(port)]
        if port == 7496:
            cmd += ['--live']

    elif task == 'journal_sync':
        cmd += ['sync', '--port', str(port)]

    # update_us et update_canada n'ont pas besoin d'arguments supplémentaires

    logger.info(f">>> Lancement: {' '.join(cmd)}")

    if dry_run:
        logger.info("[DRY-RUN] Commande non exécutée.")
        return True

    cwd = WORKING_DIRS.get(task, PROJECT_ROOT)
    logger.info(f"Répertoire de travail: {cwd}")

    try:
        if background:
            subprocess.Popen(cmd, cwd=cwd)
            logger.info(f"Tâche '{task}' lancée en arrière-plan")
            return True
        else:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=False,   # Affiche la sortie dans le terminal en temps réel
                text=True,
                timeout=1800            # 30 min max par tâche
            )
            if result.returncode == 0:
                logger.info(f"Tâche '{task}' terminée avec succès (code {result.returncode})")
                return True
            else:
                logger.error(f"Tâche '{task}' échouée (code {result.returncode})")
                return False
    except subprocess.TimeoutExpired:
        logger.error(f"Tâche '{task}' : timeout (30 min dépassé)")
        return False
    except Exception as e:
        logger.error(f"Tâche '{task}' : erreur inattendue: {e}")
        return False


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

def is_trading_day() -> bool:
    """Retourne False le weekend (pas de bourse)."""
    return datetime.now().weekday() < 5  # 0=lun … 4=ven


def parse_time(hhmm: str) -> tuple[int, int]:
    h, m = hhmm.split(':')
    return int(h), int(m)


def main():
    parser = argparse.ArgumentParser(description="Scheduler quotidien de trading IB")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--paper', action='store_true', default=True,
                       help='Paper trading TWS (port 7497, défaut)')
    group.add_argument('--live', action='store_true',
                       help='Live trading TWS (port 7496)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Affiche le planning sans exécuter les scripts')
    args = parser.parse_args()

    port = 7496 if args.live else 7497
    dry_run = args.dry_run

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Scheduler de trading démarré")
    logger.info(f"Mode: {'LIVE' if args.live else 'PAPER'} | Port: {port}")
    logger.info(f"Dry-run: {dry_run}")
    logger.info("Planning :")
    for hhmm, task, desc, bg in SCHEDULE:
        logger.info(f"  {hhmm} → {desc}" + (" [arrière-plan]" if bg else ""))
    logger.info("=" * 60)

    # Suivi des tâches déjà exécutées aujourd'hui
    tasks_done: set[str] = set()   # format: "YYYY-MM-DD_task"

    try:
        while True:
            try:
                now = datetime.now()
                today = now.strftime('%Y-%m-%d')

                if not is_trading_day():
                    from datetime import timedelta
                    seconds_to_tomorrow = (
                        (now.replace(hour=8, minute=0, second=0, microsecond=0)
                         + timedelta(days=1) - now).total_seconds()
                    )
                    logger.info(f"Weekend — prochaine vérification demain 8h00 "
                                f"(dans {int(seconds_to_tomorrow // 3600)}h)")
                    time.sleep(min(seconds_to_tomorrow, 3600))
                    continue

                for hhmm, task, desc, bg in SCHEDULE:
                    task_key = f"{today}_{task}"
                    if task_key in tasks_done:
                        continue  # Déjà exécutée aujourd'hui

                    h, m = parse_time(hhmm)
                    scheduled_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)

                    if now >= scheduled_dt:
                        logger.info(f"--- {desc} ({hhmm}) ---")
                        success = run_task(task, port, dry_run, logger, background=bg)
                        tasks_done.add(task_key)
                        if not success:
                            logger.warning(f"Tâche '{task}' échouée, elle ne sera pas relancée automatiquement.")

                # Vérifier si toutes les tâches du jour sont faites
                all_done = all(f"{today}_{t}" in tasks_done for _, t, _, _ in SCHEDULE)
                if all_done:
                    from datetime import timedelta
                    tomorrow_9am = (now + timedelta(days=1)).replace(
                        hour=9, minute=0, second=0, microsecond=0
                    )
                    wait_sec = (tomorrow_9am - now).total_seconds()
                    logger.info(f"Toutes les tâches du jour terminées. "
                                f"Prochain cycle demain à 09h00 (dans {int(wait_sec // 3600)}h{int((wait_sec % 3600) // 60):02d}m)")
                    time.sleep(wait_sec)
                    tasks_done.clear()  # Réinitialiser pour le nouveau jour
                    continue

                time.sleep(30)  # Vérifie toutes les 30 secondes

            except Exception as e:
                logger.error(f"Erreur inattendue dans la boucle principale : {e}", exc_info=True)
                logger.info("Reprise dans 60 secondes...")
                time.sleep(60)  # Pause courte avant de reprendre

    except KeyboardInterrupt:
        logger.info("Scheduler arrêté par l'utilisateur (Ctrl+C)")


if __name__ == "__main__":
    main()
