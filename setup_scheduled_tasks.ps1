# Crée les tâches planifiées Windows pour les tâches de fin de journée IB Trading Bot
# Exécuter en tant qu'Administrateur

$ProjectRoot = "D:\QuantProjects\ib_trading_bot"
$Python      = "$ProjectRoot\.venv\Scripts\python.exe"
$Days        = "MON,TUE,WED,THU,FRI"

$tasks = @(
    @{
        Name    = "IB - Journal Sync"
        Script  = "$ProjectRoot\src\app\live_journal.py"
        Args    = "sync --port 7497"
        Time    = "16:15"
        WorkDir = "$ProjectRoot\src\app"
    },
    @{
        Name    = "IB - Update US Data"
        Script  = "$ProjectRoot\src\app\database\update_historical_data.py"
        Args    = ""
        Time    = "16:30"
        WorkDir = $ProjectRoot
    },
    @{
        Name    = "IB - Update Canada Data"
        Script  = "$ProjectRoot\src\app\database\update_canada_data.py"
        Args    = ""
        Time    = "16:45"
        WorkDir = $ProjectRoot
    }
)

foreach ($t in $tasks) {
    $cmd = "`"$Python`" `"$($t.Script)`""
    if ($t.Args) { $cmd += " $($t.Args)" }

    # Supprimer si déjà existante
    schtasks /delete /tn $t.Name /f 2>$null

    schtasks /create `
        /tn  $t.Name `
        /tr  $cmd `
        /sc  WEEKLY `
        /d   $Days `
        /st  $t.Time `
        /sd  "2026-04-23" `
        /f

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Tâche créée : $($t.Name) à $($t.Time)" -ForegroundColor Green
    } else {
        Write-Host "[ERREUR] Échec création : $($t.Name)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Tâches créées. Vérification :"
schtasks /query /fo LIST /tn "IB - Journal Sync"   2>$null | Select-String "Nom de la tâche|Prochaine"
schtasks /query /fo LIST /tn "IB - Update US Data" 2>$null | Select-String "Nom de la tâche|Prochaine"
schtasks /query /fo LIST /tn "IB - Update Canada Data" 2>$null | Select-String "Nom de la tâche|Prochaine"
