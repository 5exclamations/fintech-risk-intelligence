<# Windows equivalent of the Makefile.  Usage:  .\scripts\dev.ps1 <setup|all|backtest|test|lint|api|dashboard>
   Needs Python 3.11-3.13 on PATH (or the `py` launcher). Python 3.14+ may lack wheels for some dependencies. #>
param([Parameter(Mandatory)][ValidateSet("setup", "all", "data", "train", "backtest", "test", "lint", "api", "dashboard")][string]$Task)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
switch ($Task) {
    "setup" {
        if (-not (Test-Path $py)) { if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv } else { python -m venv .venv } }
        & $py -m pip install -U pip
        & $py -m pip install -r requirements-dev.txt -c constraints.txt
    }
    "all"       { & $py -m riskplatform.pipeline all }
    "data"      { & $py -m riskplatform.pipeline generate }
    "train"     { & $py -m riskplatform.pipeline train }
    "backtest"  { & $py -m riskplatform.pipeline backtest }
    "test"      { & $py -m pytest -q }
    "lint"      { & $py -m ruff check . }
    "api"       { if (-not $env:RISK_API_KEY) { $env:RISK_API_KEY = "dev-demo-key" }; & $py -m uvicorn riskplatform.api:app --port 8000 }
    "dashboard" { & $py -m streamlit run dashboard/app.py }
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }
