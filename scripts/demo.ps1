<#
.SYNOPSIS
  One command to run the whole SurgiGuard AI demo on a PC with no ESP32 and no Android Studio.

.DESCRIPTION
  1. Docker: MongoDB + local MQTT broker (Mosquitto) + SurgiGuard .NET API (built from source in Docker)
  2. Demo accounts (fictional patient Sarah + device ESP32_001, family Ahmed, doctor Dr Ruiz)
  3. Patient/family web app build (React Native -> web), if missing or built for another API port
  4. AI backend (FastAPI) on :8000, in its own window
  5. Virtual ESP32 publishing over MQTT, in its own window
  6. Opens http://localhost:8000/story

.EXAMPLE
  ./scripts/demo.ps1                     # everything, normal vitals
  ./scripts/demo.ps1 -Profile story      # scripted scenario: normal -> tachycardia -> desaturation -> sensor off -> normal
  ./scripts/demo.ps1 -AiMode stub        # force offline AI even if keys are in backend/.env
  ./scripts/demo.ps1 -Stop               # stop the containers (the two windows close with Ctrl+C)
#>
param(
    [ValidateSet('normal', 'desat', 'tachy', 'nofinger', 'agitated', 'story')][string]$Profile = 'normal',
    [ValidateSet('auto', 'stub', 'live')][string]$AiMode = 'auto',
    [int]$ApiPort = 0,
    [switch]$NoSimulator,
    [switch]$NoBrowser,
    [switch]$Stop
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$infra = Join-Path $root 'SurgiGuard/Backend/Infrastructure_hackathon'
$python = Join-Path $root 'backend/.venv/Scripts/python.exe'

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Test-Port($port) {
    $c = [System.Net.Sockets.TcpClient]::new()
    try { $c.Connect('127.0.0.1', $port); return $true } catch { return $false } finally { $c.Dispose() }
}
function Wait-Http($url, $seconds = 120) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        try { if ((Invoke-WebRequest $url -TimeoutSec 3 -UseBasicParsing).StatusCode -lt 500) { return } } catch {}
        Start-Sleep -Seconds 2
    }
    throw "Not ready after $seconds s: $url"
}

if ($Stop) {
    Push-Location $infra; try { docker compose stop } finally { Pop-Location }
    return
}

# ---------------------------------------------------------------- prerequisites
if (!(Test-Path $python)) {
    throw "Python environment missing. Run once:`n  cd backend; python -m venv .venv --system-site-packages; .venv\Scripts\pip install -r requirements.txt; .venv\Scripts\pip install -e ..\d1[wound,dev]"
}
Step 'Docker'
docker info --format '{{.ServerVersion}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Starting Docker Desktop...'
    Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    $deadline = (Get-Date).AddMinutes(3)
    do { Start-Sleep 3; docker info --format '{{.ServerVersion}}' 2>$null | Out-Null } until ($LASTEXITCODE -eq 0 -or (Get-Date) -gt $deadline)
    if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop did not start.' }
}

# The API container listens on 8080; publish it on 8080 unless something else (e.g. XAMPP/Apache) already uses it.
$apiContainerUp = (docker ps --filter 'name=^api$' --format '{{.Ports}}') -match '0.0.0.0:(\d+)->8080'
if ($ApiPort -eq 0) {
    if ($apiContainerUp) { $ApiPort = [int]$Matches[1] }
    elseif (!(Test-Port 8080)) { $ApiPort = 8080 }
    else { $ApiPort = 5051; Write-Host 'Port 8080 is busy (another program): the API will use 5051.' -ForegroundColor Yellow }
}
$api = "http://localhost:$ApiPort"

Step "MongoDB + MQTT broker + SurgiGuard API ($api)"
$env:API_PORT = "$ApiPort"
$env:MONGO_DB = 'DB_story'          # throw-away demo database, never the real one
Push-Location $infra
try { docker compose up -d --build } finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw 'docker compose failed' }
Wait-Http "$api/health" 180
Write-Host "API healthy. Swagger: $api/swagger"

Step 'Demo accounts'
& $python (Join-Path $root 'SurgiGuard/Backend/scripts/seed_story.py') --api $api

Step 'Patient / family web app'
$dist = Join-Path $root 'mobile/dist-web'
$marker = Join-Path $dist '.account-url'
$needBuild = !(Test-Path (Join-Path $dist 'app.js')) -or !(Test-Path $marker) -or ((Get-Content $marker -Raw).Trim() -ne $api)
if ($needBuild) {
    Push-Location (Join-Path $root 'mobile')
    try {
        if (!(Test-Path 'node_modules')) { npm install --no-audit --no-fund }
        $env:SG_ACCOUNT_URL = $api
        npm run build:web
        if ($LASTEXITCODE -ne 0) { throw 'web build failed' }
        Set-Content -Path $marker -Value $api -Encoding ascii
    } finally { Pop-Location }
} else { Write-Host 'Up to date.' }

Step 'AI backend (http://localhost:8000)'
if (Test-Port 8000) {
    Write-Host 'Something already listens on 8000: reusing it.' -ForegroundColor Yellow
} else {
    $envs = "`$env:SURGIGUARD_API_URL='$api'; `$env:PYTHONIOENCODING='utf-8'; "
    if ($AiMode -ne 'auto') { $envs += "`$env:LLM_MODE='$AiMode'; " }
    $cmd = "$envs `$host.UI.RawUI.WindowTitle='SurgiGuard AI backend :8000'; Set-Location '$root/backend'; & '$python' -m uvicorn main:app --port 8000"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $cmd | Out-Null
}
Wait-Http 'http://localhost:8000/docs' 180

if (!$NoSimulator) {
    Step "Virtual ESP32 over MQTT (profile: $Profile)"
    $cmd = "`$host.UI.RawUI.WindowTitle='Virtual ESP32 (MQTT)'; & '$python' '$root/scripts/simulate_device_mqtt.py' --profile $Profile"
    Start-Process powershell -ArgumentList '-NoExit', '-Command', $cmd | Out-Null
}

Write-Host @"

SurgiGuard AI is running (fictional data, no hardware):
  Whole story ........ http://localhost:8000/story
  Patient/family app . http://localhost:8000/app/      sarah@story.test / Story2026!   (family: ahmed@story.test)
  Clinic (wounds) .... http://localhost:8000/agent-demo/
  D1/D2/D3 console ... http://localhost:8000/prototype
  AI API docs ........ http://localhost:8000/docs
  SurgiGuard API ..... $api/swagger
Stop: Ctrl+C in the two windows, then ./scripts/demo.ps1 -Stop
"@ -ForegroundColor Green
if (!$NoBrowser) { Start-Process 'http://localhost:8000/story' }
