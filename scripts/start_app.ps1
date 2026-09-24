param(
    [ValidateSet('start', 'stop', 'restart', 'status', 'help')]
    [string]$Command = 'start',
    [int]$BackendPort = 0,
    [int]$FrontendPort = 0,
    [string]$HostAddress = '127.0.0.1',
    [switch]$Detached,
    [switch]$StrictPorts,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDirectory
$RuntimeDirectory = Join-Path $ProjectRoot '.run'
$PidFile = Join-Path $RuntimeDirectory 'ai-systems-lab-windows.json'
$LogDirectory = Join-Path $ProjectRoot 'logs'

function Show-Usage {
    @'
AI Systems Lab Windows launcher

Usage:
  .\scripts\start_app.cmd start
  .\scripts\start_app.cmd start -Detached
  .\scripts\start_app.cmd stop
  .\scripts\start_app.cmd restart
  .\scripts\start_app.cmd status

Options:
  -BackendPort PORT    Backend base port (default: 8000)
  -FrontendPort PORT   Frontend base port (default: 3000)
  -HostAddress HOST    Bind host (default: 127.0.0.1)
  -Detached            Run in background and write logs to logs/
  -StrictPorts         Fail instead of selecting the next free port
  -Help                Show this help

When a requested port is busy, the launcher selects the next available port
unless -StrictPorts is provided. Existing processes are never terminated.
'@ | Write-Host
}

function Test-PortInUse([int]$Port) {
    $connection = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Get-NextFreePort([int]$Port) {
    while ($Port -le 65535 -and (Test-PortInUse $Port)) {
        $Port++
    }
    if ($Port -gt 65535) {
        throw 'Boş port bulunamadı.'
    }
    return $Port
}

function Test-ValidPort([int]$Port) {
    return $Port -ge 1 -and $Port -le 65535
}

function Get-ProcessEnvironmentValue([string]$Name) {
    return [Environment]::GetEnvironmentVariable($Name, 'Process')
}

function Start-ConfiguredProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$OutputPath,
        [string]$ErrorPath,
        [hashtable]$Environment
    )

    $previousValues = @{}
    foreach ($name in $Environment.Keys) {
        $previousValues[$name] = Get-ProcessEnvironmentValue $name
        [Environment]::SetEnvironmentVariable($name, [string]$Environment[$name], 'Process')
    }

    try {
        return Start-Process -FilePath $FilePath `
            -ArgumentList $ArgumentList `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $OutputPath `
            -RedirectStandardError $ErrorPath `
            -PassThru
    }
    finally {
        foreach ($name in $Environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previousValues[$name], 'Process')
        }
    }
}

function Get-PidData {
    if (-not (Test-Path $PidFile)) {
        return $null
    }
    return Get-Content $PidFile -Raw | ConvertFrom-Json
}

function Test-ProcessRunning([int]$ProcessId) {
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

function Stop-ProcessTree([int]$ProcessId) {
    if (Test-ProcessRunning $ProcessId) {
        & taskkill.exe /PID $ProcessId /T /F *> $null
    }
}

function Stop-Application {
    $data = Get-PidData
    if ($null -eq $data) {
        Write-Host 'Uygulama çalışmıyor.'
        return
    }

    Stop-ProcessTree ([int]$data.BackendPid)
    Stop-ProcessTree ([int]$data.FrontendPid)
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host 'Uygulama durduruldu.'
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [string]$Label
    )

    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 1
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Host "✓ $Label hazır: $Url"
                return $true
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Warning "$Label 30 saniye içinde doğrulanamadı: $Url"
    return $false
}

function Require-Dependencies {
    $script:PythonPath = Join-Path $ProjectRoot 'venv\Scripts\python.exe'
    if (-not (Test-Path $PythonPath)) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $pythonCommand) {
            throw 'Python bulunamadı. Önce scripts\setup_dev.sh veya Python bağımlılık kurulumunu çalıştırın.'
        }
        $script:PythonPath = $pythonCommand.Source
    }

    $script:NpmPath = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if ([string]::IsNullOrWhiteSpace($NpmPath)) {
        $script:NpmPath = (Get-Command npm -ErrorAction SilentlyContinue).Source
    }
    if ([string]::IsNullOrWhiteSpace($NpmPath)) {
        throw 'npm bulunamadı.'
    }
    if (-not (Test-Path (Join-Path $ProjectRoot 'frontend\node_modules'))) {
        throw 'Frontend bağımlılıkları yok. Önce cd frontend; npm install çalıştırın.'
    }

    & $PythonPath -c 'import uvicorn' 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw 'uvicorn bulunamadı. Python bağımlılıklarını kurun.'
    }
}

function Resolve-Ports {
    param(
        [int]$RequestedBackendPort,
        [int]$RequestedFrontendPort
    )

    if (-not (Test-ValidPort $RequestedBackendPort)) {
        throw "Geçersiz backend portu: $RequestedBackendPort"
    }
    if (-not (Test-ValidPort $RequestedFrontendPort)) {
        throw "Geçersiz frontend portu: $RequestedFrontendPort"
    }

    $resolvedBackend = $RequestedBackendPort
    $resolvedFrontend = $RequestedFrontendPort

    if (Test-PortInUse $resolvedBackend) {
        if ($StrictPorts) { throw "Backend portu kullanımda: $resolvedBackend" }
        $resolvedBackend = Get-NextFreePort ($resolvedBackend + 1)
        Write-Host "Bilgi: Backend için boş port seçildi: $resolvedBackend"
    }

    if ($resolvedFrontend -eq $resolvedBackend) {
        if ($StrictPorts) { throw 'Backend ve frontend aynı portu kullanamaz.' }
        $resolvedFrontend++
    }

    if (Test-PortInUse $resolvedFrontend) {
        if ($StrictPorts) { throw "Frontend portu kullanımda: $resolvedFrontend" }
        $resolvedFrontend = Get-NextFreePort ($resolvedFrontend + 1)
        Write-Host "Bilgi: Frontend için boş port seçildi: $resolvedFrontend"
    }

    return @($resolvedBackend, $resolvedFrontend)
}

function Start-Application {
    param(
        [int]$RequestedBackendPort,
        [int]$RequestedFrontendPort
    )

    $existing = Get-PidData
    if ($null -ne $existing -and (Test-ProcessRunning ([int]$existing.BackendPid) -or Test-ProcessRunning ([int]$existing.FrontendPid))) {
        throw "Uygulama zaten çalışıyor. Önce '.\scripts\start_app.cmd stop' çalıştırın."
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force -Path $RuntimeDirectory, $LogDirectory | Out-Null

    $ports = Resolve-Ports $RequestedBackendPort $RequestedFrontendPort
    $backendPort = [int]$ports[0]
    $frontendPort = [int]$ports[1]
    $apiUrl = "http://${HostAddress}:${backendPort}"
    $frontendUrl = "http://${HostAddress}:${frontendPort}"
    $origins = @(
        "http://${HostAddress}:${frontendPort}",
        "http://localhost:${frontendPort}",
        "http://127.0.0.1:${frontendPort}",
        "http://${HostAddress}:${backendPort}",
        "http://localhost:${backendPort}"
    ) | ConvertTo-Json -Compress

    $backendLog = Join-Path $LogDirectory 'backend-windows.log'
    $backendErrorLog = Join-Path $LogDirectory 'backend-windows.error.log'
    $frontendLog = Join-Path $LogDirectory 'frontend-windows.log'
    $frontendErrorLog = Join-Path $LogDirectory 'frontend-windows.error.log'

    Write-Host "`nAI Systems Lab başlatılıyor..."
    Write-Host "  Backend : $apiUrl"
    Write-Host "  Frontend: $frontendUrl`n"

    $backendEnvironment = @{
        BACKEND_HOST = $HostAddress
        BACKEND_PORT = $backendPort
        BACKEND_RELOAD = 'true'
        BACKEND_WORKERS = '1'
        SEED_DEMO_USERS = if ($env:SEED_DEMO_USERS) { $env:SEED_DEMO_USERS } else { 'true' }
        ALLOWED_ORIGINS = $origins
    }
    $frontendEnvironment = @{
        API_URL = $apiUrl
        NEXT_PUBLIC_API_URL = $apiUrl
    }

    $backend = Start-ConfiguredProcess `
        -FilePath $PythonPath `
        -ArgumentList @('-m', 'uvicorn', 'backend.main:app', '--host', $HostAddress, '--port', [string]$backendPort, '--reload') `
        -WorkingDirectory $ProjectRoot `
        -OutputPath $backendLog `
        -ErrorPath $backendErrorLog `
        -Environment $backendEnvironment

    $frontend = Start-ConfiguredProcess `
        -FilePath $NpmPath `
        -ArgumentList @('run', 'dev', '--', '--hostname', $HostAddress, '--port', [string]$frontendPort) `
        -WorkingDirectory (Join-Path $ProjectRoot 'frontend') `
        -OutputPath $frontendLog `
        -ErrorPath $frontendErrorLog `
        -Environment $frontendEnvironment

    @{ BackendPid = $backend.Id; FrontendPid = $frontend.Id; BackendPort = $backendPort; FrontendPort = $frontendPort } |
        ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

    $backendReady = Wait-ForUrl "$apiUrl/health" 'Backend'
    $frontendReady = Wait-ForUrl "$frontendUrl/" 'Frontend'
    if (-not ($backendReady -and $frontendReady)) {
        Stop-ProcessTree $backend.Id
        Stop-ProcessTree $frontend.Id
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        throw "Uygulama başlatılamadı. Loglar: $backendLog, $frontendLog"
    }

    Write-Host "`nUygulama hazır."
    Write-Host "  Frontend: $frontendUrl"
    Write-Host "  Backend : $apiUrl"
    Write-Host "  API docs: $apiUrl/docs"

    if ($Detached) {
        Write-Host "  Loglar  : $backendLog, $frontendLog"
        Write-Host "Durdurmak için: .\scripts\start_app.cmd stop"
        return
    }

    Write-Host "`nÇıkmak için Ctrl-C kullanın."
    try {
        while ((Test-ProcessRunning $backend.Id) -and (Test-ProcessRunning $frontend.Id)) {
            Start-Sleep -Seconds 1
        }
    }
    finally {
        Stop-ProcessTree $backend.Id
        Stop-ProcessTree $frontend.Id
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

function Show-Status {
    $data = Get-PidData
    if ($null -eq $data) {
        Write-Host 'Uygulama çalışmıyor.'
        return
    }
    $backendState = if (Test-ProcessRunning ([int]$data.BackendPid)) { 'çalışıyor' } else { 'çalışmıyor' }
    $frontendState = if (Test-ProcessRunning ([int]$data.FrontendPid)) { 'çalışıyor' } else { 'çalışmıyor' }
    Write-Host "Backend : $($data.BackendPort) (PID $($data.BackendPid)) $backendState"
    Write-Host "Frontend: $($data.FrontendPort) (PID $($data.FrontendPid)) $frontendState"
}

if ($Help -or $Command -eq 'help') {
    Show-Usage
    exit 0
}

try {
    if ($BackendPort -eq 0) {
        $BackendPort = if ($env:BACKEND_PORT) { [int]$env:BACKEND_PORT } else { 8000 }
    }
    if ($FrontendPort -eq 0) {
        $FrontendPort = if ($env:FRONTEND_PORT) { [int]$env:FRONTEND_PORT } else { 3000 }
    }

    switch ($Command) {
        'start' {
            Require-Dependencies
            Start-Application $BackendPort $FrontendPort
        }
        'stop' { Stop-Application }
        'restart' {
            Require-Dependencies
            Stop-Application
            Start-Application $BackendPort $FrontendPort
        }
        'status' { Show-Status }
        default { throw "Bilinmeyen komut: $Command" }
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
