param(
    [string]$Resume,
    [ValidateSet('dev', 'test')][string]$Split = 'test',
    [ValidateSet('adaptive', 'retrieval-only', 'static', 'uniform')][string[]]$ResearchModes = @('adaptive', 'retrieval-only', 'static', 'uniform'),
    [string]$OutputRoot = 'results/research_v2',
    [string]$ReviewsDir,
    [switch]$MetricsOnly,
    [switch]$PrepareOnly,
    [switch]$StopDemo,
    [switch]$Help
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$coursePython = Join-Path (Split-Path (Split-Path $projectRoot -Parent) -Parent) '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $runtimePython = $localPython }
elseif (Test-Path -LiteralPath $coursePython) { $runtimePython = $coursePython }
else { throw 'Python environment missing. Install the project research dependencies first.' }

if ($MetricsOnly -and -not $Resume) { throw '-MetricsOnly requires -Resume with an existing run directory.' }
if ($MetricsOnly -and $PrepareOnly) { throw 'Choose -MetricsOnly or -PrepareOnly.' }

$previousPath = $env:PYTHONPATH
$previousUtf8 = $env:PYTHONUTF8
$previousHfHome = $env:HF_HOME
$previousTorchHome = $env:TORCH_HOME
$previousLaunchContext = $env:CLIPSCOUT_LAUNCH_CONTEXT
$env:PYTHONPATH = "$projectRoot;$projectRoot\.runtime-packages"
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$launchContext = @{ stop_demo_requested = [bool]$StopDemo; stopped_demo_pids = @() }

Push-Location -LiteralPath $projectRoot
try {
    if ($Help) {
        & $runtimePython -u -m scripts.run_research_study --help
        $clipScoutExit = $LASTEXITCODE
    } else {
        # Only an explicitly requested stop of this local demo is permitted.
        # Match both the listening PID and ClipScout's module/port arguments.
        if ($StopDemo -and -not $PrepareOnly -and -not $MetricsOnly) {
            $listeners = @(Get-NetTCPConnection -LocalPort 7871 -State Listen -ErrorAction SilentlyContinue)
            foreach ($listener in $listeners) {
                $demoProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($listener.OwningProcess)"
                $commandLine = [string]$demoProcess.CommandLine
                if ($commandLine -notmatch '(?i)-m\s+clipscout\s+serve\b' -or $commandLine -notmatch '--port\s+7871(?:\s|$)') {
                    throw 'Port 7871 belongs to another application. No process was stopped.'
                }
                Stop-Process -Id $listener.OwningProcess -ErrorAction Stop
                $launchContext.stopped_demo_pids += $listener.OwningProcess
                Write-Host "Stopped ClipScout demo PID $($listener.OwningProcess) to release its GPU memory."
            }
        }
        $env:CLIPSCOUT_LAUNCH_CONTEXT = $launchContext | ConvertTo-Json -Compress
        $experimentArgs = @('-u', '-m', 'scripts.run_research_study', '--prepare-assets', '--download-models',
                            '--local-files-only', '--output-root', $OutputRoot, '--split', $Split, '--modes')
        $experimentArgs += $ResearchModes
        if ($Resume) { $experimentArgs += @('--resume', $Resume) }
        if ($ReviewsDir) { $experimentArgs += @('--reviews-dir', $ReviewsDir) }
        if ($MetricsOnly) { $experimentArgs += '--metrics-only' }
        if ($PrepareOnly) { $experimentArgs += '--prepare-only' }
        & $runtimePython @experimentArgs
        $clipScoutExit = $LASTEXITCODE
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $previousPath
    $env:PYTHONUTF8 = $previousUtf8
    $env:HF_HOME = $previousHfHome
    $env:TORCH_HOME = $previousTorchHome
    $env:CLIPSCOUT_LAUNCH_CONTEXT = $previousLaunchContext
}
exit $clipScoutExit
