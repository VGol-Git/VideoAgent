param([Parameter(ValueFromRemainingArguments=$true)][string[]]$ClipScoutArgs)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$coursePython = Join-Path (Split-Path (Split-Path $projectRoot -Parent) -Parent) '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $runtimePython = $localPython }
elseif (Test-Path -LiteralPath $coursePython) { $runtimePython = $coursePython }
else { $runtimePython = 'python' }
$previousPath = $env:PYTHONPATH
$env:PYTHONPATH = "$projectRoot;$projectRoot\.runtime-packages"
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:PYTHONUTF8 = '1'
Push-Location -LiteralPath $projectRoot
try { & $runtimePython -m clipscout @ClipScoutArgs; $clipScoutExit = $LASTEXITCODE }
finally { Pop-Location; $env:PYTHONPATH = $previousPath }
exit $clipScoutExit
