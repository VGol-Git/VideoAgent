param(
    [ValidateSet('test', 'dev', 'all')][string]$Split = 'test',
    [string]$QuestionsPath = (Join-Path (Split-Path $PSScriptRoot -Parent) 'questions.json'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'results\assignment_v6_adaptive'),
    [ValidateSet('adaptive', 'retrieval-only', 'static', 'uniform')][string[]]$Modes = @('adaptive'),
    [switch]$StopDemo,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$questionPath = (Resolve-Path -LiteralPath $QuestionsPath).Path
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$coursePython = Join-Path (Split-Path (Split-Path $projectRoot -Parent) -Parent) '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $runtimePython = $localPython }
elseif (Test-Path -LiteralPath $coursePython) { $runtimePython = $coursePython }
else { throw 'Python environment missing.' }
$branch = & git -C $projectRoot branch --show-current
if ($LASTEXITCODE -ne 0 -or $branch -ne 'v6') { throw 'Switch to branch v6 before running this launcher.' }
$Modes = @($Modes | ForEach-Object { $_.ToLowerInvariant() })
if ($Modes.Count -eq 0 -or @($Modes | Select-Object -Unique).Count -ne $Modes.Count) { throw 'Select at least one research mode, without duplicates.' }

$previousEnvironment = @{}
foreach ($name in @('PYTHONPATH', 'PYTHONUTF8', 'HF_HOME', 'TORCH_HOME')) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$previousOutputEncoding = $OutputEncoding
$previousConsoleEncoding = [Console]::OutputEncoding
function Save-RunJson([string]$Path, $Value) {
    $json = ConvertTo-Json -InputObject $Value -Depth 100
    [IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}
try {
    $env:PYTHONPATH = "$projectRoot;$projectRoot\.runtime-packages"
    $env:PYTHONUTF8 = '1'
    $env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
    $env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    $questions = Get-Content -LiteralPath $questionPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $selected = @($questions | Where-Object { $Split -eq 'all' -or $_.split -eq $Split })
    if ($selected.Count -eq 0) { throw 'No questions in the selected split.' }

    # Only the evaluator needs this format conversion. The v6 agent receives
    # the unchanged question text and ID, never any reference answer or support.
    $categoryMap = @{ reasoning = 'reasoning'; onset = 'temporal'; timestamp_search = 'localization'; unanswerable = 'unanswerable' }
    $adapted = @(foreach ($question in $selected) {
        if ($null -ne $question.PSObject.Properties['category'] -and
            $null -ne $question.PSObject.Properties['reference_answers'] -and
            $null -ne $question.PSObject.Properties['evidence_intervals']) {
            # Native v6 sets keep their alternative answers and required terms.
            $question
            continue
        }
        if (-not $categoryMap.ContainsKey([string]$question.task)) { throw "Unknown question task: $($question.task)" }
        if ($question.answerable -isnot [bool]) { throw "Invalid answerable flag: $($question.id)" }
        $answers = @()
        $intervals = @()
        if ($question.answerable) {
            $atoms = @($question.claim_atoms | ForEach-Object { [string]$_.text })
            if ($atoms.Count -gt 0) { $answers = @(($atoms -join ' ')) }
            elseif ($question.event.reference_bracket.Count -eq 2 -and $question.event.label) {
                $onsetStart = ([double]$question.event.reference_bracket[0]).ToString('R', [Globalization.CultureInfo]::InvariantCulture)
                $onsetEnd = ([double]$question.event.reference_bracket[1]).ToString('R', [Globalization.CultureInfo]::InvariantCulture)
                $answers = @("$($question.event.label) Onset bracket: $onsetStart to $onsetEnd seconds.")
            } else { throw "Missing evaluation reference: $($question.id)" }
            foreach ($support in @($question.supports)) {
                if ($support.required -eq $true) { $intervals += ,@([double]$support.start, [double]$support.end) }
            }
        }
        [pscustomobject]@{
            id = $question.id; split = $question.split; question = $question.question
            category = $categoryMap[[string]$question.task]; answerable = $question.answerable
            reference_answers = $answers; evidence_intervals = $intervals; required_terms = @()
            source_task = $question.task
        }
    })
    $validationCode = @'
import json, sys
from clipscout.agent import PROMPT_VERSION
from clipscout.evaluation import evaluate
if PROMPT_VERSION != 'clipscout-v6-temporal-scopes':
    raise ValueError('This launcher requires the v6 agent, not a different version.')
questions = json.load(sys.stdin)
evaluate(questions, [], bootstrap_samples=1)
print('Question format validated without loading a model.')
'@
    $adaptedJson = ConvertTo-Json -InputObject $adapted -Depth 100
    $adaptedJson | & $runtimePython -c $validationCode
    if ($LASTEXITCODE -ne 0) { throw 'Question validation failed. No experiment was started.' }
    $plannedPredictions = $selected.Count * $Modes.Count
    Write-Host "Current v6; $($selected.Count) questions; modes=$($Modes -join ', '); $plannedPredictions predictions; split=$Split."
    if ($PlanOnly) {
        $selected | Select-Object id, task | Format-Table -AutoSize
        Write-Host 'Plan only: no output folder created, model loaded, or process stopped.'
        return
    }

    $listeners = @(Get-NetTCPConnection -LocalPort 7871 -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0 -and -not $StopDemo) {
        throw 'Port 7871 is in use. Close the demo, or use -StopDemo to release its GPU memory.'
    }
    foreach ($demoProcessId in @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
        $demoProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $demoProcessId"
        $commandLine = [string]$demoProcess.CommandLine
        if ($commandLine -notmatch '(?i)-m\s+clipscout\s+serve\b' -or $commandLine -notmatch '--port\s+7871(?:\s|$)') {
            throw 'Port 7871 belongs to another application. No process was stopped.'
        }
        Stop-Process -Id $demoProcessId -ErrorAction Stop
        Write-Host "Stopped ClipScout demo PID $demoProcessId to release GPU memory."
    }

    $runOutputRoot = [IO.Path]::GetFullPath($OutputRoot)
    [IO.Directory]::CreateDirectory($runOutputRoot) | Out-Null
    $runDir = $null
    for ($runNumber = 1; $runNumber -lt 1000000; $runNumber++) {
        $candidate = Join-Path $runOutputRoot ('run_{0:D3}' -f $runNumber)
        try { New-Item -ItemType Directory -Path $candidate -ErrorAction Stop | Out-Null; $runDir = $candidate; break }
        catch { if (Test-Path -LiteralPath $candidate -PathType Container) { continue }; throw }
    }
    if (-not $runDir) { throw 'No free run number.' }
    Copy-Item -LiteralPath $questionPath -Destination (Join-Path $runDir 'questions.original.json')
    Save-RunJson (Join-Path $runDir 'questions.selected.original.json') $selected
    $adaptedPath = Join-Path $runDir 'questions.v6.json'
    Save-RunJson $adaptedPath $adapted
    $datasetDirectory = Split-Path -Parent $questionPath
    $datasetProtocolPath = Join-Path $datasetDirectory 'protocol.json'
    if (Test-Path -LiteralPath $datasetProtocolPath) {
        Copy-Item -LiteralPath $datasetProtocolPath -Destination (Join-Path $runDir 'dataset.protocol.json')
        $referenceDirectory = Join-Path $datasetDirectory 'reference_frames'
        if (Test-Path -LiteralPath $referenceDirectory -PathType Container) {
            Copy-Item -LiteralPath $referenceDirectory -Destination (Join-Path $runDir 'reference_frames') -Recurse
        }
    }
    $sourceFiles = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot 'clipscout') -Recurse -File -Filter '*.py')
    $sourceFiles += @(Get-Item -LiteralPath $PSCommandPath, (Join-Path $projectRoot 'run.ps1'), (Join-Path $projectRoot 'pyproject.toml'))
    $sourceHashes = @{}
    foreach ($source in $sourceFiles) {
        $relative = $source.FullName.Substring($projectRoot.Length).TrimStart([char[]]'\/')
        $destination = Join-Path (Join-Path $runDir 'source') $relative
        [IO.Directory]::CreateDirectory((Split-Path -Parent $destination)) | Out-Null
        Copy-Item -LiteralPath $source.FullName -Destination $destination
        $sourceHashes[$relative] = (Get-FileHash -LiteralPath $source.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $state = @{
        status = 'running'; started_at_utc = [DateTime]::UtcNow.ToString('o'); branch = $branch
        git_commit = (& git -C $projectRoot rev-parse HEAD); agent = 'v6'; modes = $Modes; split = $Split
        planned_predictions = $plannedPredictions
        question_count = $selected.Count; question_ids = @($selected.id); source_sha256 = $sourceHashes
        original_questions_sha256 = (Get-FileHash -LiteralPath $questionPath -Algorithm SHA256).Hash.ToLowerInvariant()
        adapted_questions_sha256 = (Get-FileHash -LiteralPath $adaptedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        adapter = 'native-v6-or-assignment-v7-schema-to-v6-evaluator-v2'
        evaluation_note = 'v6 lexical and citation-geometry proxies only, not semantic grading or exact onset-bracket scoring. Native v6 reference answers and required terms are preserved; assignment-schema claim atoms are concatenated and onset brackets converted only for the evaluator. Exact-match/task-success scores may be restrictive for free-form answers. Review raw answers and citations against the preserved original questions.'
        stop_demo_requested = [bool]$StopDemo
    }
    $statePath = Join-Path $runDir 'launch_manifest.json'
    Save-RunJson $statePath $state
    Write-Host "Run folder: $runDir"
    $runExitCode = 1
    $consolePath = Join-Path $runDir 'console.log'
    $consoleWriter = New-Object System.IO.StreamWriter($consolePath, $false, (New-Object System.Text.UTF8Encoding($false)))
    $consoleWriter.AutoFlush = $true
    $nativeErrorPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5.1 represents native stderr as ErrorRecord objects.
        # Log warnings as well as stdout; the process exit code decides failure.
        $ErrorActionPreference = 'Continue'
        & $runtimePython -u -m clipscout evaluate --index (Join-Path $projectRoot 'data\indexes\user_video.json') --questions $adaptedPath --output $runDir --modes @Modes --split $Split --device cuda --local-files-only --max-new-tokens 768 --frames-per-clip 4 --max-frames 16 --image-size 560 2>&1 | ForEach-Object {
            $line = [string]$_
            $consoleWriter.WriteLine($line)
            Write-Host $line
        }
        $runExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $nativeErrorPreference
        $consoleWriter.Dispose()
        $state['finished_at_utc'] = [DateTime]::UtcNow.ToString('o')
        $state['exit_code'] = $runExitCode
        if ($runExitCode -eq 0) { $state['status'] = 'completed' } else { $state['status'] = 'failed' }
        $modeSummaries = @{}
        $totalSaved = 0
        $totalErrors = 0
        foreach ($mode in $Modes) {
            $predictionsPath = Join-Path $runDir ($mode + '.predictions.json')
            if (Test-Path -LiteralPath $predictionsPath) {
                $savedPredictions = Get-Content -LiteralPath $predictionsPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $modeSaved = @($savedPredictions).Count
                $modeErrors = @($savedPredictions | Where-Object status -eq 'error').Count
                $modeSummaries[$mode] = @{ saved_predictions = $modeSaved; response_errors = $modeErrors }
                $totalSaved += $modeSaved
                $totalErrors += $modeErrors
                Write-Host "${mode}: saved $modeSaved/$($selected.Count); $modeErrors response errors."
            }
        }
        $state['per_mode'] = $modeSummaries
        $state['saved_predictions'] = $totalSaved
        $state['response_errors'] = $totalErrors
        if ($runExitCode -eq 0 -and $totalErrors -gt 0) { $state['status'] = 'completed_with_response_errors' }
        Write-Host "Saved $totalSaved/$plannedPredictions predictions; $totalErrors response errors."
        Save-RunJson $statePath $state
        Write-Host "Outputs: $runDir"
    }
    if ($runExitCode -ne 0) { throw "Evaluation exited with code $runExitCode. See $consolePath" }
} finally {
    foreach ($name in $previousEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process') }
    $OutputEncoding = $previousOutputEncoding
    [Console]::OutputEncoding = $previousConsoleEncoding
}
