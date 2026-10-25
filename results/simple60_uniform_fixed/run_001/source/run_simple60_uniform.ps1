param(
    [string]$OutputRoot = (Join-Path $PSScriptRoot 'results\simple60_uniform_fixed'),
    [ValidateRange(1, 2048)][int]$MaxNewTokens = 768,
    [switch]$StopDemo,
    [switch]$PlanOnly
)
$ErrorActionPreference = 'Stop'
$questionPath = Join-Path $PSScriptRoot 'data\simple60\questions.json'
$protocolPath = Join-Path $PSScriptRoot 'data\simple60\protocol.json'
$protocol = Get-Content -LiteralPath $protocolPath -Raw -Encoding UTF8 | ConvertFrom-Json
$questionHash = (Get-FileHash -LiteralPath $questionPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($questionHash -ne $protocol.questions_sha256) { throw 'The frozen simple60 question file differs. No experiment was started.' }
$questions = Get-Content -LiteralPath $questionPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($questions.Count -ne 60 -or @($questions | Where-Object split -ne 'test').Count -ne 0) {
    throw 'Expected 60 test questions. No experiment was started.'
}
foreach ($task in @('timestamp_search', 'reasoning', 'onset', 'unanswerable')) {
    if (@($questions | Where-Object task -eq $task).Count -ne 15) { throw "Expected fifteen $task questions." }
}
foreach ($reference in $protocol.reference_frame_sha256.PSObject.Properties) {
    $framePath = Join-Path $PSScriptRoot $reference.Name
    if ((Get-FileHash -LiteralPath $framePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $reference.Value) {
        throw "Reference frame differs: $($reference.Name). No experiment was started."
    }
}
$launchOptions = @{
    QuestionsPath = $questionPath
    OutputRoot = $OutputRoot
    Modes = @('uniform')
    Split = 'test'
    MaxNewTokens = $MaxNewTokens
    InvocationScript = $PSCommandPath
    StopDemo = $StopDemo
    PlanOnly = $PlanOnly
}
& (Join-Path $PSScriptRoot 'run_v6_adaptive.ps1') @launchOptions
