param(
    [string]$VideoPath,
    [ValidateSet('dev','test','all')][string]$Split = 'test',
    [string]$OutputDirectory = 'results/reproduction'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$source = Get-Content -LiteralPath (Join-Path $projectRoot 'data\source.json') -Raw | ConvertFrom-Json
if (-not $VideoPath) {
    $VideoPath = Join-Path $projectRoot $source.video_path
}
$VideoPath = (Resolve-Path -LiteralPath $VideoPath).Path
# This script reproduces the frozen case study. VideoPath supports relocation,
# not substituting a new video while retaining the old transcript and labels.
$videoHash = (Get-FileHash -LiteralPath $VideoPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($videoHash -ne $source.sha256) {
    throw 'Source video SHA-256 differs from data/source.json. Prepare separate annotations and index for a new video.'
}
Push-Location -LiteralPath $projectRoot
try {
    & .\run.ps1 index --video "$VideoPath" --subtitles data/transcripts/user_video.vtt --captions data/captions/user_video.json --output data/indexes/user_video.json
    if ($LASTEXITCODE -ne 0) { throw 'Indexing failed' }
    & .\run.ps1 evaluate --index data/indexes/user_video.json --questions data/questions.json --split $Split --output $OutputDirectory --max-new-tokens 384 --image-size 560 --local-files-only
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
} finally { Pop-Location }
