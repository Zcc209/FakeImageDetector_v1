param(
  [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

$modelsDir = Join-Path $ProjectRoot 'models'
$scrfdDir = Join-Path $modelsDir 'scrfd'
$truforDir = Join-Path $modelsDir 'trufor'
$truforPretrainedDir = Join-Path $truforDir 'pretrained_models'

New-Item -ItemType Directory -Path $scrfdDir -Force | Out-Null
New-Item -ItemType Directory -Path $truforPretrainedDir -Force | Out-Null

# SCRFD (insightface buffalo_l)
$scrfdZip = Join-Path $scrfdDir 'buffalo_l.zip'
$scrfdExtractDir = Join-Path $scrfdDir 'models'
$scrfdUrl = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'

Write-Host "[1/2] Downloading SCRFD weights..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $scrfdUrl -OutFile $scrfdZip

Write-Host "Extracting SCRFD weights to $scrfdExtractDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Path $scrfdExtractDir -Force | Out-Null
Expand-Archive -Path $scrfdZip -DestinationPath $scrfdExtractDir -Force

# TruFor
$truforZip = Join-Path $truforDir 'TruFor_weights.zip'
$truforUrl = 'https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip'

Write-Host "[2/2] Downloading TruFor weights..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $truforUrl -OutFile $truforZip

Write-Host "Extracting TruFor weights to $truforDir" -ForegroundColor Cyan
Expand-Archive -Path $truforZip -DestinationPath $truforDir -Force

# normalize expected file path
$found = Get-ChildItem -Path $truforDir -Recurse -File -Filter 'trufor.pth.tar' | Select-Object -First 1
if (-not $found) {
  throw "Cannot find trufor.pth.tar after extraction."
}
$targetModel = Join-Path $truforPretrainedDir 'trufor.pth.tar'
Copy-Item -Path $found.FullName -Destination $targetModel -Force

Write-Host "\nDone." -ForegroundColor Green
Write-Host "SCRFD root: $scrfdDir" -ForegroundColor Green
Write-Host "TruFor model: $targetModel" -ForegroundColor Green
