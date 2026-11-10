<#
.SYNOPSIS
  Everything to run AFTER the wound outlines have been drawn with outline_wounds.py (guide: docs/wound/07_HUMAN_OUTLINES.md).

.DESCRIPTION
  1. Counts the outlines (test set required, training set optional).
  2. If at least 20 training photos are outlined: fine-tunes the segmenter on them (model unet_v3), with automatic
     resume after a crash and a separate export step.
  3. Compares classical, v1, v2 and v3 on YOUR test outlines (models/wound_seg/compare_incisions_human.json).
  4. If v3 exists: retrains the wound-attribute classifier with crops from v3 (old model backed up first).
  5. Runs the d1 test suite.
  Run from anywhere:  powershell -ExecutionPolicy Bypass -File d1\scripts\after_outlines.ps1
  Add -DryRun to only print what would be run, -SkipClassifier to stop after the comparison.
#>
param(
    [switch]$DryRun,
    [switch]$SkipClassifier,
    [int]$Epochs = 30,
    [int]$MinTrainOutlines = 20
)
$ErrorActionPreference = "Stop"
$D1 = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $D1
$Data = "C:\Users\juba839\surgiguard_data"
$TrainPy = "$Data\venv-train\Scripts\python.exe"                       # GPU torch (training)
$BackPy = Resolve-Path "..\backend\.venv\Scripts\python.exe"            # CLIP + onnxruntime (classifier, tests)
$FUSeg = "$Data\raw\wound-segmentation\data\Foot Ulcer Segmentation Challenge"
$Out = "$Data\outline"
$env:OPENBLAS_NUM_THREADS = "1"; $env:HF_HUB_OFFLINE = "1"; $env:LLM_MODE = "stub"; $env:PYTHONUNBUFFERED = "1"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Run($exe, [string[]]$argv) {
    Write-Host "> $exe $($argv -join ' ')" -ForegroundColor DarkGray
    if ($DryRun) { return 0 }
    & $exe @argv
    return $LASTEXITCODE
}
function Count($dir, $pattern) { if (Test-Path $dir) { (Get-ChildItem $dir -File -Filter $pattern).Count } else { 0 } }

Step "1. Outlines found"
$testImages = Count "$Out\test\images" "*.*"; $testLabels = Count "$Out\test\labels" "*.png"
$trainImages = Count "$Out\train\images" "*.*"; $trainLabels = Count "$Out\train\labels" "*.png"
Write-Host "test : $testLabels / $testImages outlined"
Write-Host "train: $trainLabels / $trainImages outlined"
if ($testLabels -lt $testImages) {
    Write-Host "The TEST set must be complete first (it is what the models are graded on)." -ForegroundColor Yellow
    Write-Host "Run: $TrainPy d1\scripts\outline_wounds.py $Out\test" -ForegroundColor Yellow
    if (-not $DryRun) { exit 1 }
}

$os = Get-CimInstance Win32_OperatingSystem
$freeCommitGB = [math]::Round($os.FreeVirtualMemory / 1MB, 1)
if ($freeCommitGB -lt 5) {
    Write-Host "Only $freeCommitGB GB of memory budget free. Close Docker Desktop and VS Code Java projects first (journal V17)." -ForegroundColor Yellow
}

$v3 = "models\wound_seg\unet_v3.onnx"
if ($trainLabels -ge $MinTrainOutlines) {
    Step "2. Fine-tune the segmenter on your outlines (about 30 min on the GPU)"
    $train = @("scripts\train_segmenter.py", "--train", "$FUSeg\train", "$Out\train", "--weights", "1", "4",
               "--test", "$Out\test", "--init", "models\wound_seg\unet.pt", "--out", "models\wound_seg\unet_v3",
               "--version", "fuseg+surgwound-human-v3", "--epochs", "$Epochs", "--lr", "3e-4", "--batch", "8", "--resume")
    $ok = $false
    for ($i = 1; $i -le 5 -and -not $ok; $i++) {
        $ok = ((Run $TrainPy $train) -eq 0)
        if (-not $ok) { Write-Host "run $i stopped (memory?), resuming from the last epoch..." -ForegroundColor Yellow; Start-Sleep 10 }
    }
    if (-not $DryRun -and -not (Test-Path $v3)) {
        Write-Host "training finished but no ONNX file: exporting separately" -ForegroundColor Yellow
        Run $TrainPy ($train + "--export-only") | Out-Null
    }
} else {
    Step "2. Skipped: fewer than $MinTrainOutlines training photos outlined (models are only compared)"
}

Step "3. Compare all segmenters on YOUR test outlines"
$models = @("models\wound_seg\unet.onnx", "models\wound_seg\unet_v2.onnx")
if ($DryRun -or (Test-Path $v3)) { $models += $v3 }
Run $BackPy (@("scripts\compare_segmenters.py", "--test", "$Out\test", "--models") + $models +
             @("--out", "models\wound_seg\compare_incisions_human.json")) | Out-Null

if (-not $SkipClassifier -and ($DryRun -or (Test-Path $v3))) {
    Step "4. Retrain the wound-attribute classifier with crops from v3 (about 15 min)"
    $backup = "$Data\backup_classifier_$(Get-Date -Format yyyyMMdd_HHmm)"
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force $backup | Out-Null
        Copy-Item models\wound_heads.* , models\wound_model_metrics.json -Destination $backup
        Write-Host "previous classifier backed up in $backup"
    }
    Run $BackPy @("scripts\train_wound_model.py", "$Data\surgwound", "--encoder", "openai/clip-vit-base-patch16",
                  "--segmenter", $v3) | Out-Null
} else {
    Step "4. Classifier unchanged (no v3 model, or -SkipClassifier)"
}

Step "5. Tests"
Run $BackPy @("-m", "pytest", "tests", "-q", "-p", "no:warnings") | Out-Null

Step "Done"
Write-Host "Results: d1\models\wound_seg\compare_incisions_human.json (and unet_v3.json if trained)."
Write-Host "Next: tell Claude 'outlines done' (docs, journal, commit), or follow section 5 of docs/wound/07_HUMAN_OUTLINES.md."
