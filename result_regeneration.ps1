# result_regeneration.ps1 — 从已发布的预测 CSV 重算论文全部表格与审计
# 不训练、不导出；任何一步失败（非零退出码）立即终止。
# 从训练开始的全链路复现见 reproduce_from_training.ps1。
param(
    [string]$GpuId = "0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$Env:CUDA_VISIBLE_DEVICES = $GpuId

$AuditDir = Join-Path $RepoRoot "audit"
$LogDir = Join-Path $AuditDir "table_logs"
$PredDir = Join-Path $AuditDir "predictions"
New-Item -ItemType Directory -Force -Path $LogDir, $PredDir | Out-Null

$Timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$ReportPath = Join-Path $AuditDir "regeneration_report_$Timestamp.txt"

function Write-Report {
    param([string]$Line)
    Add-Content -Path $ReportPath -Value $Line
    Write-Host $Line
}

function Run-Step {
    param([string]$StepName, [string]$Command, [string]$WorkingDir)
    Write-Host "=== $StepName ===" -ForegroundColor Green
    if ($WorkingDir) {
        Push-Location (Join-Path $RepoRoot $WorkingDir)
    }
    Invoke-Expression $Command
    $StepExit = $LASTEXITCODE
    if ($WorkingDir) {
        Pop-Location
    }
    if ($StepExit -ne 0) {
        Write-Report "FAIL: $StepName (exit code $StepExit)"
        Write-Host "ABORTING: $StepName failed. Pipeline stopped." -ForegroundColor Red
        exit 1
    }
    Write-Report "PASS: $StepName"
}

Write-Host "=== Step 0: Environment Snapshot ===" -ForegroundColor Green
$GitCommit = (git rev-parse HEAD 2>$null)
Write-Report "Git commit: $GitCommit"
Write-Report "Timestamp: $Timestamp"
Write-Report "Python: $(python --version 2>&1)"

# Step 1: 审计（manifest SHA256 + 泄漏审计 + sanitized graph）
Run-Step "Step 1a: Verify PharDDIE manifests" "python shared/verify_manifests.py --hash-log PharDDIE/dataset1/neg_manifests/manifest_hashes.json --manifest-dir PharDDIE/dataset1/neg_manifests --dataset PharDDIE/dataset1"
Run-Step "Step 1b: Verify EviDDIE manifests" "python shared/verify_manifests.py --hash-log EviDDIE/neg_manifests/manifest_hashes.json --manifest-dir EviDDIE/neg_manifests --dataset EviDDIE/dataset1"
Run-Step "Step 1c: Build sanitized path graph" "python shared/build_sanitized_path_graph.py --dataset PharDDIE/dataset1"
Run-Step "Step 1c-2: Build sanitized path graph (EviDDIE)" "python shared/build_sanitized_path_graph.py --dataset EviDDIE/dataset1"
Run-Step "Step 1d: Six-part leakage audit" "python shared/audit_leakage.py --dataset PharDDIE/dataset1"

# Step 2: 论文表格（全部从已发布 CSV 重算）
Run-Step "Step 2a: Table 2 (few-shot main results)" "python pharddie_table2.py" -WorkingDir "PharDDIE"
Run-Step "Step 2b: Table 3 zero-shot rows (canonical)" "python ../shared/calibration_table.py --csv results/predictions/predictions_eviddie_new_ablation.csv --methods EviDDIE 'Softmax baseline' 'EviDDIE w/o EVI' 'EviDDIE w/o BSA' --out results/calibration_table_variants.csv --fig results/reliability_variants.png" -WorkingDir "EviDDIE"
Run-Step "Step 2c: Table 3 frozen-EDL-head rows" "python ../shared/calibration_table.py --csv results/predictions/predictions_evi_full_frozen.csv --methods 'EviDDIE (frozen EDL head)' --out results/calibration_table_evi_full.csv --fig results/reliability_evi_full.png" -WorkingDir "EviDDIE"
Run-Step "Step 2d: Table 3 PharDDIE rows (1-shot)" "python ../shared/calibration_table.py --csv results/predictions/predictions_dataset1_PharDDIE.csv --methods PharDDIE --settings rare --shot 1 --out results/validation/table3_pharddie_1shot.csv --fig results/validation/reliability_pharddie_1shot.png" -WorkingDir "PharDDIE"
Run-Step "Step 2e: Table 3 PharDDIE rows (5-shot)" "python ../shared/calibration_table.py --csv results/predictions/predictions_dataset1_PharDDIE.csv --methods PharDDIE --settings rare --shot 5 --out results/validation/table3_pharddie_5shot.csv --fig results/validation/reliability_pharddie_5shot.png" -WorkingDir "PharDDIE"
Run-Step "Step 2f: Audit real evaluation episodes" "python shared/audit_leakage.py --dataset PharDDIE/dataset1 --episode-manifests PharDDIE/results/predictions/episode_manifests"

Write-Host "=== TABLE REGENERATION COMPLETE ===" -ForegroundColor Green
Write-Report "Full report: $ReportPath"
Write-Report "Audit logs: $LogDir/"
