# reproduce.ps1 — UAID-DDI Full Reproduction Pipeline
# 任何一步失败（非零退出码）立即终止整个流水线，绝不继续生成表格。
param(
    [switch]$SkipTraining = $false,
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
$ReportPath = Join-Path $AuditDir "reproducibility_report_$Timestamp.txt"

function Write-Report {
    param([string]$Line)
    Add-Content -Path $ReportPath -Value $Line
    Write-Host $Line
}

function Run-Step {
    param([string]$StepName, [string]$Command)
    Write-Host "=== $StepName ===" -ForegroundColor Green
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Report "FAIL: $StepName (exit code $LASTEXITCODE)"
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

# Step 1: 校验固定负样本 manifest（SHA256 + 条目数量），不一致立即终止
Run-Step "Step 1a: Verify PharDDIE manifests (SHA256 + entry counts)" "python shared/verify_manifests.py --hash-log PharDDIE/dataset1/neg_manifests/manifest_hashes.json --manifest-dir PharDDIE/dataset1/neg_manifests --dataset PharDDIE/dataset1"
Run-Step "Step 1b: Verify EviDDIE manifests (SHA256 + entry counts)" "python shared/verify_manifests.py --hash-log EviDDIE/neg_manifests/manifest_hashes.json --manifest-dir EviDDIE/neg_manifests --dataset EviDDIE/dataset1"
Run-Step "Step 1c: Six-part leakage audit" "python shared/audit_leakage.py --dataset PharDDIE/dataset1"

# Step 2: 导出逐样本预测（全部直接读取固定 manifest）
Run-Step "Step 2a: PharDDIE full export (manifest-based)" "python PharDDIE/pharddie_export_full.py"
Run-Step "Step 2b: PharDDIE w/o uncertainty export (manifest-based)" "python PharDDIE/pharddie_export.py"
Run-Step "Step 2c: EviDDIE zero-shot export (manifest-based)" "python EviDDIE/eviddie_export_zs_v2.py"

# Step 3: 论文表格
Run-Step "Step 3a: Table 2" "python PharDDIE/pharddie_table2.py"
Run-Step "Step 3b: Table 3" "python PharDDIE/pharddie_table3_complete.py"
Run-Step "Step 3c: Table 4" "python PharDDIE/pharddie_table4_paper.py"
Run-Step "Step 3d: Table 5" "python EviDDIE/eviddie_case_study.py"

Write-Host "=== REPRODUCTION COMPLETE ===" -ForegroundColor Green
Write-Report "Full report: $ReportPath"
Write-Report "Audit logs: $LogDir/"
