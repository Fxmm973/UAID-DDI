# reproduce_from_training.ps1 — 从训练开始的完整复现入口
# 流程：五种子训练（train_all_seeds.ps1）-> 逐样本导出 -> 表格重算（result_regeneration.ps1）。
# 任何一步失败立即终止。需要全部 checkpoint；训练耗时约数天（RTX 4090）。
param(
    [string]$GpuId = "0",
    [int[]]$Seeds = @(19940419, 20230801, 20240115, 20240520, 20240910)
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$Env:CUDA_VISIBLE_DEVICES = $GpuId

function Run-Script {
    param([string]$Name, [string]$Script, [string[]]$Extra)
    Write-Host "=== $Name ===" -ForegroundColor Green
    & powershell -NoProfile -ExecutionPolicy Bypass -File $Script @Extra
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ABORTING: $Name failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# Step 1: 训练（已有 checkpoint 自动跳过）
Run-Script "Step 1: train five seeds (PharDDIE 1/5-shot + EviDDIE 0-shot)" `
    (Join-Path $RepoRoot "train_all_seeds.ps1") @("-GpuId", $GpuId)

# Step 2: 导出逐样本预测（固定 manifest，fail closed）
function Run-Step {
    param([string]$StepName, [string]$Command, [string]$WorkingDir)
    Write-Host "=== $StepName ===" -ForegroundColor Green
    if ($WorkingDir) { Push-Location (Join-Path $RepoRoot $WorkingDir) }
    Invoke-Expression $Command
    $StepExit = $LASTEXITCODE
    if ($WorkingDir) { Pop-Location }
    if ($StepExit -ne 0) {
        Write-Host "ABORTING: $StepName failed (exit code $StepExit)" -ForegroundColor Red
        exit 1
    }
    Write-Host "PASS: $StepName" -ForegroundColor Green
}

Run-Step "Step 2a: PharDDIE full export" "python pharddie_export_full.py" -WorkingDir "PharDDIE"
Run-Step "Step 2b: PharDDIE w/o uncertainty export" "python pharddie_export.py" -WorkingDir "PharDDIE"
Run-Step "Step 2c: EviDDIE zero-shot export" "python eviddie_export_zs_v2.py" -WorkingDir "EviDDIE"

# Step 3: 表格重算（复用 result_regeneration.ps1）
Run-Script "Step 3: regenerate all tables and audits" `
    (Join-Path $RepoRoot "result_regeneration.ps1") @("-GpuId", $GpuId)

Write-Host "=== FULL REPRODUCTION COMPLETE ===" -ForegroundColor Green
