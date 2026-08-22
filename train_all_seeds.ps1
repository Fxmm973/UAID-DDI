# train_all_seeds.ps1 — 五种子训练 orchestrator（PharDDIE 1/5-shot + EviDDIE 0-shot）
# 每个 checkpoint 保存到导出器要求的确切路径：
#   PharDDIE:  models/dataset1/models_drugbank_{1,5}shot_str_seed{seed}/bestmodel
#   EviDDIE:   models/dataset1/eviddie_0shot_seed{seed}/bestmodel{,_G}
param(
    [string]$GpuId = "0",
    [int[]]$Seeds = @(19940419, 20230801, 20240115, 20240520, 20240910),
    [int[]]$Shots = @(1, 5),
    [int]$MaxBatches = 40000,
    [switch]$SkipPharDDIE,
    [switch]$SkipEviDDIE
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot
$Env:CUDA_VISIBLE_DEVICES = $GpuId

function Run-Cmd {
    param([string]$Name, [string]$WorkingDir, [string[]]$Cmd)
    Write-Host "=== $Name ===" -ForegroundColor Green
    Push-Location (Join-Path $RepoRoot $WorkingDir)
    & $Cmd[0] @($Cmd[1..($Cmd.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Host "ABORTING: $Name failed (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
    Pop-Location
    Write-Host "PASS: $Name" -ForegroundColor Green
}

# ---- PharDDIE 1/5-shot，五种子 ----
if (-not $SkipPharDDIE) {
    foreach ($shot in $Shots) {
        foreach ($seed in $Seeds) {
            $prefix = "dataset1/models_drugbank_${shot}shot_str_seed${seed}"
            $ckpt = "models/dataset1/models_drugbank_${shot}shot_str_seed${seed}/bestmodel"
            if (Test-Path $ckpt) {
                Write-Host "[SKIP] $ckpt already exists" -ForegroundColor Yellow
                continue
            }
            Run-Cmd "PharDDIE ${shot}-shot seed ${seed}" "PharDDIE" @(
                "python", "pharddie_trainer.py", "--dataset", "dataset1",
                "--prefix", $prefix, "--seed", "$seed", "--few", "$shot",
                "--train_few", "$shot", "--max_batches", "$MaxBatches")
        }
    }
}

# ---- EviDDIE 0-shot，五种子 ----
if (-not $SkipEviDDIE) {
    foreach ($seed in $Seeds) {
        $ckpt = "models/dataset1/eviddie_0shot_seed${seed}/bestmodel"
        if (Test-Path $ckpt) {
            Write-Host "[SKIP] $ckpt already exists" -ForegroundColor Yellow
            continue
        }
        Run-Cmd "EviDDIE 0-shot seed ${seed}" "EviDDIE" @(
            "python", "eviddie_trainer.py", "--dataset", "dataset1",
            "--prefix", "dataset1/eviddie_0shot", "--seed", "$seed",
            "--max_batches", "20000")
    }
}

Write-Host "=== TRAINING COMPLETE ===" -ForegroundColor Green
