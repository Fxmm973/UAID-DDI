# reproduce.ps1 — UAID-DDI Full Reproduction Pipeline
param(
    [switch]$SkipTraining = $false,
    [switch]$SkipHashCheck = $false,
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

Write-Host "=== Step 0: Environment Snapshot ===" -ForegroundColor Green
$GitCommit = (git rev-parse HEAD 2>$null) -replace '.*', 'unknown'
Write-Report "Git commit: $GitCommit"
Write-Report "Timestamp: $Timestamp"
Write-Report "Python: $(python --version 2>&1)"

Write-Host "=== Step 1: Negative Manifests ===" -ForegroundColor Green
try {
    python shared/neg_manifest.py --dataset PharDDIE/dataset1
    python shared/neg_manifest.py --dataset EviDDIE/dataset1
    Write-Report "PASS: Negative manifests generated"
} catch {
    Write-Report "FAIL: Negative manifests - $_"
}

Write-Host "=== Step 2: Prediction CSVs ===" -ForegroundColor Green
try { python PharDDIE/pharddie_export_full.py
    Write-Report "PASS: PharDDIE full export" } catch { Write-Report "FAIL: PharDDIE - $_" }
try { python PharDDIE/pharddie_export.py
    Write-Report "PASS: PharDDIE w/o uncertainty" } catch { Write-Report "FAIL: PharDDIE w/o - $_" }
try { python EviDDIE/eviddie_export_zs_v2.py
    Write-Report "PASS: EviDDIE zero-shot" } catch { Write-Report "FAIL: EviDDIE - $_" }

Write-Host "=== Step 3: Paper Tables ===" -ForegroundColor Green
try { python PharDDIE/pharddie_table2.py
    Write-Report "PASS: Table 2" } catch { Write-Report "FAIL: Table 2 - $_" }
try { python PharDDIE/pharddie_table3.py
    Write-Report "PASS: Table 3" } catch { Write-Report "FAIL: Table 3 - $_" }
try { python PharDDIE/pharddie_table4_paper.py
    Write-Report "PASS: Table 4" } catch { Write-Report "FAIL: Table 4 - $_" }
try { python EviDDIE/eviddie_case_study.py
    Write-Report "PASS: Table 5" } catch { Write-Report "FAIL: Table 5 - $_" }

Write-Host "=== REPRODUCTION COMPLETE ===" -ForegroundColor Green
Write-Report "Full report: $ReportPath"
Write-Report "Audit logs: $LogDir/"
