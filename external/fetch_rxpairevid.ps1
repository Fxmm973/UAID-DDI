# external/fetch_rxpairevid.ps1 — RxPairEvid-50K 完整性校验（下载由用户手动完成）
$ErrorActionPreference = "Stop"
$Raw = Join-Path $PSScriptRoot "raw"
$Required = @("ddi_pairs_50k.csv", "codebook.md", "provenance.md", "checksums.txt")
$Optional = @("LICENSE.txt", "README.md", "schema.sql",
              "audit_subset_signal_quantiles.csv", "audit_subset_strata_counts.csv")

foreach ($f in $Required) {
    if (-not (Test-Path (Join-Path $Raw $f))) { throw "MISSING required file: $f (place it under external/raw/)" }
}
Push-Location $Raw
try {
    $out = certutil -hashfile "ddi_pairs_50k.csv" SHA256 | Select-String -Pattern "[0-9A-Fa-f]{64}"
    $actual = $out.Line.Trim().ToLower()
    $expected = (Select-String -Path "checksums.txt" -Pattern "ddi_pairs_50k.csv").Line.Split(" ")[0].ToLower()
    if ($actual -ne $expected) { throw "SHA256 mismatch for ddi_pairs_50k.csv" }
    Write-Host "PASS: ddi_pairs_50k.csv SHA256 verified."
} finally { Pop-Location }

foreach ($f in $Optional) {
    if (-not (Test-Path (Join-Path $Raw $f))) { Write-Warning "Optional file missing: $f (copy from Mendeley zip)" }
}
Write-Host "DONE."
