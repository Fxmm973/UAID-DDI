#!/usr/bin/env bash
# Robust resumable downloader for BioSentVec_PubMed_MIMICIII-bigram_d700.bin
# (22,475,736,490 bytes, split into 6 append-resumable range chunks).
# Each part resumes at its current size; curl failures are retried until the
# part reaches its exact expected length. Safe to run at any point: ranges are
# exact and appends are idempotent (offset == current file size).
set -u
cd "/d/PharDDIE and EviDDIE/PharDDIE_github_8_10/external/biosentvec" || exit 1
TOTAL=22475736490
N=6
CHUNK=$((TOTAL / N))
URL="https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
MAX_ATTEMPTS=200

for i in 0 1 2 3 4 5; do
  S=$((i * CHUNK))
  if [ "$i" -eq $((N - 1)) ]; then E=$((TOTAL - 1)); else E=$(((i + 1) * CHUNK - 1)); fi
  EXP=$((E - S + 1))
  attempt=0
  while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    ACT=$(stat -c%s part_$i 2>/dev/null || echo 0)
    [ "$ACT" -ge "$EXP" ] && break
    attempt=$((attempt + 1))
    echo "[resume] part_$i attempt=$attempt ACT=$ACT EXP=$EXP ($((100 * ACT / EXP))%)"
    curl -s --connect-timeout 30 --max-time 1500 -r $((S + ACT))-$E -o - "$URL" >> part_$i || true
  done
  ACT=$(stat -c%s part_$i 2>/dev/null || echo 0)
  if [ "$ACT" = "$EXP" ]; then echo "part_$i OK ($ACT)"; else echo "part_$i SHORT ($ACT/$EXP)"; fi
done
echo "RESUME_DONE total=$(stat -c%s part_0 2>/dev/null || echo 0),$(stat -c%s part_1 2>/dev/null || echo 0),$(stat -c%s part_2 2>/dev/null || echo 0),$(stat -c%s part_3 2>/dev/null || echo 0),$(stat -c%s part_4 2>/dev/null || echo 0),$(stat -c%s part_5 2>/dev/null || echo 0)"
