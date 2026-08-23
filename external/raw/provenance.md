# Provenance and build steps

## 1. Identifier canonicalization
- Resolve drug mentions to InChIKey14 (IK14) identifiers.
- Keep preferred generic names for display; do not redistribute proprietary synonym lists.

## 2. FAERS ingestion and normalization
- Parse quarterly FAERS DRUG and REAC files.
- Normalize name tokens; build case-level drug sets.
- Enumerate co-medication pairs; join to MedDRA PTs by code.

## 3. Pair–PT contingency and disproportionality
- Build 2x2 tables (a,b,c,d); apply +0.5 continuity correction.
- Compute PRR, ROR, and 95% lower confidence limit of ROR.

## 4. Roll-ups
- Loose: maximize per pair across PTs.
- Strict: floors a_raw>=3, pair>=10, pt>=10; keep per-pair maxima and PT coverage.

## 5. Public subset sampling (50k)
- Stratify by lower-bound bins, PT coverage, and ATC class diversity.
- Deduplicate by sorted IK14 pair key.

## 6. Export
- Emit ddi_pairs_50k.csv, schema.sql, audits, and this documentation.

## 7. Integrity
- All files UTF-8; SHA-256 digests in checksums.txt.
