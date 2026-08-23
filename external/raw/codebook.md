
# RxPairEvid-50K Codebook


> Public, license-clean 50,000-row subset for modeling and audit.  
> **DOI:** [INSERT DOI BEFORE SUBMISSION]  
> **License (recommended):** CC BY 4.0 for original work

---

## 1. Scope of the public subset

This CSV contains **one row per drug–drug pair** canonicalized to the **InChIKey-14** stem (IK14).  
Evidence fields are **derived from FAERS** under a **strict** quality regime. To stay license‑clean, we provide **MedDRA PT codes** (not the term strings) and **do not** redistribute payloads from DrugBank, KEGG, PDB/PDBbind, SIDER, or similar third‑party sources.

**File:** `ddi_pairs_50k.csv` (UTF‑8, header present)

---

## 2. Column dictionary

| Column | Type | Nulls | Description | Example |
|---|---|---:|---|---|
| `drug_a_ik14` | CHAR(14) | no | InChIKey connection-layer stem (first 14 chars) for drug A. CSV is canonicalized so `drug_a_ik14 <= drug_b_ik14` lexicographically. | `AAAAAA BBBBBB C` |
| `drug_b_ik14` | CHAR(14) | no | IK14 for drug B. | `CCCCCC DDDDDD E` |
| `a_name` | TEXT | no | Human‑readable preferred generic name for drug A. Authored by us to be license‑clean. | `metformin` |
| `b_name` | TEXT | no | Human‑readable preferred generic name for drug B. | `cefuroxime` |
| `pair_id` | TEXT | no | Deterministic key for the pair: `LEAST(A,B) || '::' || GREATEST(A,B)` where A,B are IK14s. | `AAAAAA...::CCCCCC...` |
| `n_faers_reports` | INTEGER | no | Total FAERS reports contributing to the pair evidence in this release. | `27` |
| `faers_prr_max_strict` | DOUBLE | yes | Maximum PRR across MedDRA PTs that pass strict floors. | `4.21` |
| `faers_ror95_lcl_max_strict` | DOUBLE | yes | Maximum **lower 95% CI** of ROR across strict PTs, after +0.5 continuity correction. | `2.13` |
| `faers_pt_covered_strict` | INTEGER | no | Count of distinct MedDRA PTs meeting strict floors for this pair. | `5` |
| `faers_best_pt_code_strict` | TEXT (digits) | yes | **MedDRA PT code** (numeric string) for the PT yielding the max lower‑bound ROR under strict floors. Term text is not redistributed. | `10020772` |

### 2.1 Types and parsing tips
- Use **strings** for IK14 columns in data frames to avoid trimming leading characters.
- `faers_best_pt_code_strict` is **numeric string**; don’t coerce to integer if you plan to join to MedDRA lookups that ship codes as strings.
- `faers_prr_max_strict` and `faers_ror95_lcl_max_strict` can be `NULL` when no strict PT passed the floors.

---

## 3. Evidence construction (strict regime)

For each pair and each MedDRA Preferred Term (PT), we build a 2×2 table of FAERS counts: `a, b, c, d`. We apply **Haldane–Anscombe +0.5 continuity correction**:

- PRR = (a'/(a'+b')) / (c'/(c'+d'))  
- ROR = (a'/b') / (c'/d')  
- SE[ln ROR] = sqrt(1/a' + 1/b' + 1/c' + 1/d')  
- **ROR95_LCL** = exp( ln ROR − 1.96 × SE )

**Strict floors** per PT:  
- `a_raw ≥ 3` (uncorrected)  
- Pair‑support and PT‑support minimums as specified in the provenance (see `provenance.md`).

Per‑pair roll‑ups (strict):
- `faers_prr_max_strict` = max PRR over strict PTs.  
- `faers_ror95_lcl_max_strict` = max lower 95% CI of ROR over strict PTs.  
- `faers_pt_covered_strict` = number of strict PTs.  
- `faers_best_pt_code_strict` = PT code that achieved `faers_ror95_lcl_max_strict`.

---

## 4. Sampling to the 50k subset

The public subset is a **stratified sample** from the strict matrix. Strata balance:  
- strength bins of `faers_ror95_lcl_max_strict`,  
- `faers_pt_covered_strict`,  
- diversity across major ATC/therapeutic classes (approximated via IK14‑to‑name mapping),  
- and report support (`n_faers_reports`).

Sampling avoids near‑duplicates by relying on deterministic `pair_id` and ensures high‑support pairs are represented without dominating the file.

---

## 5. Integrity rules you can rely on

- **Canonical order:** Every row satisfies `drug_a_ik14 <= drug_b_ik14`.  
- **Pair key:** `pair_id` equals `LEAST(drug_a_ik14,drug_b_ik14) || '::' || GREATEST(drug_a_ik14,drug_b_ik14)`.  
- **Non‑negative counts:** `n_faers_reports >= 0`, `faers_pt_covered_strict >= 0`.  
- **Signals non‑negative when present:** `faers_prr_max_strict >= 0`, `faers_ror95_lcl_max_strict >= 0`.  
- **PT code format:** if present, `faers_best_pt_code_strict` matches `^[0-9]+$`.

These constraints are mirrored in `schema.sql` to support validation on load.

---

## 6. Quick starts

### 6.1 PostgreSQL
```sql
-- Create schema/table, then load CSV
-- \i schema.sql
-- \copy rxpairevid.ddi_pairs_50k FROM '/path/to/ddi_pairs_50k.csv' CSV HEADER;

-- Sanity summary
SELECT * FROM rxpairevid.v_pairs_50k_summary;

-- Top 50 pairs by strict lower bound
SELECT a_name, b_name, faers_ror95_lcl_max_strict
FROM rxpairevid.ddi_pairs_50k
WHERE faers_ror95_lcl_max_strict IS NOT NULL
ORDER BY faers_ror95_lcl_max_strict DESC
LIMIT 50;
```

### 6.2 Python (pandas)
```python
import pandas as pd
df = pd.read_csv("ddi_pairs_50k.csv", dtype={
    "drug_a_ik14": "string",
    "drug_b_ik14": "string",
    "faers_best_pt_code_strict": "string"
})
df["pair_id_check"] = df.apply(
    lambda r: "::".join(sorted([r["drug_a_ik14"], r["drug_b_ik14"]])), axis=1)
assert (df["pair_id"] == df["pair_id_check"]).all()
```

---

## 7. What is **not** in the public file

Excluded to remain license‑clean and compact:
- **MedDRA term text** (we provide only **PT codes**).
- DrugBank/KEGG/PDB/PDBbind/SIDER payloads or derived overlaps.
- Per‑source raw tables or verbatim case records.

The `schema.sql` documents **attachment points** for re‑creating these joins if you have legitimate access to the original sources.

---

## 8. Recommended usage patterns

- Treat `faers_ror95_lcl_max_strict` and the PT code as **evidence**, not labels.  
- When splitting data, group by IK14s to reduce leakage from synonyms.  
- Consider log transforms for heavy‑tailed `n_faers_reports`.  
- Use class‑imbalance‑aware evaluation if you binarize evidence into labels.

---

## 9. Provenance and reproducibility

See `provenance.md` for:
- FAERS ingestion and name normalization
- PT‑level contingency construction
- Continuity correction and interval formulas
- Strict floor values and their justifications
- Sampling procedure for the 50k subset
- Versioning and checksums

---

## 10. Citation

If you use this dataset, please cite the Data in Brief article and the repository DOI once available. Also cite the underlying sources you reconnect (e.g., FAERS, MedDRA, DrugBank, KEGG, STRING/STITCH, SIDER, PDB/PDBbind, LINCS).
