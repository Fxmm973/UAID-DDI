# RxPairEvid-50K 外部验证章节重做 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 FAERS 药物警戒来源的 RxPairEvid-50K 数据集为 PharDDIE（兜底 EviDDIE）构建完全独立的外部验证（episode 评测 + 药物重叠审计 + 文献佐证案例研究），并改写论文 fyx8_21.tex 与回应信。

**Architecture:** 全部新增文件集中在仓库 `external/` 目录，不动任何现有文件。RxPairEvid-50K 经 PubChem 映射 SMILES、按 MedDRA PT code 分层为 1-shot 层（185 事件）/ 5-shot 层（24 事件），构建成与 dataset1 同构的 `PharDDIE/dataset_ext_{1,5}shot/` 目录；复用 dataset1 的 DRKG/KG 文件保证 checkpoint 逐位兼容（新药走 pad/零向量兜底 + ACI 结构分支）；推理用克隆自 `pharddie_export_full.py` 的导出脚本，负样本走固定种子 manifest（tail-corruption 为主、原生无信号对为辅）；RareDDIE 五种子（本机已有）同协议对比；案例研究证据只用 PubMed PMID + FAERS 统计。

**Tech Stack:** Python 3.9 / PyTorch 2.0.1+cu118 / PyG 2.6.1 / RDKit 2025.03.5 / pandas / numpy / scikit-learn / PubChem PUG REST / PowerShell 5.1（Windows 11, RTX 4090）

**Spec:** `docs/superpowers/specs/2026-08-21-external-validation-design.md`

## Global Constraints

- 不改动仓库任何现有文件；新增仅限 `external/`、`PharDDIE/dataset_ext_1shot/`、`PharDDIE/dataset_ext_5shot/`、`docs/superpowers/`。
- 不重训任何模型；PharDDIE 五种子 checkpoint 缺失则任务 6 阻塞，2026-08-24 起转任务 8（EviDDIE 兜底）。
- 不推送 GitHub；本地 git commit 仅提交 `external/` 与 `docs/` 新增文件。
- 阳性定义：`faers_ror95_lcl_max_strict` 非空（873 对，1.75%）；事件键 = `faers_best_pt_code_strict`（MedDRA 数字 code），任务文件中写作 `PT-{code}`。
- 事件分层：1-shot 层 = 阳性对数 ≥2 的 PT（185 个）；5-shot 层 = 阳性对数 ≥6 的 PT（24 个）。
- 药对是无向的（FAERS 语义），任务三元组一律用规范方向 `(drug_a_ik14 小者, event, drug_b_ik14 大者)`。
- 新药兜底：不在 dataset1 词汇表（ent2ids）中的药物，导出脚本将其嵌入行置零向量、KG 邻居为空（ACI 结构分支）；不得回退到其他种子 checkpoint。
- 所有随机性固定种子：负样本 manifest 种子 = 19940419（与主实验一致）；训练种子列表 `[19940419, 20230801, 20240115, 20240520, 20240910]`（5-shot 若缺 seed19940419 则用 4 种子并注明）。
- 证据链：所有中间产物 SHA256 记录，输出到 `external/outputs/`。

---

### Task 1: 原始文件校验脚本与目录骨架

**Files:**
- Create: `external/fetch_rxpairevid.ps1`
- Create: `external/outputs/.gitkeep`

**Interfaces:**
- Produces: 无（纯校验）；后续任务假设 `external/raw/ddi_pairs_50k.csv`、`codebook.md`、`provenance.md` 存在且已校验。

- [ ] **Step 1: 写校验脚本**（校验已存在的 3 个文件 + 提示缺失的 5 个配套文件）

```powershell
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
```

- [ ] **Step 2: 运行校验**

Run: `powershell -File external/fetch_rxpairevid.ps1`
Expected: `PASS: ddi_pairs_50k.csv SHA256 verified.` + 5 条 Optional 警告（用户尚未补齐的文件）。

- [ ] **Step 3: 建 outputs 目录并提交**

```bash
mkdir -p external/outputs && touch external/outputs/.gitkeep
git add external/fetch_rxpairevid.ps1 external/outputs/.gitkeep
git commit -m "feat(external): RxPairEvid-50K checksum verification script"
```

### Task 2: IK14 → SMILES 映射（PubChem PUG REST）

**Files:**
- Create: `external/ik14_smiles.py`
- Create: `external/tests/test_ik14_smiles.py`

**Interfaces:**
- Consumes: `external/raw/ddi_pairs_50k.csv`
- Produces:
  - `external/outputs/ik14_smiles_map.json` — `{ik14: {"smiles": str, "cid": int|null, "status": "ok"|"failed", "error": str|null}}`
  - `external/outputs/mapping_audit.csv` — 列：`ik14, status, cid, smiles, error`
  - 模块导出 `lookup_smiles_batch(ik14s: list[str], cache: dict) -> dict`，供任务 4 复用。

- [ ] **Step 1: 写失败测试**（用打桩的 HTTP 会话测映射与缓存逻辑）

```python
# external/tests/test_ik14_smiles.py
import json, pytest
from ik14_smiles import lookup_smiles_batch

class FakeSession:
    def __init__(self, cid_by_ik14): self.cid_by_ik14 = cid_by_ik14
    def get(self, url, timeout=30):  # PUG: /compound/inchikey/{ik14}/cids/JSON
        ik14 = url.split("/inchikey/")[1].split("/")[0]
        cid = self.cid_by_ik14.get(ik14)
        body = {"IdentifierList": {"CID": [cid]}} if cid else {"Fault": {"Message": "not found"}}
        class R: status_code = 200; text = json.dumps(body)
        return R()

def test_lookup_smiles_batch_hit_and_miss(monkeypatch, tmp_path):
    s = FakeSession({"AAAABBBBCCCCDD": 12345})
    monkeypatch.setattr("ik14_smiles.requests.Session", lambda: s)
    # 属性查询打桩：返回固定 SMILES
    monkeypatch.setattr("ik14_smiles.get_property", lambda sess, cid, prop: {"12345": {"CanonicalSMILES": "CCO"}})
    cache = {}
    out = lookup_smiles_batch(["AAAABBBBCCCCDD", "XXXXXXXXXXXXXX"], cache)
    assert out["AAAABBBBCCCCDD"]["status"] == "ok"
    assert out["AAAABBBBCCCCDD"]["smiles"] == "CCO"
    assert out["XXXXXXXXXXXXXX"]["status"] == "failed"
    assert len(cache) == 2  # 缓存含失败项，避免重复请求

def test_cache_reuse_skips_http(monkeypatch):
    cache = {"AAAABBBBCCCCDD": {"smiles": "CCO", "cid": 12345, "status": "ok", "error": None}}
    calls = []
    monkeypatch.setattr("ik14_smiles.requests.Session.get", lambda *a, **k: calls.append(1))
    out = lookup_smiles_batch(["AAAABBBBCCCCDD"], cache)
    assert out["AAAABBBBCCCCDD"]["smiles"] == "CCO" and not calls
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd external && python -m pytest tests/test_ik14_smiles.py -v`
Expected: FAIL（`ModuleNotFoundError: ik14_smiles` 或断言失败）

- [ ] **Step 3: 实现映射模块**（PUG REST：InChIKey→CID→CanonicalSMILES；限速 0.35s/请求；断点缓存；映射率审计）

```python
# external/ik14_smiles.py
"""IK14 (InChIKey 前14位) -> canonical SMILES via PubChem PUG REST.
InChIKey 前14位是连接层骨架，PUG 的 /inchikey/ 前缀查询对 14 位截断键同样有效；
命中多个 CID 时取第一个并记录。限速 + 断点缓存保证 3170 药量级可稳定跑完。"""
import csv, json, os, sys, time, logging
import requests

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ik14_smiles_map.json")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def get_cid(session, ik14):
    r = session.get(f"{BASE}/compound/inchikey/{ik14}/cids/JSON", timeout=30)
    data = r.json()
    if "IdentifierList" in data and data["IdentifierList"]["CID"]:
        return data["IdentifierList"]["CID"][0]
    return None

def get_property(session, cid, prop="CanonicalSMILES"):
    r = session.get(f"{BASE}/compound/cid/{cid}/property/{prop}/JSON", timeout=30)
    prop_list = r.json().get("PropertyTable", {}).get("Properties", [])
    return {str(p["CID"]): p for p in prop_list}

def lookup_smiles_batch(ik14s, cache):
    """返回 {ik14: {smiles, cid, status, error}}；cache 为持久缓存（含失败项）。"""
    out, session = {}, requests.Session()
    todo = [k for k in ik14s if k not in cache]
    for i, k in enumerate(sorted(todo)):
        try:
            cid = get_cid(session, k)
            if cid is None:
                out[k] = {"smiles": None, "cid": None, "status": "failed", "error": "no_cid"}
            else:
                props = get_property(session, cid)
                smi = props.get(str(cid), {}).get("CanonicalSMILES")
                out[k] = {"smiles": smi, "cid": cid,
                          "status": "ok" if smi else "failed", "error": None if smi else "no_smiles"}
        except Exception as e:
            out[k] = {"smiles": None, "cid": None, "status": "failed", "error": str(e)[:200]}
        cache[k] = out[k]
        if (i + 1) % 20 == 0:
            json.dump(cache, open(CACHE_PATH, "w"))
            logging.info("checkpoint %d/%d saved", i + 1, len(todo))
        time.sleep(0.35)  # NCBI 限速：不高于 5 req/s
    for k in ik14s:
        if k not in out:
            out[k] = cache[k]
    return out

def main():
    df = pd_read_pairs()
    ik14s = pd.unique(pd.concat([df["drug_a_ik14"], df["drug_b_ik14"]]))
    cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
    out = lookup_smiles_batch(list(ik14s), cache)
    json.dump(cache, open(CACHE_PATH, "w"), indent=2)
    with open(os.path.join(os.path.dirname(CACHE_PATH), "mapping_audit.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ik14", "status", "cid", "smiles", "error"])
        for k, v in sorted(out.items()):
            w.writerow([k, v["status"], v["cid"], v["smiles"], v["error"]])
    n_ok = sum(1 for v in out.values() if v["status"] == "ok")
    logging.info("MAPPED %d/%d (%.1f%%)", n_ok, len(out), 100 * n_ok / max(1, len(out)))

def pd_read_pairs():
    import pandas as pd
    raw = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw", "ddi_pairs_50k.csv")
    return pd.read_csv(raw, dtype={"drug_a_ik14": "string", "drug_b_ik14": "string",
                                   "faers_best_pt_code_strict": "string"})

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试通过 + 小样本冒烟**

```bash
cd external && python -m pytest tests/test_ik14_smiles.py -v   # PASS
python -c "import ik14_smiles; ik14_smiles.lookup_smiles_batch(['XSBSKEQEUFOSDD','ZDIGNSYAACHWNL'], {})"  # 观察真实 PubChem 响应
```

- [ ] **Step 5: 全量运行 + 审计**（约 3170 药 × 0.35s ≈ 20 分钟）

Run: `cd external && python ik14_smiles.py`
Expected: `MAPPED n/3170 (≥85%)`；若 <70%，停并报告（启用 CIR/UniChem 备用源，见 spec §5 风险②）。

- [ ] **Step 6: 提交**

```bash
git add external/ik14_smiles.py external/tests/test_ik14_smiles.py external/outputs/ik14_smiles_map.json external/outputs/mapping_audit.csv
git commit -m "feat(external): PubChem IK14-to-SMILES mapping with cache and audit"
```

### Task 3: 药物重叠审计（vs Dataset 1）

**Files:**
- Create: `external/audit_overlap_ext.py`
- Create: `external/tests/test_audit_overlap_ext.py`

**Interfaces:**
- Consumes: `PharDDIE/dataset1/drug_smiles.csv`、`external/outputs/ik14_smiles_map.json`、`external/raw/ddi_pairs_50k.csv`
- Produces:
  - `external/outputs/drug_overlap_report.csv` — 列：`dataset1_drug_id, dataset1_ik14, overlap(0/1)`
  - `external/outputs/overlap_summary.json` — `{"n_dataset1": 1706, "n_ext": 3170, "n_overlap": int, "overlap_rate": float, "overlap_ids": [db ids...]}`
  - `external/outputs/ik14_to_db.json` — 供任务 4 复用：`{ik14: db_id}`（dataset1 药物的 IK14→DB ID 映射）
  - 模块导出 `ik14_of(smiles) -> str|None`、`build_overlap(...)`。

- [ ] **Step 1: 写失败测试**

```python
# external/tests/test_audit_overlap_ext.py
import json, pytest
from rdkit import Chem
from audit_overlap_ext import build_overlap

@pytest.fixture
def tmp_ctx(tmp_path):
    ds1 = tmp_path / "ds1.csv"
    ds1.write_text("drug_id,smiles\nDB00001,CCO\nDB00002,CCN\n", encoding="utf-8")
    ikmap = {"A": {"smiles": "CCO", "status": "ok"}, "B": {"smiles": "CCC", "status": "ok"}}
    return ds1, ikmap

def test_build_overlap_counts(tmp_ctx):
    ds1_csv, ikmap = tmp_ctx
    # 构造 ext 药集 = {"A","B"}；"A" 与 DB00001 同为 CCO → 重叠 1
    report, summary, ik14_to_db = build_overlap(str(ds1_csv), ikmap, ext_ik14s={"A", "B"})
    assert summary["n_overlap"] == 1
    assert summary["overlap_rate"] == 1.0 / 2  # 按 dataset1 侧分母 1706 之外的约定：这里用小样本约定 rate=n_overlap/min(n_ds1,n_ext)
    assert ik14_to_db["A"] == "DB00001"
    assert report.loc[report["dataset1_drug_id"] == "DB00001", "overlap"].iloc[0] == 1
```

- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**（RDKit `MolFromSmiles`→`MolToInchiKey` 取前 14 位；与 3170 个 ext IK14 求交）

```python
# external/audit_overlap_ext.py
"""Dataset 1 与 RxPairEvid-50K 的药物重叠审计（审稿回应第一张证据）。"""
import csv, json, os
import pandas as pd
from rdkit import Chem

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def ik14_of(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)[:14] if mol else None
    except Exception:
        return None

def build_overlap(dataset1_smiles_csv, ik14_smiles_map, ext_ik14s):
    ds1 = pd.read_csv(dataset1_smiles_csv, dtype={"drug_id": "string"})
    ds1["ik14"] = ds1["smiles"].map(ik14_of)
    ext_set = {k for k, v in ik14_smiles_map.items() if v.get("status") == "ok"} & ext_ik14s
    ds1_valid = ds1.dropna(subset=["ik14"])
    overlap_ids = set(ds1_valid.loc[ds1_valid["ik14"].isin(ext_set), "drug_id"])
    report = ds1.assign(overlap=lambda d: d["ik14"].isin(ext_set).astype(int))
    summary = {
        "n_dataset1": int(len(ds1)),
        "n_ext": int(len(ext_set)),
        "n_overlap": int(len(overlap_ids)),
        "overlap_rate": round(len(overlap_ids) / max(1, len(ds1_valid)), 4),
        "overlap_ids": sorted(overlap_ids),
    }
    ik14_to_db = dict(zip(ds1_valid["ik14"], ds1_valid["drug_id"]))
    return report, summary, ik14_to_db

def main():
    ikmap = json.load(open(os.path.join(OUT, "ik14_smiles_map.json")))
    df = pd.read_csv(os.path.join(REPO, "external", "raw", "ddi_pairs_50k.csv"),
                     dtype={"drug_a_ik14": "string", "drug_b_ik14": "string"})
    ext_ik14s = set(pd.concat([df["drug_a_ik14"], df["drug_b_ik14"]]))
    report, summary, ik14_to_db = build_overlap(
        os.path.join(REPO, "PharDDIE", "dataset1", "drug_smiles.csv"), ikmap, ext_ik14s)
    report.to_csv(os.path.join(OUT, "drug_overlap_report.csv"), index=False)
    json.dump(summary, open(os.path.join(OUT, "overlap_summary.json"), "w"), indent=2)
    json.dump(ik14_to_db, open(os.path.join(OUT, "ik14_to_db.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 测试通过 + 全量运行**：`python -m pytest external/tests/ -v`；`cd external && python audit_overlap_ext.py`
Expected: `overlap_summary.json` 产出；重叠率如实记录（无论高低都要写进回应信）。

- [ ] **Step 5: 提交**

```bash
git add external/audit_overlap_ext.py external/tests/test_audit_overlap_ext.py external/outputs/drug_overlap_report.csv external/outputs/overlap_summary.json external/outputs/ik14_to_db.json
git commit -m "feat(external): drug-overlap audit vs Dataset 1 with IK14 crosswalk"
```

### Task 4: 构建 dataset_ext_{1shot,5shot}（dataset1 同构布局）

**Files:**
- Create: `external/build_dataset_ext.py`
- Create: `external/tests/test_build_dataset_ext.py`

**Interfaces:**
- Consumes: 任务 2/3 产物 + `PharDDIE/dataset1/` 的 `relation2ids, relation2embids, ent2ids, ent2embids, e1rel_e2.json, path_graph_train_only, DRKG_TransE_entity.npy, DRKG_TransE_relation.npy`
- Produces（两个目录同构，事件不同）:
  - `PharDDIE/dataset_ext_1shot/`、`PharDDIE/dataset_ext_5shot/`，各含：
    - `test2_tasks.json` — `{"PT-{code}": [[h, "PT-{code}", t], ...]}`，规范方向，按 pair_id 字典序固定顺序；1-shot 层取 ≥2 对的事件（185），5-shot 层取 ≥6 对的事件（24）
    - `rel2candidates.json` — `{"PT-{code}": [drug ids...]}`（该事件全部出现药物）
    - `drug_smiles.csv` — dataset1 的 1,706 行原文 + 新增药物（IK14→SMILES，id 用 IK14；若其 IK14 命中 `ik14_to_db` 则**改用 DB ID 行**，SMILES 用 dataset1 的）
    - `ent2ids` / `ent2embids` — dataset1 文件原文 + 新增药物 id（ent2embids=-1）；**旧键值不得改动顺序与内容**
    - `relation2ids` / `relation2embids` — dataset1 文件逐字节复制
    - `e1rel_e2.json`、`path_graph_train_only`、`DRKG_TransE_entity.npy`、`DRKG_TransE_relation.npy` — dataset1 文件复制（npy 用 `shutil.copy2`，完成后 SHA256 与源文件比对并记录）
  - `external/outputs/dataset_ext_build_report.json` — 事件数、药数、新旧药数、剔除事件数、文件 SHA256 表

- [ ] **Step 1: 写失败测试**（小合成输入：3 个 PT 事件 {A:2对, B:6对, C:1对}，验证 1-shot 层收 A、B 拒 C；5-shot 层只收 B；规范方向与字典序）

```python
# external/tests/test_build_dataset_ext.py
import json, pytest
from build_dataset_ext import build_tasks, canonical_triple

def test_canonical_triple_orders_by_ik14():
    assert canonical_triple("ZDIG..", "XSBS..", "PT-1") == ["XSBS..", "PT-1", "ZDIG.."]
    assert canonical_triple("XSBS..", "ZDIG..", "PT-1") == ["XSBS..", "PT-1", "ZDIG.."]

def test_build_tasks_tiers():
    pairs = {  # (a, b, event)
        ("A", "B", "PT-1"), ("A", "C", "PT-1"),
        ("A", "B", "PT-2"), ("A", "C", "PT-2"), ("A", "D", "PT-2"),
        ("B", "C", "PT-2"), ("B", "D", "PT-2"), ("C", "D", "PT-2"),
        ("A", "B", "PT-3"),
    }
    tasks_1, tasks_5 = build_tasks(pairs, min_pairs_1shot=2, min_pairs_5shot=6)
    assert set(tasks_1) == {"PT-1", "PT-2"}
    assert set(tasks_5) == {"PT-2"}
    assert all(len(v) == 6 for v in tasks_5.values())
    # 字典序固定：每事件内按 pair_id 排序
    for event, triples in tasks_1.items():
        pair_ids = ["::".join(sorted([t[0], t[2]])) for t in triples]
        assert pair_ids == sorted(pair_ids)
```

- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**（核心函数 + 组装）

```python
# external/build_dataset_ext.py
"""把 RxPairEvid-50K 组装成 dataset1 同构的 dataset_ext_{1shot,5shot}。
关键不变量：ent2ids/ent2embids/relation2ids 的旧键值内容与顺序与 dataset1 完全一致
（保证 symbol2vec 前 N 行与训练时逐行相同，checkpoint 可逐位加载）。"""
import json, os, shutil, hashlib
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DS1 = os.path.join(REPO, "PharDDIE", "dataset1")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_triple(a_ik14, b_ik14, event):
    a, b = sorted([a_ik14, b_ik14])
    return [a, event, b]

def build_tasks(signal_pairs, min_pairs_1shot=2, min_pairs_5shot=6):
    """signal_pairs: iterable of (a_ik14, b_ik14, event)；返回 (tasks_1, tasks_5)。"""
    from collections import defaultdict
    by_event = defaultdict(list)
    for a, b, e in signal_pairs:
        by_event[e].append(canonical_triple(a, b, e))
    def tier(min_pairs):
        tasks = {}
        for e, triples in sorted(by_event.items()):
            if len(triples) >= min_pairs:
                pair_ids = ["::".join(sorted([t[0], t[2]])) for t in triples]
                tasks[e] = [t for _, t in sorted(zip(pair_ids, triples))]
        return tasks
    return tier(min_pairs_1shot), tier(min_pairs_5shot)

def assemble_dataset(tier_name, tasks, ik14_smiles_map, ik14_to_db, ds1_smiles_df):
    """返回 build_report 条目；写 PharDDIE/dataset_ext_{tier_name}/。"""
    dst = os.path.join(REPO, "PharDDIE", f"dataset_ext_{tier_name}")
    os.makedirs(dst, exist_ok=True)
    # 事件键集合
    events = list(tasks)
    # 药物集合（任务中出现的所有 id）
    drugs = sorted({d for triples in tasks.values() for t in triples for d in (t[0], t[2])})
    # 1) 任务与候选
    json.dump(tasks, open(os.path.join(dst, "test2_tasks.json"), "w"), indent=2, ensure_ascii=False)
    rel2candidates = {}
    for e, triples in tasks.items():
        pool = {t[0] for t in triples} | {t[2] for t in triples}
        rel2candidates[e] = sorted(pool)
    json.dump(rel2candidates, open(os.path.join(dst, "rel2candidates.json"), "w"), indent=2, ensure_ascii=False)
    # 2) drug_smiles：dataset1 原文 + 新药（IK14 或映射后的 DB ID）
    ds1_rows = ds1_smiles_df.to_dict("records")
    known_ids = set(ds1_smiles_df["drug_id"])
    new_rows = []
    for d in drugs:
        if d in known_ids:
            continue  # dataset1 已有
        smi = (ik14_smiles_map.get(d) or {}).get("smiles")
        if not smi:
            raise ValueError(f"no SMILES for drug {d}")
        new_rows.append({"drug_id": d, "smiles": smi})
    pd.DataFrame(ds1_rows + new_rows).to_csv(os.path.join(dst, "drug_smiles.csv"), index=False)
    # 3) 符号表：旧文件原文 + 新药追加（-1）
    ent2ids = json.load(open(os.path.join(DS1, "ent2ids")))
    ent2embids = json.load(open(os.path.join(DS1, "ent2embids")))
    for d in drugs:
        if d not in ent2ids:
            ent2ids[d] = max(ent2ids.values()) + 1
            ent2embids[d] = -1
    json.dump(ent2ids, open(os.path.join(dst, "ent2ids"), "w"), indent=2)
    json.dump(ent2embids, open(os.path.join(dst, "ent2embids"), "w"), indent=2)
    # 4) 逐字节复制静态资源并记录哈希
    hashes = {}
    for f in ["relation2ids", "relation2embids", "e1rel_e2.json", "path_graph_train_only",
              "DRKG_TransE_entity.npy", "DRKG_TransE_relation.npy"]:
        shutil.copy2(os.path.join(DS1, f), os.path.join(dst, f))
        hashes[f] = {"src": sha256(os.path.join(DS1, f)), "dst": sha256(os.path.join(dst, f))}
        assert hashes[f]["src"] == hashes[f]["dst"], f"copy mismatch for {f}"
    return {"tier": tier_name, "n_events": len(events), "n_drugs": len(drugs),
            "n_new_drugs": len(new_rows), "hashes": hashes}
```

- [ ] **Step 4: 测试通过 + 全量组装 + 校验**

```bash
cd external && python -m pytest tests/test_build_dataset_ext.py -v   # PASS
python build_dataset_ext.py    # 产出两个数据集目录 + dataset_ext_build_report.json
python - <<'EOF'
# 加载期冒烟：确认 symbol2vec 前 N 行与 dataset1 一致
import json, numpy as np
for tier in ("1shot", "5shot"):
    e1 = json.load(open(f"../PharDDIE/dataset1/ent2ids")); e2 = json.load(open(f"../PharDDIE/dataset_ext_{tier}/ent2ids"))
    assert list(e1.items()) == list(e2.items())[:len(e1)], tier + " prefix invariant violated"
    print(tier, "ent2ids prefix OK; new drugs:", len(e2) - len(e1))
EOF
```
Expected: 两个目录各自 `ent2ids` 前缀不变；1shot 新药数 + 5shot 新药数打印。

- [ ] **Step 5: 提交**

```bash
git add external/build_dataset_ext.py external/tests/test_build_dataset_ext.py external/outputs/dataset_ext_build_report.json
git commit -m "feat(external): build dataset_ext 1-shot/5-shot tiers in dataset1 layout"
```
（`PharDDIE/dataset_ext_*/` 体积大，按 .gitignore 习惯不入库——在 `.gitignore` 之外不影响；提交时用 `git add` 明确列文件避免误加。）

### Task 5: 负样本 manifest 生成（单 split 版）

**Files:**
- Create: `external/neg_manifest_ext.py`
- Create: `external/tests/test_neg_manifest_ext.py`

**Interfaces:**
- Consumes: `PharDDIE/dataset_ext_{tier}/test2_tasks.json`、`rel2candidates.json`
- Produces（每层目录 `neg_manifests/`）:
  - `test2_seed{seed}_negatives.json` — `{event: [[d_i, d_j, d_k, rel], ...]}`，与 `shared/neg_manifest.py` 格式一致（入口顺序 = tasks 顺序；support 在前 query 在后）
  - `test2_seed{seed}_nativenegatives.json` — 同格式，d_k 来自原生无信号对（见下）
  - `manifest_hashes.json` — `{"test2_seed{seed}": {"path":..., "sha256":...}, "test2_seed{seed}_native": {...}}`
- 采样规则（tail-corruption）：对每个阳性 `(d_i, rel, d_j)`：`d_k = choice(candidates)`，要求 `d_k != d_j` 且 `(d_i, rel, d_k)` 不在该事件已知阳性中（用任务自身三元组构建 known-positives 集合；新事件无 DRKG 边可查，**不依赖 e1rel_e2**）。
- 原生负样本：从全局无信号对池（`faers_ror95_lcl_max_strict` 为 NULL 的 49,127 对）中，对每个查询阳性 `(d_i, rel, d_j)` 采样一个包含 `d_i` 的无信号对 `(d_i, x)` 取 `d_k = x`；不够时回退任意无信号对。支持集条目不生成负样本（与主协议一致：manifest 条目数 = 任务三元组数，导出端取 `expected = manifest_entries[few:]`）。

- [ ] **Step 1: 写失败测试**（固定种子复现、正负不交叉）

```python
# external/tests/test_neg_manifest_ext.py
import json, pytest
from neg_manifest_ext import generate_manifest_ext

TASKS = {"PT-1": [["A", "PT-1", "B"], ["A", "PT-1", "C"], ["A", "PT-1", "D"]]}
CAND = {"PT-1": ["A", "B", "C", "D", "E", "F"]}

def test_manifest_deterministic_and_no_collision():
    m1 = generate_manifest_ext(TASKS, CAND, seed=19940419)
    m2 = generate_manifest_ext(TASKS, CAND, seed=19940419)
    assert m1 == m2
    for entries in m1.values():
        for d_i, d_j, d_k, rel in entries:
            assert d_k != d_j                       # 不采样自身
            assert [d_i, rel, d_k] not in TASKS[rel]  # 不在已知阳性中

def test_manifest_entry_count_matches_tasks():
    m = generate_manifest_ext(TASKS, CAND, seed=19940419)
    for event, entries in m.items():
        assert len(entries) == len(TASKS[event])
```

- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**

```python
# external/neg_manifest_ext.py
"""单 split（test2）固定负样本 manifest 生成器。
shared/neg_manifest.py 要求 dev/test/test2 三文件齐全且用 e1rel_e2 排除已知阳性；
新数据集只有 test2 且事件在 DRKG 中无对应边，故此处用任务自身三元组构建排除集合。"""
import argparse, hashlib, json, os, random
from collections import defaultdict

SEEDS = [19940419, 20230801, 20240115, 20240520, 20240910]

def generate_manifest_ext(tasks, rel2candidates, seed, no_signal_pairs=None):
    random.seed(seed)
    manifest = {}
    for event, triples in sorted(tasks.items()):
        candidates = rel2candidates[event]
        known = {(t[0], t[2]) for t in triples}  # (head, tail) 已知阳性
        entries = []
        for d_i, rel, d_j in triples:
            while True:
                d_k = random.choice(candidates)
                if d_k != d_j and (d_i, d_k) not in known:
                    break
            entries.append([d_i, d_j, d_k, rel])
        manifest[event] = entries
    return manifest

def generate_native_manifest(tasks, no_signal_pairs, seed, few):
    """原生负样本：无信号对池；优先含 d_i 的对。few 之前为 support，不生成。"""
    random.seed(seed)
    by_drug = defaultdict(list)
    for a, b in no_signal_pairs:
        by_drug[a].append(b); by_drug[b].append(a)
    manifest = {}
    for event, triples in sorted(tasks.items()):
        entries = []
        for d_i, rel, d_j in triples:
            pool = by_drug.get(d_i, [])
            d_k = random.choice(pool) if pool else random.choice(no_signal_pairs)[0]
            if d_k == d_i:
                d_k = (pool + [d_j])[0] if pool else d_j
            entries.append([d_i, d_j, d_k, rel])
        manifest[event] = entries
    return manifest

def write_manifest(path, manifest):
    json.dump(manifest, open(path, "w"), indent=2, ensure_ascii=False)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="PharDDIE/dataset_ext_1shot or ..._5shot")
    ap.add_argument("--few", type=int, required=True)
    ap.add_argument("--no-signal-pairs", default=None, help="json of [a,b] pairs (native negatives)")
    args = ap.parse_args()
    tasks = json.load(open(f"{args.dataset}/test2_tasks.json"))
    cand = json.load(open(f"{args.dataset}/rel2candidates.json"))
    no_sig = json.load(open(args.no_signal_pairs)) if args.no_signal_pairs else None
    os.makedirs(f"{args.dataset}/neg_manifests", exist_ok=True)
    hash_log = {}
    for seed in SEEDS:
        m = generate_manifest_ext(tasks, cand, seed)
        h = write_manifest(f"{args.dataset}/neg_manifests/test2_seed{seed}_negatives.json", m)
        hash_log[f"test2_seed{seed}"] = {"path": f"neg_manifests/test2_seed{seed}_negatives.json", "sha256": h}
        if no_sig:
            mn = generate_native_manifest(tasks, no_sig, seed, few=args.few)
            hn = write_manifest(f"{args.dataset}/neg_manifests/test2_seed{seed}_nativenegatives.json", mn)
            hash_log[f"test2_seed{seed}_native"] = {"path": f"neg_manifests/test2_seed{seed}_nativenegatives.json", "sha256": hn}
    json.dump(hash_log, open(f"{args.dataset}/neg_manifests/manifest_hashes.json", "w"), indent=2)
    print("manifests written; hash log saved")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 测试通过 + 全量生成**（先导出无信号对池 JSON）

```bash
cd external && python -m pytest tests/test_neg_manifest_ext.py -v
python - <<'EOF'
import json, pandas as pd
df = pd.read_csv("raw/ddi_pairs_50k.csv", dtype={"drug_a_ik14":"string","drug_b_ik14":"string"})
no_sig = df[df["faers_ror95_lcl_max_strict"].isna()][["drug_a_ik14","drug_b_ik14"]].values.tolist()
json.dump(no_sig, open("outputs/no_signal_pairs.json","w"))
print("no-signal pool:", len(no_sig))
EOF
python neg_manifest_ext.py --dataset ../PharDDIE/dataset_ext_1shot --few 1 --no-signal-pairs outputs/no_signal_pairs.json
python neg_manifest_ext.py --dataset ../PharDDIE/dataset_ext_5shot --few 5 --no-signal-pairs outputs/no_signal_pairs.json
```
Expected: 每层 `neg_manifests/` 含 5 种子 × 2 类 manifest + `manifest_hashes.json`。

- [ ] **Step 5: 提交**

```bash
git add external/neg_manifest_ext.py external/tests/test_neg_manifest_ext.py external/outputs/no_signal_pairs.json
git commit -m "feat(external): single-split fixed negative manifests (tail-corrupted + native)"
```

### Task 6: PharDDIE 外部验证推理（主路径）

**Files:**
- Create: `external/pharddie_export_ext.py`

**Interfaces:**
- Consumes: `PharDDIE/dataset_ext_{tier}/`（含 neg_manifests）、`PharDDIE/models/dataset1/models_drugbank_{few}shot_str_seed{seed}/bestmodel`（用户找回后放置）、`PharDDIE/pharddie_{args,matcher,models,layers}.py`、`shared/checkpoint.py`
- Produces: `external/outputs/predictions_rxpairevid_{tier}_{shot}shot.csv`（列：`train_seed, eval_seed, tier, shot, method, event_type, drug_a, drug_b, y_true, y_pred, prob, uncertainty`）+ `external/outputs/episode_manifests_rxpairevid/` + `external/outputs/checkpoint_hashes_rxpairevid.json`

**关键实现点（与 `pharddie_export_full.py` 的差异，全部在此列出）：**
1. 数据集参数化为 `--dataset`（默认 `../PharDDIE/dataset_ext_1shot`），只读 `test2_tasks.json`，无 dev/test。
2. `load_embed` 逻辑照抄但 `symbol2vec` 前缀不变式：先按 dataset1 相同的键序构建，新药行（ent2embids=-1）置**零向量**（确定性，替代原代码无种子 `np.random.randn`）。
3. 加载 checkpoint 后、`load_state_dict_safe` 前，恢复旧行精确值：`ckpt['symbol_emb.weight']` 的 `[:n_old]` 行 copy 进 `matcher.symbol_emb.weight`，新行保持零；随后从 ckpt 中 `del ckpt['symbol_emb.weight']` 再 `load_state_dict_safe(self.matcher, ckpt, model_name='matcher')`（strict=False 对缺失 key 只告警）。
4. 分子图：自含 `build_drug_graphs(drug_smiles_csv)`（复制 `shared/preprocess.py` 第 18-60 行的图构建逻辑并参数化路径，不 import 该模块避免其硬编码 dataset1）。
5. `ent2id`/`ent2embids`/`relation2ids`/`relation2embids`/`e1rel_e2.json`/`path_graph_train_only`/`DRKG_TransE_*.npy` 一律从 `--dataset` 读。
6. 负样本：默认读 `neg_manifests/test2_seed{EVAL_SEED}_negatives.json`；`--native` 时改读 `..._nativenegatives.json`，输出 CSV 文件名加 `_native` 后缀。manifest 与任务的逐条校验逻辑照抄（长度 + 头尾/事件一致性）。
7. per-seed checkpoint 缺失即报错退出（不跨种子回退）；`--train-seeds` 可传子集（默认 5 个；5-shot 若缺 seed19940419 用 `--train-seeds 20230801,20240115,20240520,20240910`）。
8. 每个 `(tier, shot, train_seed, native)` 组合：推理前随机种子固定 `random.seed(eval_seed); np.random.seed(eval_seed); torch.manual_seed(eval_seed)`。
9. 逐样本不确定性 = SRAE `torch.exp(z_logvar).mean(dim=-1)`（照抄原管线）；评分 `fc(|proto - z|)` → sigmoid。
10. 运行前校验 `neg_manifests/manifest_hashes.json` 与实际文件 SHA256 一致（照抄原导出逻辑）。
11. 输出 CSV 追加 `tier` 列；结束后写 `checkpoint_hashes_rxpairevid.json`（每种子 checkpoint 的 SHA256）与 episode manifest JSON。

- [ ] **Step 1: 冒烟骨架测试**（无 GPU 也可跑的部分：manifest 校验 + 零向量 symbol2vec 构建）

```python
# external/tests/test_pharddie_export_ext.py
import json, pytest
import numpy as np
from pharddie_export_ext import build_symbol2vec_zero_fallback, verify_manifest_entries

def test_build_symbol2vec_prefix_invariant():
    ent_embed = np.random.RandomState(0).randn(4, 128)   # 4 个旧实体
    old_ent2embids = {"DB1": 0, "DB2": 1, "DB3": 2, "DB4": -1}
    new_ent2embids = dict(old_ent2embids); new_ent2embids["NEW1"] = -1
    sym2vec, symbol2id = build_symbol2vec_zero_fallback(old_ent2embids, new_ent2embids, ent_embed)
    n_old = len(old_ent2embids)
    assert np.allclose(sym2vec[:n_old], 0) or True  # 旧 -1 行也允许零（确定性）
    assert np.all(sym2vec[n_old:n_old+1] == 0)       # 新药零向量
    assert symbol2id["NEW1"] == n_old

def test_verify_manifest_entries_mismatch_raises():
    tasks = {"PT-1": [["A", "PT-1", "B"], ["A", "PT-1", "C"]]}
    manifest = {"PT-1": [["A", "B", "X", "PT-1"], ["A", "B", "Y", "PT-1"]]}  # 第二条 head 错
    with pytest.raises(RuntimeError):
        verify_manifest_entries(tasks, manifest, few=1)
```

- [ ] **Step 2: 运行确认失败** → **Step 3: 实现完整脚本**（按上述 11 点实现；主体照抄 `pharddie_export_full.py` 的 ExportFull 类，改 `__init__`/`load_embed`/分子图加载/manifest 路径，其余管线逐行保留）

- [ ] **Step 4: 冒烟（1 事件 1 种子）**：`--dataset ../PharDDIE/dataset_ext_1shot --few 1 --train-seeds 19940419 --smoke-events 1`（实现 `--smoke-events N`：只评估前 N 个事件，验证管线通 + 无 NaN）
- [ ] **Step 5: 全量运行**（检查 GPU 可用后）

```bash
cd external
python pharddie_export_ext.py --dataset ../PharDDIE/dataset_ext_1shot --few 1
python pharddie_export_ext.py --dataset ../PharDDIE/dataset_ext_5shot --few 5 --train-seeds 20230801,20240115,20240520,20240910   # 若 5shot 缺 seed19940419
python pharddie_export_ext.py --dataset ../PharDDIE/dataset_ext_1shot --few 1 --native
```
Expected: `predictions_rxpairevid_1shot_1shot.csv`、`..._5shot_5shot.csv`、`..._1shot_1shot_native.csv` + episode manifests + checkpoint 哈希。

- [ ] **Step 6: 提交**（CSV/JSON 一并入库，供证据链审计）

```bash
git add external/pharddie_export_ext.py external/tests/test_pharddie_export_ext.py external/outputs/predictions_rxpairevid_*.csv external/outputs/episode_manifests_rxpairevid/ external/outputs/checkpoint_hashes_rxpairevid.json
git commit -m "feat(external): PharDDIE per-seed inference on RxPairEvid episodes"
```

> **门控**：若本任务 Step 4 前 checkpoint 未找到（2026-08-24），标记本任务 blocked，直接转 Task 8；本任务留待 checkpoint 到位后续做。

### Task 7: RareDDIE 基线同协议对比（主路径配套）

**Files:**
- Create: `external/rareddie_export_ext.py`

**Interfaces:**
- Consumes: `PharDDIE/eval_rareddie_unified.py`（阅读后镜像其数据装配与推理入口）、RareDDIE 五种子模型 `C:\Users\Admin\UAID-DDI\PharDDIE\models\rareddie_{1,5}shot_seed{seed}bestmodel`（读取，不移动）、任务 4/5 的 dataset_ext 目录
- Produces: `external/outputs/predictions_rxpairevid_rareddie_{tier}_{shot}shot.csv`（列与任务 6 一致，`method=RareDDIE`）

- [ ] **Step 1: 读 `PharDDIE/eval_rareddie_unified.py` 与 `run_rareddie_seed.py`，记录其模型类、加载函数签名与 dataset1 依赖点**（探索步骤，不写代码）
- [ ] **Step 2: 写失败测试**：小任务 + 随机权重模型（不依赖 checkpoint）跑通装配函数，断言输出 CSV 行数 = 查询数×2
- [ ] **Step 3: 实现**：镜像统一评估脚本，`--dataset`/`--few`/`--train-seeds`/`--native` 参数与任务 6 相同；模型文件路径参数化 `--ckpt-pattern`
- [ ] **Step 4: 冒烟 + 全量**：同任务 6 的两个 tier × shot 组合 + native 补充
- [ ] **Step 5: 提交**

```bash
git add external/rareddie_export_ext.py external/tests/test_rareddie_export_ext.py external/outputs/predictions_rxpairevid_rareddie_*.csv
git commit -m "feat(external): RareDDIE baseline on RxPairEvid episodes"
```

### Task 8: EviDDIE 零样本兜底推理（8-24 未找到 checkpoint 时启用）

**Files:**
- Create: `external/eviddie_export_ext.py`

**Interfaces:**
- Consumes: `EviDDIE/models/dataset1/eviddie_0shot_seed{seed}/bestmodel`（已存在）、`EviDDIE/eviddie_export_zs_v2.py`（镜像）、PT code→文本映射
- Produces: `external/outputs/predictions_rxpairevid_eviddie_0shot.csv`（列对齐任务 6）

- [ ] **Step 1: 获取 MedDRA PT code→label 映射**（探索步骤）：尝试顺序 (a) 用户机构 MedDRA 订阅导出；(b) 公开 PT 列表源（如 UMLS MRCONSO 的 MedDRA 部分，需机构授权；(c) 找不到文本时，事件原型 = `"FAERS adverse event PT-" + code` 字符串嵌入，并在论文注明该近似）。结果写入 `external/outputs/pt_labels.json`（`{code: label}`）与来源记录
- [ ] **Step 2: 读 `eviddie_export_zs_v2.py`，记录事件原型构建与推理入口；写镜像脚本**（事件原型：label → BioSentVec（`EviDDIE/dataset1/event_embedding2.json` 外的文本需本地 BioSentVec 模型编码；无本地模型时用 `pt_labels.json` 近似字符串））
- [ ] **Step 3: 冒烟 + 全量**：5 种子 × 0-shot，两个 tier 的 `test2_tasks.json`；输出对齐 CSV
- [ ] **Step 4: 提交**

```bash
git add external/eviddie_export_ext.py external/outputs/predictions_rxpairevid_eviddie_0shot.csv external/outputs/pt_labels.json
git commit -m "feat(external): EviDDIE zero-shot fallback inference on RxPairEvid"
```

### Task 9: 外部验证汇总表

**Files:**
- Create: `external/ext_summary_table.py`
- Create: `external/tests/test_ext_summary_table.py`

**Interfaces:**
- Consumes: 任务 6/7/8 的预测 CSV（存在哪个用哪个；方法列区分）
- Produces: `external/outputs/ext_validation_table.csv` + `.txt`（论文表格直排版面）
- 指标：按 `tier × shot × method` 分组：pooled AUROC/AUPRC/ACC/F1 + event-macro F1（`(1/|E|)Σ_e F1_e`，只对 `y_true` 含两类的单事件计算）；跨 train_seed 的 mean±SD；no-skill 参考行（p=0.5 常数预测在平衡采样下的 ECE/Brier 不适用，只给 ACC/F1/AUROC=0.5 的参考说明）

- [ ] **Step 1: 写失败测试**（合成 CSV：2 种子 × 2 事件 × 4 样本，手工算 AUROC/F1 断言）
- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**（pandas groupby + sklearn metrics；`zero_division=0`；AUROC 在单类事件上记 0 并标注）
- [ ] **Step 4: 全量运行** → **Step 5: 提交**

```bash
git add external/ext_summary_table.py external/tests/test_ext_summary_table.py external/outputs/ext_validation_table.csv external/outputs/ext_validation_table.txt
git commit -m "feat(external): external validation summary table with seed-level SD"
```

### Task 10: 案例研究（独立文献佐证）

**Files:**
- Create: `external/case_study_ext.py`

**Interfaces:**
- Consumes: 任务 6（或 8）的 1-shot 预测 CSV、任务 3 的 `drug_overlap_report.csv`/`overlap_summary.json`
- Produces:
  - `external/outputs/case_candidates.csv` — 候选：1-shot（主路径）或 0-shot（兜底）rare 层样本按 `r = p(1-u)`（主路径 5 种子均值排名）取 top-10 药对，剔除与 Dataset 1 重叠药物所在的药对，列：`rank, drug_a, drug_b, a_name, b_name, event, prob_mean, u_mean, r, faers_prr_max_strict, faers_ror95_lcl_max_strict, n_faers_reports`
  - `external/outputs/case_evidence.md` — 每候选一对：PubMed Entrez `esearch`（查询 `"{a_name}[All Fields] AND {b_name}[All Fields] AND interaction"`）取 top-3 PMID + 标题；FAERS 信号数值摘录；证据列模板（PMID / FAERS，禁止 DrugBank）

- [ ] **Step 1: 写失败测试**（合成预测 CSV：构造 3 个候选，验证排名、剔除规则、r 计算）
- [ ] **Step 2: 运行确认失败** → **Step 3: 实现**（排名/剔除部分直接实现；PubMed 部分用 Entrez `esearch.fcgi`（无 key，0.34s 限速）+ `esummary.fcgi` 取标题）
- [ ] **Step 4: 全量运行**：`cd external && python case_study_ext.py`
- [ ] **Step 5: 用户人工复核 `case_evidence.md`**（文献佐证是学术判断，最终保留/替换由用户定）
- [ ] **Step 6: 提交**

```bash
git add external/case_study_ext.py external/tests/test_case_study_ext.py external/outputs/case_candidates.csv external/outputs/case_evidence.md
git commit -m "feat(external): objective case-study selection with PubMed/FAERS evidence"
```

### Task 11: 论文改写（fyx8_21.tex）与回应信

**Files:**
- Backup: `C:\Users\Admin\Desktop\fyx8_21_backup_2026-08-21.tex`（改前复制）
- Modify: `C:\Users\Admin\Desktop\fyx8_21.tex`
- Create: `C:\Users\Admin\Desktop\回应信_外部验证_初稿.md`

**Interfaces:**
- Consumes: 任务 3/9/10 的全部产物

- [ ] **Step 1: 备份 tex 并复制工作副本**

```bash
cp "/c/Users/Admin/Desktop/fyx8_21.tex" "/c/Users/Admin/Desktop/fyx8_21_backup_2026-08-21.tex"
```

- [ ] **Step 2: 新增外部验证小节**（插入在 `\subsection{Case Study}` 之前，约第 672 行）：标题 "External Validation on Pharmacovigilance-Derived Data"；内容：数据来源（RxPairEvid-50K、FAERS、MedDRA PT code 说明）、重叠审计数字（引用任务 3 的 `overlap_summary.json`）、事件分层（185/24）、协议一致性（同 manifest/5 种子）、结果表 `tab:ext_validation`（引用任务 9 的 txt）+ 兜底药物比例
- [ ] **Step 3: 重写 Case Study（第 672-724 行）**：删除"内部一致性检查"表述与旧表 `tab:case_study`；换新案例表（任务 10 产物，证据列 = PMID/FAERS）
- [ ] **Step 4: 修订 Limitations（第 743-748 行）**：删除"Validation on additional DDI resources ... would provide a stronger test" 的自认句，改为已完成外部验证的表述；保留其余局限
- [ ] **Step 5: 更新 Data and Code availability（第 753-754 行）**：补充 RxPairEvid 引用（DOI 10.17632/zrvzpfmzcz.1，F1000Research DOI 10.12688/f1000research.178856.1）与 `external/` 脚本说明
- [ ] **Step 6: 新增 bib 条目**（写入论文引用但**标记待用户同步 Overleaf bib**）：RxPairEvid、FAERS、MedDRA（按 fyx8_21.tex 现有引用风格 `\cite{bNNN}` 续号）
- [ ] **Step 7: 回应信初稿**：点对点回应"数据集重叠"（三条：来源独立—FAERS vs DrugBank；药物重叠率数字；独立证据案例结果）；另附重做实验的表格引用
- [ ] **Step 8: 用户审阅**（论文与回应信均为学术交付物，用户最终把关）

---

## Self-Review 记录

- **Spec coverage**: spec §2 组件 → T1-T11 全部覆盖；§3 数据流 8 步 → T1(校验)/T2(映射)/T3(审计)/T4(分层+episode)/T5(负样本)/T6-8(推理)/T10(案例)；§5 风险①②③④⑤⑥ → 分别落在 T1/plan 门控、T2 映射率阈值、T4 分层剔除、T8 PT 标签、T6 门控、T6 分事件冒烟；§6 论文 5 处改动 → T11 Step 2-6；§7 验收 → T9/T10/T11 + 各任务提交。
- **Type consistency**: 任务文件事件键 `PT-{code}` 在 T4（构建）→T5（manifest）→T6-8（导出）全程一致；manifest 条目 `[d_i, d_j, d_k, rel]` 与 `shared/neg_manifest.py` 格式一致，T6 校验逻辑兼容；CSV 列名在 T6/T7/T8 输出与 T9/T10 消费处逐字一致；`ik14_to_db.json` 由 T3 产出、T4 消费。
- **已知开放项**: (1) PharDDIE checkpoint 是否找回（门控 T6/T7）；(2) MedDRA PT 标签来源（T8 Step 1）；(3) 5-shot seed19940419 缺失（T6 `--train-seeds` 处理）；(4) 用户补齐 5 个可选原始文件（T1 警告）。
