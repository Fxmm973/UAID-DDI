# external/tests/test_audit_overlap_ext.py
import json, pytest
from rdkit import Chem
from audit_overlap_ext import build_overlap

@pytest.fixture
def tmp_ctx(tmp_path):
    ds1 = tmp_path / "ds1.csv"
    ds1.write_text("drug_id,smiles\nDB00001,CCO\nDB00002,CCN\n", encoding="utf-8")
    # 注意：fixture 的 ext 键必须是真实的 IK14（ik14_of 实时计算，占位键无法交叠）
    # LFQSCWFLJHTTHZ = CCO(乙醇)，ATUOYWHBWRKTHZ = CCC(丙烷)
    ikmap = {"LFQSCWFLJHTTHZ": {"smiles": "CCO", "status": "ok"},
             "ATUOYWHBWRKTHZ": {"smiles": "CCC", "status": "ok"}}
    return ds1, ikmap

def test_build_overlap_counts(tmp_ctx):
    ds1_csv, ikmap = tmp_ctx
    # 构造 ext 药集 = {乙醇, 丙烷} 的 ik14；乙醇与 DB00001 同为 CCO → 重叠 1
    report, summary, ik14_to_db = build_overlap(
        str(ds1_csv), ikmap, ext_ik14s={"LFQSCWFLJHTTHZ", "ATUOYWHBWRKTHZ"})
    assert summary["n_overlap"] == 1
    assert summary["overlap_rate"] == 1.0 / 2  # 按 dataset1 侧分母 1706 之外的约定：这里用小样本约定 rate=n_overlap/min(n_ds1,n_ext)
    assert ik14_to_db["LFQSCWFLJHTTHZ"] == "DB00001"
    assert report.loc[report["dataset1_drug_id"] == "DB00001", "overlap"].iloc[0] == 1
