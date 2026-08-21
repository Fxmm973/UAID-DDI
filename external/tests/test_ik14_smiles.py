# external/tests/test_ik14_smiles.py
import json, csv
import pytest
import pandas as pd
import ik14_smiles
from ik14_smiles import lookup_smiles_batch

class FakeSession:
    def __init__(self, cid_by_ik14): self.cid_by_ik14 = cid_by_ik14
    def get(self, url, timeout=30):  # PUG: /compound/inchikey/{ik14}/cids/JSON
        ik14 = url.split("/inchikey/")[1].split("/")[0]
        cid = self.cid_by_ik14.get(ik14)
        body = {"IdentifierList": {"CID": [cid]}} if cid else {"Fault": {"Message": "not found"}}
        class R:
            status_code = 200
            text = json.dumps(body)
            def json(self): return json.loads(self.text)
        return R()

def test_lookup_smiles_batch_hit_and_miss(monkeypatch, tmp_path):
    s = FakeSession({"AAAABBBBCCCCDD": 12345})
    monkeypatch.setattr("ik14_smiles.requests.Session", lambda: s)
    # 属性查询打桩：返回固定 SMILES
    monkeypatch.setattr("ik14_smiles.get_property", lambda sess, cid, prop="CanonicalSMILES": {"12345": {"CanonicalSMILES": "CCO"}})
    cache = {}
    out = lookup_smiles_batch(["AAAABBBBCCCCDD", "XXXXXXXXXXXXXX"], cache)
    assert out["AAAABBBBCCCCDD"]["status"] == "ok"
    assert out["AAAABBBBCCCCDD"]["smiles"] == "CCO"
    assert out["AAAABBBBCCCCDD"]["prop"] == "CanonicalSMILES"  # 逐条记录实际属性名
    assert out["XXXXXXXXXXXXXX"]["status"] == "failed"
    assert len(cache) == 2  # 缓存含失败项（内容缺失），避免重复请求

def test_cache_reuse_skips_http(monkeypatch):
    cache = {"AAAABBBBCCCCDD": {"smiles": "CCO", "cid": 12345, "status": "ok", "error": None}}
    calls = []
    monkeypatch.setattr("ik14_smiles.requests.Session.get", lambda *a, **k: calls.append(1))
    out = lookup_smiles_batch(["AAAABBBBCCCCDD"], cache)
    assert out["AAAABBBBCCCCDD"]["smiles"] == "CCO" and not calls

def test_transient_http_error_retries_then_ok(monkeypatch):
    class FlakySession:
        def __init__(self): self.calls = 0
        def get(self, url, timeout=30):
            self.calls += 1
            if self.calls == 1:
                class R: status_code = 500; text = "boom"
                return R()
            ik14 = url.split("/inchikey/")[1].split("/")[0]
            body = {"IdentifierList": {"CID": [12345]}} if ik14 == "AAAABBBBCCCCDD" else {"Fault": {}}
            class R2:
                status_code = 200
                text = json.dumps(body)
                def json(self): return json.loads(self.text)
            return R2()
    s = FlakySession()
    monkeypatch.setattr("ik14_smiles.requests.Session", lambda: s)
    monkeypatch.setattr("ik14_smiles.get_property", lambda sess, cid, prop="CanonicalSMILES,ConnectivitySMILES": {"12345": {"CanonicalSMILES": "CCO"}})
    cache = {}
    out = lookup_smiles_batch(["AAAABBBBCCCCDD"], cache)
    assert out["AAAABBBBCCCCDD"]["status"] == "ok"
    assert s.calls == 2  # 瞬时错误重试 1 次后成功
    assert cache["AAAABBBBCCCCDD"]["status"] == "ok"

def test_transient_error_not_cached_after_retries(monkeypatch):
    class Always500Session:
        def get(self, url, timeout=30):
            class R: status_code = 500; text = "boom"
            return R()
    monkeypatch.setattr("ik14_smiles.requests.Session", lambda: Always500Session())
    cache = {}
    out = lookup_smiles_batch(["QQQQQQQQQQQQQQ"], cache)
    v = out["QQQQQQQQQQQQQQ"]
    assert v["status"] == "failed" and v["error"].startswith("http_error:")
    assert cache == {}  # 瞬时错误不入缓存，下次运行重试

def test_lookup_batch_writes_no_files(monkeypatch, tmp_path):
    # 库函数只改传入的 cache dict，不写盘（断点持久化在 main 的 on_progress 回调里）
    monkeypatch.setattr("ik14_smiles.CACHE_PATH", str(tmp_path / "cache.json"))
    s = FakeSession({"AAAABBBBCCCCDD": 12345})
    monkeypatch.setattr("ik14_smiles.requests.Session", lambda: s)
    monkeypatch.setattr("ik14_smiles.get_property", lambda sess, cid, prop="CanonicalSMILES": {"12345": {"CanonicalSMILES": "CCO"}})
    cache = {}
    lookup_smiles_batch(["AAAABBBBCCCCDD"], cache)
    assert not (tmp_path / "cache.json").exists()
    assert cache["AAAABBBBCCCCDD"]["status"] == "ok"

def test_main_adds_meta_and_audit(monkeypatch, tmp_path):
    # main() 结尾写入 __meta__ 来源标注 + mapping_audit.csv
    df = pd.DataFrame({"drug_a_ik14": ["AAAABBBBCCCCDD"], "drug_b_ik14": ["QQQQQQQQQQQQQQ"]})
    monkeypatch.setattr("ik14_smiles.pd_read_pairs", lambda: df)
    out_dir = tmp_path / "outputs"; out_dir.mkdir()
    monkeypatch.setattr("ik14_smiles.CACHE_PATH", str(out_dir / "ik14_smiles_map.json"))
    def fake_lookup(ik14s, cache, on_progress=None):
        cache["AAAABBBBCCCCDD"] = {"smiles": "CCO", "cid": 12345, "prop": "CanonicalSMILES", "status": "ok", "error": None}
        cache["QQQQQQQQQQQQQQ"] = {"smiles": None, "cid": None, "prop": None, "status": "failed", "error": "no_cid"}
        if on_progress is not None:
            on_progress(2, 2)
        return dict(cache)
    monkeypatch.setattr("ik14_smiles.lookup_smiles_batch", fake_lookup)
    ik14_smiles.main()
    data = json.load(open(out_dir / "ik14_smiles_map.json"))
    assert data["__meta__"]["smiles_property"] == "connectivity_fallback"
    assert data["__meta__"]["note"] == "CanonicalSMILES deprecated by PubChem 2024-08; entries are ConnectivitySMILES"
    assert data["AAAABBBBCCCCDD"]["status"] == "ok"
    rows = list(csv.reader(open(out_dir / "mapping_audit.csv", encoding="utf-8")))
    assert rows[0] == ["ik14", "status", "cid", "smiles", "error"]
    assert len(rows) == 3  # 2 条数据 + 表头
