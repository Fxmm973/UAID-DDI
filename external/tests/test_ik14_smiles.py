# external/tests/test_ik14_smiles.py
import json, pytest
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
    assert out["XXXXXXXXXXXXXX"]["status"] == "failed"
    assert len(cache) == 2  # 缓存含失败项，避免重复请求

def test_cache_reuse_skips_http(monkeypatch):
    cache = {"AAAABBBBCCCCDD": {"smiles": "CCO", "cid": 12345, "status": "ok", "error": None}}
    calls = []
    monkeypatch.setattr("ik14_smiles.requests.Session.get", lambda *a, **k: calls.append(1))
    out = lookup_smiles_batch(["AAAABBBBCCCCDD"], cache)
    assert out["AAAABBBBCCCCDD"]["smiles"] == "CCO" and not calls
