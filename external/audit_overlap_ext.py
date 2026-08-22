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
    report = ds1.assign(overlap=lambda d: d["ik14"].isin(ext_set).astype(int)).rename(
        columns={"drug_id": "dataset1_drug_id", "ik14": "dataset1_ik14"})
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
