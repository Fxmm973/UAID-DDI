# Figure Audit Checklist — Version Conflicts to Fix

## Figure 1: Framework Overview (`kuangjiatu.jpg`)
- [ ] BioSentVec dim: change "768" to "700" everywhere in the figure
- [ ] SRAE label: change "VAP" or "VAE" to "SRAE" to match paper terminology
- [ ] Few/zero-shot labels: change "1-shot / 5-shot / 10-shot" to "Few-shot" or match paper's "K=1, K=5"
- [ ] Triage label: change "Agent decision" or "Agent-based" to "Rule-based triage"
- [ ] EviDDIE branch: ensure it says "zero-shot" not "few-shot"
- [ ] Module labels: ensure "MME", "ACI", "SRAE", "BSA", "EVI" are consistent

## Figure 2/3: Fusion Weight Selection
- [ ] X-axis label: "Pharmacophore weight" consistent with text
- [ ] Y-axis: ensure "AUC / ACC / F1" labels present
- [ ] Legend: ensure "AUC", "ACC", "F1" (not "AUROC", "Accuracy", "F1-score")

## Figure 4/5: PharDDIE Ablation
- [ ] Ablation labels: "w/o MME", "w/o ACI", "w/o SRAE", "Full"
- [ ] Error bars should reflect training-seed std

## Figure: EviDDIE Ablation (`EviDDIE_Ablation_Study.png`)
- [ ] Legend: "Softmax baseline", "EviDDIE w/o EVI", "EviDDIE"
- [ ] Ensure referenced in RQ2 section

## General
- [ ] All figures: consistent color scheme
- [ ] All figures: font size >= 8pt
- [ ] All figures: >=300dpi for raster, vector preferred
