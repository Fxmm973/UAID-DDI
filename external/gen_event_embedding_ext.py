#!/usr/bin/env python
# coding=utf-8
"""Generate the real semantic event embeddings for the ext tiers.

Run ONLY after the 92-vector BioSentVec validation gate passes.  The gate
passed 2026-08-21: all 92 reference vectors in
EviDDIE/dataset1/event_embedding2.json reproduce at cosine = 1.000000 with
the loader DEFAULT pipeline (tokenizer='nltkish': nltk word_tokenize -> NLTK
stopword removal -> punctuation-token removal -> '#'-token drop -> lowercase;
rolling bigrams; mean pooling, no normalization).

Fallback event text: "FAERS adverse event PT-{code}" (per task brief).  The
event keys are the union of both ext tiers' test2_tasks.json keys, sorted.

Output: external/outputs/event_embedding_ext.json (nested 1x700 per key,
same layout as event_embedding2.json) plus the file's sha256 on stdout.

Deterministic: the loader is fully deterministic (fixed model bytes, fixed
tokenizer, fixed key order), so re-running reproduces the same sha256.

Usage: python external/gen_event_embedding_ext.py
"""
import sys
import os
import json
import hashlib

sys.stdout.reconfigure(encoding='utf-8')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'external'))

from biosentvec_loader import Sent2VecModel  # noqa: E402

BIN = os.path.join(_REPO, 'external/biosentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin')
OUT = os.path.join(_REPO, 'external/outputs/event_embedding_ext.json')


def main():
    # union of events across both tiers, sorted
    events = {}
    for tier in ['1shot', '5shot']:
        tasks = json.load(open(os.path.join(_REPO, 'PharDDIE/dataset_ext_%s/test2_tasks.json' % tier),
                               encoding='utf-8'))
        for k in tasks:
            events[k] = None
    keys = sorted(events)
    print('[GEN] %d events across both tiers; mode=nltkish/rolling/none (defaults)' % len(keys),
          flush=True)
    model = Sent2VecModel(BIN)  # defaults: nltkish tokenizer, rolling bigrams, mean, no L2
    payload = {}
    for i, k in enumerate(keys):
        text = 'FAERS adverse event %s' % k      # "PT-{code}"
        v = model.embed(text)
        payload[k] = [v.tolist()]                # nested 1x700 like event_embedding2.json
        if (i + 1) % 50 == 0:
            print('[GEN] %d/%d embedded (%s)' % (i + 1, len(keys), k), flush=True)
    model.close()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    h = hashlib.sha256(open(OUT, 'rb').read()).hexdigest()
    nzero = sum(1 for k in keys if any(payload[k][0]))
    print('[GEN] wrote %s' % OUT)
    print('[GEN] sha256=%s' % h)
    print('[GEN] keys=%d nonzero=%d dims_ok=%s' %
          (len(payload), nzero, all(len(payload[k][0]) == 700 for k in payload)))


if __name__ == '__main__':
    main()
