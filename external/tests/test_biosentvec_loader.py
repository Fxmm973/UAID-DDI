#!/usr/bin/env python
# coding=utf-8
"""Tests for external/biosentvec_loader.py (pure-Python sent2vec loader).

The authoritative acceptance gate for the loader is the reproduction of the
92 precomputed vectors in EviDDIE/dataset1/event_embedding2.json (mean cosine
>= 0.999), which requires the real BioSentVec model file. These unit tests
cover the pure logic: FNV-1a hashing, whitespace tokenization, the bigram
rolling-hash construction, and full header/matrix parsing against a small
synthetic .bin written in the reverse-engineered binary format.
"""
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from biosentvec_loader import (  # noqa: E402
    FASTTEXT_MAGIC,
    NGRAM_HASH_MULT,
    Sent2VecModel,
    bigram_rows_rolling,
    ft_hash,
    tokenize_ws,
    tokenize_ws_lower,
    validate,
)


def fnv1a_reference(s):
    """Independently written FNV-1a 32-bit reference implementation."""
    h = 2166136261
    for b in s.encode("utf-8"):
        h = h ^ b
        h = (h * 16777619) % (2 ** 32)
    return h


def test_ft_hash_fnv1a_properties():
    assert ft_hash("") == 2166136261
    assert ft_hash("a") == fnv1a_reference("a")
    assert ft_hash("the") == fnv1a_reference("the")
    assert ft_hash("#Drug1 may increase") == fnv1a_reference("#Drug1 may increase")
    assert ft_hash("héllo") == fnv1a_reference("héllo")  # multi-byte UTF-8
    # case-sensitivity: fastText hashes are case-sensitive
    assert ft_hash("Drug") != ft_hash("drug")
    # 32-bit range
    assert 0 <= ft_hash("BioSentVec_PubMed_MIMICIII") < 2 ** 32


def test_tokenize_ws_semantics():
    # C++ readWord: whitespace-only split, punctuation stays attached
    assert tokenize_ws("#Drug1 may increase the photosensitizing activities of #Drug2.") == [
        "#Drug1", "may", "increase", "the", "photosensitizing",
        "activities", "of", "#Drug2."]
    assert tokenize_ws("a\tb\nc\r\nd\x0b e\x0c") == ["a", "b", "c", "d", "e"]
    assert tokenize_ws("") == []
    assert tokenize_ws("   ") == []
    assert tokenize_ws_lower("#Drug1") == ["#drug1"]
    assert tokenize_ws("Drug") == ["Drug"]  # no lowercasing in C++ readWord


def test_bigram_rows_rolling_formula():
    # C++ Dictionary::addNgrams(line, n=2): h = line[i]; then h = h*116049371 + line[i+1]
    ids = [5, 7]
    rows = bigram_rows_rolling(ids, nwords=100, bucket=16)
    h = (5 * NGRAM_HASH_MULT + 7) & 0xFFFFFFFFFFFFFFFF
    assert rows == [100 + (h % 16)]
    # three tokens -> two bigrams
    rows2 = bigram_rows_rolling([1, 2, 3], nwords=100, bucket=16)
    h1 = (1 * NGRAM_HASH_MULT + 2) & 0xFFFFFFFFFFFFFFFF
    h2 = (2 * NGRAM_HASH_MULT + 3) & 0xFFFFFFFFFFFFFFFF
    assert rows2 == [100 + (h1 % 16), 100 + (h2 % 16)]
    # single token -> no bigrams
    assert bigram_rows_rolling([9], nwords=100, bucket=16) == []
    assert bigram_rows_rolling([], nwords=100, bucket=16) == []


def write_synthetic_model(path, dim=8, bucket=16, words=("the", "quick", "brown", "fox")):
    """Write a minimal sent2vec .bin in the reverse-engineered binary format.

    Returns (input_rows, vocab_word_index) with deterministic float data so
    the test can compute expected embeddings by hand.
    """
    nwords = len(words)
    size = nwords
    ntokens = 1000
    pruneidx = -1
    m = nwords + bucket

    rng = np.random.RandomState(1234)
    input_data = rng.standard_normal((m, dim)).astype(np.float32)
    output_data = rng.standard_normal((m, dim)).astype(np.float32)

    with open(path, "wb") as f:
        f.write(struct.pack("<i", FASTTEXT_MAGIC))
        f.write(struct.pack("<i", 11))  # version
        # Args: dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket,
        #       minn, maxn, lrUpdateRate, t (double)
        f.write(struct.pack("<12i", dim, 5, 5, 5, 10, 2, 2, 4, bucket, 0, 0, 100))
        f.write(struct.pack("<d", 0.001))
        # Dictionary: size, nwords, nlabels, ntokens, pruneidx_size_
        f.write(struct.pack("<3i", size, nwords, 0))
        f.write(struct.pack("<q", ntokens))
        f.write(struct.pack("<q", pruneidx))
        for i, w in enumerate(words):
            f.write(w.encode("utf-8") + b"\x00")
            f.write(struct.pack("<q", i * 100 + 10))
            f.write(struct.pack("<i", 0))  # type: word
        f.write(struct.pack("<?", 0))  # quant
        f.write(struct.pack("<2q", m, dim))
        f.write(input_data.tobytes())
        f.write(struct.pack("<?", 0))  # qout
        f.write(struct.pack("<2q", m, dim))
        f.write(output_data.tobytes())
    return input_data, {w: i for i, w in enumerate(words)}


def test_synthetic_header_and_embed(tmp_path):
    path = str(tmp_path / "mini.bin")
    input_data, widx = write_synthetic_model(path)
    model = Sent2VecModel(path)

    assert model.version == 11
    assert model.dim == 8
    assert model.wordNgrams == 2
    assert model.bucket == 16
    assert model.nwords == 4
    assert model.nlabels == 0
    assert model.ntokens == 1000
    assert model.pruneidx_size == -1
    assert model.quant is False
    assert model.input_m == 4 + 16
    assert model.input_n == 8

    # word rows are file-order positions, not FNV hash positions
    for w, i in widx.items():
        assert model.word_row(w) == i
        h = ft_hash(w) % model.nwords
        assert model.word_row(w) == h or True  # (informational; file order wins)

    # embedding: mean of [word rows + bigram rows]
    emb = model.embed("the quick brown fox")
    ids = [widx["the"], widx["quick"], widx["brown"], widx["fox"]]
    bigrams = bigram_rows_rolling(ids, model.nwords, model.bucket)
    rows = np.asarray(ids + bigrams, dtype=np.int64)
    expected = input_data[rows].mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(emb, expected, rtol=1e-6)

    # OOV token dropped
    emb2 = model.embed("the QUICK brown fox")
    ids2 = [widx["the"], widx["brown"], widx["fox"]]
    bigrams2 = bigram_rows_rolling(ids2, model.nwords, model.bucket)
    rows2 = np.asarray(ids2 + bigrams2, dtype=np.int64)
    np.testing.assert_allclose(emb2, input_data[rows2].mean(axis=0).astype(np.float32), rtol=1e-6)

    # no normalization applied by default
    assert not np.isclose(np.linalg.norm(emb), 1.0)

    # single known word: word row + no bigram (n=2 needs >= 2 tokens)
    emb3 = model.embed("fox")
    np.testing.assert_allclose(emb3, input_data[widx["fox"]].astype(np.float32), rtol=1e-6)

    # empty text -> zero vector
    np.testing.assert_allclose(model.embed(""), np.zeros(8, dtype=np.float32))
    model.close()


def test_validate_function(tmp_path):
    path = str(tmp_path / "mini.bin")
    input_data, widx = write_synthetic_model(path)
    model = Sent2VecModel(path)
    ids = [widx["the"], widx["quick"], widx["brown"], widx["fox"]]
    bigrams = bigram_rows_rolling(ids, model.nwords, model.bucket)
    rows = np.asarray(ids + bigrams, dtype=np.int64)
    expected = input_data[rows].mean(axis=0).astype(np.float32)
    ref = {"the quick brown fox": expected.tolist()}
    ref_path = str(tmp_path / "ref.json")
    import json
    json.dump(ref, open(ref_path, "w"))
    ok, mean_cos, min_cos, worst = validate(model, ref_path)
    assert ok
    assert mean_cos >= 0.999
    assert worst == "the quick brown fox"
    model.close()


def test_missing_magic(tmp_path):
    path = str(tmp_path / "bad.bin")
    with open(path, "wb") as f:
        f.write(struct.pack("<i", 12345))
    with pytest.raises(ValueError, match="not a fastText"):
        Sent2VecModel(path)
