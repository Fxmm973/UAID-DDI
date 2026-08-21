#!/usr/bin/env python
# coding=utf-8
"""
Pure-Python loader for epfml/sent2vec (fastText fork) .bin model files.

Reverse-engineered from the actual BioSentVec file header and the epfml/sent2vec
source at commit cdf8d832ac (2019-04-05, the era BioSentVec was trained):

  Binary layout (all little-endian):
    int32  magic   = 793712314 (FASTTEXT_FILEFORMAT_MAGIC_INT32)
    int32  version = 11
    Args:
      int32  dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket
      int32  minn, maxn, lrUpdateRate
      float64 t
    Dictionary:
      int32  size, nwords, nlabels
      int64  ntokens
      int64  pruneidx_size_        (fork default: -1, a sentinel)
      per entry: word\\0 + int64 count + int32 type   (x size entries)
    int8   quant
    Matrix input:  int64 m, int64 n, float32 data[m*n]      (m = nwords + bucket)
    int8   qout
    Matrix output: int64 m, int64 n, float32 data[m*n]

  Sentence embedding (C++ textVector semantics, no subsampling):
    - tokenize by whitespace only (' ', '\\n', '\\r', '\\t', '\\v', '\\f', '\\0'),
      case-sensitive; punctuation stays attached ("#Drug1." is one token)
    - known words -> row id = the word's position in the dictionary file order
      (Dictionary::load: word2int_[find(w)] = i; the file is written in array
      order by Dictionary::save). OOV tokens are dropped (sent2vec).
    - wordNgrams=2 (Dictionary::addNgrams(line, n) 2-arg overload used by
      FastText::textVector): rolling uint64 hash  h = h*116049371 + line[j]
      over consecutive word IDs -> row = nwords + (h % bucket)
    - sentence vector = mean over (word rows + bigram rows); NO L2 norm
      (Vector::mul(1.0/line.size()))

  For diagnosis the loader supports tokenizer / bigram / normalization mode
  switches; the authoritative acceptance gate is the reproduction of the
  92 precomputed vectors in EviDDIE/dataset1/event_embedding2.json with
  mean cosine similarity >= 0.999 (see the --validate CLI).
"""
import os
import struct
import sys

import numpy as np

FASTTEXT_MAGIC = 793712314  # FASTTEXT_FILEFORMAT_MAGIC_INT32
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619
NGRAM_HASH_MULT = 116049371  # fastText addNgrams rolling hash multiplier

WS_CHARS = set(' \n\r\t\x0b\x0c\x00')


def ft_hash(word):
    """fastText hashing: FNV-1a 32-bit over the UTF-8 bytes of `word`."""
    h = FNV_OFFSET
    for b in word.encode('utf-8'):
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def tokenize_ws(text):
    """C++ readWord semantics: whitespace-only split, case-sensitive."""
    return [tok for tok in text.split() if tok] if text else []


def tokenize_ws_lower(text):
    return [tok.lower() for tok in tokenize_ws(text)]


def tokenize_punct(text):
    """Whitespace split plus punctuation detachment (diagnostic mode only)."""
    toks = []
    for tok in tokenize_ws(text):
        while tok and tok[-1] in '.,;:!?()[]"\'':
            toks.append(tok[-1])
            tok = tok[:-1]
        if tok:
            toks.append(tok)
    return toks


def tokenize_punct_lower(text):
    return [t.lower() for t in tokenize_punct(text)]


_TOKENIZERS = {
    'ws': tokenize_ws,
    'ws_lower': tokenize_ws_lower,
    'punct': tokenize_punct,
    'punct_lower': tokenize_punct_lower,
}


def bigram_rows_rolling(token_ids, nwords, bucket):
    """C++ Dictionary::addNgrams(line, n=2) 2-arg overload (sent2vec path):
    rolling uint64 hash over consecutive word IDs, one bigram row per adjacent
    pair: h starts at line[i] (=word id) then h = h*116049371 + line[i+1]."""
    rows = []
    for i in range(len(token_ids) - 1):
        h = token_ids[i]
        h = (h * NGRAM_HASH_MULT + token_ids[i + 1]) & 0xFFFFFFFFFFFFFFFF
        rows.append(nwords + (h % bucket))
    return rows


def bigram_rows_string_hash(tokens, nwords, bucket):
    """Alternative bigram construction: FNV-1a over the joined token string
    (fse-style; diagnostic mode only)."""
    rows = []
    for i in range(len(tokens) - 1):
        s = tokens[i] + ' ' + tokens[i + 1]
        rows.append(nwords + (ft_hash(s) % bucket))
    return rows


_BIGRAM_BUILDERS = {
    'rolling': bigram_rows_rolling,
    'string_hash': bigram_rows_string_hash,
}


class Sent2VecModel(object):
    """Minimal pure-Python reader for a sent2vec/fastText .bin model."""

    def __init__(self, path, tokenizer='ws', bigram='rolling', normalize='none'):
        self.path = path
        self.tokenizer = _TOKENIZERS[tokenizer]
        self.bigram_builder = _BIGRAM_BUILDERS[bigram]
        self.normalize = normalize
        self._parse_header(path)
        self._open_matrix(path)

    # ------------------------------------------------------------------ header
    def _parse_header(self, path):
        with open(path, 'rb') as f:
            magic, = struct.unpack('<i', f.read(4))
            if magic != FASTTEXT_MAGIC:
                raise ValueError(
                    'not a fastText/sent2vec model: magic=%d expected %d' % (magic, FASTTEXT_MAGIC))
            self.magic = magic
            self.version, = struct.unpack('<i', f.read(4))
            (self.dim, self.ws, self.epoch, self.minCount, self.neg,
             self.wordNgrams, self.loss, self.model, self.bucket,
             self.minn, self.maxn, self.lrUpdateRate) = struct.unpack('<12i', f.read(48))
            self.t, = struct.unpack('<d', f.read(8))
            self.size, self.nwords, self.nlabels = struct.unpack('<3i', f.read(12))
            self.ntokens, = struct.unpack('<q', f.read(8))
            self.pruneidx_size, = struct.unpack('<q', f.read(8))
            self.words = {}
            for _ in range(self.size):
                w = b''
                while True:
                    c = f.read(1)
                    if not c or c == b'\x00':
                        break
                    w += c
                count, = struct.unpack('<q', f.read(8))
                typ, = struct.unpack('<i', f.read(4))
                self.words[w.decode('utf-8', errors='replace')] = count
            self.quant, = struct.unpack('<?', f.read(1))
            self.input_m, self.input_n = struct.unpack('<2q', f.read(16))
            self.input_offset = f.tell()
            if self.input_n != self.dim:
                raise ValueError('input matrix n=%d != dim=%d' % (self.input_n, self.dim))
            if self.input_m != self.nwords + self.bucket:
                raise ValueError('input matrix m=%d != nwords+bucket=%d'
                                 % (self.input_m, self.nwords + self.bucket))
            self.word_row_index = {}  # word -> row id (file-order position)
            for i, w in enumerate(self.words):
                self.word_row_index[w] = i

    def _open_matrix(self, path):
        self._mm = open(path, 'rb')
        self._mm.seek(self.input_offset)
        # mmap the whole matrix region as a float32 view (no copy; pages load lazily)
        import mmap
        m = self.input_m
        n = self.input_n
        self._mmp = mmap.mmap(self._mm.fileno(), 0, access=mmap.ACCESS_READ)
        data = np.frombuffer(self._mmp, dtype='<f4', count=m * n, offset=self.input_offset)
        self.input_matrix = data.reshape(m, n)
        out_off = self.input_offset + m * n * 4
        self.qout, = struct.unpack('<?', self._mmp[out_off:out_off + 1])
        om, on = struct.unpack('<2q', self._mmp[out_off + 1:out_off + 17])
        if om != m:
            raise ValueError('output matrix m=%d != input m=%d' % (om, m))
        self.output_offset = out_off + 17

    def close(self):
        if getattr(self, '_mmp', None) is not None:
            # numpy view over the mmap must be released before closing it
            self.input_matrix = None
            self._mmp.close()
            self._mmp = None
        if getattr(self, '_mm', None) is not None:
            self._mm.close()
            self._mm = None

    # ----------------------------------------------------------------- lookup
    def word_row(self, word):
        """Row id of a known word = its position in the dictionary file order
        (C++ Dictionary::load: word2int_[find(word)] = i)."""
        return self.word_row_index.get(word)

    def rows_for_tokens(self, tokens):
        ids = []
        for tok in tokens:
            if tok in self.words:
                ids.append(self.word_row(tok))
        if not ids:
            return np.zeros(0, dtype=np.int64)
        rows = np.asarray(ids, dtype=np.int64)
        if self.wordNgrams >= 2 and len(ids) >= 2:
            rows = np.concatenate([rows, self.bigram_builder(ids, self.nwords, self.bucket)])
        return rows

    # --------------------------------------------------------------- embedding
    def embed_tokens(self, tokens):
        rows = self.rows_for_tokens(tokens)
        if len(rows) == 0:
            return np.zeros(self.dim, dtype=np.float32)
        vec = self.input_matrix[rows].mean(axis=0).astype(np.float32)
        if self.normalize == 'l2':
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed(self, text):
        return self.embed_tokens(self.tokenizer(text))

    def embed_sentences(self, texts):
        return np.vstack([self.embed(t) for t in texts])


# --------------------------------------------------------------------- utility
def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def validate(model, ref_json, min_mean=0.999, out=None):
    """Embed the keys of ref_json and compare cosine similarity to the stored
    vectors. Returns (ok, mean_cos, min_cos, worst_key)."""
    ref = __import__('json').load(open(ref_json, encoding='utf-8'))
    keys = sorted(ref.keys())
    sims = []
    for k in keys:
        v = np.asarray(ref[k], dtype=np.float32)
        e = model.embed(k)
        sims.append(cosine(e, v))
    sims = np.asarray(sims)
    mean_cos = float(sims.mean())
    min_cos = float(sims.min())
    worst = keys[int(sims.argmin())]
    line = ('VALIDATE ref=%s keys=%d mean_cos=%.6f min_cos=%.6f worst=%r'
            % (ref_json, len(keys), mean_cos, min_cos, worst))
    if out is not None:
        out.write(line + '\n')
    else:
        print(line)
    return (mean_cos >= min_mean), mean_cos, min_cos, worst


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description='Pure-Python sent2vec .bin loader/validator')
    ap.add_argument('--bin', required=True, help='path to the .bin model file')
    ap.add_argument('--validate', metavar='REF_JSON', default=None,
                    help='validate against a {text: [700 floats]} reference JSON')
    ap.add_argument('--tokenizer', default='ws', choices=sorted(_TOKENIZERS))
    ap.add_argument('--bigram', default='rolling', choices=sorted(_BIGRAM_BUILDERS))
    ap.add_argument('--normalize', default='none', choices=['none', 'l2'])
    ap.add_argument('--text', default=None, help='embed one text and print the vector')
    args = ap.parse_args(argv)

    model = Sent2VecModel(args.bin, tokenizer=args.tokenizer,
                          bigram=args.bigram, normalize=args.normalize)
    print('[META] version=%d dim=%d ws=%d epoch=%d minCount=%d neg=%d wordNgrams=%d '
          'loss=%d model=%d bucket=%d minn=%d maxn=%d lrUpdateRate=%d t=%.4f'
          % (model.version, model.dim, model.ws, model.epoch, model.minCount,
             model.neg, model.wordNgrams, model.loss, model.model, model.bucket,
             model.minn, model.maxn, model.lrUpdateRate, model.t))
    print('[META] size=%d nwords=%d nlabels=%d ntokens=%d pruneidx=%d quant=%r '
          'input=(%d,%d) input_offset=%d'
          % (model.size, model.nwords, model.nlabels, model.ntokens,
             model.pruneidx_size, model.quant, model.input_m, model.input_n,
             model.input_offset))
    if args.text is not None:
        v = model.embed(args.text)
        print('[EMB] %r -> %s' % (args.text, ','.join('%.6f' % x for x in v[:16])))
    if args.validate:
        ok, mean_cos, min_cos, worst = validate(model, args.validate)
        print('[RESULT] %s' % ('PASS' if ok else 'FAIL'))
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
