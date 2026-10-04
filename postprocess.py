"""Snap out-of-vocabulary predicted words to close words from the training transcriptions.

Tune:   python postprocess.py tune runs/qwen3vl2b/val_preds_epoch2.json
Apply:  python postprocess.py apply submissions/qwen3vl2b.json --max_dist 1 --min_count 2
"""
import argparse
import json
from collections import Counter

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from common import clean, load_train, score, write_submission


def build_vocab(targets):
    return Counter(w for t in targets for w in t.split())


def correct(line, vocab, words, max_dist, min_count, min_len=4):
    out = []
    for w in line.split():
        if w in vocab or len(w) < min_len or not w.isalpha():
            out.append(w)
            continue
        cands = process.extract(w, words, scorer=Levenshtein.distance, score_cutoff=max_dist, limit=5)
        cands = [(c, d) for c, d, _ in cands if vocab[c] >= min_count]
        if cands:
            best = min(cands, key=lambda cd: (cd[1], -vocab[cd[0]]))[0]
            out.append(best)
        else:
            out.append(w)
    return clean(" ".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "apply"])
    ap.add_argument("preds")
    ap.add_argument("--max_dist", type=int, default=1)
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.mode == "tune":
        recs = json.load(open(a.preds))
        val_ids = {r["ID"] for r in recs}
        vocab = build_vocab(r["Target"] for r in load_train() if r["ID"] not in val_ids)
        words = list(vocab)
        refs, hyps = [r["ref"] for r in recs], [r["hyp"] for r in recs]
        print("raw", score(refs, hyps))
        for md in (1, 2):
            for mc in (1, 2, 3, 5):
                fixed = [correct(h, vocab, words, md, mc) for h in hyps]
                print(f"max_dist={md} min_count={mc}", score(refs, fixed))
    else:
        preds = json.load(open(a.preds))
        vocab = build_vocab(r["Target"] for r in load_train())
        words = list(vocab)
        ids = list(preds)
        fixed = [correct(preds[i], vocab, words, a.max_dist, a.min_count) for i in ids]
        out = a.out or a.preds.replace(".json", "_pp.csv")
        write_submission(out, ids, fixed)
        print("wrote", out)


if __name__ == "__main__":
    main()
