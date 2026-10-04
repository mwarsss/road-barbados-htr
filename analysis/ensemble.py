"""Combine several line-level HTR prediction files into one.

Inputs: JSON files in either format
  - list of {"ID", "hyp", ["ref"]}  (val_preds_epoch*.json)
  - dict ID -> hyp                  (val_b*_a*.json, submissions/*.json)
Methods:
  mbr      per line pick the hypothesis with min weighted expected loss
           (char edits + ALPHA * word edits, ~ the 0.5*CER+0.5*WER metric) to the others
  rover    word-level confusion network around a pivot (MBR choice) + weighted vote
  crover   same at character level
  first    just the first file (baseline)
Weights: --weights w1 w2 ... (default uniform). The first file doubles as tie-breaker
(it gets +1e-3 weight), so list the strongest system first.

Examples
  python ensemble.py --method mbr a.json b.json c.json --out sub.csv
  python ensemble.py --method mbr a.json b.json c.json --eval     # needs refs (val files)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import clean, collapse_repeats, load_train, score, write_submission  # noqa: E402

ALPHA = 5.5   # ~ mean chars per word incl. space: puts a word edit on the same scale as char edits
TIE = 1e-3


def load_preds(path):
    """Return (hyps: dict ID->hyp, refs: dict ID->ref or {})."""
    d = json.load(open(path))
    if isinstance(d, list):
        return {r["ID"]: clean(r["hyp"]) for r in d}, {r["ID"]: r["ref"] for r in d if "ref" in r}
    return {k: clean(v) for k, v in d.items()}, {}


def line_loss(a, b):
    return Levenshtein.distance(a, b) + ALPHA * Levenshtein.distance(a.split(), b.split())


def mbr(hyps, weights):
    """hyps: list of strings (one per system). Returns index of the MBR hypothesis."""
    n = len(hyps)
    best, best_i = None, 0
    for i in range(n):
        risk = sum(weights[j] * line_loss(hyps[i], hyps[j]) for j in range(n) if j != i and hyps[j] != hyps[i])
        risk -= TIE * weights[i] * (n - i)   # tie-break: earlier / heavier systems
        if best is None or risk < best:
            best, best_i = risk, i
    return best_i


def _vote(pivot, hyps, weights):
    """Confusion network with 2n+1 slots around `pivot` (a token sequence).
    Slot 2k+1 = pivot token k (value: token or None=deletion);
    slot 2k   = insertion gap before pivot token k (value: tuple of inserted tokens, () = none)."""
    n = len(pivot)
    slots = [defaultdict(float) for _ in range(2 * n + 1)]
    for h, w in zip(hyps, weights):
        sub = [None] * n                 # what h has at each pivot position
        gaps = [[] for _ in range(n + 1)]
        ops = Levenshtein.opcodes(pivot, h)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal" or tag == "replace":
                # replace blocks of unequal length: pair up in order, spill extra into gap / deletions
                m = min(i2 - i1, j2 - j1)
                for k in range(m):
                    sub[i1 + k] = h[j1 + k]
                if j2 - j1 > m:
                    gaps[i1 + m].extend(h[j1 + m:j2])
            elif tag == "insert":
                gaps[i1].extend(h[j1:j2])
            # delete: sub stays None
        for k in range(n):
            slots[2 * k + 1][sub[k]] += w
        for k in range(n + 1):
            slots[2 * k][tuple(gaps[k])] += w
    out = []
    for k, s in enumerate(slots):
        v = max(s.items(), key=lambda kv: kv[1])[0]
        if k % 2:
            if v is not None:
                out.append(v)
        else:
            out.extend(v)
    return out


def rover(hyps, weights, level="word", pivot_idx=None):
    if pivot_idx is None:
        pivot_idx = mbr(hyps, weights)
    w = [x + (TIE if i == pivot_idx else 0) for i, x in enumerate(weights)]
    if level == "word":
        toks = [h.split() for h in hyps]
        return clean(" ".join(_vote(toks[pivot_idx], toks, w)))
    return clean("".join(_vote(list(hyps[pivot_idx]), [list(h) for h in hyps], w)))


def combine_line(hyps, weights, method):
    if method == "first":
        return hyps[0]
    if method == "mbr":
        return hyps[mbr(hyps, weights)]
    if method == "rover":
        return rover(hyps, weights, "word")
    if method == "crover":
        return rover(hyps, weights, "char")
    raise ValueError(method)


def combine(pred_dicts, weights=None, method="mbr", ids=None):
    ids = ids or list(pred_dicts[0])
    weights = weights or [1.0] * len(pred_dicts)
    return {i: combine_line([p[i] for p in pred_dicts], weights, method) for i in ids}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--method", default="mbr", choices=["mbr", "rover", "crover", "first"])
    ap.add_argument("--weights", type=float, nargs="*")
    ap.add_argument("--collapse", action="store_true", help="apply common.collapse_repeats to the output")
    ap.add_argument("--out", help="output .csv (submission) and matching .json dict")
    ap.add_argument("--eval", action="store_true", help="score against refs (from list-format files or the val split)")
    a = ap.parse_args()

    loaded = [load_preds(f) for f in a.files]
    preds = [p for p, _ in loaded]
    ids = [i for i in preds[0] if all(i in p for p in preds)]
    if len(ids) != len(preds[0]):
        print(f"warning: only {len(ids)}/{len(preds[0])} IDs present in all files", file=sys.stderr)
    out = combine(preds, a.weights, a.method, ids)
    if a.collapse:
        out = {k: collapse_repeats(v) for k, v in out.items()}

    if a.eval:
        refs = {}
        for _, r in loaded:
            refs.update(r)
        if not refs:
            refs = {r["ID"]: r["Target"] for r in load_train()}
        R = [refs[i] for i in ids]
        for f, p in zip(a.files, preds):
            print(f"{score(R, [p[i] for i in ids])['final']:.4f}  {f}")
        s = score(R, [out[i] for i in ids])
        print(f"{s['final']:.4f}  ENSEMBLE {a.method}  (cer {s['cer']:.4f} wer {s['wer']:.4f})")
    if a.out:
        write_submission(a.out, ids, [out[i] for i in ids])
        json.dump(out, open(str(Path(a.out).with_suffix(".json")), "w"), indent=1, ensure_ascii=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
