"""Build pseudo-labels for test lines where many ensemble members agree exactly on the consensus transcription.

python make_pseudo.py --min_agree 5 --out data/pseudo.json submissions/a.json submissions/b.json ...
"""
import argparse
import json

from rapidfuzz.distance import Levenshtein

from common import clean

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--min_agree", type=int, default=5)
ap.add_argument("--out", default="data/pseudo.json")
a = ap.parse_args()

sets = [{k: clean(v) for k, v in json.load(open(f)).items()} for f in a.files]
ids = [i for i in sets[0] if all(i in s for s in sets)]


def cost(h, hyps):
    return sum(Levenshtein.distance(h, o) + 5.5 * Levenshtein.distance(h.split(), o.split()) for o in hyps)


out = []
for i in ids:
    hyps = [s[i] for s in sets]
    best = min(hyps, key=lambda h: cost(h, hyps))
    if best and sum(h == best for h in hyps) >= a.min_agree:
        out.append({"ID": i, "Target": best})
json.dump(out, open(a.out, "w"), indent=1)
print(f"{len(out)}/{len(ids)} test lines pseudo-labelled (>= {a.min_agree}/{len(sets)} members agree) -> {a.out}")
