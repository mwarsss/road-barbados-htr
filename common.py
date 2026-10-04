import csv
import os
import random
import re
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageStat
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ROAD_DATA", ROOT / "data"))
IMAGES = Path(os.environ.get("ROAD_IMAGES", DATA / "images"))

PROMPT = "Transcribe the handwritten text in this image exactly as written."


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean(text):
    return " ".join(str(text).split())


def collapse_repeats(text, max_n=6, min_repeats=3, max_filler=5):
    """Collapse generation loops like 'given given given' or 'of the said X of the said X ...' to one copy."""
    words = clean(text).split()
    changed = True
    while changed:
        changed = False
        for n in range(1, max_n + 1):
            i = 0
            while i + n * min_repeats <= len(words):
                unit = words[i:i + n]
                k = 1
                while words[i + k * n:i + (k + 1) * n] == unit:
                    k += 1
                has_word = any(sum(c.isalpha() for c in w) >= 2 for w in unit)
                if k >= min_repeats and has_word:
                    words = words[:i + n] + words[i + k * n:]
                    changed = True
                i += 1
    capped = []
    for w in words:
        is_filler = sum(c.isalpha() for c in w) < 2
        if is_filler and len(capped) >= max_filler and all(p == w for p in capped[-max_filler:]):
            continue
        capped.append(w)
    return " ".join(capped)


def load_train():
    return [{"ID": r["ID"].strip(), "Target": clean(r["Target"])} for r in read_csv(DATA / "Train.csv")]


def load_test_ids():
    return [r["ID"].strip() for r in read_csv(DATA / "Test.csv")]


def split(rows, val_frac=0.08, seed=42):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = int(len(rows) * val_frac)
    return rows[n_val:], rows[:n_val]


BAD_LABEL = re.compile(r"missing text|illegible", re.I)


def augment(img, rng):
    """Mild, text-preserving augmentation for training only."""
    if rng.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
        g = rng.uniform(0.8, 1.2)
        img = img.point(lambda v: 255 * (v / 255) ** g)
    if rng.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    bg = tuple(int(c) for c in ImageStat.Stat(img).median)
    if rng.random() < 0.5:
        img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=True, fillcolor=bg)
    if rng.random() < 0.5:
        w, h = img.size
        pt, pb = (int(h * rng.uniform(0, 0.08)) for _ in range(2))
        pl, pr = (int(w * rng.uniform(0, 0.03)) for _ in range(2))
        canvas = Image.new("RGB", (w + pl + pr, h + pt + pb), bg)
        canvas.paste(img, (pl, pt))
        img = canvas
    return img


def load_image(image_id, max_area=300_000, max_upscale=2.5, max_side=3072, rng=None):
    img = Image.open(IMAGES / f"{image_id}.jpg").convert("RGB")
    if rng is not None:
        img = augment(img, rng)
        max_area = int(max_area * rng.uniform(0.73, 1.4))
    w, h = img.size
    s = min((max_area / (w * h)) ** 0.5, max_upscale, max_side / max(w, h))
    nw, nh = max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)
    return img.resize((nw, nh), Image.BICUBIC)


def score(refs, hyps):
    """Length-weighted CER/WER (total edits / total reference length), final = 0.5*CER + 0.5*WER."""
    ce = cn = we = wn = 0
    for r, h in zip(refs, hyps):
        r, h = clean(r), clean(h)
        ce += Levenshtein.distance(r, h)
        cn += max(len(r), 1)
        rw, hw = r.split(), h.split()
        we += Levenshtein.distance(rw, hw)
        wn += max(len(rw), 1)
    cer, wer = ce / cn, we / wn
    return {"cer": cer, "wer": wer, "final": 0.5 * cer + 0.5 * wer}


def write_submission(path, ids, preds):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Target"])
        for i, p in zip(ids, preds):
            w.writerow([i, clean(p)])
