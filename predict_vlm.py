import argparse
import json

from peft import PeftModel

from common import ROOT, load_test_ids, write_submission
from vlm import generate, load_model, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--adapter", default="runs/qwen3vl2b/best")
ap.add_argument("--out", default="submissions/qwen3vl2b.csv")
ap.add_argument("--beams", type=int, default=1)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max_area", type=int, default=300_000)
args = ap.parse_args()

proc = load_processor(args.model)
model = PeftModel.from_pretrained(load_model(args.model), args.adapter)
ids = load_test_ids()
preds = generate(model, proc, ids, batch_size=args.bs, num_beams=args.beams, max_area=args.max_area)

out = ROOT / args.out
out.parent.mkdir(parents=True, exist_ok=True)
write_submission(out, ids, preds)
with open(out.with_suffix(".json"), "w") as f:
    json.dump(dict(zip(ids, preds)), f, indent=1)
print(f"wrote {out} ({len(ids)} rows, {sum(not p for p in preds)} empty)")
