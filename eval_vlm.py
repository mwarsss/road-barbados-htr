import argparse
import json
import time

from peft import PeftModel

from common import load_train, score, split
from vlm import generate, load_model, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--adapter", default="runs/qwen3vl2b/best")
ap.add_argument("--configs", default="1:300000,3:300000,1:600000", help="beams:max_area,...")
ap.add_argument("--bs", type=int, default=8)
args = ap.parse_args()

_, val = split(load_train())
ids, refs = [r["ID"] for r in val], [r["Target"] for r in val]
proc = load_processor(args.model)
model = PeftModel.from_pretrained(load_model(args.model), args.adapter)
for cfg in args.configs.split(","):
    beams, area = map(int, cfg.split(":"))
    t0 = time.time()
    preds = generate(model, proc, ids, batch_size=args.bs if beams == 1 else max(1, args.bs // beams),
                     num_beams=beams, max_area=area)
    res = score(refs, preds)
    print(json.dumps({"beams": beams, "max_area": area, "minutes": round((time.time() - t0) / 60, 1), **res}), flush=True)
    json.dump(dict(zip(ids, preds)), open(f"{args.adapter}/val_b{beams}_a{area}.json", "w"), indent=1)
