import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from common import BAD_LABEL, load_train, score, split
from vlm import DTYPE, generate, load_model, load_processor, train_batch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
ap.add_argument("--out", default="runs/qwen3vl2b")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--full_data", action="store_true", help="train on train+val (no validation)")
ap.add_argument("--limit", type=int, default=0, help="debug: use only N training rows")
ap.add_argument("--no_4bit", action="store_true")
ap.add_argument("--resume_from", help="adapter dir of the last finished epoch, e.g. runs/x/epoch0")
ap.add_argument("--start_epoch", type=int, default=0)
ap.add_argument("--max_area", type=int, default=300_000)
ap.add_argument("--augment", action="store_true")
ap.add_argument("--lora_dropout", type=float, default=0.05)
ap.add_argument("--weight_decay", type=float, default=0.0)
ap.add_argument("--drop_bad_labels", action="store_true")
ap.add_argument("--pseudo", help="JSON list of {ID, Target} pseudo-labelled test lines to add to training")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time_budget_min", type=float, default=0, help="stop training after this many minutes (Kaggle 12h limit)")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)

rows = load_train()
train, val = (rows, []) if args.full_data else split(rows)
if args.drop_bad_labels:
    train = [r for r in train if not BAD_LABEL.search(r["Target"])]
if args.pseudo:
    train = train + json.load(open(args.pseudo))
if args.limit:
    train, val = train[:args.limit], val[:max(16, args.limit // 10)]
print(f"train={len(train)} val={len(val)}", flush=True)

proc = load_processor(args.model)
model = load_model(args.model, four_bit=not args.no_4bit)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
if args.resume_from:
    model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=args.lora_dropout,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
steps_per_epoch = math.ceil(len(train) / (args.bs * args.accum))
sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps_per_epoch * args.epochs),
                                        steps_per_epoch * args.epochs)

for _ in range(args.start_epoch * steps_per_epoch):
    sched.step()

scaler = torch.amp.GradScaler("cuda", enabled=DTYPE == torch.float16)
t_start = time.time()
out_of_time = False
best = None
if (out / "log.jsonl").exists():
    finals = [json.loads(l).get("final") for l in open(out / "log.jsonl") if l.strip()]
    best = min((f for f in finals if f is not None), default=None)
log = open(out / "log.jsonl", "a")
for epoch in range(args.start_epoch, args.epochs):
    model.train()
    random.Random(1000 * args.seed + epoch).shuffle(train)
    aug_rng = random.Random(7919 * args.seed + epoch) if args.augment else None
    t0, running = time.time(), 0.0
    n_batches = math.ceil(len(train) / args.bs)
    for b in range(n_batches):
        chunk = train[b * args.bs:(b + 1) * args.bs]
        enc = train_batch(proc, [r["ID"] for r in chunk], [r["Target"] for r in chunk], args.max_area, aug_rng).to(model.device)
        with torch.autocast("cuda", dtype=DTYPE):
            loss = model(**enc).loss / args.accum
        scaler.scale(loss).backward()
        running += loss.item()
        if (b + 1) % args.accum == 0 or b + 1 == n_batches:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if (b + 1) % (25 * args.accum) == 0:
            el = time.time() - t0
            print(f"ep {epoch} batch {b + 1}/{n_batches} loss {running / 25:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} eta {el / (b + 1) * (n_batches - b - 1) / 60:.1f}m", flush=True)
            running = 0.0
        if args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min:
            print(f"time budget reached at batch {b + 1}; saving and stopping", flush=True)
            out_of_time = True
            break

    model.save_pretrained(out / f"epoch{epoch}")
    rec = {"epoch": epoch, "minutes": (time.time() - t0) / 60}
    if val:
        model.config.use_cache = True
        preds = generate(model, proc, [r["ID"] for r in val], max_area=args.max_area)
        model.config.use_cache = False
        rec.update(score([r["Target"] for r in val], preds))
        with open(out / f"val_preds_epoch{epoch}.json", "w") as f:
            json.dump([{"ID": r["ID"], "ref": r["Target"], "hyp": p} for r, p in zip(val, preds)], f, indent=1)
        if best is None or rec["final"] < best:
            best = rec["final"]
            model.save_pretrained(out / "best")
    else:
        model.save_pretrained(out / "best")
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    log.flush()
    if out_of_time:
        break
