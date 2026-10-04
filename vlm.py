import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from common import PROMPT, collapse_repeats, load_image

END = "<|im_end|>"
DTYPE = torch.float16 if os.environ.get("ROAD_FP16") else torch.bfloat16


def load_processor(model_id):
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = "right"
    return proc


def load_model(model_id, four_bit=True):
    kw = dict(dtype=DTYPE, device_map={"": 0}, low_cpu_mem_usage=True)
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE,
        )
    return AutoModelForImageTextToText.from_pretrained(model_id, **kw)


def prompt_text(proc):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def train_batch(proc, ids, targets, max_area=300_000, rng=None):
    p = prompt_text(proc)
    images = [load_image(i, max_area=max_area, rng=rng) for i in ids]
    texts = [p + t + END for t in targets]
    enc = proc(text=texts, images=images, padding=True, return_tensors="pt")
    labels = torch.full_like(enc["input_ids"], -100)
    lengths = enc["attention_mask"].sum(1)
    for k, t in enumerate(targets):
        n = len(proc.tokenizer(t + END, add_special_tokens=False)["input_ids"])
        end = int(lengths[k])
        labels[k, end - n:end] = enc["input_ids"][k, end - n:end]
    enc["labels"] = labels
    return enc


@torch.no_grad()
def generate(model, proc, ids, batch_size=8, num_beams=1, max_new_tokens=96, max_area=300_000):
    model.eval()
    proc.tokenizer.padding_side = "left"
    p = prompt_text(proc)
    out = []
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        enc = proc(text=[p] * len(chunk), images=[load_image(i, max_area=max_area) for i in chunk],
                   padding=True, return_tensors="pt").to(model.device)
        with torch.autocast("cuda", dtype=DTYPE):
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=num_beams, repetition_penalty=1.0)
        gen = gen[:, enc["input_ids"].shape[1]:]
        out += [collapse_repeats(t) for t in proc.batch_decode(gen, skip_special_tokens=True)]
    proc.tokenizer.padding_side = "right"
    return out
