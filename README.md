# R.O.A.D. Barbados Historic Handwriting Recognition

Fine-tuning Qwen3-VL vision-language models with QLoRA to transcribe single lines of 17th–18th century Barbadian archive manuscripts, for the [Zindi R.O.A.D. Barbados Historic Handwriting Challenge](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge) (September–October 2026).

**Result:** public leaderboard score **0.8827 → 0.9196** in five days, about a third fewer errors, using a laptop with a **6 GB GPU** (RTX 3050) plus **free Kaggle T4s**. The final submission combined 17 models. Around 240th of ~690 active teams on the public leaderboard.

## The task

- 4,098 labelled training lines and 1,374 test lines, each a cropped image of one line of handwriting, mostly 17th-century secretary hand.
- Transcribe exactly: archaic spelling ("publique", "heires"), superscript abbreviations written with a caret (`y^e`, `S:^d`, `W^m`), inconsistent capitalisation.
- Metric: 50% weighted word error rate + 50% weighted character error rate. The public score works out to `0.5·(1 − word_edits/12) + 0.5·(1 − char_edits/55)`, so one wrong word costs ~4.6× one wrong character.

## Approach

```
line images ──► resize to ~300k px ──► Qwen3-VL 2B/4B (laptop) ┐
                                       Qwen3-VL 8B  (Kaggle T4) ┴► greedy decode + loop clean-up
                                                                      │
                     high-agreement test lines ◄── MBR selection across models ──► submission
                     (pseudo-labels, fed back into training)
```

1. **QLoRA fine-tuning.** 4-bit NF4 weights, LoRA rank 16 / alpha 32 on all linear layers including the vision encoder, AdamW 2e-4 with cosine schedule, effective batch 8, gradient checkpointing, loss on the transcription tokens only.
2. **A trustworthy local score.** A fixed 8% hold-out (327 lines) scored with the same formula tracked the leaderboard within ~0.003 across all submissions.
3. **Measured negative results.** Vocabulary-based spelling correction, case/caret/punctuation rules, higher-resolution retraining and image augmentation all made held-out error worse.
4. **Minimum Bayes risk (MBR) ensembling.** For each line, keep the candidate with the smallest total distance (character edits + ~5.5 × word edits) to all other models' candidates. Diverse models matter far more than more copies of one model.
5. **Automated pseudo-labelling.** Test lines where most models agree exactly (~95% accurate on held-out data) are added to training for "all-data" models.
6. **A bigger model on free GPUs.** Qwen3-VL-8B trained on Kaggle T4s with float16 + gradient scaling (the T4 has no bfloat16), with a time budget for the 12-hour session limit and two runs in parallel, one per GPU.

### Held-out error (lower is better)

| Model | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Qwen3-VL-2B | 0.113 | 0.102 | 0.100 |
| Qwen3-VL-4B | 0.120 | 0.0995 | 0.0993 |
| Qwen3-VL-8B (Kaggle) | 0.101 | **0.0915** | 0.0935 |

| Ensemble | Held-out error |
|---|---|
| Best single laptop model | 0.098 |
| 9 laptop checkpoints | 0.0861 |
| + three 8B checkpoints | 0.0836 |

### Leaderboard progression

| Submission | Public score | Change |
|---|---|---|
| #1 | 0.8827 | 2B, 1 epoch |
| #2 | 0.8970 | 2B, 3 epochs |
| #4 | 0.9079 | 8 checkpoints, MBR |
| #5 | 0.9116 | + 4B on all data + pseudo-labels |
| #8 | 0.9140 | + 8B (Kaggle) |
| #10 | 0.9182 | + 8B on all data + pseudo-labels |
| #11 | **0.9196** | 17 models |

## Repository layout

| File | Purpose |
|---|---|
| `common.py` | Data loading, hold-out split, image resizing/augmentation, metric, loop clean-up |
| `vlm.py` | Model/processor loading (4-bit), training batches with label masking, batched generation |
| `train_vlm.py` | QLoRA training with resume, augmentation, pseudo-labels, all-data mode, time budget |
| `predict_vlm.py` | Test-set inference to CSV + JSON |
| `eval_vlm.py` | Compare decoding settings (beams, resolution) on the hold-out |
| `make_pseudo.py` | Build pseudo-labels from agreement between models' test predictions |
| `analysis/ensemble.py` | MBR / ROVER ensembling and hold-out evaluation |
| `postprocess.py` | Vocabulary-snapping experiment (kept as a documented negative result) |
| `kaggle/run_template.py` | Kaggle kernel runner (single, all-data, prediction and dual-GPU modes) |

## Reproducing

The competition data is not redistributable. Download it from the Zindi data page and place it as:

```
data/Train.csv
data/Test.csv
data/images/<ID>.jpg
```

Set up and train (examples):

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2B on the training split, scored on the hold-out each epoch
python train_vlm.py --epochs 3 --out runs/qwen3vl2b

# 4B fits a 6 GB GPU at batch size 1
python train_vlm.py --model Qwen/Qwen3-VL-4B-Instruct --bs 1 --accum 8 --out runs/qwen3vl4b

# test predictions from the best epoch
python predict_vlm.py --adapter runs/qwen3vl2b/best --out submissions/qwen3vl2b.csv

# pseudo-labels from several models' test predictions, then an all-data model
python make_pseudo.py --min_agree 5 --out data/pseudo.json submissions/*.json
python train_vlm.py --full_data --pseudo data/pseudo.json --drop_bad_labels --out runs/qwen3vl2b_all

# combine models (list the strongest first; it breaks ties)
python analysis/ensemble.py --method mbr --collapse submissions/a.json submissions/b.json submissions/c.json --out submission.csv
# or evaluate a combination on hold-out predictions
python analysis/ensemble.py --method mbr --eval runs/*/val_preds_epoch2.json
```

Environment variables used on Kaggle: `ROAD_FP16=1` (float16 for GPUs without bfloat16), `ROAD_DATA` and `ROAD_IMAGES` (data locations). If `/tmp` is small (e.g. WSL), set `TMPDIR` to a folder on disk before installing PyTorch.

## Sources

- Hu et al., 2021 — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Dettmers et al., 2023 — [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Kumar & Byrne, 2004 — Minimum Bayes-Risk Decoding for Statistical Machine Translation (HLT-NAACL)
- Fiscus, 1997 — Recognizer Output Voting Error Reduction (ROVER), IEEE ASRU
- Lee, 2013 — Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML Workshop)
- Models: [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- Competition discussions: [leaderboard formula & negative results](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34861), [pseudo-labelling ruling](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34459), [external data ruling](https://zindi.world/competitions/road-barbados-historic-handwriting-challenge/discussions/34734)

## License

Code: MIT (see `LICENSE`). Competition data and the Qwen3-VL model weights are under their own terms.
