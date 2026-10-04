import glob, os, shutil, subprocess, sys, time
MODE = "__MODE__"
def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print(f"command failed ({r.returncode})", flush=True)
    return r.returncode
t0 = time.time()
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q -U 'transformers==5.17.0' 'peft==0.21.0' 'bitsandbytes==0.50.2' accelerate rapidfuzz 2>&1 | tail -3")
data_root = os.path.dirname(glob.glob("/kaggle/input/**/Train.csv", recursive=True)[0])
code_root = os.path.dirname(glob.glob("/kaggle/input/**/train_vlm.py", recursive=True)[0])
img_dir = os.path.dirname(glob.glob(f"{data_root}/**/02atqnjsbbPPWAfx.jpg", recursive=True)[0])
work = "/kaggle/working/road-htr"
shutil.copytree(code_root, work, dirs_exist_ok=True)
os.chdir(work)
os.environ.update(ROAD_DATA=data_root, ROAD_IMAGES=img_dir, ROAD_FP16="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
print("data", data_root, "images", img_dir, "code", code_root, flush=True)
M = "Qwen/Qwen3-VL-8B-Instruct"
if MODE == "smoke":
    sh(f"python train_vlm.py --model {M} --limit 40 --epochs 1 --bs 1 --accum 8 --out runs/smoke 2>&1 | grep -vE 'Warning|Loading weights'")
    sh("nvidia-smi --query-gpu=memory.used --format=csv")
elif MODE == "predict":
    for e in (2, 0):
        ad = os.path.dirname(glob.glob(f"/kaggle/input/**/runs/qwen3vl8b/epoch{e}/adapter_config.json", recursive=True)[0])
        sh(f"python predict_vlm.py --model {M} --adapter {ad} --bs 4 --out submissions/qwen3vl8b_e{e}.csv 2>&1 | tail -2")
elif MODE == "full_all":
    budget = 590 - (time.time() - t0) / 60
    sh(f"python train_vlm.py --model {M} --full_data --pseudo pseudo.json --drop_bad_labels --seed 1 --lora_dropout 0.1 --weight_decay 0.01 "
       f"--bs 1 --accum 8 --epochs 2 --time_budget_min {budget:.0f} --out runs/qwen3vl8b_all 2>&1 | grep -vE 'Warning|Loading weights'")
    sh(f"python predict_vlm.py --model {M} --adapter runs/qwen3vl8b_all/best --bs 4 --out submissions/qwen3vl8b_all.csv 2>&1 | tail -2")
elif MODE == "dual":
    from huggingface_hub import snapshot_download
    snapshot_download(M)
    budget = 585 - (time.time() - t0) / 60
    common = (f"python train_vlm.py --model {M} --full_data --pseudo pseudo.json --drop_bad_labels --lora_dropout 0.1 --weight_decay 0.01 "
              f"--bs 1 --accum 8 --epochs 2 --time_budget_min {budget:.0f}")
    jobs = [
        ("0", f"{common} --seed 2 --out runs/qwen3vl8b_s2", "runs/qwen3vl8b_s2", "qwen3vl8b_s2", 300000),
        ("1", f"{common} --seed 3 --max_area 450000 --out runs/qwen3vl8b_hr", "runs/qwen3vl8b_hr", "qwen3vl8b_hr", 450000),
    ]
    procs = []
    for gpu, cmd, run_dir, name, area in jobs:
        full = (f"CUDA_VISIBLE_DEVICES={gpu} {cmd} > log_{name}.txt 2>&1; "
                f"CUDA_VISIBLE_DEVICES={gpu} python predict_vlm.py --model {M} --adapter {run_dir}/best --bs 4 --max_area {area} "
                f"--out submissions/{name}.csv >> log_{name}.txt 2>&1")
        print("$", full, flush=True)
        procs.append(subprocess.Popen(full, shell=True))
    for p in procs:
        p.wait()
    sh("grep -hE '^\\{|wrote|Traceback|Error|time budget' log_*.txt")
else:
    budget = 590 - (time.time() - t0) / 60
    sh(f"python train_vlm.py --model {M} --pseudo pseudo.json --drop_bad_labels --lora_dropout 0.1 --weight_decay 0.01 "
       f"--bs 1 --accum 8 --epochs 3 --time_budget_min {budget:.0f} --out runs/qwen3vl8b 2>&1 | grep -vE 'Warning|Loading weights'")
    sh(f"python predict_vlm.py --model {M} --adapter runs/qwen3vl8b/best --bs 4 --out submissions/qwen3vl8b.csv 2>&1 | tail -2")
print(f"total minutes {(time.time() - t0) / 60:.1f}", flush=True)
