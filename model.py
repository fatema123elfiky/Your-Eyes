# train_local_yolo.py
"""
Two-stage YOLOv8 local training script tuned for a laptop GPU (RTX 3050 4GB).
Features:
 - Auto-detect device (GPU/CPU) and VRAM -> adjust batch/imgsz/fp16/workers.
 - Stage1 = head warmup (freeze some backbone), Stage2 = unfreeze & fine-tune.
 - Save checkpoints periodically (save_period computed from SAVE_EVERY_IMAGES).
 - Early stopping via patience.
 - Helpful prints and safe defaults for low-VRAM machines.
 
Edit ROOT and DATA_YAML if your dataset path differs.
Run inside your venv after installing ultralytics and torch.
"""

import os
import math
import time
import json
import platform

# ---------- USER CONFIG (adjust these paths) ----------
# Point ROOT to your local dataset folder that contains data.yaml or images/labels
ROOT = r"D:\DEPI_Project"   # <<-- change if needed
DATA_YAML = os.path.join(ROOT, "data.yaml")

# Project / results folder (where runs will be saved)
PROJECT = os.path.join(ROOT, "yolo_runs_local")

# Model choice: you requested yolov8s. We keep yolov8s but auto-fallback logic exists.
MODEL_ARCH = "yolov8s.pt"

# Save every X images (approx) -> converted to save_period in epochs
SAVE_EVERY_IMAGES = 6000

# Early stopping patience (in epochs)
EARLY_STOP_PATIENCE = 20

# Default stage configs (will be tuned automatically based on device)
STAGE1 = {"epochs": 6, "batch": 2, "imgsz": 416, "lr0": 0.01, "freeze_layers": 10, "name": "stage1_head_warmup"}
STAGE2 = {"epochs": 20, "batch": 1, "imgsz": 416, "lr0": 0.0015, "freeze_layers": 0, "name": "stage2_finetune_full"}

# -----------------------------------------------------

def detect_system():
    """Detect CPU cores, RAM, GPU availability & VRAM, and torch if installed.
    Returns a dict with cpu/ram/gpu/torch info for later decisions.
    """
    info = {}
    # CPU cores
    try:
        info["cpu_logical"] = os.cpu_count() or 1
    except Exception:
        info["cpu_logical"] = 1

    # RAM (bytes)
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total"] = vm.total
        info["ram_available"] = vm.available
    except Exception:
        info["ram_total"] = None
        info["ram_available"] = None

    # Torch & CUDA
    try:
        import torch
        info["torch_installed"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        gpus = []
        if info["cuda_available"]:
            for i in range(info["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                gpus.append({"id": i, "name": props.name, "total_memory": int(props.total_memory)})
        info["gpus"] = gpus
    except Exception:
        info["torch_installed"] = False
        info["torch_version"] = None
        info["cuda_available"] = False
        info["gpu_count"] = 0
        info["gpus"] = []

    # nvidia-smi fallback (if torch not available)
    if not info["gpus"]:
        try:
            import subprocess
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode().strip()
            if out:
                gpus = []
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    gpus.append({"id": int(parts[0]), "name": parts[1], "total_memory": int(float(parts[2]) * 1024**2)})
                info["gpus"] = gpus
                info["cuda_available"] = True if gpus else info["cuda_available"]
                info["gpu_count"] = len(gpus)
        except Exception:
            pass

    # Platform
    info["platform"] = platform.platform()

    return info

def choose_hyperparams_by_device(sysinfo):
    """Choose model device, batch, imgsz, fp16 and workers based on detected GPU VRAM and RAM.
    Returns a dict with device, model_arch, batch, imgsz, fp16, workers.
    """
    cfg = {}
    # Default assume CPU-only
    if sysinfo.get("cuda_available") and sysinfo.get("gpus"):
        # pick first (id 0) GPU
        gpu = sysinfo["gpus"][0]
        vram_gb = gpu["total_memory"] / (1024**3)
        cfg["device"] = f"cuda:{gpu['id']}"
        # set recommended model (you asked yolov8s; still pick yolov8s but batch/imgsz adapt)
        cfg["model_arch"] = MODEL_ARCH
        # heuristics for batch/imgsz
        if vram_gb >= 12:
            cfg["imgsz"] = 640
            cfg["batch_stage1"] = 8
            cfg["batch_stage2"] = 8
            cfg["fp16"] = True
            cfg["workers"] = max(1, min(6, sysinfo["cpu_logical"] // 2))
        elif vram_gb >= 8:
            cfg["imgsz"] = 640
            cfg["batch_stage1"] = 4
            cfg["batch_stage2"] = 2
            cfg["fp16"] = True
            cfg["workers"] = max(1, min(4, sysinfo["cpu_logical"] // 3))
        elif vram_gb >= 6:
            cfg["imgsz"] = 512
            cfg["batch_stage1"] = 2
            cfg["batch_stage2"] = 1
            cfg["fp16"] = True
            cfg["workers"] = 1
        else:
            # small VRAM (like your 4GB)
            cfg["imgsz"] = 416   # smaller to reduce mem
            cfg["batch_stage1"] = 2
            cfg["batch_stage2"] = 1
            cfg["fp16"] = False   # fp16 might help but on some laptops can be unstable without correct torch build
            cfg["workers"] = 0
    else:
        cfg["device"] = "cpu"
        cfg["model_arch"] = "yolov8n.pt" if not sysinfo.get("torch_installed") else MODEL_ARCH
        cfg["imgsz"] = 416
        cfg["batch_stage1"] = 1
        cfg["batch_stage2"] = 1
        cfg["fp16"] = False
        cfg["workers"] = 0

    # safety clamps
    cfg["workers"] = max(0, int(cfg.get("workers", 0)))
    cfg["batch_stage1"] = max(1, int(cfg.get("batch_stage1", 1)))
    cfg["batch_stage2"] = max(1, int(cfg.get("batch_stage2", 1)))
    return cfg

def count_images_in_train(data_yaml_path):
    """Try to read data.yaml and count images in the train path. Returns (n_images, train_path)"""
    try:
        import yaml
    except Exception:
        yaml = None

    train_path = None
    base = os.path.dirname(data_yaml_path)
    # naive parse if pyyaml not present
    try:
        if yaml:
            with open(data_yaml_path) as f:
                d = yaml.safe_load(f)
            train_rel = d.get("train")
            base_path = d.get("path") or base
            if train_rel:
                train_path = train_rel if os.path.isabs(train_rel) else os.path.join(base_path, train_rel)
        else:
            with open(data_yaml_path) as f:
                for line in f:
                    if line.strip().startswith("train:"):
                        train_rel = line.split(":",1)[1].strip()
                        base_path = None
                        # if file path given absolute use direct, else join with base
                        if os.path.isabs(train_rel):
                            train_path = train_rel
                        else:
                            train_path = os.path.join(base, train_rel)
                        break
    except Exception:
        train_path = None

    # fallback to common layout
    if not train_path:
        cand = os.path.join(base, "images", "train")
        if os.path.exists(cand):
            train_path = cand

    if not train_path or not os.path.exists(train_path):
        return 0, train_path

    exts = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(train_path) if f.lower().endswith(exts)]
    return len(files), train_path

def compute_save_period(num_train_images, save_every_images):
    """Convert 'save every X images' to 'save every N epochs' (integer >=1)."""
    if num_train_images <= 0:
        return 1
    return max(1, math.ceil(save_every_images / num_train_images))

def run_stage(model, stage_cfg, data_yaml, project, device, workers, save_every_images, early_stop_patience, fp16=False):
    """Run a single training stage using Ultralytics YOLO API.
    stage_cfg: dict with keys epochs,batch,imgsz,lr0,freeze_layers,name
    """
    from ultralytics import YOLO  # import here to avoid global failure if not installed
    print(f"\n--- RUN STAGE: {stage_cfg.get('name')} ---")
    epochs = int(stage_cfg.get("epochs", 10))
    batch = int(stage_cfg.get("batch", 1))
    imgsz = int(stage_cfg.get("imgsz", 640))
    lr0 = stage_cfg.get("lr0", None)
    freeze = int(stage_cfg.get("freeze_layers", 0))
    name = stage_cfg.get("name", "run")

    # compute save_period
    n_train, train_path = count_images_in_train(data_yaml)
    save_period = compute_save_period(n_train, save_every_images)
    print(f"Train images: {n_train}  (train path: {train_path})")
    print(f"Saving every ~{save_every_images} images -> save_period (epochs) = {save_period}")
    print(f"Early stopping patience (epochs) = {early_stop_patience}")
    print(f"Using device: {device} | workers: {workers} | fp16: {fp16}")
    print(f"Stage params: epochs={epochs}, batch={batch}, imgsz={imgsz}, freeze={freeze}, lr0={lr0}")

    # call Ultralytics training
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        cache=False,           # avoid huge RAM usage on laptop; set True if you have fast NVMe and plenty of RAM
        augment=True,
        save=True,
        name=name,
        project=project,
        workers=workers,
        freeze=freeze,
        patience=early_stop_patience,
        save_period=save_period,
        half=fp16              # use mixed precision if enabled
    )
    print(f"Stage {name} finished.")

def train_two_stage(data_yaml=DATA_YAML, project=PROJECT, save_every_images=SAVE_EVERY_IMAGES, early_stop_patience=EARLY_STOP_PATIENCE):
    """Main orchestration: detect system -> choose hyperparams -> run Stage1 then Stage2."""
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    sysinfo = detect_system()
    print("\n=== System detection summary ===")
    print(json.dumps(sysinfo, indent=2, default=str))

    hwcfg = choose_hyperparams_by_device(sysinfo)
    print("\n=== Hardware-based recommendations ===")
    print(json.dumps(hwcfg, indent=2))

    # adapt stage configs from defaults + hw recommendations
    s1 = STAGE1.copy()
    s2 = STAGE2.copy()
    # use recommended imgsz and batches
    s1["imgsz"] = hwcfg["imgsz"]
    s2["imgsz"] = hwcfg["imgsz"]
    s1["batch"] = hwcfg["batch_stage1"]
    s2["batch"] = hwcfg["batch_stage2"]

    # device & fp16
    device = hwcfg["device"]
    use_fp16 = hwcfg["fp16"]

    # adjust workers for Windows safe default
    workers = hwcfg["workers"]

    # instantiate model
    from ultralytics import YOLO
    print("\nLoading model:", MODEL_ARCH)
    model = YOLO(MODEL_ARCH)

    # create project dir
    os.makedirs(project, exist_ok=True)

    # Stage1
    run_stage(model, s1, data_yaml, project, device=device, workers=workers, save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)

    # Stage2
    run_stage(model, s2, data_yaml, project, device=device, workers=workers, save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)

    print("\nAll stages finished. Check results under:", project)
    print("Best weights (likely) at:", os.path.join(project, s2["name"], "weights", "best.pt"))
    print("You can run evaluation (example):")
    print(f'yolo detect val model="{os.path.join(project, s2["name"], "weights", "best.pt")}" data="{data_yaml}" imgsz={s2["imgsz"]}')

if __name__ == "__main__":
    t0 = time.time()
    train_two_stage()
    print("Total elapsed (s):", int(time.time() - t0))
