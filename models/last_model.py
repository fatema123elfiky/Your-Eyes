# train_local_yolo.py
"""
Two-stage YOLOv8 local training script tuned for a laptop GPU (RTX 3050 4GB).
Updated:
 - respect user-defined STAGE1/STAGE2 unless you mark them "auto": True
 - more robust count_images_in_train() (recursive, supports lists, txt files)
"""

import os
import math
import time
import json
import platform

# ---------- USER CONFIG (adjust these paths) ----------
ROOT = r"D:\DEPI_Project"   # <<-- change if needed
DATA_YAML = os.path.join(ROOT, "data.yaml")

PROJECT = os.path.join(ROOT, "yolo_runs_local")
MODEL_ARCH = "yolov8s.pt"
SAVE_EVERY_IMAGES = 6000
EARLY_STOP_PATIENCE = 7

# NOTE: if you want the script to override these automatically based on GPU,
# set "auto": True in the stage dict below. Otherwise the values you provide
# will be used as-is (this is the default now).

# Default stage configs (will be tuned automatically based on device)
# STAGE1 = {"epochs": 6, "batch": 2, "imgsz": 416, "lr0": 0.01, "freeze_layers": 10, "name": "stage1_head_warmup"}
# STAGE2 = {"epochs": 20, "batch": 1, "imgsz": 416, "lr0": 0.0015, "freeze_layers": 0, "name": "stage2_finetune_full"}

STAGE1 = {"epochs": 10, "batch": 2, "imgsz": 416, "lr0": 0.01,   "freeze_layers": 10, "name": "stage1_head_warmup", "auto": False}
STAGE2 = {"epochs": 13, "batch": 1, "imgsz": 416, "lr0": 0.00015, "freeze_layers": 0,  "name": "stage2_finetune_full", "auto": False}

RESUME_STRATEGY = "load_weights"

# -----------------------------------------------------

def detect_system():
    info = {}
    try:
        info["cpu_logical"] = os.cpu_count() or 1
    except Exception:
        info["cpu_logical"] = 1

    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total"] = vm.total
        info["ram_available"] = vm.available
    except Exception:
        info["ram_total"] = None
        info["ram_available"] = None

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

    if not info["gpus"]:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL).decode().strip()
            if out:
                gpus = []
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    # memory.total comes in MiB sometimes; we standardize to bytes
                    mem_mib = float(parts[2])
                    gpus.append({"id": int(parts[0]), "name": parts[1], "total_memory": int(mem_mib * 1024**2)})
                info["gpus"] = gpus
                info["cuda_available"] = True if gpus else info["cuda_available"]
                info["gpu_count"] = len(gpus)
        except Exception:
            pass

    info["platform"] = platform.platform()
    return info

def choose_hyperparams_by_device(sysinfo):
    cfg = {}
    if sysinfo.get("cuda_available") and sysinfo.get("gpus"):
        gpu = sysinfo["gpus"][0]
        vram_gb = gpu["total_memory"] / (1024**3)
        cfg["device"] = f"cuda:{gpu['id']}"
        cfg["model_arch"] = MODEL_ARCH
        if vram_gb >= 12:
            cfg["imgsz"] = 640; cfg["batch_stage1"] = 8; cfg["batch_stage2"] = 8; cfg["fp16"] = True; cfg["workers"] = max(1, min(6, sysinfo["cpu_logical"] // 2))
        elif vram_gb >= 8:
            cfg["imgsz"] = 640; cfg["batch_stage1"] = 4; cfg["batch_stage2"] = 2; cfg["fp16"] = True; cfg["workers"] = max(1, min(4, sysinfo["cpu_logical"] // 3))
        elif vram_gb >= 6:
            cfg["imgsz"] = 512; cfg["batch_stage1"] = 2; cfg["batch_stage2"] = 1; cfg["fp16"] = True; cfg["workers"] = 1
        else:
            cfg["imgsz"] = 416; cfg["batch_stage1"] = 2; cfg["batch_stage2"] = 1; cfg["fp16"] = False; cfg["workers"] = 0
    else:
        cfg["device"] = "cpu"
        cfg["model_arch"] = "yolov8n.pt" if not sysinfo.get("torch_installed") else MODEL_ARCH
        cfg["imgsz"] = 416; cfg["batch_stage1"] = 1; cfg["batch_stage2"] = 1; cfg["fp16"] = False; cfg["workers"] = 0

    cfg["workers"] = max(0, int(cfg.get("workers", 0)))
    cfg["batch_stage1"] = max(1, int(cfg.get("batch_stage1", 1)))
    cfg["batch_stage2"] = max(1, int(cfg.get("batch_stage2", 1)))
    return cfg

def count_images_in_train(data_yaml_path):
    """
    Robust counting of training images:
     - parses data.yaml if available (supports train: string or list)
     - supports train path being a directory (counts recursively) or a .txt file listing images
     - returns (n_images, train_path_list) where train_path_list is a list of resolved paths used
    """
    try:
        import yaml
    except Exception:
        yaml = None

    base = os.path.dirname(data_yaml_path)
    train_entries = []

    # try YAML parsing
    try:
        if yaml:
            with open(data_yaml_path, 'r') as f:
                cfg = yaml.safe_load(f)
            # common YOLO fields: train can be str or list
            train_field = cfg.get("train") if isinstance(cfg, dict) else None
            if isinstance(train_field, list):
                for t in train_field:
                    train_entries.append(t)
            elif isinstance(train_field, str):
                train_entries.append(train_field)
            else:
                # sometimes dataset uses 'path' + 'train' relative
                train_rel = cfg.get("train")
                base_path = cfg.get("path") or base
                if train_rel:
                    train_entries.append(train_rel if os.path.isabs(train_rel) else os.path.join(base_path, train_rel))
        else:
            # naive parse: look for a line starting with train:
            with open(data_yaml_path, 'r') as f:
                for line in f:
                    if line.strip().startswith("train:"):
                        train_rel = line.split(":",1)[1].strip()
                        if train_rel:
                            train_entries.append(train_rel)
                        break
    except Exception:
        train_entries = []

    # fallback common layout
    if not train_entries:
        cand = os.path.join(base, "images", "train")
        if os.path.exists(cand):
            train_entries = [cand]
        else:
            # last resort: look for images/train or images
            cand2 = os.path.join(base, "images")
            if os.path.exists(cand2):
                train_entries = [cand2]

    resolved_paths = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    total = 0

    for entry in train_entries:
        if not entry:
            continue
        path = entry if os.path.isabs(entry) else os.path.join(base, entry)
        path = os.path.normpath(path)
        if os.path.exists(path):
            # if it's a text file listing images (common in some setups)
            if os.path.isfile(path) and path.lower().endswith(('.txt', '.lst')):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = [l.strip() for l in f if l.strip()]
                    # filter image lines
                    n = sum(1 for l in lines if os.path.splitext(l)[1].lower() in exts or os.path.exists(l))
                    total += n
                    resolved_paths.append(path)
                except Exception:
                    continue
            elif os.path.isdir(path):
                # walk recursively and count image files
                n = 0
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.lower().endswith(exts):
                            n += 1
                total += n
                resolved_paths.append(path)
            else:
                # single file (image)
                if os.path.splitext(path)[1].lower() in exts and os.path.isfile(path):
                    total += 1
                    resolved_paths.append(path)
        else:
            # path doesn't exist -- skip
            continue

    return total, resolved_paths if resolved_paths else None

def compute_save_period(num_train_images, save_every_images):
    """Convert 'save every X images' to 'save every N epochs' (integer >=1).
    If images_per_epoch is smaller than save_every_images, result will be >=1.
    """
    if num_train_images <= 0:
        return 1
    # compute epochs needed to accumulate save_every_images images
    epochs = save_every_images / float(num_train_images)
    # we want to save every N epochs -> take ceil of reciprocal logic:
    # e.g., if epochs = 0.1 (i.e. save_every_images < num_train_images) -> ceil(0.1) = 1
    return max(1, math.ceil(epochs))
def run_stage(model,
              stage_cfg,
              data_yaml,
              project,
              device,
              workers,
              save_every_images,
              early_stop_patience,
              fp16=False,
              weights_arg=None,
              resume_flag=False):
    """
    weights_arg: path to weights file to pass as 'weights' argument to model.train (optional)
    resume_flag: if True, pass resume=True to model.train (attempt full resume)
    """
    from ultralytics import YOLO
    print(f"\n--- RUN STAGE: {stage_cfg.get('name')} ---")
    epochs = int(stage_cfg.get("epochs", 10))
    batch = int(stage_cfg.get("batch", 1))
    imgsz = int(stage_cfg.get("imgsz", 640))
    lr0 = stage_cfg.get("lr0", None)
    freeze = int(stage_cfg.get("freeze_layers", 0))
    name = stage_cfg.get("name", "run")

    n_train, train_path = count_images_in_train(data_yaml)
    save_period = compute_save_period(n_train, save_every_images)
    print(f"Train images: {n_train}  (train path(s): {train_path})")
    print(f"Saving every ~{save_every_images} images -> save_period (epochs) = {save_period}")
    print(f"Early stopping patience (epochs) = {early_stop_patience}")
    print(f"Using device: {device} | workers: {workers} | fp16: {fp16}")
    print(f"Stage params: epochs={epochs}, batch={batch}, imgsz={imgsz}, freeze={freeze}, lr0={lr0}")
    if weights_arg:
        print(f"Starting from weights: {weights_arg} (passed as 'weights')")
    if resume_flag:
        print("Calling model.train with resume=True (attempt full resume)")

    # build kwargs
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        cache=False,
        augment=True,
        save=True,
        name=name,
        project=project,
        workers=workers,
        freeze=freeze,
        patience=early_stop_patience,
        save_period=save_period,
        half=fp16
    )

    if weights_arg:
        train_kwargs["weights"] = weights_arg
    if resume_flag:
        train_kwargs["resume"] = True
    # ensure model argument is set so ultralytics has explicit model path/name
    train_kwargs["model"] = MODEL_ARCH

    model.train(**train_kwargs)
    print(f"Stage {name} finished.")

def train_two_stage(data_yaml=DATA_YAML, project=PROJECT, save_every_images=SAVE_EVERY_IMAGES, early_stop_patience=EARLY_STOP_PATIENCE):
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

    # Only override user STAGE settings if stage dict explicitly asks for auto
    if s1.get("auto", False):
        s1["imgsz"] = hwcfg["imgsz"]
        s1["batch"] = hwcfg["batch_stage1"]
        print("Stage1: using auto hw recommendations (overrode user values).")
    else:
        print("Stage1: using user-provided hyperparameters (no auto-override).")

    if s2.get("auto", False):
        s2["imgsz"] = hwcfg["imgsz"]
        s2["batch"] = hwcfg["batch_stage2"]
        print("Stage2: using auto hw recommendations (overrode user values).")
    else:
        print("Stage2: using user-provided hyperparameters (no auto-override).")

    device = hwcfg["device"]
    use_fp16 = hwcfg["fp16"]
    workers = hwcfg["workers"]

    from ultralytics import YOLO
    print("\nLoading model:", MODEL_ARCH)
    model = YOLO(MODEL_ARCH)

    os.makedirs(project, exist_ok=True)

    # run_stage(model, s1, data_yaml, project, device=device, workers=workers, save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)
    # run_stage(model, s2, data_yaml, project, device=device, workers=workers, save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)

    # RUN Stage1
    run_stage(model, s1, data_yaml, project, device=device, workers=workers,
              save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)

    # after Stage1: try to find last.pt or best.pt
    s1_weights_dir = os.path.join(project, s1["name"], "weights")
    last_w = os.path.join(s1_weights_dir, "last.pt")
    best_w = os.path.join(s1_weights_dir, "best.pt")
    chosen_weights = None
    if os.path.exists(last_w):
        chosen_weights = last_w
        print(f"Found last checkpoint: {last_w}")
    elif os.path.exists(best_w):
        chosen_weights = best_w
        print(f"Found best checkpoint: {best_w}")
    else:
        print("No checkpoint (last.pt/best.pt) found from Stage1 — Stage2 will start from the current in-memory model.")

    # Strategy: load_weights (safe) or resume_true (attempt)
    if chosen_weights and RESUME_STRATEGY == "load_weights":
        print("Stage2 will start from Stage1 weights (loading weights into a fresh YOLO instance).")
        from ultralytics import YOLO
        model2 = YOLO(chosen_weights)  # new model initialized from checkpoint weights
        run_stage(model2, s2, data_yaml, project, device=device, workers=workers,
                  save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)
    elif chosen_weights and RESUME_STRATEGY == "resume_true":
        print("Attempting full resume for Stage2 using resume=True.")
        run_stage(model, s2, data_yaml, project, device=device, workers=workers,
                  save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16,
                  weights_arg=chosen_weights, resume_flag=True)
    else:
        # fallback: use same in-memory model
        print("Stage2 will continue using the in-memory model object (no checkpoint loaded).")
        run_stage(model, s2, data_yaml, project, device=device, workers=workers,
                  save_every_images=save_every_images, early_stop_patience=early_stop_patience, fp16=use_fp16)

    print("\nAll stages finished. Check results under:", project)
    print("Best weights (likely) at:", os.path.join(project, s2["name"], "weights", "best.pt"))
    print("You can run evaluation (example):")
    print(f'yolo detect val model="{os.path.join(project, s2["name"], "weights", "best.pt")}" data="{data_yaml}" imgsz={s2["imgsz"]}')

if __name__ == "__main__":
    t0 = time.time()
    train_two_stage()
    print("Total elapsed (s):", int(time.time() - t0))
