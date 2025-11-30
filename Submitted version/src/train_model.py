import argparse
from pathlib import Path
from ultralytics import YOLO


def train_youreyes_model(
    model_size: str = "n",
    data_yaml: str = "data.yaml",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    project_name: str = "youreyes_model",
    resume: bool = False
):
    """
    Train YOLO model on Your Eyes dataset
    
    Args:
        model_size: YOLO model size (n, s, m, l, x)
        data_yaml: Path to data configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        project_name: Name for this training run
        resume: Resume from last checkpoint
    """
    
    print("=" * 70)
    print("Your Eyes - YOLO Model Training")
    print("=" * 70)
    print()
    
    # Validate data.yaml exists
    if not Path(data_yaml).exists():
        print(f"❌ Error: {data_yaml} not found!")
        print("Please create data.yaml with your dataset configuration")
        return
    
    # Select base model
    model_name = f"yolov8{model_size}.pt"
    print(f"📦 Base model: {model_name}")
    print(f"📊 Dataset config: {data_yaml}")
    print(f"🔄 Epochs: {epochs}")
    print(f"📏 Batch size: {batch_size}")
    print(f"🖼️  Image size: {img_size}")
    print(f"📁 Project name: {project_name}")
    print()
    
    # Load model
    print("Loading base model...")
    model = YOLO(model_name)
    
    # Training parameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'name': project_name,
        'patience': 10,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': False,  # Set to True if you have enough RAM
        'device': 0,  # Use GPU 0, or 'cpu' for CPU training
        'workers': 8,
        'project': 'runs/detect',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': resume,
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True
    }
    
    print("🚀 Starting training...")
    print("=" * 70)
    print()
    
    # Train
    try:
        results = model.train(**train_args)
        
        print()
        print("=" * 70)
        print("✅ Training completed successfully!")
        print("=" * 70)
        print()
        print(f"📁 Results saved to: runs/detect/{project_name}")
        print(f"🏆 Best model: runs/detect/{project_name}/weights/best.pt")
        print(f"📊 Last model: runs/detect/{project_name}/weights/last.pt")
        print()
        print("To use your trained model:")
        print(f"1. In the app, go to Settings")
        print(f"2. Set model path to: runs/detect/{project_name}/weights/best.pt")
        print(f"3. Click 'Load/Reload Model'")
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Training failed: {e}")
        print("=" * 70)
        print()


def validate_model(model_path: str, data_yaml: str = "data.yaml"):
    """
    Validate trained model on validation set
    
    Args:
        model_path: Path to trained model weights
        data_yaml: Path to data configuration
    """
    print("=" * 70)
    print("Validating Model")
    print("=" * 70)
    print()
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print()
    print("Validation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Your Eyes YOLO model")
    
    parser.add_argument(
        "--model",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to data.yaml configuration file"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="youreyes_model",
        help="Project name for this training run"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint"
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate a trained model (provide path to weights)"
    )
    
    args = parser.parse_args()
    
    if args.validate:
        validate_model(args.validate, args.data)
    else:
        train_youreyes_model(
            model_size=args.model,
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            project_name=args.name,
            resume=args.resume
        )

