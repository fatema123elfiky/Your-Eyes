import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class YourEyesDetector:

    PRIORITY_CLASSES = {
        'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train',
        'traffic light', 'stop sign', 'person', 'dog', 'cat'
    }

    INDOOR_CLASSES = {
        'person', 'chair', 'dining table', 'cup', 'bottle', 'bowl',
        'tv', 'couch', 'bed', 'laptop', 'book', 'cell phone',
        'handbag', 'backpack', 'knife', 'fork', 'spoon'
    }

    OUTDOOR_CLASSES = {
        'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train',
        'traffic light', 'stop sign', 'person', 'dog', 'cat', 'bench'
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def process_image(
        self,
        image_source,
        conf_threshold: Optional[float] = None,
        mode: str = "all"
    ) -> Tuple[List[Dict], np.ndarray]:
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model(image_source, conf=conf, verbose=False)
        result = results[0]

        detected_objects = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = self.model.names[cls_id]

            if mode == "indoor" and label not in self.INDOOR_CLASSES:
                continue
            elif mode == "outdoor" and label not in self.OUTDOOR_CLASSES:
                continue

            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            area = width * height

            detected_objects.append({
                "label": label,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "center_x": center_x,
                "area": area,
                "is_priority": label in self.PRIORITY_CLASSES
            })

        annotated_image = result.plot()

        return detected_objects, annotated_image

    def estimate_distance(self, area: float, image_width: int, image_height: int) -> str:
        image_area = image_width * image_height
        relative_size = area / image_area

        if relative_size > 0.3:
            return "very close"
        elif relative_size > 0.15:
            return "close"
        elif relative_size > 0.05:
            return "medium distance"
        else:
            return "far"

    def get_position(self, center_x: float, image_width: int) -> str:
        relative_x = center_x / image_width

        if relative_x < 0.33:
            return "on your left"
        elif relative_x < 0.67:
            return "in front of you"
        else:
            return "on your right"

