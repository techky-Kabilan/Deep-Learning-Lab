from pathlib import Path
from threading import Lock
from .core import CODES, CATEGORIES, Detection

def validate_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if set(names) != set(range(4)) or tuple(names[i] for i in range(4)) != CODES:
        raise ValueError("Checkpoint must have names 0:D00, 1:D10, 2:D20, 3:D40. Use the supplied training notebook.")

class Detector:
    def __init__(self, weights):
        path = Path(weights)
        if not path.is_file() or path.suffix != ".pt":
            raise ValueError("Trained checkpoint missing. Complete Colab training and copy best.pt into weights/.")
        from ultralytics import YOLO
        # Load only checkpoints from a trusted training run; .pt files can contain executable data.
        self.model = YOLO(str(path), task="detect")
        validate_names(self.model.names)
        self.lock = Lock()

    def predict(self, image, confidence=0.25, iou=0.5):
        with self.lock:
            result = self.model.predict(source=image, conf=confidence, iou=iou, imgsz=640, verbose=False)[0]
        detections = []
        for box, score, category in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            x1, y1, x2, y2 = box
            x1, x2 = max(0, x1), min(image.width, x2)
            y1, y2 = max(0, y1), min(image.height, y2)
            if x2 > x1 and y2 > y1:
                detections.append(Detection(CODES[int(category)], score, x1, y1, x2, y2))
        return detections
