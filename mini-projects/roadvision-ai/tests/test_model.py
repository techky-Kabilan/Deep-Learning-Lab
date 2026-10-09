from types import SimpleNamespace
from threading import Lock
from PIL import Image
from roadvision.model import Detector

class Array:
    def __init__(self, data): self.data = data
    def cpu(self): return self
    def tolist(self): return self.data

def test_prediction_converts_labels_clips_boxes_and_forwards_thresholds():
    calls = []
    def predict(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=Array([[-5,-1,30,40],[7,7,7,10]]), conf=Array([.82,.6]), cls=Array([3,0])))]
    detector = Detector.__new__(Detector)
    detector.model = SimpleNamespace(predict=predict)
    detector.lock = Lock()
    image = Image.new('RGB',(20,20))
    results = detector.predict(image, confidence=.35, iou=.45)
    assert len(results) == 1
    assert results[0].category == 'D40' and results[0].xmax == 20
    assert results[0].xmin == 0 and results[0].ymin == 0
    assert calls[0]['conf'] == .35 and calls[0]['iou'] == .45
