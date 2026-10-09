from io import BytesIO
import csv
import pytest
from PIL import Image
from roadvision.core import *
from roadvision.model import validate_names

def test_upload_decodes_real_image_not_extension():
    stream = BytesIO()
    Image.new("RGBA", (40,30)).save(stream, format="PNG")
    assert load_image(stream.getvalue()).mode == "RGB"
    with pytest.raises(ValueError): load_image(b"fake.jpg")
    stream = BytesIO()
    Image.new("RGB", (10,10)).save(stream, format="GIF")
    with pytest.raises(ValueError): load_image(stream.getvalue())

def test_empty_and_populated_exports():
    assert counts([]) == {code:0 for code in CODES}
    assert len(csv_bytes([], "empty.png").decode().splitlines()) == 1
    detection = Detection("D40", .81, 2,3,20,25)
    row = next(csv.DictReader(csv_bytes([detection], "=SUM(1).png").decode().splitlines()))
    assert row["image"].startswith("'") and row["label"] == "Pothole"
    assert counts([detection])["D40"] == 1
    original = Image.new("RGB", (60,60), "gray")
    marked = annotate(original, [detection])
    assert original.getpixel((2,3)) != marked.getpixel((2,3))
    assert Image.open(BytesIO(png_bytes(marked))).size == (60,60)

@pytest.mark.parametrize("args", [("car",.7,0,0,10,10),("D00",1.1,0,0,10,10),("D00",.5,5,0,1,10),("D00",.5,0,0,float('nan'),10)])
def test_invalid_detections(args):
    with pytest.raises(ValueError): Detection(*args)

def test_rejects_coco_and_wrong_class_order():
    validate_names(dict(enumerate(CODES)))
    with pytest.raises(ValueError): validate_names({0:"person"})
    with pytest.raises(ValueError): validate_names(dict(enumerate(reversed(CODES))))
