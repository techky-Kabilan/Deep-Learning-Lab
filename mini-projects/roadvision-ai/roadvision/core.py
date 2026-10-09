from dataclasses import dataclass, asdict
from io import BytesIO
import csv
import io
import warnings
from PIL import Image, ImageOps, ImageDraw

CATEGORIES = {"D00": "Longitudinal crack", "D10": "Transverse crack", "D20": "Alligator crack", "D40": "Pothole"}
CODES = tuple(CATEGORIES)
COLORS = {"D00": "#225c9c", "D10": "#9c4815", "D20": "#6850a0", "D40": "#ad2339"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 25_000_000

@dataclass(frozen=True)
class Detection:
    category: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self):
        import math
        if self.category not in CATEGORIES or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid category or confidence")
        if not all(math.isfinite(v) for v in (self.xmin, self.ymin, self.xmax, self.ymax)) or not (0 <= self.xmin < self.xmax and 0 <= self.ymin < self.ymax):
            raise ValueError("Invalid bounding box")

def load_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Upload a JPG/PNG file smaller than 20 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only actual JPEG and PNG images are accepted.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 25 megapixel limit.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Image cannot be decoded safely.") from exc

def counts(detections):
    result = dict.fromkeys(CODES, 0)
    for detection in detections:
        result[detection.category] += 1
    return result

def csv_bytes(detections, filename):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["image", "category", "label", "confidence", "xmin", "ymin", "xmax", "ymax"])
    writer.writeheader()
    # Prevent spreadsheet formula evaluation in user-supplied file names.
    safe_name = "'" + filename if filename.lstrip().startswith(("=", "+", "-", "@")) else filename
    for detection in detections:
        writer.writerow({"image": safe_name, "label": CATEGORIES[detection.category], **asdict(detection)})
    return stream.getvalue().encode("utf-8")

def annotate(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for d in detections:
        color = COLORS[d.category]
        draw.rectangle((d.xmin, d.ymin, d.xmax, d.ymax), outline=color, width=3)
        label = f"{d.category} {CATEGORIES[d.category]} {d.confidence:.1%}"
        x, y = min(d.xmin, max(0, image.width - 280)), max(0, d.ymin - 18)
        box = draw.textbbox((x, y), label)
        draw.rectangle(box, fill=color)
        draw.text((x, y), label, fill="white")
    return annotated

def png_bytes(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
