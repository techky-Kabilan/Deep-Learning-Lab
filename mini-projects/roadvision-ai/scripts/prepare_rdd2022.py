"""Convert extracted RDD2022 VOC annotations, deduplicate, and split deterministically."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
from defusedxml import ElementTree as ET
from PIL import Image
import yaml
from roadvision.core import CODES

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def convert_xml(path, width, height):
    tree = ET.parse(path).getroot()
    labels, ignored = [], Counter()
    for obj in tree.findall("object"):
        code = (obj.findtext("name") or "").strip()
        if code not in CODES:
            ignored[code] += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing box: {path}")
        # VOC coordinates are 1-based inclusive; YOLO uses continuous image coordinates.
        x1 = max(0., min(width, float(box.findtext("xmin")) - 1))
        y1 = max(0., min(height, float(box.findtext("ymin")) - 1))
        x2 = max(0., min(width, float(box.findtext("xmax"))))
        y2 = max(0., min(height, float(box.findtext("ymax"))))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Degenerate box: {path}")
        labels.append(f"{CODES.index(code)} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return labels, ignored

def prepare(source, destination, seed=42, countries=None, group_manifest=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Output already exists. Choose a new output directory; no data is overwritten.")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Input and output must be separate directories.")
    groups = json.loads(Path(group_manifest).read_text()) if group_manifest else {}
    records, seen, ignored = [], set(), Counter()
    for xml in sorted(source.rglob("*.xml")):
        if "test" in {p.lower() for p in xml.parts}:
            continue
        train_dir = next((parent for parent in xml.parents if parent.name.lower() == "train"), None)
        if train_dir is None:
            raise ValueError(f"Annotation is not inside a train directory: {xml}")
        image_dir = train_dir / "images"
        image = next((image_dir / (xml.stem + suffix) for suffix in (".jpg", ".JPG", ".png", ".jpeg") if (image_dir / (xml.stem + suffix)).is_file()), None)
        if image is None:
            raise ValueError(f"Image missing for {xml}")
        country = train_dir.parent.name
        if countries and country not in countries:
            continue
        digest = sha256(image)
        if digest in seen:
            continue
        seen.add(digest)
        with Image.open(image) as im:
            labels, skipped = convert_xml(xml, *im.size)
        ignored.update(skipped)
        relative = image.relative_to(source).as_posix()
        group = groups.get(relative, digest)
        records.append({"image": image, "relative": relative, "country": country, "sha256": digest, "group": group, "labels": labels})
    if len(records) < 10:
        raise ValueError("At least 10 labelled images are required for a usable split.")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group"]].append(record)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    if n < 3:
        raise ValueError("At least 3 independent groups are required.")
    # Split groups, not images, so route/sequence groups cannot cross partitions.
    first, second = max(1, int(n*.7)), max(2, int(n*.85))
    second = min(second, n-1)
    assignment = {k: "train" if i < first else "val" if i < second else "test" for i, k in enumerate(keys)}
    manifest, class_counts = [], {split: Counter() for split in ("train", "val", "test")}
    for record in records:
        split = assignment[record["group"]]
        name = record["sha256"]
        images, labels_dir = destination / "images" / split, destination / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["image"], images / (name + record["image"].suffix.lower()))
        (labels_dir / f"{name}.txt").write_text("\n".join(record["labels"]), encoding="utf-8")
        class_counts[split].update(CODES[int(line.split()[0])] for line in record["labels"])
        manifest.append({k: record[k] for k in ("relative", "country", "sha256", "group")} | {"split": split})
    dataset = {"path": str(destination), "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(CODES))}
    (destination / "data.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
    (destination / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "split_summary.json").write_text(json.dumps({"seed":seed,"grouped_by_sequence":bool(group_manifest),"images":dict(Counter(r["split"] for r in manifest)),"objects":class_counts,"ignored_categories":ignored}, indent=2), encoding="utf-8")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--countries", nargs="+")
    parser.add_argument("--group-manifest", help="JSON mapping relative image paths to route/sequence IDs")
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed, args.countries, args.group_manifest)
