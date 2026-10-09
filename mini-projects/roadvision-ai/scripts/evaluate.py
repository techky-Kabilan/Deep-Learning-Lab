"""Evaluate a trusted fine-tuned checkpoint; never supplies canned metrics."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml
from roadvision.model import validate_names

def evaluate(weights, data, output, split="test", device="cpu"):
    import ultralytics
    from ultralytics import YOLO
    weights, data, output = Path(weights).resolve(), Path(data).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Evaluation destination exists; choose a new directory.")
    model = YOLO(str(weights), task="detect")
    validate_names(model.names)
    config = yaml.safe_load(data.read_text())
    validate_names(config["names"])
    root = Path(config["path"])
    if not root.is_absolute():
        root = (data.parent / root).resolve()
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {s: {r["sha256"] for r in manifest if r["split"] == s} for s in ("train", "val", "test")}
    if any(hashes[a] & hashes[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Split leakage detected")
    if not hashes[split]:
        raise ValueError("Selected evaluation split is empty")
    metrics = model.val(data=str(data), split=split, imgsz=640, conf=.001, iou=.6, plots=True, device=device, project=str(output.parent), name=output.name, exist_ok=False)
    actual = Path(model.validator.save_dir)
    per_class = []
    for index, class_id in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        per_class.append({"category":model.names[int(class_id)],"precision":float(p),"recall":float(r),"mAP50":float(ap50),"mAP50_95":float(ap)})
    pd.DataFrame(per_class).to_csv(actual / "per_class.csv", index=False)
    summary = {"status":"evaluated", "split":split,"metrics":{k:float(v) for k,v in metrics.results_dict.items()}, "weights_sha256":hashlib.sha256(weights.read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),"timestamp_utc":datetime.now(timezone.utc).isoformat(),"ultralytics":ultralytics.__version__,"python":platform.python_version(),"validation_settings":{"imgsz":640,"conf":.001,"iou":.6},"interpretation":"Precision/recall are library summary operating-point values. mAP integrates the confidence ranking. See PR curves and per-class support; absent classes cannot establish performance."}
    (actual / "metrics.json").write_text(json.dumps(summary, indent=2))
    return actual

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/evaluation")
    parser.add_argument("--split", choices=["val","test"], default="test")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(evaluate(args.weights, args.data, args.output, args.split, args.device))
