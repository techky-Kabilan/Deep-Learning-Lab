# RoadVision AI model card

**Model status:** Not trained. No checkpoint, quantitative performance or operational approval available.

Intended model: pretrained Ultralytics YOLOv8n detection backbone, fine-tuned on the four RDD2022 classes in the fixed mapping in README. Planned input: RGB photograph, 640-pixel training/inference size. Output: class, confidence and pixel bounding box.

Intended use: educational exploration and human-reviewed inspection of photographs. Not an autonomous road-maintenance decision system. Confidence does not quantify danger or physical severity.

Planned data: official RDD2022 labelled training images; India subset by default in the notebook. Official unlabelled challenge test images are excluded from local scored evaluation. Seed 42, group split 70/15/15; actual image ratios depend on group sizes. Route metadata is optional and image splitting alone cannot eliminate correlated scenes.

Evaluation pending: precision, recall, per-class AP, mAP@0.5, mAP@0.5:0.95, raw/normalized confusion matrices, PR curves, qualitative false positives/negatives and latency. Record actual checkpoint hash, split-manifest hash, training commit, GPU, epochs completed, training arguments and environment after a run.

Potential limitations: thin cracks, shadows, road markings, occlusion, camera angle, compression, wet roads, rare damage classes, label ambiguity and country/domain shift. No detections does not prove a damage-free road. No validated severity or road-condition score is implemented.

Licensing: see THIRD_PARTY_NOTICES.md. Checkpoint artifacts must have their own verified provenance and terms.
