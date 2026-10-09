# Methodology and viva notes

## Problem and scope
Locate and classify visible road-surface defects from one image and summarize the detection evidence. Image-only scope keeps the project manageable while retaining a full machine-learning workflow. Avoid a binary crack/non-crack classifier because it cannot localize multiple damage types.

## Pipeline
RGB photograph -> validated decoding and EXIF orientation -> YOLO resize/letterbox -> feature extraction and detection heads -> confidence filtering and non-maximum suppression -> pixel boxes -> annotation, analytics and downloads.

Training: official VOC XML -> whitelist four target codes -> clamp valid coordinates and normalize centers/sizes -> exact-image deduplication -> route/sequence groups when available -> seeded train/validation/test partition -> pretrained YOLOv8n fine-tuning -> validation-based model selection -> one final held-out test evaluation.

## Metrics
IoU = intersection area / union area. A true positive requires the correct category and sufficient box overlap, with one-to-one matching. Precision = TP/(TP+FP); recall = TP/(TP+FN). AP summarizes a precision-recall curve; mAP averages AP over classes. mAP@0.5 uses IoU 0.5; mAP@0.5:0.95 averages across thresholds from 0.5 to 0.95 in steps of 0.05. A stricter overlap requirement tests localization quality. Use Ultralytics' evaluator rather than a hand-written accuracy percentage.

Detection confusion matrices include background: missed ground-truth objects and unmatched predictions. Inspect the generated axis labels before describing cells. Precision/recall and matrices depend on the operating point; AP depends on the ranking across confidence scores. Class imbalance makes overall averages insufficient. Include per-class support and examples from each failure type.

## Reproducibility and validity
Record data origin, subset, split manifest hash, weights hash, commit, environment, seed, GPU, epochs actually completed and evaluation arguments. Colab/GPU kernels may remain nondeterministic even with a seed. Images from the same driving route are correlated: group split where metadata exists, audit near-duplicates, and describe limitations if only image splitting is possible. Do not tune thresholds on the final test set.

## Viva questions
**Why transfer learning?** Road damage examples are fewer and more specialized than generic detection pretraining; pretrained visual features give a useful starting point.

**Why YOLOv8n?** A small, established pretrained detector is practical for Colab and CPU demos. Model choice is a resource tradeoff, not a claim of best accuracy.

**Why not classification accuracy?** An image may contain several objects from several categories. Object detection needs localization-aware metrics.

**Is confidence severity?** No. It is a model score; physical severity needs additional validated labels and measurement.

**Can zero detections prove a road is safe?** No. Small, occluded or unfamiliar defects can be missed.

**What remains before submission as a trained-model project?** Execute the notebook on licensed data, retain the checkpoint and measured evaluation, inspect failure cases, update the model card/report with real evidence, and capture real inference screenshots.
