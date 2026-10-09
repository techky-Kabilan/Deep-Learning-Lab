# RoadVision AI
## Intelligent Road Damage Detection and Analytics Using Deep Learning

Image-only college mini-project: fine-tune pretrained YOLOv8n on RDD2022, inspect JPG/PNG photographs in Streamlit, and export annotated images and detection CSVs.

**Status: implementation provided; RDD2022 training and model evaluation have NOT been performed. No trained checkpoint or benchmark metrics are bundled.** The app runs in upload-preview mode until a trusted fine-tuned checkpoint is installed. Tests use synthetic fixtures and do not establish model accuracy.

### Quick start
Use Python 3.12. From this folder:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```
For CPU-only installations, install matching torch/torchvision wheels from https://pytorch.org/get-started/locally/ before the requirements. GPU builds must match the available CUDA runtime. Training is intended for a Colab GPU; local app inference can use CPU.

### Train in Google Colab
Open [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) in Colab, select a GPU runtime, and run cells in order. It clones this repository and branch, installs dependencies, downloads the official India archive only after explicit execution, converts XML annotations, trains a pretrained YOLOv8n, evaluates the held-out labelled test split and saves artifacts to Google Drive. India is a manageable starting subset; it is not a claim of six-country generalization. Change countries deliberately and report the actual data scope.

1. Inspect dataset terms and available disk space. Do not commit dataset images or archives.
2. Review `split_summary.json`. Verify all four classes occur in training, validation and test; enlarge the subset if any are missing.
3. Train; tune only against validation. Keep the test split untouched until final evaluation.
4. Copy `best.pt` to `weights/` and restart Streamlit. Only load trusted `.pt` files.
5. Retain metrics, confusion matrices, PR curves, split manifest, training arguments, environment and checkpoint SHA-256.

### Features
- Actual JPEG/PNG decoding, EXIF orientation, 20 MB / 25 megapixel limits.
- Bounding boxes, D00/D10/D20/D40 labels and per-object confidence.
- Confidence and IoU controls; results invalidated when inputs change.
- Current-image object counts by all four classes and confidence histogram.
- Annotated PNG and CSV downloads; empty CSV retains its header.
- No video, GPS, severity scoring or pavement-condition index. Counts represent boxes, not affected road length or severity.

### Classes
| ID | Code | Dataset category |
|---|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

### Evaluation
```bash
python -m scripts.evaluate --weights weights/best.pt --data data/prepared/data.yaml --output artifacts/final-test --device cpu
```
This creates overall precision, recall, mAP@0.5, mAP@0.5:0.95, per-class results and Ultralytics confusion-matrix/PR plots. Precision/recall use the library summary operating point; validation uses a low confidence cutoff to preserve the ranking for AP. Dashboard confidence defaults to 0.25 and should not be confused with the AP evaluation cutoff. Include background false positives/false negatives in confusion-matrix interpretation. Report class support and failure cases, not just a single average. The test set here is a held-out portion of labelled training data, NOT the official challenge test set (which has no public annotations).

Exact image duplicates are removed before seeded 70/15/15 group splitting. Without a route manifest each distinct image is its own group; adjacent frames can still leak similar scenes. Supply `--group-manifest` with JSON `{relative_image_path: route_or_sequence_id}` for a stronger evaluation. Audit near-duplicates and country distribution manually. Never claim route-independent generalization from the default image split.

### Repository integration
This folder is isolated under `mini-projects/roadvision-ai` in [Deep-Learning-Lab](https://github.com/techky-Kabilan/Deep-Learning-Lab). Existing experiments remain intact. Its code license is scoped to this folder. CI runs only when this project or its workflow changes.

### Documentation
- [College report](output/pdf/RoadVision_AI_Project_Report.pdf)
- [Methodology and viva notes](docs/methodology.md)
- [Third-party licensing](THIRD_PARTY_NOTICES.md)
- [Model card](MODEL_CARD.md)
- [Screenshots](docs/screenshots/)

### Sources
- Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, Geoscience Data Journal 11(4), 846-862, 2024. Dataset and class definitions: https://github.com/sekilab/RoadDamageDetector
- Ultralytics validation: https://docs.ultralytics.com/modes/val/
- Ultralytics YOLOv8: https://docs.ultralytics.com/models/yolov8/
- Ultralytics licensing: https://www.ultralytics.com/license
- Streamlit documentation: https://docs.streamlit.io/
