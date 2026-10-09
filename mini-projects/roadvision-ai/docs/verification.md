# Verification record

Verified locally on 09 October 2026 with Python 3.12.14 on Windows.

- `python -m pytest -q`: 13 passed.
- Pinned Ultralytics 8.3.221 / Torch 2.8.0 / torchvision 0.23.0 installed and imported.
- Real Streamlit startup without weights: no application exceptions.
- Browser workflow with a separate synthetic fixture: image upload, analysis,
  annotations, category counts, rendered confidence chart and desktop/mobile screenshots.
- Browser PNG and CSV downloads completed. Decoded PNG dimensions were 480 x 340;
  CSV contained the two scripted categories D00 and D40. These are software fixtures.
- Final 12-page PDF rendered and visually inspected. Model metrics explicitly pending.
- Official RDD2022 directory manifest checked: train/annotations/xmls and train/images.
- Official India S3 URL returned HTTP 403; notebook includes an official-source/Drive fallback.

Not verified: actual RDD2022 training, trained-checkpoint inference, measured precision/
recall/mAP, real confusion-matrix output, GPU performance, or route-independent accuracy.
GitHub Actions status must be read from its actual run; local passes are not a CI claim.
