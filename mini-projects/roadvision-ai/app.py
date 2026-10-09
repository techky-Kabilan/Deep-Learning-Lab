import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from roadvision.core import CATEGORIES, counts, load_image, annotate, png_bytes, csv_bytes
from roadvision.model import Detector

st.set_page_config(page_title="RoadVision AI", page_icon="🛣️", layout="wide")
st.markdown("""<style>
:root {--rv-bg:oklch(97% .008 245);--rv-ink:oklch(24% .03 245);--rv-accent:oklch(43% .10 245)}
.stApp {background:var(--rv-bg);color:var(--rv-ink);font-family:'Segoe UI',sans-serif}
h1,h2,h3 {letter-spacing:-.02em} h1 {font-size:36px!important}
button {transition:transform 120ms cubic-bezier(.16,1,.3,1)}
button:active {transform:scale(.98)}
button:focus-visible {outline:3px solid var(--rv-accent)!important}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media(max-width:768px){h1{font-size:30px!important}}
</style>""", unsafe_allow_html=True)
st.caption("ROAD INSPECTION / IMAGE ANALYSIS")
st.title("RoadVision AI")
st.write("Inspect road damage, review each detection, and export the evidence.")
root = Path(__file__).resolve().parent
weights = Path(os.environ.get("ROADVISION_WEIGHTS", str(root / "weights/best.pt")))
with st.sidebar:
    st.header("Inspection settings")
    confidence = st.slider("Minimum confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("Overlap suppression (IoU)", 0.1, 0.9, 0.5, 0.05)
    st.caption("Confidence is a model score, not damage severity or a calibrated probability.")
    st.markdown("**RDD2022 classes**")
    for code, label in CATEGORIES.items():
        st.write(f"{code} · {label}")
    st.caption("Images are processed in this session. No upload history is stored.")

@st.cache_resource
def get_detector(path, modified_ns):
    return Detector(path)

ready = weights.is_file()
if not ready:
    st.info("Training pending. Run notebooks/train_colab.ipynb, then place your trusted RDD2022 best.pt in weights/. Uploads are available for preview; detection activates when weights are present.")
upload = st.file_uploader("Upload a road photograph", type=["jpg", "jpeg", "png"], help="Maximum 20 MB and 25 megapixels.")
if upload is None:
    st.subheader("One photograph. A clear inspection record.")
    st.write("Upload a JPG or PNG to see bounding boxes, class labels and confidence scores. The dashboard counts detected objects in the current image.")
else:
    try:
        data = upload.getvalue()
        image = load_image(data)
        key = (data, confidence, iou, weights.stat().st_mtime_ns if ready else None)
        previous = st.session_state.get("inspection")
        with st.expander("Original photograph", expanded=not (previous and previous[0] == key)):
            st.image(image, caption=f"Original · {image.width} × {image.height}", width=480)
        if st.button("Analyze image", type="primary", disabled=not ready):
            with st.spinner("Inspecting the road surface…"):
                detections = get_detector(str(weights), weights.stat().st_mtime_ns).predict(image, confidence, iou)
            st.session_state["inspection"] = (key, detections)
            st.rerun()
        saved = st.session_state.get("inspection")
        if saved and saved[0] == key:
            detections = saved[1]
            annotated = annotate(image, detections)
            left, right = st.columns([3, 2])
            with left:
                st.subheader("Inspection evidence")
                st.image(annotated, caption="Detected objects at the selected confidence threshold", width="stretch")
                st.download_button("Download annotated PNG", png_bytes(annotated), "roadvision-annotated.png", "image/png", on_click="ignore")
            with right:
                st.subheader("Damage overview")
                st.metric("Detected objects", len(detections))
                summary = pd.DataFrame([{"Category": f"{c} · {CATEGORIES[c]}", "Count": n} for c, n in counts(detections).items()])
                st.bar_chart(summary.set_index("Category"), color="#225c9c")
                if detections:
                    fig = px.histogram(pd.DataFrame({"Confidence": [d.confidence for d in detections]}), x="Confidence", nbins=10, range_x=[0, 1], color_discrete_sequence=["#225c9c"])
                    fig.update_layout(title="Confidence distribution", yaxis_title="Detected objects", bargap=.08, height=280)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No detections above the threshold. This does not establish that the road is damage-free.")
            rows = [{"Category": d.category, "Label": CATEGORIES[d.category], "Confidence": d.confidence, "Box (pixels)": f"{d.xmin:.0f}, {d.ymin:.0f}, {d.xmax:.0f}, {d.ymax:.0f}"} for d in detections]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.download_button("Export detections CSV", csv_bytes(detections, upload.name), "roadvision-detections.csv", "text/csv", on_click="ignore")
    except Exception as exc:
        st.error(f"Inspection could not be completed: {exc}")
