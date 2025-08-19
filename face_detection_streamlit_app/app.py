import streamlit as st
import numpy as np
import cv2
from pathlib import Path
import datetime
import tempfile
from src.detection import load_cascade, detect_faces
from src.filters import (
     blur_faces, crop_faces, draw_boxes,
    apply_grayscale, apply_edge, apply_sepia, apply_cartoon,
    apply_pencil_sketch, apply_contrast, apply_denoise  
)

st.set_page_config(page_title="Face Detection & Image Manipulation", layout="wide")

st.title("👤 Face Detection & Image Manipulation — All-in-One App")
st.write("Face detection (HaarCascade) with image & video tools: blur faces, filters, cropping, and downloads.")

# Sidebar controls
st.sidebar.header("Options")
mode = st.sidebar.radio("Choose input type", ["Image", "Video"])

st.sidebar.subheader("Face detection")
scale = st.sidebar.slider("scaleFactor", 1.05, 1.5, 1.1, 0.01)
neighbors = st.sidebar.slider("minNeighbors", 3, 10, 5, 1)
min_size = st.sidebar.slider("minSize (px)", 20, 120, 30, 5)

st.sidebar.subheader("Manipulations")
do_boxes = st.sidebar.checkbox("Draw boxes", value=True)
do_blur = st.sidebar.checkbox("Blur faces")
filt = st.sidebar.selectbox(
    "Filter",
    [
        "None", 
        "Grayscale", 
        "Edges", 
        "Sepia", 
        "Cartoon (OpenCV)",
        "Pencil Sketch",
        "High Contrast",
        "Denoise"
    ]
)
do_crop = st.sidebar.checkbox("Crop faces (for download)")

cascade = load_cascade()

# ----------------------------------
# Common Functions
# ----------------------------------
def apply_all(image_bgr, faces):
    out = image_bgr.copy()
    if do_blur:
        out = blur_faces(out, faces)
    if do_boxes:
        out = draw_boxes(out, faces)

    # Filters
    if filt == "Grayscale":
        out = apply_grayscale(out)
    elif filt == "Edges":
        out = apply_edge(out)
    elif filt == "Sepia":
        out = apply_sepia(out)
    elif filt == "Cartoon (OpenCV)":
        out = apply_cartoon(out)
    elif filt == "Pencil Sketch":
        out = apply_pencil_sketch(out)
    elif filt == "High Contrast":
        out = apply_contrast(out, alpha=1.7, beta=20)
    elif filt == "Denoise":
        out = apply_denoise(out)

    return out

def convert_bgr_to_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------
# IMAGE MODE
# ---------------------------
if mode == "Image":
    st.subheader("Upload or Capture Image")

    img_source = st.radio("Select image source", ["Upload Image", "Capture Image"])

    image = None
    if img_source == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif img_source == "Capture Image":
        captured_img = st.camera_input("Take a picture with your webcam")
        if captured_img is not None:
            file_bytes = np.asarray(bytearray(captured_img.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        faces = detect_faces(image, cascade, scaleFactor=scale, minNeighbors=neighbors, minSize=(min_size,min_size))
        processed = apply_all(image, faces)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Original")
            st.image(convert_bgr_to_rgb(image), channels="RGB", use_column_width=True)
            st.caption(f"Detected faces: {len(faces)}")
        with c2:
            st.subheader("Processed")
            st.image(convert_bgr_to_rgb(processed), channels="RGB", use_column_width=True)

        # Crops
        if do_crop and len(faces) > 0:
            crops = crop_faces(image, faces)
            st.subheader("Cropped faces")
            for i, crop in enumerate(crops, 1):
                st.image(convert_bgr_to_rgb(crop), channels="RGB", caption=f"Face {i}", width=160)

        # Download processed image
        is_gray = len(processed.shape) == 2
        to_save = processed if not is_gray else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", to_save)
        if ok:
            st.download_button("📥 Download processed image", data=buf.tobytes(),
                               file_name="processed.jpg", mime="image/jpeg")

# ---------------------------
# VIDEO MODE
# ---------------------------
elif mode == "Video":
    st.subheader("Upload a Video for Processing")
    vfile = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])

    if vfile:
        tdir = tempfile.mkdtemp()
        in_path = Path(tdir)/"input.mp4"
        with open(in_path, "wb") as f:
            f.write(vfile.read())

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            st.error("Could not open video.")
        else:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

            out_path = Path(tdir)/"output.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            progress = st.progress(0)
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                faces = detect_faces(frame, cascade, scaleFactor=scale, minNeighbors=neighbors, minSize=(min_size,min_size))
                proc = apply_all(frame, faces)
                if len(proc.shape) == 2:
                    proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
                writer.write(proc)
                i += 1
                if frame_count > 0:
                    progress.progress(min(i/frame_count, 1.0))

            cap.release()
            writer.release()
            progress.empty()

            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button("📥 Download processed video", data=f.read(),
                                   file_name="processed.mp4", mime="video/mp4")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built with OpenCV, HaarCascade, and Streamlit. © {:%Y}".format(datetime.datetime.now()))
