# Face Detection & Image Manipulation — Streamlit App

One-click web app to demo everything in your project.

## Features
- Upload **Image** or **Video**, or use **Webcam (live)**
- Detect faces with **HaarCascade**
- Manipulations: **Blur faces**, **Draw boxes**, **Grayscale**, **Edges**, **Sepia**, **Cartoon**
- **Crop faces** preview (for images)
- Download processed image/video

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- If you run on a server without a webcam, use Image/Video modes.
- The app looks for `src/haarcascade_frontalface_default.xml`. If missing, it falls back to OpenCV's built-in cascade.