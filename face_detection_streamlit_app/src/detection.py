import cv2
from pathlib import Path

def load_cascade(haar_path: str | None = None) -> "cv2.CascadeClassifier":
    if haar_path is None:
        # try local file first for reproducibility in deployments
        local = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"
        if local.exists():
            haar_path = str(local)
        else:
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(haar_path)
    if cascade.empty():
        raise FileNotFoundError(f"Could not load Haar cascade from {haar_path}")
    return cascade

def detect_faces(image_bgr, cascade, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    return faces