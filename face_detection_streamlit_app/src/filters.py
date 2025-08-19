import cv2
import numpy as np

def blur_faces(image_bgr, faces, ksize=(35,35)):
    out = image_bgr.copy()
    for (x,y,w,h) in faces:
        roi = out[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, ksize, 0)
        out[y:y+h, x:x+w] = roi
    return out

def crop_faces(image_bgr, faces, margin=0.15):
    crops = []
    h_img, w_img = image_bgr.shape[:2]
    for (x,y,w,h) in faces:
        dx = int(w * margin); dy = int(h * margin)
        x1 = max(0, x - dx); y1 = max(0, y - dy)
        x2 = min(w_img, x + w + dx); y2 = min(h_img, y + h + dy)
        crops.append(image_bgr[y1:y2, x1:x2].copy())
    return crops

def draw_boxes(image_bgr, faces, color=(0,255,0), thickness=2):
    out = image_bgr.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out, (x,y), (x+w,y+h), color, thickness)
    return out

def apply_grayscale(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

def apply_edge(image_bgr, low=100, high=200):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, low, high)

def apply_sepia(image_bgr):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image_bgr, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_cartoon(image_bgr):
    # Step 1: edge mask
    
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)  # smoother edges
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)

    # Step 2: color quantization (reduce colors)
    data = np.float32(image_bgr).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, 9, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(image_bgr.shape).astype(np.uint8)

    # Step 3: smooth color regions
    smooth = cv2.bilateralFilter(quantized, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: combine edges with smooth image
    cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)
    return cartoon
def apply_pencil_sketch(image_bgr):
    """
    Pencil sketch effect with stronger edges.
    Uses dodge blend of grayscale and inverted blur.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)       # 1. Convert to grayscale
    inv = 255 - gray                                         # 2. Invert grayscale
    blur = cv2.GaussianBlur(inv, (21, 21), 0)                # 3. Blur the inverted image
    sketch = cv2.divide(gray, 255 - blur, scale=256)         # 4. Dodge blend for edges
    return sketch


def apply_contrast(image_bgr, alpha=1.5, beta=0):
    """
    Increase contrast.
    alpha > 1 increases contrast, < 1 decreases.
    beta adjusts brightness.
    """
    adjusted = cv2.convertScaleAbs(image_bgr, alpha=alpha, beta=beta)
    return adjusted

def apply_denoise(image_bgr):
    """Remove noise using fastNlMeansDenoisingColored."""
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, 7, 5, 7, 21)
