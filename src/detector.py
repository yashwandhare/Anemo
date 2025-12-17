# src/detector.py
import cv2
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from src.config import YOLO_MODEL_PATH, RESULTS_DIR

# Set up logging
logger = logging.getLogger(__name__)

# File size and type limits
MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Load YOLO model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
try:
    yolo_model = YOLO(str(YOLO_MODEL_PATH))
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise RuntimeError(f"Failed to load YOLO model: {e}") from e

TARGET_CLASSES = {
    "palpebral",
    "forniceal_palpebral",
    "eye" # fallback in case the model labels it differently
}

def detect_and_crop(image_path):
    """
    Find the eye area, draw a box around it, and return the cropped image.
    """
    image_path = Path(image_path)
    
    # Check if the path is valid
    if not isinstance(image_path, Path):
        logger.error("Invalid image_path type")
        raise ValueError("Invalid image path")
    
    # Don't allow tricks like ".." to escape the directory
    if ".." in str(image_path):
        logger.error(f"Directory traversal attempt detected")
        raise ValueError("Invalid image path")
    
    # Make sure the file actually exists
    if not image_path.exists() or not image_path.is_file():
        logger.error(f"Image file not found or inaccessible")
        raise ValueError(f"Could not access image file")
    
    # Only accept image files
    if image_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        logger.error(f"Invalid file extension attempted")
        raise ValueError(f"Unsupported file type")
    
    # Don't try to process huge files
    try:
        file_size = image_path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            logger.error(f"Image file exceeds size limit")
            raise ValueError(f"Image file too large")
    except OSError as e:
        logger.error(f"Can't read the file")
        raise ValueError("Cannot access image file")
    
    image_bgr = cv2.imread(str(image_path))
    
    if image_bgr is None:
        logger.error(f"Can't open the image - it might be broken or not a real image")
        raise ValueError(f"Could not read image - file may be corrupted")
    
    # Shrink really big images so they don't use too much memory
    # Max dimension ~2048px, preserve aspect ratio
    h, w = image_bgr.shape[:2]
    max_dim = 2048
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w}x{h} to {new_w}x{new_h} to prevent memory issues")

    # Run Inference
    results = yolo_model.predict(source=image_bgr, conf=0.25, verbose=False)
    
    h, w, _ = image_bgr.shape
    best_box = None
    best_area = 0

    # Look for the biggest eye area in the results
    if results and len(results[0].boxes) > 0:
        names = results[0].names
        boxes = results[0].boxes

        for box in boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id)

            if cls_name in TARGET_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)

    # Create a safe filename for the output
    boxed_filename = f"boxed_{image_path.name}"
    # Remove any path separators from filename
    boxed_filename = boxed_filename.replace("/", "_").replace("\\", "_")
    boxed_path = RESULTS_DIR / boxed_filename
    
    crop_rgb = None

    if best_box:
        x1, y1, x2, y2 = best_box
        # Clamp coordinates to image size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Draw Green Box
        boxed_img = image_bgr.copy()
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(boxed_path), boxed_img)
        print(f"Saved detection visualization to: {boxed_path}")

        # Crop
        crop_bgr = image_bgr[y1:y2, x1:x2]
        
        # Make sure we actually got an image
        if crop_bgr.size != 0:
            # Resize to 224x224 here to ensure consistency
            crop_bgr = cv2.resize(crop_bgr, (224, 224))
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    
    else:
        print("No conjunctiva detected. Using full image as fallback.")
        cv2.imwrite(str(boxed_path), image_bgr)
        
        # If no eye is detected, just use the whole image
        resized_full = cv2.resize(image_bgr, (224, 224))
        crop_rgb = cv2.cvtColor(resized_full, cv2.COLOR_BGR2RGB)

    return crop_rgb