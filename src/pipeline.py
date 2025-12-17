from pathlib import Path
import logging
import cv2

from src.detector import detect_and_crop
from src.preprocess import preprocess_image
from src.classifier import predict_anemia, get_model
from src.config import RESULTS_DIR
from src.explain import generate_gradcam

logger = logging.getLogger(__name__)

def run_pipeline(image_path, explain=False):
    """
    Run the full analysis on an image.
    """
    image_path = Path(image_path)
    
    # Check if the image file exists
    if not image_path.exists():
        logger.error(f"Pipeline: Image file validation failed")
        raise ValueError(f"Image file not found")
    
    # Make sure explain is a true/false value
    if not isinstance(explain, bool):
        logger.warning(f"Pipeline: Invalid explain type, coercing to bool")
        explain = bool(explain)

    # 1. Detect & Crop
    crop_rgb = detect_and_crop(image_path)
    
    # If detection failed, stop here
    if crop_rgb is None:
        raise ValueError("Failed to extract image data for classification")

    note = None
    boxed_image_path = RESULTS_DIR / f"boxed_{image_path.name}"

    # 2. Preprocess
    input_tensor = preprocess_image(crop_rgb)
    
    # If preprocessing failed, stop here
    if input_tensor is None:
        raise ValueError("Failed to preprocess image")

    # Run the classifier
    label, confidence = predict_anemia(input_tensor)

    # Build result dictionary
    result = {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "boxed_image_path": str(boxed_image_path),
        "note": note
    }
    
    # Make a heatmap if the user asked for it
    if explain:
        heatmap_path = RESULTS_DIR / f"heatmap_{image_path.name}"
        try:
            model = get_model()
            # generate_gradcam auto-detects the best conv layer in MobileNetV2
            heatmap_result = generate_gradcam(
                model=model,
                input_tensor=input_tensor,
                original_rgb=crop_rgb,
                output_path=heatmap_path
            )
            if heatmap_result:
                result["heatmap_path"] = heatmap_result
                print(f"✓ Grad-CAM heatmap generated: {heatmap_result}")
            else:
                print(f"✗ Grad-CAM heatmap generation returned None")
        except Exception as e:
            print(f"✗ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail the request, just skip heatmap
    
    return result
