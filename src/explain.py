# src/explain.py
"""
Grad-CAM Explainability Module
Generates visual explanations of CNN predictions without modifying model weights.
"""

import cv2
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

# Set up error logging
logger = logging.getLogger(__name__)

# File size limits
MAX_HEATMAP_SIZE = 50 * 1024 * 1024  # 50MB max output file
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def generate_gradcam(model, input_tensor, original_rgb, last_conv_layer_name=None, output_path=None):
    """
    Create a visual heatmap showing what the model is looking at.
    Shows which parts of the image influence the anemia prediction.
    """
    try:
        # SECURITY: Validate inputs
        if not isinstance(input_tensor, (np.ndarray, type(tf.constant([])))):
            logger.error("Invalid input_tensor type")
            return None
        
        if not isinstance(original_rgb, np.ndarray):
            logger.error("Invalid original_rgb type")
            return None
        
        if input_tensor.shape != (1, 224, 224, 3):
            logger.error(f"Invalid input_tensor shape: {input_tensor.shape}, expected (1, 224, 224, 3)")
            return None
        
        if original_rgb.shape[2] != 3:
            logger.error(f"Invalid RGB channels: {original_rgb.shape[2]}, expected 3")
            return None
        
        if output_path is None:
            logger.error("Output path is required")
            return None
        
        # Make sure the output path is safe to use
        output_path = Path(output_path)
        
        # Check the file extension
        if output_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            logger.error(f"Invalid output extension: {output_path.suffix}")
            return None
        
        # Prevent directory traversal attacks
        if ".." in str(output_path):
            logger.error("Directory traversal attempt detected in output_path")
            return None
        
        # Debug logging
        print(f"[Grad-CAM] Starting with input shape: {input_tensor.shape}")
        print(f"[Grad-CAM] Output path: {output_path}")
        
        # Get the underlying MobileNetV2 model 
        base_model = model.get_layer('mobilenetv2_1.00_224')
        
        # Track gradients to see what the model cares about
        input_tensor_var = tf.Variable(input_tensor)
        
        with tf.GradientTape() as tape:
            # Forward through model
            predictions = model(input_tensor_var, training=False)
            loss = predictions[0, 0]  # Get the positive class probability
        
        print(f"[Grad-CAM] Prediction score: {loss.numpy()}")
        
        # Get gradients w.r.t input
        input_grads = tape.gradient(loss, input_tensor_var)
        
        # Calculate how sensitive the prediction is to each pixel
        if input_grads is None:
            logger.warning("[Grad-CAM] Could not compute input gradients")
            return None
        
        print(f"[Grad-CAM] Gradient shape: {input_grads.shape}")
        
        # Use the gradient strength to create an attention map
        grad_magnitude = tf.reduce_max(tf.abs(input_grads[0]), axis=-1)

        # Stretch the heatmap to show contrast better
        heatmap = grad_magnitude.numpy()
        print(f"[Grad-CAM] Heatmap shape before processing: {heatmap.shape}")
        print(f"[Grad-CAM] Heatmap value range: {heatmap.min()} to {heatmap.max()}")

        heatmap = np.maximum(heatmap, 0)
        p_low = np.percentile(heatmap, 40)
        p_high = np.percentile(heatmap, 99)
        print(f"[Grad-CAM] Percentiles -> p40: {p_low}, p99: {p_high}")

        if p_high <= p_low:
            logger.warning("[Grad-CAM] Invalid percentile range; heatmap collapsed")
            return None

        heatmap = np.clip(heatmap, p_low, p_high)
        heatmap = (heatmap - p_low) / max(p_high - p_low, 1e-6)
        heatmap = np.clip(heatmap, 0, 1)

        # Hide the faint parts so the important areas stand out
        heatmap[heatmap < 0.4] = 0

        # Smooth out the heatmap so it looks cleaner
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), sigmaX=1.5, sigmaY=1.5)

        # Resize to match cropped image dimensions
        target_h, target_w = original_rgb.shape[:2]
        if heatmap.shape != (target_h, target_w):
            heatmap = cv2.resize(heatmap, (target_w, target_h))

        # Apply perceptually clear colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap),
            cv2.COLORMAP_INFERNO
        )
        
        # Convert to grayscale to reduce visual clutter
        original_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        # Convert back to 3-channel for blending
        original_gray_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
        
        # Blend the heatmap with the image so you can see both
        overlay = cv2.addWeighted(original_gray_bgr, 0.45, heatmap_colored, 0.65, 0)
        
        # Create the output folder if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the heatmap
        cv2.imwrite(str(output_path), overlay)
        
        # Check file was created and is within size limits
        if not output_path.exists():
            logger.error(f"Failed to write heatmap file: {output_path}")
            return None
        
        file_size = output_path.stat().st_size
        if file_size > MAX_HEATMAP_SIZE:
            logger.error(f"Generated heatmap exceeds size limit: {file_size} bytes")
            output_path.unlink(missing_ok=True)
            return None
        
        print(f"✓ [Grad-CAM] Heatmap saved to {output_path} ({file_size} bytes)")
        return str(output_path)
        
    except Exception as e:
        # Track the error but don't expose details to the user
        logger.error(f"Grad-CAM generation error: {e}", exc_info=True)
        return None
