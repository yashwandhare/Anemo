# src/preprocess.py
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def preprocess_image(img_rgb):
    """
    Get the image ready for the AI model.
    """
    # Make sure we have an image to work with
    if img_rgb is None:
        logger.error("Preprocess received None input")
        return None
    
    if not isinstance(img_rgb, np.ndarray):
        logger.error(f"Invalid input type: {type(img_rgb)}")
        return None
    
    if img_rgb.shape != (224, 224, 3):
        logger.error(f"Invalid input shape: {img_rgb.shape}, expected (224, 224, 3)")
        return None
    
    if not np.issubdtype(img_rgb.dtype, np.integer) or img_rgb.max() > 255:
        logger.error(f"Invalid input dtype or value range")
        return None

    try:
        # Prepare the image for processing
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Enhance the contrast on the green channel
        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        g2 = clahe.apply(g)
        img = cv2.merge([b, g2, r])

        # Remove noise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        # Make details clearer
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale pixel values to 0-1 range
        img = img.astype("float32") / 255.0
        
        # Add batch dimension so the model knows it's one image
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None