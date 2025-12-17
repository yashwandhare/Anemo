import os
import tensorflow as tf
from src.config import KERAS_MODEL_PATH

# Silence TensorFlow's startup messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

print(f"Loading Keras model from {KERAS_MODEL_PATH}...")

try:
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load Keras model: {e}")

def predict_anemia(input_tensor):
    """
    Input: (1, 224, 224, 3) float32 tensor
    Output: (label: str, confidence: float)
    """
    p = float(model.predict(input_tensor, verbose=0)[0][0])

    print(f"DEBUG: Raw Model Probability (p): {p:.4f}")

    # If probability < 0.5, person doesn't have anemia
    if p < 0.5:
        return "NON-ANEMIC", 1.0 - p
    else:
        return "ANEMIC", p


def get_model():
    """
    Return the model for visualization (no weights are changed).
    
    Returns:
        Loaded Keras model
    """
    return model
