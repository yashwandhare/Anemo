# main_cli.py
import sys
import argparse
from pathlib import Path
from src.detector import detect_and_crop
from src.preprocess import preprocess_image
from src.classifier import predict_anemia

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Anemia Detection CLI")
    parser.add_argument("image_path", help="Path to the input eye image")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Error: File not found at {image_path}")
        return

    print(f"\n--- Processing {image_path.name} ---")

    # 1. Detect & Crop
    print("Step 1: Running YOLO Detector...")
    crop_rgb = detect_and_crop(image_path)

    if crop_rgb is None:
        print("Failed to process image.")
        return

    # 2. Preprocess
    print("Step 2: Preprocessing...")
    input_tensor = preprocess_image(crop_rgb)

    # 3. Predict
    print("Step 3: Running Classifier...")
    label, confidence = predict_anemia(input_tensor)

    # 4. Output Result
    print("\n" + "="*30)
    print(f"RESULT: {label}")
    print(f"CONFIDENCE: {confidence * 100:.2f}%")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()