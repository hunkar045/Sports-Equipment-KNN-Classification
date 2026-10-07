"""
Sports ball classification - PREDICTION

Usage:
    python prediction.py                 -> opens a window to choose an image
    python prediction.py path/to/img.jpg -> predicts that image

Run training.py first so that knn_model.joblib exists.
"""
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

from training import MODEL_PATH, extract_features


def choose_image():
    """File-picker window (falls back to typing the path)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        root.destroy()
        return path
    except Exception:
        return input("Enter image path: ").strip().strip('"')


def main():
    if not MODEL_PATH.exists():
        sys.exit("Model not found. Run  python training.py  first.")
    model = joblib.load(MODEL_PATH)

    path = sys.argv[1] if len(sys.argv) > 1 else choose_image()
    if not path:
        sys.exit("No image selected.")

    # Extract features and predict class probabilities
    features = extract_features(path).reshape(1, -1)
    proba = model.predict_proba(features)[0]
    order = np.argsort(proba)[::-1][:3]
    label = str(model.classes_[order[0]])

    print(f"\nPrediction: {label.replace('_', ' ').upper()}")
    for i in order:
        print(f"  {model.classes_[i]:<25} {proba[i]:.1%}")

    # Display ONLY the selected image with the predicted label and confidence
    import matplotlib.pyplot as plt
    img_data = np.fromfile(str(path), np.uint8)
    original = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if original is None:
        sys.exit(f"Could not load image: {path}")
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(original_rgb)
    ax.axis("off")
    title_text = f"Prediction: {label.replace('_', ' ').title()}  ({proba[order[0]]:.0%})"
    ax.set_title(title_text, fontsize=14, weight="bold", pad=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
