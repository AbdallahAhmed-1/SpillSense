import numpy as np
from pathlib import Path
from PIL import Image
from joblib import load
from tools.image_tools import extract_single_image_features
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = Path("models/image/best_image_model.joblib")
ENCODER_PATH = Path("models/image/label_encoder.joblib")


def predict_spill_from_image(image_path: Path, verbose: bool = True) -> str:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")

    model = load(MODEL_PATH)
    encoder = load(ENCODER_PATH)

    # Extract features from image
    features = extract_single_image_features(image_path)
    if features is None:
        return "❌ Could not extract features from the image."

    # Predict class
    X = np.array([features])
    pred_class = model.predict(X)[0]
    class_label = encoder.inverse_transform([pred_class])[0]

    if verbose:
        print(f"🔍 Predicted class ID: {class_label}")

    # Classify as spill or not
    spill_classes = [51, 124, 204, 221, 255]
    has_spill = class_label in spill_classes
    return "🛢️ Spill detected!" if has_spill else "✅ No spill detected."
