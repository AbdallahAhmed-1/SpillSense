import numpy as np
from pathlib import Path
from PIL import Image
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
from state.state_utils import load_state
from tools.image_tools import extract_image_features
from sklearn.preprocessing import LabelEncoder

MASK_DIR = Path("data/processed/masks")
IMG_DIR = Path("data/processed/jpg")
STATE_PATH = Path("state/image_state.joblib")
MODEL_DIR = Path("models/image")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("\n📦 Loading image state...")
print(f"📂 Loading state from {STATE_PATH}")
state = load_state(STATE_PATH)

# Step 1: Scan class IDs
print("\n🧪 Scanning available class IDs across all masks...")
class_ids = set()
for mask_path in MASK_DIR.glob("*.png"):
    mask = np.array(Image.open(mask_path))
    ids = np.unique(mask)
    class_ids.update(ids[ids > 0])

class_ids = sorted(list(class_ids))
print(f"  - Detected class IDs: {class_ids}")

# Step 2: Extract features and labels
print("🔍 Extracting features and labels...")
X, y = extract_image_features(IMG_DIR, MASK_DIR, class_ids)

if len(X) == 0 or len(set(y)) < 2:
    print(f"✅ Feature shape: {np.array(X).shape}, Labels: {len(set(y))} classes")
    print("🚫 No usable features found or only one class present.")
    exit()

X = np.array(X)
y = np.array(y)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
dump(label_encoder, MODEL_DIR / "label_encoder.joblib")
print(f"✅ Feature shape: {X.shape}, Labels: {len(set(y))} classes")

# Step 3: Balance dataset
print("\n📊 Applying SMOTE to balance classes...")
try:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print(f"  - After SMOTE: {X.shape[0]} samples")
except Exception as e:
    print(f"⚠️ SMOTE failed: {e}")

# Step 4: Train and evaluate models
print("\n🏁 Training and evaluating models...")
scorer = make_scorer(f1_score, average='macro')

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC()
}

scores = {}
for name, model in models.items():
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scorer)
        avg_score = np.nanmean(cv_scores)
    except Exception as e:
        print(f"⚠️  {name} failed: {e}")
        avg_score = -1.0
    scores[name] = avg_score
    print(f"🔹 {name}: F1-macro = {avg_score:.4f}")

# Step 5: Select and train best model
best_model_name = max(scores, key=scores.get)
best_score = scores[best_model_name]
if best_score < 0:
    print("❌ No valid model could be trained.")
    exit()

best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name} (F1-macro = {best_score:.4f})")
best_model.fit(X, y)

# Step 6: Save best model
model_path = MODEL_DIR / "best_image_model.joblib"
dump(best_model, model_path)
print(f"🗒️ Saved best model to: {model_path}")