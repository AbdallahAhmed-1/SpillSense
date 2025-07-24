import pandas as pd
import os
import time
from state.state_utils import load_state
from joblib import load


def predict_from_csv(csv_path: str) -> str:
    print("[Predictor] 🚀 Starting prediction...")

    # Load trained state
    state = load_state("state/rca_state_after_modeling.joblib")
    print("📂 Loading state from state/rca_state_after_modeling.joblib")

    model_path = state["model_artifacts"].get("best_model")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Best model path not found: {model_path}")
    model = load(model_path)

    try:
        encoder = load("models/encoder.joblib")
        print("[Predictor] 🏷️ TargetEncoder loaded.")
    except:
        encoder = None
        print("[Predictor] 💤 No encoder found or needed.")

    try:
        feature_order = load("models/feature_order.joblib")
    except:
        feature_order = state["model_artifacts"].get("feature_columns", [])

    # Load and clean input
    df = pd.read_csv(csv_path)
    if "predicted_severity" in df.columns:
        print("⚠️ Already predicted, skipping.")
        return None
    df_clean = clean_input_csv(df)

    # Apply encoding
    if encoder:
        cat_cols = [c for c in encoder.cols if c in df_clean.columns]
        encoded = encoder.transform(df_clean[cat_cols])
        df_clean = pd.concat([df_clean.select_dtypes(include="number"), encoded], axis=1)
    else:
        df_clean = df_clean.select_dtypes(include="number")

    # Ensure all expected features exist
    missing = [col for col in feature_order if col not in df_clean.columns]
    if missing:
        print(f"[Predictor] ❌ Missing required features: {missing}")
        for col in missing:
            df_clean[col] = 0
    df_clean = df_clean[feature_order]

    # Predict
    preds = model.predict(df_clean)
    try:
        probs = model.predict_proba(df_clean).max(axis=1)
        df["prediction_confidence"] = probs
    except:
        pass

    df["predicted_severity"] = preds

    # Save with timestamp
    base = os.path.basename(csv_path).replace(".csv", "")
    ts = time.strftime("%Y%m%d-%H%M%S")
    new_file = f"{base}_predicted_{ts}.csv"
    out_path = os.path.join("static", "reports", new_file)
    df.to_csv(out_path, index=False)

    print(f"[Predictor] ✅ Saved to: {out_path}")
    return new_file


def clean_input_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop empty or unnamed columns
    df.drop(columns=[col for col in df.columns if "Unnamed" in col or df[col].isna().all()], inplace=True)

    # Clean up strings
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Fill NaNs
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(0)

    return df
