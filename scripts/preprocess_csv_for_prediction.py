import pandas as pd
import joblib
import os
from state.state_utils import load_state

RAW_INPUT_PATH = "data/raw/csv/synthetic_spill_data.csv"
OUTPUT_PATH = "/Users/abdalla/Desktop/SpillSense/SpillSense/predictions/cleaned_for_prediction.csv"
STATE_PATH = "state/rca_state_after_modeling.joblib"

def main():
    print("📂 Loading state...")
    state = load_state(STATE_PATH)

    # Load raw data
    df = pd.read_csv(RAW_INPUT_PATH)

    # Get trained features
    trained_feats = state["model_artifacts"]["X_train"].columns.tolist()
    print(f"🔍 Model expects {len(trained_feats)} features")

    # Fill missing columns with 0
    for col in trained_feats:
        if col not in df.columns:
            df[col] = 0

    # Retain only the trained features
    df_clean = df[trained_feats]

    # Save for prediction
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Preprocessed input saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
