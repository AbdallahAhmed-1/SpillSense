import os
import pandas as pd
import joblib
from state.state_utils import load_state
from state.session_data import session
from datetime import datetime
from pathlib import Path

STATE_PATH = "state/rca_state_after_modeling.joblib"

def load_model(model_path: str):
    print(f"📦 Loading best model from: {model_path}")
    model_dict = joblib.load(model_path)
    return model_dict["model"] if isinstance(model_dict, dict) and "model" in model_dict else model_dict

def prepare_features_for_prediction(df_raw: pd.DataFrame, state: dict) -> pd.DataFrame:
    model_artifacts = state.get("model_artifacts", {})
    col_map = state.get("column_mappings", {})
    raw_feats = state["analysis_params"].get("features", [])
    mapped_feats = [col_map.get(f, f) for f in raw_feats]
    df = df_raw.copy()

    # Subset to only columns required for prediction
    df = df[[f for f in mapped_feats if f in df.columns]]

    encoders = model_artifacts.get("encoders", {})
    target_encoder = encoders.get("target_encoder")
    target_encoded_columns = encoders.get("target_encoded_columns", [])

    if not target_encoder:
        raise ValueError("❌ Target encoder not found in state")

    # Split into numeric/categorical
    numeric_feats = df.select_dtypes(include="number").columns.tolist()
    categorical_feats = [col for col in df.columns if col not in numeric_feats]

    # Apply encoding
    df_encoded = target_encoder.transform(df[target_encoded_columns]) if target_encoded_columns else pd.DataFrame()
    df_numeric = df[numeric_feats]

    final_X = pd.concat([df_numeric, df_encoded], axis=1)
    final_X = final_X.loc[:, ~final_X.columns.duplicated()]

    # Match training columns
    trained_columns = model_artifacts.get("X_train", pd.DataFrame()).columns.tolist()
    final_X = final_X.reindex(columns=trained_columns, fill_value=0)

    return final_X


# scripts/predict_new_batch.py

def predict_from_csv_file(
    csv_path: str,
    out_dir: str,
    state_path: str = "state/rca_state_after_modeling.joblib"
) -> str:
    """
    Non-interactive wrapper. Returns output CSV path.
    """
    state = load_state(state_path)
    model_blob = joblib.load(state["model_artifacts"]["best_model"])
    model = model_blob.get("model", model_blob)

    df_in = pd.read_csv(csv_path)
    X = prepare_features_for_prediction(df_in, state)
    preds = model.predict(X)

    # optionally add confidence
    try:
        df_in["prediction_confidence"] = model.predict_proba(X).max(axis=1)
    except Exception:
        pass

    df_in["predicted_severity"] = preds

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f"predicted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = Path(out_dir) / out_name
    df_in.to_csv(out_path, index=False)
    return str(out_path)


def predict_from_text(model, state: dict):
    print("📝 Manual prediction mode (type 'exit' at any time to quit)\n")
    previous_inputs = {}
    raw_feats = state["analysis_params"].get("features", [])
    col_map = state.get("column_mappings", {})
    mapped_feats = [col_map.get(f, f) for f in raw_feats]
    feature_types = state["model_artifacts"]["X_train"].dtypes.to_dict()

    while True:
        row_data = {}
        for col in mapped_feats:
            dtype = feature_types.get(col, "float64")
            default_val = f"[{previous_inputs.get(col, '')}]" if col in previous_inputs else ""
            prompt = f"Enter value for '{col}' {default_val}: "
            while True:
                val = input(prompt).strip()
                if val.lower() == 'exit':
                    return
                if val == "" and col in previous_inputs:
                    row_data[col] = previous_inputs[col]
                    break
                try:
                    if pd.api.types.is_numeric_dtype(dtype):
                        row_data[col] = float(val)
                    else:
                        row_data[col] = val
                    previous_inputs[col] = row_data[col]
                    break
                except ValueError:
                    print("❌ Invalid input. Please try again.")

        input_df = pd.DataFrame([row_data])
        try:
            X_input = prepare_features_for_prediction(input_df, state)
            pred = model.predict(X_input)[0]
            print(f"\n✅ Predicted severity: {pred}\n")
        except Exception as e:
            print(f"❌ Prediction failed: {e}\n")

def main():
    print("📂 Loading state...")
    state = load_state(STATE_PATH)
    model = load_model(state["model_artifacts"]["best_model"])

    print("📊 Model trained on features:")
    print(state["model_artifacts"]["X_train"].columns.tolist())

    mode = input("🔍 Input mode? [csv/text]: ").strip().lower()
    while mode not in ("csv", "text"):
        mode = input("❌ Invalid input. Enter 'csv' or 'text': ").strip().lower()

    if mode == "csv":
        csv_path = input("📄 CSV file path: ").strip()
        output_path = input("📄 Output file path: ").strip()
        predict_from_csv(model, csv_path, output_path, state)
    elif mode == "text":
        predict_from_text(model, state)

if __name__ == "__main__":
    main()