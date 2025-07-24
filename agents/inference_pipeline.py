# inference_pipeline.py

import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt

from tools.analysis_tools import create_error_plot, fig_to_base64


def load_model(path: str):
    if not os.path.exists(path):
        print(f"❌ Model not found at: {path}")
        sys.exit(1)
    print(f"🔍 Loading model from: {path}")
    return joblib.load(path)


def clean_input_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.dropna().reset_index(drop=True)


def run_prediction_loop(model_bundle: dict):
    model = model_bundle["model"]
    features = model_bundle["features"]

    print(f"📊 Model expects features: {features}")

    while True:
        mode = input("\n📥 Enter 'csv' to load a file or 'text' to paste input manually (or 'exit' to quit): ").strip().lower()

        if mode == "exit":
            print("👋 Exiting inference loop.")
            break

        elif mode == "csv":
            path = input("📄 Enter path to CSV file: ").strip()
            if not os.path.exists(path):
                print("❌ File does not exist.")
                continue
            try:
                df = pd.read_csv(path)
                print(f"✅ Loaded data with shape: {df.shape}")
            except Exception as e:
                print(f"❌ Failed to read CSV: {e}")
                continue

        elif mode == "text":
            print(f"📝 Paste a single row of {len(features)} comma-separated values (order: {features}):")
            text = input("> ")
            try:
                values = [x.strip() for x in text.split(",")]
                df = pd.DataFrame([values], columns=features)
                print("✅ Parsed manual input into DataFrame.")
            except Exception as e:
                print(f"❌ Failed to parse input: {e}")
                continue

        else:
            print("⚠️ Invalid mode. Please type 'csv', 'text', or 'exit'.")
            continue

        try:
            df_clean = clean_input_data(df)
            X = df_clean[features]
        except Exception as e:
            print(f"❌ Cleaning failed: {e}")
            continue

        try:
            preds = model.predict(X)
            df_clean["prediction"] = preds
            print("✅ Predictions completed.")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            continue

        output_path = input("💾 Enter output path to save CSV (or press Enter for 'predictions.csv'): ").strip()
        if not output_path:
            output_path = "predictions.csv"
        try:
            df_clean.to_csv(output_path, index=False)
            print(f"📦 Predictions saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save predictions: {e}")

        try:
            fig, ax = plt.subplots()
            df_clean["prediction"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Prediction Distribution")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Count")
            plt.show()
        except Exception as e:
            print(f"⚠️ Failed to show plot: {e}")


if __name__ == "__main__":
    default_model_path = "models/best_model.joblib"
    print(f"📦 Default model path: {default_model_path}")
    user_path = input("🔧 Enter model path (or press Enter to use default): ").strip()
    model_path = user_path if user_path else default_model_path

    model_bundle = load_model(model_path)
    run_prediction_loop(model_bundle)
