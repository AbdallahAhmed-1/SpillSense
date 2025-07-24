# run_inference.py

import joblib
import pandas as pd
from state.state_utils import load_state, save_state
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("📂 Loading state...")
    state = load_state("state/rca_state_after_modeling.joblib")

    model_path = state["model_artifacts"].get("best_model")
    if not model_path:
        print("❌ No best model found.")
        return

    print(f"📦 Loading best model from: {model_path}")
    model_bundle = joblib.load("models/RandomForest_model.joblib")
    model = model_bundle["model"]


    X_test = state["model_artifacts"].get("X_test")
    y_test = state["model_artifacts"].get("y_test")

    if X_test is None or y_test is None:
        print("❌ Test data not found in state.")
        return

    print("🧠 Running inference...")
    y_pred = model.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    if not isinstance(report, dict):
        report = {"report": report}
    cmatrix = confusion_matrix(y_test, y_pred)

    print("✅ Inference complete.")
    print(pd.DataFrame(report).T)
    print("\nConfusion Matrix:")
    print(cmatrix)

    # Store results back into state
    if "evaluation_results" not in state or not isinstance(state["evaluation_results"], dict):
        state["evaluation_results"] = {}
        
    state["evaluation_results"]["inference_report"] = report
    state["evaluation_results"]["confusion_matrix"] = cmatrix.tolist()
    state["model_artifacts"]["y_pred"] = y_pred

    # Save updated state
    save_state(state, "state/rca_state_after_inference.joblib")
    print("💾 Updated state saved to: state/rca_state_after_inference.joblib")

if __name__ == "__main__":
    main()
