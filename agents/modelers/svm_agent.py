import os
from typing import Dict, Any
from sklearn.svm import SVC
from joblib import dump
from tools.Evaluation_tools import evaluate_classification_model
from state.state_utils import OilGasRCAState


class SVMClassifierAgent:
    def __init__(self):
        self.name = "SVMClassifierAgent"

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        return self.run(state)

    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        try:
            print(f"[{self.name}] Training started...")

            X_train = state["model_artifacts"].get("X_train")
            y_train = state["model_artifacts"].get("y_train")
            X_test = state["model_artifacts"].get("X_test")
            y_test = state["model_artifacts"].get("y_test")

            if X_train is None or y_train is None or X_test is None or y_test is None:
                raise ValueError("Missing training/testing data in state")

            features = X_train.columns.tolist()

            # Train SVM model (with probability enabled for evaluation)
            clf = SVC(probability=True, random_state=42)
            clf.fit(X_train, y_train)

            # Evaluate
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
            metrics = evaluate_classification_model(y_test, y_pred, y_proba)

            # Save model
            os.makedirs("models", exist_ok=True)
            model_bundle = {"model": clf, "features": features}
            model_path = "models/SVM_model.joblib"
            dump(model_bundle, model_path)

            # Update state
            state["evaluation_results"]["SVM"] = metrics
            state["model_artifacts"]["SVM"] = model_path
            state["agent_messages"].append({
                "from_agent": self.name,
                "message": f"SVM model trained and saved to {model_path}."
            })
            print(f"[{self.name}] ✅ Completed")

        except Exception as e:
            error_msg = str(e)
            print(f"[{self.name}] ❌ Error: {error_msg}")
            state["errors_encountered"].append({
                "agent": self.name,
                "error_message": error_msg
            })
            state["agent_messages"].append({
                "from_agent": self.name,
                "message": f"Training failed: {error_msg}"
            })

        return state
