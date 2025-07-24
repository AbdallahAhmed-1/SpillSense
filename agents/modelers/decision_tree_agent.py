import os
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
from tools.Evaluation_tools import evaluate_classification_model
from state.state_utils import OilGasRCAState


class DecisionTreeAgent:
    def __init__(self):
        self.name = "DecisionTreeAgent"

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

            # Train Decision Tree model
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(X_train, y_train)

            # Evaluate
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
            # Ensure y_proba is a numpy ndarray or None
            if isinstance(y_proba, list):
                import numpy as np
                y_proba = np.array(y_proba)
            metrics = evaluate_classification_model(y_test, y_pred, y_proba)

            # Save model
            os.makedirs("models", exist_ok=True)
            model_bundle = {"model": clf, "features": features}
            model_path = "models/DecisionTree_model.joblib"
            dump(model_bundle, model_path)

            # Update state
            state["evaluation_results"]["DecisionTree"] = metrics
            state["model_artifacts"]["DecisionTree"] = model_path
            state["agent_messages"].append({
                "from_agent": self.name,
                "message": f"DecisionTree model trained and saved to {model_path}."
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
