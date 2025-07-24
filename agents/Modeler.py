from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from state.state_utils import OilGasRCAState, save_state
from agents.modelers.random_forest_agent import RandomForestAgent
from agents.modelers.logistic_regression_agent import LogisticRegressionAgent
from agents.modelers.svm_agent import SVMClassifierAgent
from agents.modelers.decision_tree_agent import DecisionTreeAgent
import pandas as pd
from category_encoders.target_encoder import TargetEncoder
import os


class Modeler:
    def __init__(self):
        self.name = "Modeler"

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        return self.run(state)

    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        try:
            print(f"[{self.name}] started")

            self.prepare_train_test_split(state)
                        
            print(
                f"[Modeler] ✅ Train/Test shapes: "
                f"{state['model_artifacts']['X_train'].shape}, "
                f"{state['model_artifacts']['y_train'].shape}"
            )
            save_state(state, "state/rca_state.joblib")

            agents = [
                ("RandomForest", RandomForestAgent()),
                ("LogisticRegression", LogisticRegressionAgent()),
                ("DecisionTree", DecisionTreeAgent()),
                #("SVM", SVMClassifierAgent())
            ]

            evaluation_results: Dict[str, Any] = {}
            best_f1 = -1.0
            best_model_path = None

            for name, agent in agents:
                print(f"[{self.name}] 🧠 Running {name} in subprocess...")
                try:
                    updated_state = agent(state)
                    state = updated_state
                    metrics = updated_state["evaluation_results"].get(name, {})
                    evaluation_results[name] = metrics

                    f1 = metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model_path = updated_state["model_artifacts"].get(name)

                except Exception as e:
                    err = str(e)
                    print(f"[{self.name}] ⚠️ {name} agent failed: {err}")
                    state["errors_encountered"].append({"agent": name, "error_message": err})

            if best_model_path:
                state["model_artifacts"]["best_model"] = best_model_path

            state["evaluation_results"] = evaluation_results
            state["rca_completed"] = True
            state["agent_messages"].append({
                "from_agent": self.name,
                "message": f"All model agents completed. Best F1={best_f1:.3f}"
            })
            print(f"[{self.name}] ✅ Completed. Best model saved.")

        except Exception as e:
            err = str(e)
            print(f"[{self.name}] ❌ Error: {err}")
            state["errors_encountered"].append({"agent": self.name, "error_message": err})
            state["agent_messages"].append({"from_agent": self.name, "message": f"Modeler failed: {err}"})

        return state
    
    
    def prepare_train_test_split(self, state: OilGasRCAState):
        print(f"[{self.name}] 📊 Preparing train-test split...")

        df = state.get("cleaned_data")
        if df is None:
            raise ValueError("Cleaned data not found")

        # Load config
        col_map: Dict[str, str] = state.get("column_mappings", {})
        raw_feats: List[str] = state["analysis_params"].get("features", [])
        target = state["analysis_params"].get("target")

        print("++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"[Modeler] Here is the target: {target}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++")

        if not target:
            raise ValueError("Target not specified in analysis_params")

        mapped_feats = [col_map.get(f, f) for f in raw_feats]
        existing_feats = [f for f in mapped_feats if f in df.columns]

        if not existing_feats:
            existing_feats = df.select_dtypes(include="number").columns.tolist()
            print(f"[{self.name}] ℹ️ No valid features—using numeric: {existing_feats}")

        print(f"[{self.name}] 🧪 Features: {existing_feats}")

        target_col = col_map.get(target, target)
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")

        X_raw = df[existing_feats]
        y = df[target_col]

        # Separate features
        numeric_feats = X_raw.select_dtypes(include='number').columns.tolist()
        categorical_feats = [col for col in X_raw.columns if col not in numeric_feats]

        print(f"[{self.name}] 🧮 Numeric features: {numeric_feats}")
        print(f"[{self.name}] 🏷️ Categorical features (all will be target-encoded): {categorical_feats}")

        # Target encode all categorical features
        X_encoded = X_raw[numeric_feats].copy()
        from joblib import dump
        os.makedirs("models", exist_ok=True)

        # Initialize encoder even if no categorical features (for consistency)
        encoder = None
        if categorical_feats:
            encoder = TargetEncoder(cols=categorical_feats)
            X_target_encoded = encoder.fit_transform(X_raw[categorical_feats], y)
            X_encoded = pd.concat([X_encoded, X_target_encoded], axis=1)
        else:
            print(f"[{self.name}] 💤 No categorical features to encode.")

        # Save encoder to state
        enc_store = state["model_artifacts"].setdefault("encoders", {})
        enc_store["target_encoder"] = encoder
        enc_store["target_encoded_columns"] = categorical_feats

        # Always dump feature order
        dump(X_encoded.columns.tolist(), "models/feature_order.joblib")

        # Dump encoder only if it exists
        if encoder:
            dump(encoder, "models/encoder.joblib")
        else:
            # Dump a dummy encoder to keep downstream logic simple
            dump(None, "models/encoder.joblib")


        print(f"[{self.name}] ✅ Encoded feature matrix shape: {X_encoded.shape}")

        # Stratified split (fallback if needed)
        try:
            if y.value_counts().min() >= 2:
                stratify_param = y
            else:
                raise ValueError("Class imbalance too severe")
        except Exception as e:
            print(f"[{self.name}] ⚠️ Stratified split failed: {e}")
            stratify_param = None

        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=stratify_param
        )
        
        print("[Modeler] 🧾 y_train value counts:")
        print(y_train.value_counts())


        # Apply SMOTE only if all classes have enough samples
        print(f"[{self.name}] 🧾 y_train value counts:")
        print(y_train.value_counts())

        min_class_size = y_train.value_counts().min()

        if min_class_size >= 6:
            print(f"[{self.name}] 🧬 Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"[{self.name}] ✅ Resampled shape: {X_train_resampled.shape}, {y_train_resampled.shape}")
        else:
            print(f"[{self.name}] ⚠️ Skipping SMOTE — not enough samples per class (min={min_class_size})")
            X_train_resampled, y_train_resampled = X_train, y_train
        # Store in state
        state["model_artifacts"]["X_train"] = X_train_resampled
        state["model_artifacts"]["y_train"] = y_train_resampled
        state["model_artifacts"]["X_test"] = X_test
        state["model_artifacts"]["y_test"] = y_test
        state["model_artifacts"]["feature_columns"] = X_encoded.columns.tolist()

        print(f"[{self.name}] ✅ All training artifacts saved to state.")