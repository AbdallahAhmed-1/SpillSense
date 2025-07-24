# agents/cleaner.py

import json
import os
import re
import numpy as np
import pandas as pd
from typing import List
from dateutil.parser import parse
from state.state_utils import OilGasRCAState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel
import math
from pandas.api.types import is_string_dtype, is_object_dtype
import ast
from category_encoders import TargetEncoder  # ✅ NEW

class Cleaner:
    def __init__(self, llm: BaseLanguageModel):
        self.name = "Cleaner"
        self.llm = llm
        self.dict = {}

    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        return self.run(state)

    def _clean_python(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:python)?\s*\n", "", text)
        text = re.sub(r"\n```$", "", text)
        return text.strip()

    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        raw_df = state.get("raw_data")
        if raw_df is None:
            state["errors_encountered"].append({"agent": self.name, "error_message": "No raw_data found to clean."})
            print("No data to be cleaned.")
            return state

        df = raw_df.copy(deep=True)
        print(f"[Cleaner] Starting data cleaning for {df.shape[0]} rows and {df.shape[1]} columns")

        df = self.clean_generic_junk(df)

        top_columns = []
        column_names = list(df.columns)
        try:
            prompt = f"""You are a data cleaning expert. Given the following column names:\n{column_names}\nIdentify at least 6 of the most important columns related to ID, geolocation, material, date, quantity, root cause analysis. Return ONLY a valid Python list of column names."""
            response = self.llm.invoke(prompt)
            cleaned_response = self._clean_python(response.content.strip())
            if not cleaned_response.startswith("["):
                raise ValueError("LLM did not return a list.")
            try:
                top_columns = ast.literal_eval(cleaned_response)
                assert isinstance(top_columns, list)
            except Exception as e:
                print(f"[Cleaner] ⚠️ Failed to parse top columns: {e}")
                top_columns = column_names[:5]
            print(f"[LLM] Top columns: {top_columns}")
        except Exception as e:
            print(f"[LLM] Column ranking failed: {e}")
            top_columns = column_names[:5]

        keep_cols = set(top_columns)
        for col in df.columns:
            if col not in keep_cols and df[col].isna().mean() > 0.6:
                df.drop(columns=col, inplace=True)
                print(f"[Cleaner] Dropped unimportant high-null column: {col}")

        common_date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"]
        for col in df.columns:
            if is_string_dtype(df[col]) or is_object_dtype(df[col]):
                for fmt in common_date_formats:
                    try:
                        df[col] = pd.to_datetime(df[col], format=fmt)
                        print(f"[Cleaner] Standardized date column '{col}' with format '{fmt}'")
                        break
                    except (ValueError, TypeError):
                        continue

        required_cols = [col for col in top_columns if col in df.columns]
        before = len(df)
        df.dropna(subset=required_cols, inplace=True)
        after = len(df)
        if before != after:
            print(f"[Cleaner] Dropped {before - after} rows with NA in important columns: {required_cols}")

        if "Quantity" in df.columns:
            try:
                df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
                df["Severity"] = pd.cut(
                    df["Quantity"],
                    bins=[-np.inf, 10, 50, np.inf],
                    labels=["low", "medium", "high"],
                    right=False
                ).astype(str)
                print("[Cleaner] ✅ 'Severity' column added as target")
            except Exception as e:
                print(f"[Cleaner] ⚠️ Could not compute 'Severity': {e}")
        else:
            print("[Cleaner] ⚠️ 'Quantity' column not found, skipping 'Severity'")

        print("[Cleaner] Dropping unwanted columns")
        df.drop(columns=[col for col in ['Quantity', 'Units', 'Recovered', '_material_sanitized_'] if col in df.columns], inplace=True, errors="ignore")

        protected_columns = {"Severity"}
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        low_card_cols = [col for col in categorical_cols if df[col].nunique() <= 50 and col not in protected_columns]

        print(f"[Cleaner] Using TargetEncoder for: {low_card_cols}")
        target_encoder = TargetEncoder(cols=low_card_cols)
        df_encoded = target_encoder.fit_transform(df[low_card_cols], df["Severity"])
        df = pd.concat([df.drop(columns=low_card_cols), df_encoded], axis=1)

        print("[Cleaner] Finished encoding.")
        print(f"[Cleaner] Final shape: {df.shape}")
        print(f"[Cleaner] Null values after cleaning: {df.isnull().sum().sum()}")

        state["cleaned_data"] = df
        state["data_cleaned"] = True
        state["analysis_params"] = {"target": "Severity", "features": [col for col in df.columns if col != "Severity"]}
        state["model_artifacts"] = state.get("model_artifacts", {})
        state["model_artifacts"]["encoders"] = {
            "target_encoder": target_encoder,
            "target_encoded_columns": low_card_cols
        }
        state["model_artifacts"]["X_train"] = df.drop(columns=["Severity"])

        print("[Cleaner] Data cleaning completed.")
        return state

    def try_parse_date(self, x: str):
        try:
            return parse(x, fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            return pd.NaT

    def clean_generic_junk(self, df: pd.DataFrame) -> pd.DataFrame:
        invalid_pattern = re.compile(r'^(?!.*[a-zA-Z0-9]).*$')
        placeholder_values = {"", " ", "na", "n/a", "null", "none", "-", "--", "nan", ".", "*", "_", "?", "#", "@", "~", "%", "!", "''", '""', "''", '"', "[]", "{}", "()", "<>", "<", ">"}

        def is_invalid(val):
            if pd.isna(val):
                return True
            val_str = str(val).strip().lower()
            if val_str in placeholder_values:
                return True
            if invalid_pattern.match(val_str):
                return True
            return False

        df_cleaned = df.map(lambda x: np.nan if is_invalid(x) else x)
        df_cleaned.dropna(how="all", inplace=True)
        df_cleaned.drop_duplicates(inplace=True)
        return df_cleaned

    def sanitize_material(self, s):
        return str(s).strip().lower().replace(" ", "_")