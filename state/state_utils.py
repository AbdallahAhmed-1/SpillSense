# state/state_utils.py  – unified state definition for CSV, HSI (.mat), and JPG workflows
# =====================================================================================
# This version keeps the original tabular‑oriented fields *and* adds optional keys that
# the MAT / Image workflows need (raw cubes, cleaned cubes, per‑file summaries, etc.).
# All keys are optional (`total=False`) so each modality can populate only what it uses.

from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
import base64
import os
import joblib

# ---------------------------------------------------------------------------
# Small dataclasses reused across modalities
# ---------------------------------------------------------------------------

@dataclass
class DataQualityMetrics:
    total_rows: int = 0
    total_columns: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    data_types: Dict[str, str] = field(default_factory=dict)
    memory_usage: float = 0.0


@dataclass
class VisualizationArtifact:
    plot_type: str
    title: str
    description: str
    base64_image: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReportSection:
    section_name: str
    content: str
    visualizations: List[str] = field(default_factory=list)
    tables: List[pd.DataFrame] = field(default_factory=list)
    priority: int = 1


# ---------------------------------------------------------------------------
# Unified TypedDict – all keys optional so each workflow can pick & choose
# ---------------------------------------------------------------------------

class OilGasRCAState(TypedDict, total=False):
    # ==== universal context ====
    modality: str                     # 'csv' | 'mat' | 'jpg'
    paths: List[str]                  # raw file paths given to the workflow
    processed_paths: List[str]        # new locations after moving to processed/
    data_source_type: str             # csv, mat, jpg, xlsx …
    data_loaded_at: datetime

    warnings: List[str]
    errors_encountered: List[Dict[str, str]]
    agent_messages: List[Dict[str, str]]

    # ==== Tabular‑specific ====
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    data_quality: DataQualityMetrics
    engineered_features: pd.DataFrame

    model_artifacts: Dict[str, Any]
    evaluation_results: Dict[str, Dict[str, Any]]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: pd.Series
    analysis_params: Dict[str, Any]
    column_mappings: Dict[str, str]

    # ==== HSI (.mat) specific ====
    hsi_data: Dict[str, Any]          # filename → raw cube (np.ndarray or dict)
    hsi_data_clean: Dict[str, Any]    # filename → cleaned cube
    hsi_summaries: Dict[str, Any]     # filename → dict of mean spectrum, PCA path …

    # ==== Image/JPG specific ====
    img_data: Dict[str, Any]        # filename → raw image (np.ndarray)
    img_data_clean: Dict[str, Any]  # filename → cleaned / segmented image
    mask_data_clean: Dict[str, Any]  # maskname → binary mask (np.ndarray)
    image_summaries: Dict[str, Any]   # filename → histogram, mask paths 
    mask_data: Dict[str, Any]        # maskname → binary mask (np.ndarray)
    image_paths: List[str]            # List of image file paths
    mask_paths: List[str]             # List of mask file paths

    # ==== Visualization & reporting ====
    visualizations: List[VisualizationArtifact]
    insights: List[str]
    report_sections: List[ReportSection]
    pdf_report_path: str
    report_generated_at: datetime
    report_title: str
    report_author: str

    # ==== Pipeline status flags ====
    data_loaded: bool
    data_cleaned: bool
    exploration_completed: bool
    rca_completed: bool
    report_generated: bool
    image_exploration_completed: bool

    # ==== ad‑hoc analysis results (legacy) ====
    basic_summary_result: str
    missing_values_result: Dict[str, int]
    frequency_distribution_result: Dict[str, int]
    outliers_result: str
    
    # ==== Web search results ====
    web_search_input: str             # original query string
    web_search_terms: List[str]       # parsed search terms
    web_search_start: str             # start date in ISO format
    web_search_end: str               # end date in ISO format
    web_search_results: Dict[str, List[Dict[str, str]]]  # term →
    web_search_completed: bool  # True if search was performed


# ---------------------------------------------------------------------------
# Helpers to create / persist the state
# ---------------------------------------------------------------------------

STATE_FILE = "state/rca_state.joblib"


def create_initial_state(
    *,
    modality: str,
    paths: Optional[List[str]] = None,
    data_path: Optional[str] = None,
    **kwargs,
) -> OilGasRCAState:
    """Factory that works for all three modalities.

    If `modality=='csv'` **and** `data_path` is provided, load the CSV/XLSX to
    auto‑detect column types like the legacy function did.
    """
    now = datetime.now()

    # ---------------- tabular auto‑schema (CSV only) -------------------
    column_mappings: Dict[str, str] = {}
    analysis_params: Dict[str, Any] = {}

    if modality == "csv" and data_path:
        try:
            if data_path.lower().endswith(".csv"):
                df = pd.read_csv(data_path)
            elif data_path.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(data_path)
            else:
                df = pd.read_csv(data_path)

            columns = df.columns.tolist()
            date_cols = [c for c in columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower()]
            numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in columns if pd.api.types.is_object_dtype(df[c]) or df[c].dtype.name == "category"]

            for idx, col in enumerate(columns, start=1):
                column_mappings[f"col_{idx}"] = col

            target = kwargs.get("target", "Severity")
            feature_candidates = [c for c in columns if c != target]

            analysis_params = {
                "all_columns": columns,
                "date_columns": date_cols,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                # define target & full feature set excluding target
                "target": target,
                "features": feature_candidates,
                # clustering default
                "n_clusters": kwargs.get('n_clusters', 3),
                # time-series frequency default
                "freq": kwargs.get('freq', 'ME'),
                # top-n for ranking tasks
                "top_n": kwargs.get('top_n', 5),
            }
        except Exception as e:
            print(f"Warning: CSV analysis skipped: {e}")

    # ---------------- state skeleton ----------------------------------
    state: OilGasRCAState = {
        "modality": modality,
        "paths": paths or ([] if data_path is None else [data_path]),
        "processed_paths": [],
        "data_source_type": (Path(data_path).suffix.lstrip(".") if data_path else modality),
        "data_loaded_at": None,

        # tabular placeholders
        "raw_data": None,
        "cleaned_data": None,
        "data_quality": None,
        "engineered_features": None,
        

        # HSI / Image dict slots
        "hsi_data": {},
        "hsi_data_clean": {},
        "hsi_summaries": {},
        "image_data": {},

        # Reporting
        "visualizations": [],
        "insights": [],
        "report_sections": [],
        "pdf_report_path": kwargs.get("pdf_report_path", str(Path.cwd())),
        "report_generated_at": None,
        "report_title": "Oil & Gas RCA Report",
        "report_author": "RCA Framework",

        # Flags
        "data_loaded": False,
        "data_cleaned": False,
        "exploration_completed": False,
        "rca_completed": False,
        "report_generated": False,
        "image_exploration_completed": False,

        # Legacy analysis
        "basic_summary_result": None,
        "missing_values_result": None,
        "frequency_distribution_result": None,
        "outliers_result": None,

        # Modelling dicts
        "model_artifacts": {},
        "evaluation_results": {},

        # Train/test placeholders
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "y_pred": None,

        # Meta
        "warnings": [],
        "errors_encountered": [],
        "agent_messages": [],
        "column_mappings": column_mappings,
        "analysis_params": analysis_params,
        
        "img_data": {},      # filename → raw image (np.ndarray)
        "img_data_clean": {},  # filename → cleaned / segmented image
        "mask_data_clean": {},  # maskname → binary mask (np.ndarray)
        "image_summaries": {},  # filename → histogram, mask paths 
        "mask_data": {},      # maskname → binary mask (np.ndarray)
        "image_paths": [],           # List of image file paths
        "mask_paths": [],
        
        "web_search_input": "",
        "web_search_terms": [],
        "web_search_start": "",
        "web_search_end": "",
        "web_search_results": {},
        "web_search_completed": False
    }
    return state


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_state(state: OilGasRCAState, path: str = STATE_FILE):
    os.makedirs(Path(path).parent, exist_ok=True)
    joblib.dump(state, path, compress=3)
    print(f"💾 State saved to {path}")


def load_state(path: str = STATE_FILE) -> OilGasRCAState:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"📂 Loading state from {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Convenience for embedding base64 images in the state
# ---------------------------------------------------------------------------

def save_figure_to_state(state: OilGasRCAState, *, base64_image: str, plot_type: str,
                         title: str, description: str = "") -> VisualizationArtifact:
    art = VisualizationArtifact(plot_type, title, description, base64_image, metadata={"source": "analysis_tools"})
    state.setdefault("visualizations", []).append(art)
    state.setdefault("model_artifacts", {})[title] = art
    return art


def get_visualization_as_image(artifact: VisualizationArtifact) -> bytes:
    return base64.b64decode(artifact.base64_image) if artifact.base64_image else b""
