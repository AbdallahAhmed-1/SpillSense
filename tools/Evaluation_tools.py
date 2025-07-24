#tools/Evaluation_tools.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Union, Optional, List
from tools.common_utils import fig_to_base64, create_error_plot



def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[Union[str, int]]] = None,
    average: str = "macro"
) -> Dict[str, Union[Dict, str]]:
    """
    Evaluates classification model using metrics, confusion matrix, and ROC/PR curves (if applicable).

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (required for ROC/PR curves).
        labels: Optional list of class labels.
        average: Averaging strategy for multiclass metrics.

    Returns:
        Dictionary with evaluation results including images (as base64 PNGs) and metrics.
    """
    result: Dict[str, Union[Dict, str]] = {}

    # 1️- Classification Report
    try:
        result["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print(f"Classification report: {result['classification_report']}")
        print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
               
    except Exception as e:
        result["classification_report"] = {"error": f"Classification report failed: {e}"}

    # 2️- Confusion Matrix
    try:
        cm_labels = labels if labels is not None else np.unique(y_true).tolist()
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels).plot(ax=ax, cmap="Blues", colorbar=False)
        result["confusion_matrix_image"] = f"data:image/png;base64,{fig_to_base64(fig)}"
    except Exception as e:
        result["confusion_matrix_image"] = create_error_plot(f"Confusion matrix failed: {e}")

    # 3️- ROC & PR Curves (if probabilities provided)
    y_true_bin = None
    y_score = None
    
    if y_proba is not None:
        try:
            label_list = np.unique(y_true).tolist()
            is_multiclass = len(label_list) > 2

            if is_multiclass:
                y_true_bin = label_binarize(y_true, classes=label_list)
                y_score = y_proba
                fpr, tpr, _ = roc_curve(np.asarray(y_true_bin).ravel(), np.asarray(y_score).ravel())
            else:
                y_true_bin = np.asarray(label_binarize(y_true, classes=label_list)).ravel()
                y_score = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                fpr, tpr, _ = roc_curve(y_true_bin, y_score)

            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 4))
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)
            result["roc_curve_image"] = f"data:image/png;base64,{fig_to_base64(fig)}"

            # PR Curve (moved inside same try to guarantee variables are defined)
            precision, recall, _ = precision_recall_curve(
                np.asarray(y_true_bin).ravel(),
                np.asarray(y_score).ravel()
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)
            result["pr_curve_image"] = f"data:image/png;base64,{fig_to_base64(fig)}"

        except Exception as e:
            result["roc_curve_image"] = create_error_plot(f"ROC/PR curve failed: {e}")
            result["pr_curve_image"] = create_error_plot(f"ROC/PR curve failed: {e}")

        try:
            precision, recall, _ = precision_recall_curve(
            np.asarray(y_true_bin).ravel(), 
            np.asarray(y_score).ravel()
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)
            result["pr_curve_image"] = f"data:image/png;base64,{fig_to_base64(fig)}"
        except Exception as e:
            result["pr_curve_image"] = create_error_plot(f"PR curve failed: {e}")

    return result
