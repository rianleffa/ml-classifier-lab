"""The real, shared training implementation used by the API and CLI."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

CLASS_NAMES = {0: "Maligno", 1: "Benigno"}


def train_and_evaluate(
    *,
    max_depth: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train and evaluate a Decision Tree on scikit-learn's real dataset.

    Class 0 is malignant and is treated as the positive class for the headline
    precision, recall, and F1 metrics so false negatives are visible.
    """
    started = time.perf_counter()
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, pos_label=0, zero_division=0))
    recall = float(recall_score(y_test, predictions, pos_label=0, zero_division=0))
    f1 = float(f1_score(y_test, predictions, pos_label=0, zero_division=0))
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        DecisionTreeClassifier(max_depth=max_depth, random_state=random_state),
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,
    )

    class_summary: list[dict[str, Any]] = []
    for label in (0, 1):
        mask = np.asarray(y_test) == label
        class_summary.append(
            {
                "label": label,
                "name": CLASS_NAMES[label],
                "support": int(mask.sum()),
                "correct": int(np.sum((np.asarray(y_test) == label) & (predictions == label))),
                "incorrect": int(np.sum((np.asarray(y_test) == label) & (predictions != label))),
            }
        )

    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std(ddof=0)),
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_labels": [CLASS_NAMES[0], CLASS_NAMES[1]],
        "classification_report": [
            {
                "class": name,
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in (CLASS_NAMES[0], CLASS_NAMES[1])
        ],
        "cv_scores": [float(score) for score in cv_scores],
        "class_summary": class_summary,
        "counts": {
            "total": int(len(dataset.data)),
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "correct": int(np.sum(predictions == y_test)),
            "incorrect": int(np.sum(predictions != y_test)),
            "classes": {
                CLASS_NAMES[0]: int(np.sum(y_test == 0)),
                CLASS_NAMES[1]: int(np.sum(y_test == 1)),
            },
        },
        "config": {
            "algorithm": "DecisionTreeClassifier",
            "max_depth": max_depth,
            "test_size": test_size,
            "train_size": round(1 - test_size, 2),
            "random_state": random_state,
            "stratified": True,
            "cross_validation": "StratifiedKFold (5 partições)",
            "positive_class": CLASS_NAMES[0],
        },
        "elapsed_seconds": round(elapsed, 4),
    }

    if artifact_dir is not None:
        _write_charts(result, Path(artifact_dir))
    return result


def _write_charts(result: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    colors = ["#e05d5d", "#2fa879"]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    matrix = np.asarray(result["confusion_matrix"])
    image = ax.imshow(matrix, cmap="viridis")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Maligno", "Benigno"],
           yticklabels=["Maligno", "Benigno"], xlabel="Classe prevista", ylabel="Classe real",
           title="Matriz de confusão")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "#18314f",
                    fontsize=16, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(directory / "confusion-matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    scores = result["cv_scores"]
    bars = ax.bar([f"Partição {i}" for i in range(1, 6)], scores, color="#3478d4", width=0.62)
    ax.axhline(result["metrics"]["cv_mean"], color="#27a77a", linestyle="--", label="Média")
    ax.set(ylim=(0, 1.18), ylabel="Acurácia", title="Validação cruzada estratificada")
    ax.legend(frameon=False)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "cross-validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    distribution = result["counts"]["classes"]
    bars = ax.bar(list(distribution.keys()), list(distribution.values()), color=colors, width=0.58)
    ax.set(ylabel="Registros no teste", title="Distribuição das classes no teste")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(directory / "class-distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    metric_names = ["Acurácia", "Precisão*", "Recall*", "F1*", "CV média"]
    values = [result["metrics"][key] for key in ("accuracy", "precision", "recall", "f1", "cv_mean")]
    bars = ax.bar(metric_names, values, color=["#3478d4", "#6b8ed6", "#e69b45", "#8c73c7", "#2fa879"])
    ax.set(ylim=(0, 1.08), ylabel="Pontuação", title="Métricas do modelo (classe positiva: Maligno)")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(directory / "metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output = train_and_evaluate(artifact_dir=PROJECT_ROOT / "results" / "cli-run")
    print(json.dumps(output, ensure_ascii=False, indent=2))
