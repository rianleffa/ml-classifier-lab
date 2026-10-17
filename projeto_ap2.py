"""CLI companion to the ML Classifier Lab AP2 web application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.app.training import train_and_evaluate


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parent
    result = train_and_evaluate(artifact_dir=project_root / "results" / "cli-run")
    print("--- Avaliação no conjunto de teste ---")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")
    print("\nMatriz de confusão (linhas = real; colunas = previsto; ordem Maligno, Benigno):")
    print(result["confusion_matrix"])
    print("\nAcurácias por partição:", [round(score, 4) for score in result["cv_scores"]])
    print("\nConfiguração:")
    print(json.dumps(result["config"], ensure_ascii=False, indent=2))
    print("\nGráficos salvos em:", project_root / "results" / "cli-run")
