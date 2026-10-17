from __future__ import annotations

import csv
import io
import json
import os
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer
from matplotlib.backends.backend_pdf import PdfPages

from .training import CLASS_NAMES, train_and_evaluate

RUNS_DIR = PROJECT_ROOT / "results" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ML Classifier Lab API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    max_depth: int = Field(default=4, ge=1, le=20)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = Field(default=42, ge=0, le=999999)


def _dataset():
    return load_breast_cancer(as_frame=True)


def _feature_description(name: str) -> str:
    parts = name.split()
    if parts[0] in {"mean", "worst"}:
        statistic, base = parts[0], " ".join(parts[1:])
    elif parts[-1] == "error":
        statistic, base = "error", " ".join(parts[:-1])
    else:
        statistic, base = "mean", " ".join(parts)
    attribute = {
        "radius": "raio (distância do centro até pontos na borda)",
        "texture": "desvio padrão dos valores de escala de cinza",
        "perimeter": "perímetro do núcleo celular",
        "area": "área do núcleo celular",
        "smoothness": "variação local dos comprimentos dos raios",
        "compactness": "perímetro² / área − 1",
        "concavity": "severidade das porções côncavas do contorno",
        "concave points": "quantidade de porções côncavas do contorno",
        "symmetry": "simetria do núcleo celular",
        "fractal dimension": "aproximação da dimensão fractal do contorno",
    }.get(base, base)
    description = {
        "mean": f"Média de {attribute}",
        "error": f"Erro padrão de {attribute}",
        "worst": f"Maior valor (pior caso) observado de {attribute}",
    }[statistic]
    return f"{description}. Medida nuclear calculada a partir de imagem digitalizada de punção aspirativa."


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "ML Classifier Lab API"}


@app.get("/api/dataset")
def dataset_info():
    try:
        data = _dataset()
        counts = {CLASS_NAMES[int(label)]: int(count) for label, count in zip(*np.unique(data.target, return_counts=True))}
        return {
            "name": "Breast Cancer Wisconsin Diagnostic",
            "source": "Scikit-learn: sklearn.datasets.load_breast_cancer",
            "records": int(data.data.shape[0]),
            "features_count": int(data.data.shape[1]),
            "classes_count": int(len(data.target_names)),
            "class_distribution": counts,
            "classes": [{"label": 0, "name": "Maligno"}, {"label": 1, "name": "Benigno"}],
            "features": [{"name": str(name), "description": _feature_description(str(name))} for name in data.feature_names],
            "split": {"train_percent": 80, "test_percent": 20, "stratified": True, "note": "Proporção padrão; é ajustável na página Treinamento."},
            "sample_columns": [*map(str, data.feature_names), "diagnostico"],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Não foi possível carregar o dataset do Scikit-learn: {exc}") from exc


@app.get("/api/dataset/samples")
def dataset_samples(limit: int = 12):
    data = _dataset()
    frame = data.frame.head(max(1, min(limit, 100))).copy()
    frame["diagnostico"] = frame["target"].map({0: "Maligno", 1: "Benigno"})
    frame = frame.drop(columns=["target"])
    return {"columns": list(frame.columns), "rows": frame.round(4).to_dict(orient="records")}


@app.get("/api/dataset.csv")
def dataset_csv():
    data = _dataset()
    frame = data.frame.copy()
    frame["diagnostico"] = frame["target"].map({0: "Maligno", 1: "Benigno"})
    frame = frame.drop(columns=["target"])
    content = frame.to_csv(index=False).encode("utf-8-sig")
    return Response(content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": "attachment; filename=breast-cancer-wisconsin.csv"})


def _read_runs() -> list[dict[str, Any]]:
    runs = []
    for result_path in RUNS_DIR.glob("*/result.json"):
        try:
            runs.append(json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)


@app.get("/api/runs")
def list_runs():
    return {"runs": _read_runs()}


@app.post("/api/train")
def train(request: TrainRequest):
    run_id = str(uuid.uuid4())
    run_dir = RUNS_DIR / run_id
    try:
        results = train_and_evaluate(
            max_depth=request.max_depth,
            test_size=request.test_size,
            random_state=request.random_state,
            artifact_dir=run_dir / "charts",
        )
        run = {
            "id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **results,
            "charts": {name: f"/api/runs/{run_id}/charts/{name}.png" for name in ("confusion-matrix", "cross-validation", "class-distribution", "metrics")},
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        return run
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falha durante o treinamento: {exc}") from exc


def _get_run(run_id: str) -> dict[str, Any]:
    path = RUNS_DIR / run_id / "result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Execução não encontrada.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}/charts/{chart_name}.png")
def chart(run_id: str, chart_name: str):
    if chart_name not in {"confusion-matrix", "cross-validation", "class-distribution", "metrics"}:
        raise HTTPException(status_code=404, detail="Gráfico não encontrado.")
    path = RUNS_DIR / run_id / "charts" / f"{chart_name}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Gráfico não encontrado.")
    return FileResponse(path, media_type="image/png", filename=f"{chart_name}-{run_id[:8]}.png")


@app.get("/api/runs/{run_id}/results.csv")
def results_csv(run_id: str):
    run = _get_run(run_id)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["execucao", "metrica", "valor", "configuracao"])
    for metric, value in run["metrics"].items():
        writer.writerow([run["created_at"], metric, value, json.dumps(run["config"], ensure_ascii=False)])
    return Response(stream.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f"attachment; filename=metricas-{run_id[:8]}.csv"})


@app.get("/api/runs/{run_id}/report.pdf")
def report_pdf(run_id: str):
    run = _get_run(run_id)
    out = io.BytesIO()
    with PdfPages(out) as pdf:
        created = datetime.fromisoformat(run["created_at"]).astimezone().strftime("%d/%m/%Y")
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.78, "APRENDIZADO DE MÁQUINA", ha="center", fontsize=18, fontweight="bold", color="#172d3a")
        fig.text(0.5, 0.71, "Classificação de tumores mamários", ha="center", fontsize=16, fontweight="bold", color="#172d3a")
        fig.text(0.5, 0.675, "utilizando Árvore de Decisão", ha="center", fontsize=15, color="#345f70")
        cover = [
            "Disciplina: Fundamentos de Inteligência Artificial",
            "Curso: Análise e Desenvolvimento de Sistemas | ULBRA Torres",
            "Integrante: Rian Cardoso Leffa",
            f"Data da execução: {created}",
        ]
        for index, line in enumerate(cover):
            fig.text(0.5, 0.54 - index * 0.038, line, ha="center", fontsize=10, color="#52636d")
        fig.text(0.5, 0.28, "Relatório da Avaliação Parcial 2 (AP2)", ha="center", fontsize=11, color="#345f70")
        fig.text(0.5, 0.10, "Finalidade acadêmica. Não utilizar para diagnóstico médico.", ha="center", fontsize=9, color="#a34b47")
        pdf.savefig(fig)
        plt.close(fig)

        data = _dataset()
        class_counts = {CLASS_NAMES[int(label)]: int(count) for label, count in zip(*np.unique(data.target, return_counts=True))}
        accuracy_delta = run["metrics"]["accuracy"] - run["metrics"]["cv_mean"]
        sections: list[tuple[str, list[str]]] = [
            ("1. Introdução e definição do problema", [
                "Classificar registros como malignos ou benignos a partir de medidas celulares. Este estudo é acadêmico e não constitui validação clínica.",
            ]),
            ("2. Tipo de aprendizado e dataset", [
                f"Aprendizado supervisionado. Breast Cancer Wisconsin Diagnostic, disponibilizado pelo Scikit-learn: {len(data.data)} registros, {data.data.shape[1]} atributos numéricos, {class_counts['Maligno']} malignos e {class_counts['Benigno']} benignos.",
            ]),
            ("3. Fluxo do projeto de Machine Learning", [
                "Definição do problema, carregamento dos dados, preparação com divisão estratificada, treinamento, avaliação e consideração de validação independente para qualquer implantação futura.",
            ]),
            ("4. Algoritmo clássico: implementação e resultados", [
                f"{run['config']['algorithm']}; profundidade máxima {run['config']['max_depth']}; treino {run['config']['train_size']:.0%}, teste {run['config']['test_size']:.0%}; random_state={run['config']['random_state']}; tempo {run['elapsed_seconds']:.4f} s.",
                f"Acurácia: {run['metrics']['accuracy']:.4f}. Precisão para maligno: {run['metrics']['precision']:.4f}. Recall para maligno: {run['metrics']['recall']:.4f}. F1-score para maligno: {run['metrics']['f1']:.4f}.",
            ]),
            ("5. Avaliação aprofundada do modelo", [
                *[f"{row['class']}: precisão {row['precision']:.4f}; recall {row['recall']:.4f}; F1-score {row['f1']:.4f}; suporte {row['support']}." for row in run["classification_report"]],
                "Matriz de confusão (linhas = classe real; colunas = prevista; ordem Maligno, Benigno): " + str(run["confusion_matrix"]),
                "Acurácias nas cinco partições: " + ", ".join(f"{score:.4f}" for score in run["cv_scores"]),
                f"Média da validação cruzada: {run['metrics']['cv_mean']:.4f}; desvio-padrão: {run['metrics']['cv_std']:.4f}.",
            ]),
            ("6. Análise crítica do desempenho", [
                f"A acurácia no teste {run['metrics']['accuracy']:.4f} difere da média CV {run['metrics']['cv_mean']:.4f} em {accuracy_delta:+.4f}. No teste: {run['counts']['correct']} acertos e {run['counts']['incorrect']} erros; falsos negativos malignos: {run['confusion_matrix'][0][1]}; falsos positivos: {run['confusion_matrix'][1][0]}.",
                f"Distribuição completa: {class_counts['Maligno']} malignos e {class_counts['Benigno']} benignos. Acurácia isolada não descreve os erros por classe; a árvore pode variar entre amostras e não garante desempenho externo.",
            ]),
            ("7. Conclusão", [
                "Interprete acurácia junto com matriz de confusão, precisão, recall e F1-score. Resultados deste dataset não asseguram generalização para outras populações, equipamentos ou contextos.",
            ]),
            ("8. Anexo: código-fonte e execução", [
                "A implementação completa está em backend/app/training.py e é compartilhada pela API e por projeto_ap2.py. Instruções de instalação e execução estão no README.md.",
            ]),
        ]
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Resultados da execução | AP2", fontsize=16, fontweight="bold", color="#15345b", y=0.97)
        y = 0.92
        for heading, paragraphs in sections:
            fig.text(0.09, y, heading, fontsize=10, fontweight="bold", color="#15345b", va="top")
            y -= 0.027
            for paragraph in paragraphs:
                for line in textwrap.wrap(paragraph, width=104, break_long_words=False, break_on_hyphens=False):
                    fig.text(0.09, y, line, fontsize=8.2, color="#26384c", va="top")
                    y -= 0.016
                y -= 0.004
            y -= 0.008
        fig.text(0.09, 0.055, "Finalidade acadêmica. Não utilizar para diagnóstico médico.", fontsize=8, color="#a34b47")
        pdf.savefig(fig)
        plt.close(fig)

        chart_titles = {
            "confusion-matrix.png": "Matriz de confusão",
            "cross-validation.png": "Validação cruzada",
            "class-distribution.png": "Distribuição das classes",
            "metrics.png": "Métricas do modelo",
        }
        chart_paths = [RUNS_DIR / run_id / "charts" / name for name in chart_titles]
        for chart_path in chart_paths:
            image = plt.imread(chart_path)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.5, 0.93, chart_titles[chart_path.name], ha="center", fontsize=14, fontweight="bold", color="#15345b")
            ax = fig.add_axes([0.06, 0.18, 0.88, 0.68])
            ax.imshow(image)
            ax.axis("off")
            fig.text(0.09, 0.055, "Finalidade acadêmica. Não utilizar para diagnóstico médico.", fontsize=8, color="#a34b47")
            pdf.savefig(fig)
            plt.close(fig)
    out.seek(0)
    return StreamingResponse(out, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=relatorio-ap2-{run_id[:8]}.pdf"})


@app.get("/api/code.py")
def code_source():
    path = Path(__file__).with_name("training.py")
    source = path.read_text(encoding="utf-8")
    return Response(source, media_type="text/x-python; charset=utf-8", headers={"Content-Disposition": "attachment; filename=ml_classifier_training.py"})


@app.get("/api/code")
def code_json():
    path = Path(__file__).with_name("training.py")
    return {"filename": "training.py", "language": "python", "source": path.read_text(encoding="utf-8")}
