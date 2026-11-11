"""Reproduce the train/test experiments and write experiment_results.json."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    _classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)


ROOT = Path(__file__).resolve().parent


def run() -> dict[str, object]:
    students = pd.read_csv(ROOT / "Student_Performance.csv")
    students["Extracurricular Activities"] = students[
        "Extracurricular Activities"
    ].map({"Yes": 1, "No": 0})
    regression_features = students.drop(columns="Performance Index").to_numpy(dtype=float)
    regression_target = students["Performance Index"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        regression_features,
        regression_target,
        test_size=0.2,
        random_state=42,
    )
    regression_data = RegressionInput(x_train, y_train, x_test, y_test)
    regression_result = train_linear_regression(regression_data)
    regression_mse = float(np.mean((regression_result.predictions - y_test) ** 2))

    german = pd.read_csv(ROOT / "german.csv", sep=";")
    classification_target = german["Creditability"].to_numpy(dtype=int)
    classification_features = german.drop(columns="Creditability").to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        classification_features,
        classification_target,
        test_size=0.2,
        random_state=42,
        stratify=classification_target,
    )
    classification_data = ClassificationInput(x_train, y_train, x_test, y_test)
    own_logistic = train_logistic_classifier(classification_data)
    own_metrics = _classification_metrics(y_test, own_logistic.probabilities)
    baselines = compare_classical_models(classification_data)

    return {
        "linear_regression": {"test_mse": regression_mse},
        "classification": {
            "own_logistic": asdict(own_metrics),
            "logistic_regression": asdict(baselines.logistic_regression),
            "decision_tree": asdict(baselines.decision_tree),
            "knn": asdict(baselines.knn),
        },
    }


if __name__ == "__main__":
    results = run()
    output_path = ROOT / "experiment_results.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
