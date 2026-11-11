import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationMetrics,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def _as_features(values: object) -> np.ndarray:
    features = np.asarray(values, dtype=float)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError("Features must be a one- or two-dimensional array")
    return features


def _as_target(values: object) -> np.ndarray:
    target = np.asarray(values, dtype=float).reshape(-1)
    if target.size == 0:
        raise ValueError("Training target must not be empty")
    return target


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-values))


def _classification_metrics(target: np.ndarray, probabilities: np.ndarray) -> ClassificationMetrics:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    target = np.asarray(target).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    if np.unique(target).size < 2:
        roc_auc = 0.5
    else:
        roc_auc = float(roc_auc_score(target, probabilities))
    return ClassificationMetrics(
        roc_auc=roc_auc,
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    train_features = _as_features(data.train_features)
    test_features = _as_features(data.test_features)
    train_target = _as_target(data.train_target)
    if train_features.shape[0] != train_target.size:
        raise ValueError("Training features and target must have the same number of rows")
    if test_features.shape[1] != train_features.shape[1]:
        raise ValueError("Training and test features must have the same number of columns")
    if data.epochs < 0 or data.learning_rate <= 0:
        raise ValueError("epochs must be non-negative and learning_rate must be positive")

    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    normalized_features = (train_features - feature_mean) / feature_scale
    normalized_test = (test_features - feature_mean) / feature_scale

    target_mean = float(train_target.mean())
    target_scale = float(train_target.std())
    if target_scale == 0:
        target_scale = 1.0
    normalized_target = (train_target - target_mean) / target_scale

    weights = np.zeros(train_features.shape[1], dtype=float)
    bias = 0.0
    loss_history: list[float] = []
    sample_count = train_features.shape[0]

    for _ in range(data.epochs):
        errors = normalized_features @ weights + bias - normalized_target
        weights -= data.learning_rate * (2.0 / sample_count) * (normalized_features.T @ errors)
        bias -= data.learning_rate * 2.0 * float(errors.mean())

        predictions = (normalized_features @ weights + bias) * target_scale + target_mean
        loss_history.append(float(np.mean((predictions - train_target) ** 2)))

    original_weights = weights * target_scale / feature_scale
    original_bias = target_mean + target_scale * bias - float(feature_mean @ original_weights)
    test_predictions = test_features @ original_weights + original_bias
    return RegressionResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        predictions=test_predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    train_features = _as_features(data.train_features)
    test_features = _as_features(data.test_features)
    train_target = _as_target(data.train_target)
    if train_features.shape[0] != train_target.size:
        raise ValueError("Training features and target must have the same number of rows")
    if test_features.shape[1] != train_features.shape[1]:
        raise ValueError("Training and test features must have the same number of columns")
    if not np.isin(train_target, [0, 1]).all():
        raise ValueError("Logistic classifier supports binary targets encoded as 0 and 1")
    if data.epochs < 0 or data.learning_rate <= 0:
        raise ValueError("epochs must be non-negative and learning_rate must be positive")

    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    normalized_features = (train_features - feature_mean) / feature_scale
    normalized_test = (test_features - feature_mean) / feature_scale

    weights = np.zeros(train_features.shape[1], dtype=float)
    bias = 0.0
    loss_history: list[float] = []
    sample_count = train_features.shape[0]

    for _ in range(data.epochs):
        probabilities = _sigmoid(normalized_features @ weights + bias)
        errors = probabilities - train_target
        weights -= data.learning_rate * (normalized_features.T @ errors) / sample_count
        bias -= data.learning_rate * float(errors.mean())

        updated_probabilities = np.clip(
            _sigmoid(normalized_features @ weights + bias), 1e-15, 1.0 - 1e-15
        )
        loss = -np.mean(
            train_target * np.log(updated_probabilities)
            + (1.0 - train_target) * np.log(1.0 - updated_probabilities)
        )
        loss_history.append(float(loss))

    probabilities = _sigmoid(normalized_test @ weights + bias)
    original_weights = weights / feature_scale
    original_bias = bias - float(feature_mean @ original_weights)
    return ClassificationResult(
        weights=original_weights,
        bias=float(original_bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=(probabilities >= 0.5).astype(int),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    train_features = _as_features(data.train_features)
    test_features = _as_features(data.test_features)
    train_target = _as_target(data.train_target).astype(int)
    test_target = _as_target(data.test_target).astype(int)

    if train_features.shape[0] != train_target.size:
        raise ValueError("Training features and target must have the same number of rows")
    if test_features.shape[0] != test_target.size:
        raise ValueError("Test features and target must have the same number of rows")

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    tree = DecisionTreeClassifier(random_state=42)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

    logistic.fit(train_features, train_target)
    tree.fit(train_features, train_target)
    knn.fit(train_features, train_target)

    return BaselineComparisonResult(
        logistic_regression=_classification_metrics(test_target, logistic.predict_proba(test_features)[:, 1]),
        decision_tree=_classification_metrics(test_target, tree.predict_proba(test_features)[:, 1]),
        knn=_classification_metrics(test_target, knn.predict_proba(test_features)[:, 1]),
    )
