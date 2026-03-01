from __future__ import annotations

import time
from typing import Any

import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Series

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix

from src.util import split_features_target, load_data, DATA_PATH, get_column_types, OUT_PATH


def build_pipeline(
        categorical_cols: list[str],
        numerical_cols: list[str],
        C: float = 1.0,
        gamma: float | str = "scale",
) -> Pipeline:
    """
    Builds a machine learning pipeline that preprocesses the data and fits a Support Vector Machine (SVM) model.
    Args:
        categorical_cols (list[str]): A list of column names that are categorical features.
        numerical_cols (list[str]): A list of column names that are numerical features.
        C (float): Regularization parameter. Default is 1.0.
        gamma (float | str): Kernel coefficient. Default is "scale".
    Returns:
        Pipeline: A scikit-learn Pipeline object that includes the preprocessing steps and the SVM model.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ]
    )

    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        verbose=True,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def cumulative_lift_curve(
        y_true: np.ndarray,
        y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the cumulative lift curve for a binary classification model.
    Args:
        y_true (np.ndarray): The true binary labels (0 or 1) for the dataset.
        y_score (np.ndarray): The predicted probabilities for the positive class (1) from the model.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
            - x: The cumulative percentage of the population (from 0 to 1).
            - lift: The cumulative lift values corresponding to the population percentages.
    """

    order = np.argsort(-y_score)

    y_sorted = y_true[order]

    positives = np.sum(y_true)

    cum_positives = np.cumsum(y_sorted)

    population = np.arange(1, len(y_true) + 1)

    cum_response_rate = cum_positives / population
    base_rate = positives / len(y_true)

    lift = cum_response_rate / base_rate
    x = population / len(y_true)

    return x, lift


def plot_lift_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        title: str
) -> None:
    """
    Plots the cumulative lift curve for a binary classification model.
    Args:
        y_true (np.ndarray): The true binary labels (0 or 1) for the dataset.
        y_score (np.ndarray): The predicted probabilities for the positive class (1) from the model.
        title (str): The title for the plot.
    """
    x, lift = cumulative_lift_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, lift, label="SVM")
    ax.plot([0.0, 1.0], [1.0, 1.0], linestyle="--", label="Random")

    ax.set_xlabel("Procenat populacije")
    ax.set_ylabel("Kumulativni lift")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_PATH}/lift_curve_svm.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_roc_curve_svm(
        model: Any,
        features_test: DataFrame,
        labels_test: Series
) -> None:
    """
    Plots the ROC curve for a Support Vector Machine (SVM) model.
    Args:
        model (Any): The trained SVM model that has a predict_proba method.
        features_test (DataFrame): The test set features.
        labels_test (Series): The true binary labels for the test set.
    """
    y_score = model.predict_proba(features_test)[:, 1]

    fpr, tpr, _ = roc_curve(labels_test, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve – SVM (RBF)")
    ax.legend(loc="lower right")
    fig.savefig(f"{OUT_PATH}/roc_curve_svm.png", dpi=300)
    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
        y_test: "Series[int]",
        y_pred: "np.ndarray[int]",
) -> None:
    """
    Plots the confusion matrix for the SVM model's predictions.
    Args:
        y_test (Series[int]): The true binary labels for the test set.
        y_pred (np.ndarray[int]): The predicted binary labels from the SVM model.
    """
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – SVM (RBF)")
    fig.savefig(f"{OUT_PATH}/confusion_matrix_svm.png", dpi=300)
    plt.show()
    plt.close(fig)


def grid_search_svm(
        pipeline: Pipeline,
        features_train: DataFrame,
        labels_train: Series,
) -> GridSearchCV:
    """
    Performs grid search with cross-validation to find the best C and gamma parameters
    for the SVM model, following the methodology from the paper:
        γ ∈ {2^-7, 2^-6, ..., 2^8}  and  C ∈ {2^-3, 2^-2, ..., 2^7} (including C=3).
    Args:
        pipeline (Pipeline): The machine learning pipeline with preprocessing and SVM.
        features_train (DataFrame): Training features.
        labels_train (Series): Training labels.
    Returns:
        GridSearchCV: The fitted grid search object with the best parameters.
    """
    # γ ∈ {2^-7, 2^-6, ..., 2^8}  — kao u radu
    gamma_range = [2 ** i for i in range(-7, 9)]
    # C opseg — uključuje C=3 iz rada
    c_range = [2 ** i for i in range(-3, 8)]

    param_grid = {
        "model__C": c_range,
        "model__gamma": gamma_range,
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    grid.fit(features_train, labels_train)

    print(f"\nNajbolji parametri: {grid.best_params_}")
    print(f"Najbolji ROC AUC (CV): {grid.best_score_:.4f}")

    return grid


if __name__ == "__main__":
    start_time = time.time()

    data: DataFrame = load_data(DATA_PATH)
    features, labels = split_features_target(data)

    cat_cols, num_cols = get_column_types(features)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    pipeline_gs = build_pipeline(cat_cols, num_cols)
    grid = grid_search_svm(pipeline_gs, features_train, labels_train)
    best_pipeline = grid.best_estimator_

    y_pred = best_pipeline.predict(features_test)
    y_proba = best_pipeline.predict_proba(features_test)[:, 1]

    acc: float = accuracy_score(labels_test, y_pred)
    auc_score: float = roc_auc_score(labels_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {auc_score:.4f}")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

    plot_roc_curve_svm(best_pipeline, features_test, labels_test)
    plot_confusion_matrix(labels_test, y_pred)
    plot_lift_curve(labels_test.values, y_proba, "Kumulativni Lift – SVM (RBF)")
