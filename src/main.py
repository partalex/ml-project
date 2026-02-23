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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix

from src.util import split_features_target, load_data, DATA_PATH, get_column_types, OUT_PATH


def build_pipeline(categorical_cols: list[str], numerical_cols: list[str]) -> Pipeline:
    """
    Builds a machine learning pipeline that preprocesses the data and fits a Support Vector Machine (SVM) model.
    Args:
        categorical_cols (list[str]): A list of column names that are categorical features.
        numerical_cols (list[str]): A list of column names that are numerical features.
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
        C=3.0,
        gamma=0.582366793,
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

    plt.figure(figsize=(7, 5))
    plt.plot(x, lift, label="SVM")
    plt.plot([0.0, 1.0], [1.0, 1.0], linestyle="--", label="Random")

    plt.xlabel("Procenat populacije")
    plt.ylabel("Kumulativni lift")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/lift_curve_svm.png", dpi=300)
    plt.show()
    plt.close()


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

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve – SVM (RBF)")
    plt.legend(loc="lower right")
    plt.savefig(f"{OUT_PATH}/roc_curve_svm.png", dpi=300)
    plt.show()
    plt.close()


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

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – SVM (RBF)")
    plt.savefig(f"{OUT_PATH}/confusion_matrix_svm.png", dpi=300)
    plt.show()
    plt.close()


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

    pipeline = build_pipeline(cat_cols, num_cols)

    pipeline.fit(features_train, labels_train)

    y_pred = pipeline.predict(features_test)
    y_proba = pipeline.predict_proba(features_test)[:, 1]

    acc: float = accuracy_score(labels_test, y_pred)
    auc_score: float = roc_auc_score(labels_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {auc_score:.4f}")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

    plot_roc_curve_svm(pipeline, features_test, labels_test)
    plot_confusion_matrix(labels_test, y_pred)
    plot_lift_curve(labels_test.values, y_proba, "Kumulativni Lift – SVM (RBF)")
