import numpy as np
import seaborn as sns
from typing import Any
from pandas import DataFrame, Series
from matplotlib import pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc

from util import OUT_PATH


def plot_roc_curve_svm(model: Any, features_test: DataFrame, y_test: Series) -> None:
    """
    Plots the ROC curve for a Support Vector Machine (SVM) model.
    Args:
        model (Any): The trained SVM model that has a predict_proba method.
        features_test (DataFrame): The test set features.
        y_test (Series): The true binary labels for the test set.
    """
    y_score = model.predict_proba(features_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
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

    plt.xlabel("Percentage of Population")
    plt.ylabel("Cumulative Lift")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/lift_curve_svm.png", dpi=300)
    plt.show()
    plt.close()


def plot_confusion_matrix(
        y_test: Series,
        y_predicted: np.ndarray,
) -> None:
    """
    Plots the confusion matrix for the SVM model's predictions.
    Args:
        y_test (Series): The true binary labels for the test set.
        y_predicted (np.ndarray): The predicted binary labels from the SVM model.
    """
    cm = confusion_matrix(y_test, y_predicted)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – SVM (RBF)")
    plt.savefig(f"{OUT_PATH}/confusion_matrix_svm.png", dpi=300)
    plt.show()
    plt.close()


def plot_svm_rbf_2d_projection(
        features: np.ndarray,
        y: np.ndarray,
        title: str = "RBF SVM – 2D PCA Projection",
        max_samples: int = 5000,
) -> None:
    """
    Plots the decision boundaries of an RBF SVM model in a 2D PCA projection of the feature space.
    Args:
        features: The preprocessed feature matrix (after ColumnTransformer).
        y: The target labels corresponding to the features.
        title: The title for the plot.
        max_samples: Maximum number of samples to use for fitting the 2D SVM (for speed).
    """
    from sklearn.utils import resample

    # 1. PCA projection to 2D
    # PCA is necessary to visualize SVM in 2D space, since the original feature space has ~63 dimensions.
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)

    # 2. Subsample for speed (full dataset can be 40k+ rows)
    if len(features_2d) > max_samples:
        features_2d, y = resample(
            features_2d, y,
            n_samples=max_samples,
            random_state=42,
            stratify=y,
        )

    # 3. New SVM with reasonable parameters for 2D space
    #    The original C and gamma were tuned for ~63-dim space and don't make sense in 2D.
    vis_svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=42,
    )
    vis_svm.fit(features_2d, y)

    # Add a margin around the data points for better visualization of the decision boundary
    margin = 0.05
    x_range = features_2d[:, 0].max() - features_2d[:, 0].min()
    y_range = features_2d[:, 1].max() - features_2d[:, 1].min()
    x_min = features_2d[:, 0].min() - margin * x_range
    x_max = features_2d[:, 0].max() + margin * x_range
    y_min = features_2d[:, 1].min() - margin * y_range
    y_max = features_2d[:, 1].max() + margin * y_range

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = vis_svm.predict(grid).reshape(xx.shape)

    cmap_bg = plt.cm.RdYlBu
    cmap_pts = plt.cm.RdYlBu

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg)

    scatter = plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=y,
        cmap=cmap_pts,
        s=8,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.2,
    )

    plt.colorbar(scatter, label="Class (0 = no, 1 = yes)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_PATH}/svm_rbf_2d_projection.png", dpi=300)
    plt.show()
    plt.close()
