import time

from pandas import DataFrame, read_csv

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from util import C_REGULARIZATION, GAMMA
from src.util import split_features_target, DATA_PATH, get_column_types
from plots import plot_svm_rbf_2d_projection, plot_roc_curve_svm, plot_confusion_matrix, plot_lift_curve


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
        C=C_REGULARIZATION,
        gamma=GAMMA,
        # True is required to enable predict_proba for SVC, which is needed for ROC AUC and lift curve calculations.
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


if __name__ == "__main__":
    start_time = time.time()

    data: DataFrame = read_csv(DATA_PATH, sep=";")
    features, labels = split_features_target(data)

    cat_cols, num_cols = get_column_types(features)

    features_train, features_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline(cat_cols, num_cols)

    pipeline.fit(features_train, y_train)

    y_predicted = pipeline.predict(features_test)
    y_proba = pipeline.predict_proba(features_test)[:, 1]

    acc: float = accuracy_score(y_test, y_predicted)
    auc_score: float = roc_auc_score(y_test, y_proba)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {auc_score:.4f}")
    print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes")

    plot_roc_curve_svm(pipeline, features_test, y_test)
    plot_confusion_matrix(y_test, y_predicted)
    plot_lift_curve(y_test.values, y_proba, "Kumulativni Lift – SVM (RBF)")

    # still test phase
    features_transformed = pipeline.named_steps["preprocess"].transform(features_test)
    plot_svm_rbf_2d_projection(
        features=features_transformed,
        y=y_test.values,
    )
