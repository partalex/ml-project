from pathlib import Path

from pandas import DataFrame, Series

DATA_PATH: Path = Path("../data/bank-additional/bank-additional-full.csv")
DATA_SMALL: Path = Path("../data/bank/bank-full.csv")
OUT_PATH: str = "../out"
GAMMA: float = 2 ** -0.78
C_REGULARIZATION: float = 3.0


def split_features_target(df: DataFrame) -> tuple[DataFrame, Series]:
    """
    Splits the input DataFrame into features and target variable.
    Args:
        df (DataFrame): The input DataFrame containing the features and target variable.
    Returns:
        tuple: A tuple containing two elements:
            - features (DataFrame): A DataFrame containing all columns except the target variable 'y
            - y (Series): A Series containing the target variable 'y' mapped to binary values (0 for 'no' and 1 for 'yes').
    """
    features: DataFrame = df.drop(columns=["y"])
    return features, df["y"].map({"no": 0, "yes": 1})


def get_column_types(features: DataFrame) -> tuple[list[str], list[str]]:
    """
    Determines the categorical and numerical columns in the input DataFrame.
    Args:
        features (DataFrame): The input DataFrame containing the features.
    Returns:
        tuple: A tuple containing two lists:
            - categorical_cols (list[str]): A list of column names that are of type 'object' (categorical features).
            - numerical_cols (list[str]): A list of column names that are not of type 'object' (numerical features).
    """
    categorical_cols: list[str] = features.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols: list[str] = features.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    return categorical_cols, numerical_cols
