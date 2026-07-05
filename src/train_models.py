"""Training utilities for Epic 4 of Rising Waters.

This module loads the prepared flood dataset, loads the saved scaler,
creates train-test splits, scales the features, and provides reusable
training functions for classification models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "flood_dataset.xlsx"
SCALER_PATH = PROJECT_ROOT / "models" / "transform.save"
BEST_MODEL_PATH = PROJECT_ROOT / "models" / "floods.save"
TARGET_COLUMN = "flood"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the preprocessed flood dataset.

    Args:
        dataset_path: Path to the dataset file.

    Returns:
        Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return pd.read_excel(dataset_path)


def load_scaler(scaler_path: Path = SCALER_PATH):
    """Load the saved StandardScaler object.

    Args:
        scaler_path: Path to the saved scaler file.

    Returns:
        Loaded scaler object.

    Raises:
        FileNotFoundError: If the scaler file does not exist.
    """
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    return joblib.load(scaler_path)


def split_features_and_target(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into independent variables X and target variable y.

    Args:
        dataset: Flood prediction dataset.

    Returns:
        Tuple containing X features and y target.

    Raises:
        KeyError: If the target column is missing.
    """
    if TARGET_COLUMN not in dataset.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found.")

    X = dataset.drop(TARGET_COLUMN, axis=1)
    y = dataset[TARGET_COLUMN]
    return X, y


def prepare_train_test_data():
    """Load data, split it, load the scaler, and scale train/test features.

    Returns:
        Tuple containing X_train_scaled, X_test_scaled, y_train, and y_test.
    """
    dataset = load_dataset()
    scaler = load_scaler()
    X, y = split_features_and_target(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def _print_evaluation_report(model_name: str, y_test, y_pred) -> float:
    """Print standard classification metrics for a trained model.

    Args:
        model_name: Human-readable model name.
        y_test: True labels for the test set.
        y_pred: Predicted labels for the test set.

    Returns:
        Model accuracy score.
    """
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{model_name} Results")
    print("-" * (len(model_name) + 8))
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


def decision_tree(X_train, X_test, y_train, y_test):
    """Train and evaluate a Decision Tree classifier.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Returns:
        Tuple containing the trained model and its accuracy.
    """
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = _print_evaluation_report("Decision Tree", y_test, y_pred)

    return model, accuracy


def random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest classifier.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Returns:
        Tuple containing the trained model and its accuracy.
    """
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = _print_evaluation_report("Random Forest", y_test, y_pred)

    return model, accuracy


def knn(X_train, X_test, y_train, y_test):
    """Train and evaluate a K-Nearest Neighbors classifier.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Returns:
        Tuple containing the trained model and its accuracy.
    """
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = _print_evaluation_report("K-Nearest Neighbors", y_test, y_pred)

    return model, accuracy


def xgboost_model(X_train, X_test, y_train, y_test):
    """Train and evaluate an XGBoost classifier.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Returns:
        Tuple containing the trained model and its accuracy.
    """
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = _print_evaluation_report("XGBoost", y_test, y_pred)

    return model, accuracy


def compare_models(X_train, X_test, y_train, y_test):
    """Train, compare, and save the best-performing classifier.

    The function evaluates Decision Tree, Random Forest, KNN, and XGBoost
    using accuracy. It prints a formatted comparison table, identifies the
    best model, and saves only the best-performing model.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target values.
        y_test: Testing target values.

    Returns:
        Tuple containing the best model name, best model object, best accuracy,
        and a DataFrame with all model results.
    """
    model_functions = [
        ("Decision Tree", decision_tree),
        ("Random Forest", random_forest),
        ("KNN", knn),
        ("XGBoost", xgboost_model),
    ]
    results = []

    for model_name, train_function in model_functions:
        model, accuracy = train_function(X_train, X_test, y_train, y_test)
        results.append(
            {
                "Model Name": model_name,
                "Accuracy": accuracy,
                "Model": model,
            }
        )

    results_table = pd.DataFrame(results)
    display_table = results_table[["Model Name", "Accuracy"]].copy()
    display_table["Accuracy"] = display_table["Accuracy"].map("{:.4f}".format)

    print("\nModel Comparison")
    print("----------------")
    print(display_table.to_string(index=False))

    best_index = results_table["Accuracy"].idxmax()
    best_model_name = results_table.loc[best_index, "Model Name"]
    best_accuracy = results_table.loc[best_index, "Accuracy"]
    best_model = results_table.loc[best_index, "Model"]

    print("\nBest Model:")
    print(best_model_name)
    print("Best Accuracy:")
    print(f"{best_accuracy:.4f}")

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"\nBest model saved successfully at: {BEST_MODEL_PATH}")

    return best_model_name, best_model, best_accuracy, display_table


if __name__ == "__main__":
    X_train_scaled, X_test_scaled, y_train, y_test = prepare_train_test_data()

    compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
