#!/usr/bin/env python3
"""
KNN.py

KNN: K-nearest neighbours;

Use a set of validation to compare different k values' effects, display
a figure for this;
F1-score at least 92%

For that set of validation:
1) if the 3rd cli argument exists, use it for validation,
2) if not, check if "Validation_knight.csv" exists and use it,
3) when neither of the above is possible, split the input training data
so we can use it for validaiton
"""

from __future__ import annotations
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
import traceback
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TRAIN_FILE: str = "Train_knight.csv"
TEST_FILE: str = "Test_knight.csv"
VALID_FILE: str = "Validation_knight.csv"
RANDOM_SEED: int = 561


def load_csv(path: str) -> pd.DataFrame:
    """
    load the csv file
    """
    ret = pd.read_csv(path)
    return ret


def find_validation_file(f: str) -> str | None:
    """
    Try to find that file used for validation
    """
    if os.path.isfile(f):
        return f
    if os.path.isfile(VALID_FILE):
        return VALID_FILE
    return None


def split_features(
    data: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split the data into:
    - x (feature matrix) = all numeric columns
    - y (target labels) = knight column
    - feature_names (ordered list of feature names)

    Return:
    tuple (x, y, feature_names)
    """
    if target not in data.columns:
        raise ValueError("No knight column detected in the input csv file")
    feature_names: list[str] = [x for x in data.columns if x != target]
    x: pd.DataFrame = data[feature_names].copy()
    y: pd.Series = data[target].astype(str).copy()
    return (x, y, feature_names)


def enforce_feature_order(data: pd.DataFrame, ref: list[str]) -> pd.DataFrame:
    """
    Enforce that the test_data has exactly the same feature order as that
    of train_data

    If train_data contains : [Feature1, Feature2, Feature3],
    then the test_data must also has all the above features in that order
    """
    i: int = 0
    size: int = len(ref)
    if len(data.columns.tolist()) != size:
        raise ValueError(
            "The train_data and test_data do not share exactly the same number of features"
        )
    while i < size:
        if data.columns.tolist()[i] not in ref:
            raise ValueError(
                f"The featuer {data.columns.tolist()[i]} of the test_data doesn't exist in train_data"
            )
        else:
            i += 1
    i = 0
    while i < size:
        if ref[i] not in data.columns.tolist():
            raise ValueError(
                f"The feature {ref[i]} of train_data doesn't exist in test_data"
            )
        else:
            i += 1
    ret = data[ref].copy()
    return ret


def replace_empty_by_train_median(
    x_train: pd.DataFrame, others: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Replace empty/NaN by training-set median for each feature.

    x_train : training_set
    others : all other dataset(validation, test) to be processed
    medians : per-feature medians calculated using x_train
    """
    medians: pd.Series = x_train.median(numeric_only=True)
    x_train_clean: pd.DataFrame = x_train.fillna(medians)
    others_clean: list[pd.DataFrame] = [
        item.fillna(medians) for item in others
    ]
    return (x_train_clean, others_clean)


def zscore_fit(x_train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute z-score parameters from training dataset

    z = (x - x_mean) / std;
    if std == 0 return 1;
    """
    means: pd.Series = x_train.mean(numeric_only=True)
    stds: pd.Series = x_train.std(numeric_only=True, ddof=0).replace(0.0, 1.0)
    return (means, stds)


def zscore_transform(
    data: pd.DataFrame, mean: pd.Series, std: pd.Series
) -> pd.DataFrame:
    """
    Standardization of z-score
    """
    x_std: pd.DataFrame = (data - mean) / std
    return x_std


def pca_fit(data: pd.DataFrame, var_threshold: float) -> PCA:
    """
    Fit PCA On standardized training features
    """
    pca: PCA = PCA(n_components=var_threshold)
    pca.fit(data)
    return pca


def pca_transform(x_std: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    """
    Transform standardized features into PCA component space;

    pca_compo: numpy array of PCA components(n_samples, n_components)
    x_pca: DataFrame with columns like PC1, PC2, ... PCn;
    """
    pca_compo: np.ndarray = pca.transform(x_std)
    columns: list[str] = [f"PC{i + 1}" for i in range(pca_compo.shape[1])]
    x_pca: pd.DataFrame = pd.DataFrame(
        pca_compo, columns=columns, index=x_std.index
    )
    return x_pca


def compute_k_effect(
    k: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[float, float, float]:
    """
    Fit a KNN model for a given k and compute associated validation metrics;

    k: number of neighbour
    y_pred: predicted labels on validation set

    Return:
    tuple(macro_precision,macro_f1,accuracy)
    """
    knn_model: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=k, weights="distance", metric="minkowski", p=2
    )
    knn_model.fit(x_train, y_train)
    y_pred: np.ndarray = knn_model.predict(x_valid)
    macro_precision: float = float(
        precision_score(y_valid, y_pred, average="macro", zero_division=0)
    )
    macro_f1: float = float(f1_score(y_valid, y_pred, average="macro"))
    accuracy: float = float(accuracy_score(y_valid, y_pred))
    return (macro_precision, macro_f1, accuracy)


def compute_k(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    k_max: int,
) -> pd.DataFrame:
    """
    For each int from 1 to k_max, compute its effect and return
    them in a table;

    k_effect: for a given k, the associated list of [k, macro_precision, marcro_F1, accuracy]
    result:pd.DataFrame
    """
    k_limit: int = int(max(1, min(k_max, len(x_train))))
    k_effect: list[tuple[int, float, float, float]] = []

    k: int = 1
    while k < k_limit + 1:
        macro_precision, macro_f1, accuracy = compute_k_effect(
            k, x_train, y_train, x_valid, y_valid
        )
        k_effect.append((k, macro_precision, macro_f1, accuracy))
        k += 1
    result: pd.DataFrame = pd.DataFrame(
        k_effect, columns=["k", "macro_precision", "macro_f1", "accuracy"]
    )
    return result


def get_best_k(result: pd.DataFrame) -> int:
    """
    get the best k which has highest macro_f1

    returns the row
    """
    ret = result.sort_values(
        by=["macro_f1", "k"], ascending=[False, True]
    ).iloc[0]
    return int(ret["k"])


def plot_knn(result: pd.DataFrame, best_k: int) -> None:
    """
    Plot the image: Accuracy depends on k
    """
    plt.figure(figsize=(16, 9))
    plt.plot(result["k"], result["macro_precision"] * 100.0, marker="o")
    plt.plot(result["k"], result["macro_f1"] * 100.0, marker="o")
    plt.axvline(best_k, color="r", linestyle="--", label=f"best k: {best_k}")
    plt.title("KNN: metrics vs k (validation)")
    plt.xlabel("k (number of neighbours)")
    plt.ylabel("score (%)")
    plt.xticks(result["k"].tolist())
    plt.yticks(range(94, 100))
    print(result["k"].tolist())
    plt.tight_layout()
    plt.savefig("KNN.png", dpi=300)
    plt.show()
    plt.close()


def predict(
    x_train_pca: pd.DataFrame,
    y_train_final: pd.Series,
    x_test_pca: pd.DataFrame,
    best_k: int,
) -> None:
    """
    Predict and save result
    """
    if best_k < 1 or best_k > len(x_train_pca):
        raise ValueError(
            f"Invalid k: {best_k} for training size {len(x_train_pca)}"
        )
        sys.exit(7)
    model: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=best_k, weights="distance", metric="minkowski", p=2
    )
    model.fit(x_train_pca, y_train_final)
    y_pred: np.ndarray = model.predict(x_test_pca)
    i: int = 0
    with open("KNN.txt", "w", encoding="utf-8") as f:
        while i < len(y_pred):
            f.write(f"{y_pred[i]}\n")
            i += 1


def main() -> int:
    """
    main function
    """
    try:
        train_file: str = TRAIN_FILE
        test_file: str = TEST_FILE
        valid_file: str | None = VALID_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 4:
            train_file = sys.argv[1]
            test_file = sys.argv[2]
            valid_file = sys.argv[3]
        elif len(sys.argv) == 3:
            train_file = sys.argv[1]
            test_file = sys.argv[2]
        else:
            print(
                f"Usage: {sys.argv[0]} <path to train file> <path to test file> <optional path to validation file>",
                file=sys.stderr,
            )
            sys.exit(2)
        train_data: pd.DataFrame = load_csv(train_file)
        test_data: pd.DataFrame = load_csv(test_file)
        valid_file = find_validation_file(valid_file)
        # 1) same as Tree.py, split the train_data into 3 variables
        x_train, y_train, feature_names = split_features(train_data, "knight")
        # print(f"[DEBUG]:\nx_train:\n{x_train}\ny_train:\n{y_train}")
        # 2) (optional) Enforce that test_data having the same feature order
        # as train_data
        test_data = enforce_feature_order(test_data, feature_names)
        # 3) Create validation set
        if valid_file is None:
            print("[INFO]: No validation file provided, creating one...")
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=RANDOM_SEED,
                stratify=y_train,
            )
            # #print(
            #     f"[DEBUG]:\nx_train:\n{x_train}\nx_val:\n{x_valid}\ny_train:\n{y_train}\ny_val:\n{y_valid}"
            # )
        else:
            valid_data: pd.DataFrame = load_csv(valid_file)
            x_valid, y_valid, _ = split_features(valid_data, "knight")
            x_valid = enforce_feature_order(x_valid, feature_names)
        # 4) Replace missing fields by median of training set
        x_train_clean, others_clean = replace_empty_by_train_median(
            x_train, [x_valid, test_data]
        )
        x_valid_clean = others_clean[0]
        x_test_clean = others_clean[1]
        # 5) Z-score standardization
        means_by_feature: pd.Series
        stds_by_feature: pd.Series
        means_by_feature, stds_by_feature = zscore_fit(x_train_clean)
        x_train_std: pd.DataFrame = zscore_transform(
            x_train_clean, means_by_feature, stds_by_feature
        )
        x_valid_std: pd.DataFrame = zscore_transform(
            x_valid_clean, means_by_feature, stds_by_feature
        )
        x_test_std: pd.DataFrame = zscore_transform(
            x_test_clean, means_by_feature, stds_by_feature
        )
        # 6) PCA analysis, keep only those variance >= 90%
        pca_model: PCA = pca_fit(x_train_std, var_threshold=0.90)
        x_train_pca: pd.DataFrame = pca_transform(x_train_std, pca_model)
        x_valid_pca: pd.DataFrame = pca_transform(x_valid_std, pca_model)
        x_test_pca: pd.DataFrame = pca_transform(x_test_std, pca_model)
        print(
            f"[INFO]: PCA kept {x_train_pca.shape[1]} components (>= 90% varirance)"
        )
        # 7） Compute the most suitable value of k
        result: pd.DataFrame = compute_k(
            x_train_pca, y_train, x_valid_pca, y_valid, k_max=50
        )
        best_k: int = get_best_k(result)
        best_row = result[result["k"] == best_k].iloc[0]

        print(f"best k value is: {best_k}")
        print(f"macro_precision is: {best_row['macro_precision'] * 100.0:.2f}")
        print(f"macro_f1 is: {best_row['macro_f1'] * 100.0:.2f}")
        print(f"accuracy is: {best_row['accuracy'] * 100.0:.2f}")
        plot_knn(result, best_k)
        # 8) Final training
        if valid_file is None:
            x_train_final = pd.concat([x_train_clean, x_valid_clean], axis=0)
            y_train_final = pd.concat([y_train, y_valid], axis=0)
            x_train_std = zscore_transform(
                x_train_final, means_by_feature, stds_by_feature
            )
            x_train_pca = pca_transform(x_train_std, pca_model)
        else:
            x_train_final = x_train_pca
            y_train_final = y_train

        predict(x_train_pca, y_train_final, x_test_pca, best_k)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        tb = traceback.TracebackException.from_exception(
            e, capture_locals=True
        )
        print("".join(tb.format()), file=sys.stderr)
        raise
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
