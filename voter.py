#!/usr/bin/env python3
"""
voter.py

Voting Classifier: combine three different classifiers
(RandomForestClassifier, KNeighborsClassifier, GradientBoostingClassifier)
into a single VotingClassifier using soft voting.

The program takes Train_knight.csv as 1st argument,
Test_knight.csv as 2nd argument, outputs a Voting.txt file
as the result of prediction.

F1 at least 94%;

Preprocessing pipeline (same as KNN.py):
1) split features from target
2) enforce feature order on test/validation sets
3) impute NaN with training-set median
4) z-score standardization
5) PCA dimensionality reduction (>= 90% variance)
6) build VotingClassifier, evaluate on validation, predict on test
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
import traceback
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import f1_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TRAIN_FILE: str = "Train_knight.csv"
TEST_FILE: str = "Test_knight.csv"
VALID_FILE: str = "Validation_knight.csv"
RANDOM_SEED: int = 56


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


def find_best_k(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    k_max: int,
) -> int:
    """
    Sweep k from 1 to k_max and return the k that maximises
    macro F1 on the validation set.
    """
    k_limit: int = int(max(1, min(k_max, len(x_train))))
    best_k: int = 1
    best_f1: float = -1.0
    k: int = 1
    while k < k_limit + 1:
        knn_model: KNeighborsClassifier = KNeighborsClassifier(
            n_neighbors=k, weights="distance", metric="minkowski", p=2
        )
        knn_model.fit(x_train, y_train)
        y_pred: np.ndarray = knn_model.predict(x_valid)
        macro_f1: float = float(f1_score(y_valid, y_pred, average="macro"))
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_k = k
        k += 1
    print(
        f"[INFO] best k for KNN sub-model: {best_k} (macro F1 ~= {best_f1:.4f})"
    )
    return best_k


def build_voter(best_k: int) -> VotingClassifier:
    """
    Build a soft-voting VotingClassifier from three estimators:
    1) RandomForestClassifier  (same as Tree.py)
    2) KNeighborsClassifier    (same as KNN.py, with tuned k)
    3) GradientBoostingClassifier (third model of choice)
    """
    rf: RandomForestClassifier = RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
    )
    knn: KNeighborsClassifier = KNeighborsClassifier(
        n_neighbors=best_k, weights="distance", metric="minkowski", p=2
    )
    gb: GradientBoostingClassifier = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_SEED,
        subsample=0.8,
    )
    voter: VotingClassifier = VotingClassifier(
        estimators=[("rf", rf), ("knn", knn), ("gb", gb)],
        voting="soft",
        n_jobs=-1,
    )
    return voter


def predict(voter: VotingClassifier, x_test_pca: pd.DataFrame) -> None:
    """
    Predict and write Voting.txt
    """
    y_pred: np.ndarray = voter.predict(x_test_pca)
    i: int = 0
    with open("Voting.txt", "w", encoding="utf-8") as f:
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
        # 1) split the train_data into features and target
        x_train, y_train, feature_names = split_features(train_data, "knight")
        # 2) enforce that test_data has the same feature order as train_data
        test_data = enforce_feature_order(test_data, feature_names)
        # 3) create validation set
        if valid_file is None:
            print("[INFO]: No validation file provided, creating one...")
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_train,
                y_train,
                test_size=0.2,
                random_state=RANDOM_SEED,
                stratify=y_train,
            )
        else:
            valid_data: pd.DataFrame = load_csv(valid_file)
            x_valid, y_valid, _ = split_features(valid_data, "knight")
            x_valid = enforce_feature_order(x_valid, feature_names)
        # 4) replace missing fields by median of training set
        x_train_clean, others_clean = replace_empty_by_train_median(
            x_train, [x_valid, test_data]
        )
        x_valid_clean = others_clean[0]
        x_test_clean = others_clean[1]
        # 5) z-score standardization
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
            f"[INFO]: PCA kept {x_train_pca.shape[1]} components (>= 90% variance)"
        )
        # 7) find the best k for the KNN sub-model
        best_k: int = find_best_k(
            x_train_pca, y_train, x_valid_pca, y_valid, k_max=50
        )
        # 8) build voting classifier and evaluate on validation
        voter: VotingClassifier = build_voter(best_k)
        voter.fit(x_train_pca, y_train)
        y_val_pred: np.ndarray = voter.predict(x_valid_pca)
        macro_f1: float = float(f1_score(y_valid, y_val_pred, average="macro"))
        macro_precision: float = float(
            precision_score(
                y_valid, y_val_pred, average="macro", zero_division=0
            )
        )
        accuracy: float = float(accuracy_score(y_valid, y_val_pred))
        print(f"[INFO] Voting Classifier validation results:")
        print(f"  macro_precision: {macro_precision * 100.0:.2f}%")
        print(f"  macro_f1:        {macro_f1 * 100.0:.2f}%")
        print(f"  accuracy:        {accuracy * 100.0:.2f}%")
        if macro_f1 < 0.94:
            print(
                "[WARNING] macro F1 < 94% on the internal split\nConsider tuning hyper-parameters"
            )
        # 9) final training on full training data, then predict
        if valid_file is None:
            x_train_final = pd.concat([x_train_clean, x_valid_clean], axis=0)
            y_train_final = pd.concat([y_train, y_valid], axis=0)
            x_train_std_final: pd.DataFrame = zscore_transform(
                x_train_final, means_by_feature, stds_by_feature
            )
            x_train_pca_final: pd.DataFrame = pca_transform(
                x_train_std_final, pca_model
            )
        else:
            x_train_pca_final = x_train_pca
            y_train_final = y_train
        voter_final: VotingClassifier = build_voter(best_k)
        voter_final.fit(x_train_pca_final, y_train_final)
        predict(voter_final, x_test_pca)
        print("[INFO] Voting.txt written successfully")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        # tb = traceback.TracebackException.from_exception(
        #     e, capture_locals=True
        # )
        # print("".join(tb.format()), file=sys.stderr)
        # raise
        return 1


if __name__ == "__main__":
    """
    Entrypoint
    """
    raise SystemExit(main())
