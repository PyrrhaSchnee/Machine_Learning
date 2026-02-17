#!/usr/bin/env python3
"""
Tree.py

Train a decision tree classifier OR a random forest classfier
Display the tree in a graph
The program takes Train_knight.csv as 1st argument,
Test_knight.csv as 2nd argument, outputs a Tree.txt file
as the result of prediction

F1 at least 90%;
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

TRAIN_FILE: str = "Train_knight.csv"
TEST_FILE: str = "Test_knight.csv"
RANDOM_SEED: int = 56


def load_csv(path: str) -> pd.DataFrame:
    """
    load the csv file
    """
    ret = pd.read_csv(path)
    return ret


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


def model_init() -> RandomForestClassifier:
    """
    Build a RandomForestClassifier with a static random seed
    so the result is reproductible
    """
    return RandomForestClassifier(
        n_estimators=400,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
    )


def get_f1macro(
    x: pd.DataFrame, y: pd.Series, model: RandomForestClassifier
) -> float:
    """
    Compute the F1 score avg=macro
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    macro_f1: float = float(f1_score(y_val, y_pred, average="macro"))
    print(f"[INFO] macro F1 ~= {macro_f1:.4f}")
    return macro_f1


def ft_plot_tree(
    model: RandomForestClassifier, feature_names: list[str]
) -> None:
    """
    Plot and save one tree from the forest
    """
    if not hasattr(model, "estimators_") or not model.estimators_:
        raise ValueError("model lacks estimators, impossible to plot")
    tree = model.estimators_[0]
    fig = plt.figure(figsize=(16, 9))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=[str(x) for x in getattr(model, "classes_", [])],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("Tree.png", dpi=300)
    plt.show()
    plt.close(fig)


def prediction(
    model: RandomForestClassifier, data: pd.DataFrame, feature_names: list[str]
) -> None:
    """
    Do the prediction with the trained model;
    Save the result into Tree.txt;
    Plot the figure of the decision tree.
    """
    col_mismatch = [x for x in feature_names if x not in data.columns]
    if col_mismatch:
        raise ValueError(f"test csv lacks this column {col_mismatch}")
    x_test = data.copy()
    y_pred = model.predict(x_test)
    with open("Tree.txt", "w", encoding="utf-8") as f:
        for x in y_pred:
            f.write(f"{x}\n")
    ft_plot_tree(model, feature_names)


def main() -> int:
    """
    main function
    """
    try:
        train_file: str = TRAIN_FILE
        test_file: str = TEST_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 3:
            train_file = sys.argv[1]
            test_file = sys.argv[2]
        else:
            print(
                f"Usage: {sys.argv[0]} <path to train file> <path to test file>",
                file=sys.stderr,
            )
            sys.exit(2)
        train_data: pd.DataFrame = load_csv(train_file)
        test_data: pd.DataFrame = load_csv(test_file)
        x, y, feature_names = split_features(train_data, "knight")
        model: RandomForestClassifier = model_init()
        f1_macro: float = get_f1macro(x, y, model)
        if f1_macro < 0.9:
            print(
                "[WARNING] macro F1 is 0.9 on the internal split\nConsider increasing n_estimators or adjusting tree constraints"
            )
        # fit the model
        model = model_init()
        model = model.fit(x, y)
        prediction(model, test_data, feature_names)
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
