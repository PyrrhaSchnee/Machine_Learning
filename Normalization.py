#!/usr/bin/env python3
"""
Normalization.py

Normalization of data instead of standardization

Formula:
x_norm = (x - min(x)) / (max(x) - min(x))
"""

from __future__ import annotations
import sys
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_FILE: str = "Train_knight.csv"
TEST_FILE: str = "Test_knight.csv"


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Take a string or a Path object, try to access it
    """
    try:
        result: pd.DataFrame = pd.read_csv(path)
        return result
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


def retrieve_xy(
    data: pd.DataFrame, x_name: str, y_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the selected columns from the data, remvoe null rows
    Return a tuple of data in numpy.ndarray format
    """
    x_raw: np.ndarray = data[x_name].to_numpy(dtype=float)
    y_raw: np.ndarray = data[y_name].to_numpy(dtype=float)
    mark_null = np.logical_not(np.logical_or(np.isnan(x_raw), np.isnan(y_raw)))
    return (x_raw[mark_null], y_raw[mark_null])


def plot_scatter(
    ax,
    data: pd.DataFrame,
    x_name: str,
    y_name: str,
    title: str,
    label_col: str | None,
) -> None:
    """
    Plot one single scatter image
    """
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    # ax.tick_params(axis="both")
    if label_col is None:
        x, y = retrieve_xy(data, x_name, y_name)
        ax.scatter(x, y, alpha=0.5, label="knight")
        ax.legend()
        return
    labels = data[label_col].astype(str)
    unique = sorted(labels.dropna().unique().tolist())
    colours: list[str] = ["red", "skyblue", "pink", "orange"]
    i: int = 0
    while i < len(unique):
        label = unique[i]
        x, y = retrieve_xy(data[labels == label], x_name, y_name)
        ax.scatter(
            x, y, alpha=0.5, label=label, color=colours[i % len(colours)]
        )
        i += 1
    ax.legend()


def plot_points(
    data_train: pd.DataFrame,
    data_test: pd.DataFrame,
    train_file: str,
    test_file: str,
) -> None:
    """
    Plot the 4 graphs inside a single figure:
    [0,0] train_file separated
    [0,1] train_file mixed
    [1,0] test_file separated
    [1,1] test_file mixed

    Save the result
    """
    # Check if required features are present
    cols_train: tuple[str, str, str, str, str] = (
        "knight",
        "Empowered",
        "Prescience",
        "Push",
        "Midi-chlorien",
    )
    cols_test: tuple[str, str, str, str] = (
        "Empowered",
        "Prescience",
        "Push",
        "Midi-chlorien",
    )
    i: int = 0
    while i < len(cols_train):
        if cols_train[i] not in data_train.columns:
            print(
                f"Error: expected column {cols_train[i]} absent in {train_file}. Abort",
                file=sys.stderr,
            )
            sys.exit(4)
        i += 1
    i = 0
    while i < len(cols_test):
        if cols_test[i] not in data_test.columns:
            print(
                f"Error: expected column {cols_test[i]} absent in {test_file}. Abort",
                file=sys.stderr,
            )
            sys.exit(5)
        i += 1

    (x_sep, y_sep) = ("Push", "Midi-chlorien")
    (x_mix, y_mix) = ("Empowered", "Prescience")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    plot_scatter(
        axes[0, 0],
        data_train,
        x_sep,
        y_sep,
        f"Separated: {x_sep} vs {y_sep}",
        "knight",
    )
    plot_scatter(
        axes[0, 1],
        data_train,
        x_mix,
        y_mix,
        f"Mixed: {x_mix} vs {y_mix}",
        "knight",
    )
    plot_scatter(
        axes[1, 0],
        data_test,
        x_sep,
        y_sep,
        f"Separated: {x_sep} vs {y_sep}",
        None,
    )
    plot_scatter(
        axes[1, 1], data_test, x_mix, y_mix, f"Mixed: {x_mix} vs {y_mix}", None
    )
    plt.savefig("points.png", dpi=300)
    plt.show()
    plt.close()


def normalize(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Formula (for each feature):
    x_norm = (x - min(x)) / (max(x) - min(x))

    NaN values are ignored while computing mean and std
    If max(x) == min(x) then its x_norm = 0.0
    """
    result: pd.DataFrame = data.copy()
    i: int = 0
    while i < len(features):
        col: str = features[i]
        x: np.ndarray = result[col].to_numpy(dtype=float)
        x_min: float = x.min()
        x_max: float = x.max()
        if x_min == x_max:
            value: np.ndarray = np.full(x.shape, np.nan, dtype=float)
            mark_null = np.logical_not(np.isnan(value))
            value[mark_null] = 0.0
        else:
            value = (x - x_min) / (x_max - x_min)
        result[col] = value
        i += 1

    return result


def retrieve_features(data: pd.DataFrame) -> list[str]:
    """
    Retrieve the feature string besides "knight"
    """
    cols: list[str] = data.columns.to_list()
    result: list[str] = []
    i: int = 0
    while i < len(cols):
        if cols[i] != "knight":
            result.append(cols[i])
        i += 1
    return result


def pretty_print(
    data: pd.DataFrame, features: list[str], filename: str
) -> None:
    """
    Pretty print the normalized values
    """
    print(
        f"\n----------BEGINNING OF OUTPUT------------Normalized values of the file {filename}:"
    )
    features_new = [f"{word:>{max(len(word),5)}}" for word in features]
    print(" |".join(features_new))
    values: np.ndarray = data[features].to_numpy(dtype=float)
    i: int = 0
    while i < values.shape[0]:
        row: list[str] = []
        j: int = 0
        while j < values.shape[1]:
            if np.isnan(float(values[i, j])):
                row.append("NaN")
            else:
                row.append(
                    f"{float(values[i,j]):>{max(len(features[j]),5)}.2f}"
                )
            j += 1
        print(" |".join(row))
        i += 1
    print(
        f"----------END OF OUTPUT------------Normalized values of the file {filename}\n"
    )


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
                f"Usage: ./{sys.argv[0]} <path to Train_knight.csv> <Test_knight.csv>",
                file=sys.stderr,
            )
            return 3
        data_train: pd.DataFrame = load_csv(train_file)
        data_test: pd.DataFrame = load_csv(test_file)
        features_train = retrieve_features(data_train)
        features_test = retrieve_features(data_test)
        data_train_normal = normalize(data_train, features_train)
        data_test_normal = normalize(data_test, features_test)
        pretty_print(data_train_normal, features_train, train_file)
        pretty_print(data_test_normal, features_test, test_file)
        # plot_points(data_train, data_test, train_file, test_file)
        (x_sep, y_sep) = ("Push", "Midi-chlorien")
        (x_mix, y_mix) = ("Empowered", "Prescience")
        fig, axes = plt.subplots(
            1, 1, figsize=(16, 9), constrained_layout=True
        )
        plot_scatter(
            axes,
            data_train_normal,
            x_sep,
            y_sep,
            f"Separated: {x_sep} vs {y_sep}",
            "knight",
        )
        plt.savefig("normalizaed.png", dpi=300)
        plt.show()
        plt.close()
        return 0
    except Exception as e:
        print(f"Error: {e}. Exit", file=sys.stderr)
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
