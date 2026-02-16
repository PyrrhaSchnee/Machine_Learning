#!/usr/bin/env python3
"""
Correlation.py

Compute the Pearson correlation coefficient
between Knight (set to 1.0) and all other
features and display the result in text format
"""

from __future__ import annotations
import sys
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_FILE: str = "Train_knight.csv"


def get_mean(values: np.ndarray) -> float:
    """
    get the mean value of an object of numpy.ndarray type
    """


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Take a string or a Path object, try to access it
    """
    try:
        result: pd.DataFrame = pd.read_csv(path)
        return result
    except Exception as e:
        print(f"Fatal Error: {e}. Abort", file=sys.stderr)
        sys.exit(2)


def compute_pcc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient:
    corr(x, y) = cov(x, y) / (std(x) * std(y))

    If division by zero: return 0.0 directly
    """
    if x.size != y.size or x.size == 0:
        return 0.0
    mean_x: float = float(np.mean(x))
    mean_y: float = float(np.mean(y))
    dx = x - mean_x
    dy = y - mean_y
    # de = denominator
    de: float = float(np.sqrt(np.sum(dx * dx)) * np.sqrt(np.sum(dy * dy)))
    if de == 0.0:
        return 0.0
    return float(np.sum(dx * dy) / de)


def knight_to_num(input: pd.Series) -> np.ndarray:
    """
    Convert text string of knight into number

    Jedi = 1
    Sith = 0

    (The opposite of the above is also possible)
    """
    values = input.astype(str)
    unique = sorted(values.dropna().unique().tolist())
    if unique != ["Jedi", "Sith"]:
        print(
            f'Error: unexpected knight labels: {unique}.\nOnly ["Jedi", "Sith"] is expected.\nAbort',
            file=sys.stderr,
        )
        sys.exit(4)
    return (values == "Jedi").to_numpy(dtype=float)


def pcc(data: pd.DataFrame) -> None:
    """
    Display the result
    """
    if "knight" not in data.columns:
        print(
            f"Error: column 'knight' must be present in the input file. Abort",
            file=sys.stderr,
        )
        sys.exit(3)
    # target_raw : knight in its original content : Jedi or Sith
    target_raw = data["knight"]
    # target_num : convert Jedi and Sith into numbers (we cannot compute
    # anything with a string)
    target_num = knight_to_num(target_raw)
    cols = list(data.columns)
    result: list[tuple[str, float]] = []
    i: int = 0
    while i < len(cols):
        col = cols[i]
        if col != "knight":
            x = data[col].to_numpy(dtype=float)
            # mark_null is a special list of Boolean, here the list
            # doesn't mean the intrinsic basic type 'list' of python3
            # but means the numpy's implementation of advanced indexing
            # when used together with another numpy.ndarray type stuff
            # It can removes all elements where mark_null identifies it
            # as false
            mark_null = np.logical_not(np.isnan(x))
            x2 = x[mark_null]
            y2 = target_num[mark_null]
            coef = compute_pcc(x2, y2)
            result.append((col, coef))
        i += 1
    result.sort(key=lambda kv: abs(kv[1]), reverse=True)
    print(f"{'knight':<14} {1.000000:>.6f}")
    i = 0
    while i < len(result):
        name, value = result[i]
        print(f"{name:<14} {abs(value):>.6f}")
        i += 1


def main() -> int:
    """
    main function
    """
    try:
        path: str = TRAIN_FILE
        if len(sys.argv) > 2:
            print(f"Usage: {sys.argv[0]} <path to csv file>", file=sys.stderr)
            return 1
        if len(sys.argv) == 2:
            path = sys.argv[1]
        data: pd.DataFrame = load_csv(path)
        pcc(data)
        return 0
    except Exception as e:
        print(f"Error: {e}. Exit", file=sys.stderr)
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
