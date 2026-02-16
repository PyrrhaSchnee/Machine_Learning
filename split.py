#!/usr/bin/env python3
"""
split.py

It splits the incoming Train_knight.csv into
2 files, one is intended to be used for training,
the other one for validation.

Intuitively, more data should be used for training.

The spliting is stratified by the column "knight"
(Jedi/Sith) to avoid creating a bias if one kind of knight
is over/underpresented in one of the sub-dataset
"""

from __future__ import annotations
import sys
import os
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

TRAIN_RATIO: float = 0.8
OUTPUT_TRAIN_FILE: str = "Training_knight.csv"
OUTPUT_VALID_FILE: str = "Validation_knight.csv"
TRAIN_FILE: str = "Train_knight.csv"
# TEST_FILE: str = "Test_knight.csv"


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


def shuffle_rows(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Randomly reorganise the order of items of the rows inside data
    """
    i = data.index.to_numpy()
    rng.shuffle(i)
    # .loc means label-based selection of rows/columns
    # .loc[row_selector, column_selector]
    # the type of the 2 selectors can by anything among:
    # str, int, boolean, datatime etc....., the same type of the label
    # i is shuffled, so the data.loc[i] is in a different order
    return data.loc[i]


def get_partition_number(total: int, ratio: float) -> int:
    """
    Compute how many rows should be attributed to training for one label
    If total == 0: 0;
    If total == 1: 1;
    if total >= 2: make 1 <= train_nbr <= total -1
    """
    if ratio <= 0.0:
        return 0
    elif ratio >= 1.0:
        return total - 1
    if total <= 0:
        return 0
    elif total == 1:
        return 1
    else:
        return int(total * ratio)


def split(data: pd.DataFrame, train_file: str) -> None:
    """
    1) Verify if the column "knight" is present
    2) Generate a random seed (this step is not absolutely necessary)
    3) Spliting with stratification
    4) Save the results
    """
    # 1) Verify if the column "knight" is present
    if "knight" not in data.columns:
        print(
            f'Error: the column "knight" is absent in the file {train_file}.\nAbort',
            file=sys.stderr,
        )
        sys.exit(3)
    # 2) Generate a random seed; rng = random number generator
    seed_bytes: bytes = os.urandom(8)
    seed_int = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    rng: np.random.Generator = np.random.default_rng(seed_int)
    # 3) Spliting with stratification
    labels = data["knight"].astype(str)
    unique = sorted(labels.dropna().unique().tolist())
    train_part: list[pd.DataFrame] = []
    valid_part: list[pd.DataFrame] = []
    i: int = 0
    while i < len(unique):
        label = unique[i]
        subset = data[labels == label].copy()
        subset = shuffle_rows(subset, rng)
        total: int = int(subset.shape[0])
        train_nbr = get_partition_number(total, 0.8)
        # iloc means integer(number) based selection of rows/columns
        # always start at position 0 (because it is integer based)
        train_part.append(subset.iloc[:train_nbr])
        valid_part.append(subset.iloc[train_nbr:])
        i += 1
    # axis=0 means add new items line by line (one UNDER another)
    # axis=1 means add new items one NEXT TO another (never use it)
    train_result = pd.concat(train_part, axis=0)
    # drop=True is necessary here because pandas considers the existing
    # column of index is also a data, without drop=True, we will get
    # one excessive column.
    train_result = shuffle_rows(train_result, rng).reset_index(drop=True)
    train_result.to_csv(OUTPUT_TRAIN_FILE, index=False)
    valid_result = pd.concat(valid_part, axis=0)
    valid_result = shuffle_rows(valid_result, rng).reset_index(drop=True)
    valid_result.to_csv(OUTPUT_VALID_FILE, index=False)


def main() -> int:
    """
    main function
    """
    try:
        train_file: str = TRAIN_FILE
        if len(sys.argv) == 1:
            pass
        elif len(sys.argv) == 2:
            train_file = sys.argv[1]
        else:
            print(
                f"Usage: ./{sys.argv[0]} <path to Train_knight.csv>",
                file=sys.stderr,
            )
            return 3
        data_train: pd.DataFrame = load_csv(train_file)
        split(data_train, train_file)
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
